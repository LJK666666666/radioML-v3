# GPR 去噪（仅按SNR分组，per-sample谱分解提速）
# 主要优化：
# - 仅按 SNR 与 n 分组，先对无噪声核 K 做一次特征分解；对不同样本噪声用谱域缩放 Λ/(Λ+σ²) 批量计算
# - 结果按原始 (mod, snr) 键拆回，保持每键内样本顺序不变

import time
from typing import Dict, Tuple, List

import numpy as np


# ----------------------------
# 核函数与GPR辅助
# ----------------------------

def calculate_power(i_component: np.ndarray, q_component: np.ndarray) -> float:
    return float(np.mean(i_component ** 2 + q_component ** 2))


def estimate_noise_std(signal_power: float, snr_db: float) -> float:
    snr_linear = 10 ** (snr_db / 10)
    # snr_linear = 10 ** (snr_db / 20)
    noise_power = signal_power / (snr_linear + 1)
    return float(np.sqrt(noise_power / 2))


def length_scale_from_snr(snr_db: float) -> float:
    # 与 origin.py 相同
    return 5.0 if snr_db >= 0 else (5.0 - snr_db * 0.25)
    # return 5.0/4 if snr_db >= 0 else (5.0 - snr_db * 0.25)/4
    # return 10.0


def rbf_kernel_same_grid(n: int, length_scale: float) -> np.ndarray:
    # X = [[0],[1],...,[n-1]]
    idx = np.arange(n, dtype=np.float64)
    # (i-j)^2 矩阵
    d2 = (idx[:, None] - idx[None, :]) ** 2
    ls2 = (length_scale ** 2)
    K = np.exp(-0.5 * d2 / max(ls2, 1e-12))
    return K


def spectral_gp_denoise_same_inputs(
    eigvecs: np.ndarray,   # Q (n,n)
    eigvals: np.ndarray,   # Λ (n,)
    Y: np.ndarray,         # (n,m)
    noise_vars_cols: np.ndarray  # (m,) 每列对应的噪声方差
) -> np.ndarray:
    """
    谱域批处理：mean = Q @ [ (Λ/(Λ+σ_j^2)) * (Q^T Y[:,j]) ]_j
    可同时处理每列具有不同σ²的情形（per-sample）。
    """
    V = eigvecs.T @ Y           # (n,m)
    # 构造缩放矩阵 S = Λ/(Λ+σ²) 做广播
    S = eigvals[:, None] / (eigvals[:, None] + noise_vars_cols[None, :])
    G = S * V
    MU = eigvecs @ G
    return MU


# ----------------------------
# 分组与主流程（按 SNR + n 分组）
# ----------------------------

def group_by_snr_and_length_adaptive(dataset: Dict[Tuple[str, int], np.ndarray],
                                   data_format_2016: bool) -> Dict[Tuple[int, int], List[Tuple[Tuple[str, int], np.ndarray]]]:
    """
    按 (snr, n) 分组，适配不同数据格式，不按调制类型分组，避免泄露调制信息。
    返回: {(snr, n): [(key=(mod,snr), samples), ...], ...}

    Args:
        dataset: 数据集字典
        data_format_2016: True表示2016格式(samples, 2, seq_len)，False表示2018格式(samples, seq_len, 2)
    """
    groups: Dict[Tuple[int, int], List[Tuple[Tuple[str, int], np.ndarray]]] = {}
    for key, samples in dataset.items():
        mod, snr_db = key
        if len(samples) == 0:
            # 仍放入其组，后续直接原样回填
            n = 0
        else:
            if data_format_2016:
                # 2016格式: (samples, 2, seq_len) -> 序列长度是最后一维
                n = int(samples.shape[-1])
            else:
                # 2018格式: (samples, seq_len, 2) -> 序列长度是第二维
                n = int(samples.shape[1])
        gkey = (int(snr_db), n)
        groups.setdefault(gkey, []).append((key, samples))
    return groups


def group_by_snr_and_length(dataset: Dict[Tuple[str, int], np.ndarray]) -> Dict[Tuple[int, int], List[Tuple[Tuple[str, int], np.ndarray]]]:
    """
    按 (snr, n) 分组，不按调制类型分组，避免泄露调制信息。
    返回: {(snr, n): [(key=(mod,snr), samples), ...], ...}
    """
    groups: Dict[Tuple[int, int], List[Tuple[Tuple[str, int], np.ndarray]]] = {}
    for key, samples in dataset.items():
        mod, snr_db = key
        if len(samples) == 0:
            # 仍放入其组，后续直接原样回填
            n = 0
        else:
            n = int(samples.shape[-1])
        gkey = (int(snr_db), n)
        groups.setdefault(gkey, []).append((key, samples))
    return groups


def apply_gpr_denoising_efficient_per_sample(
    dataset: Dict[Tuple[str, int], np.ndarray],
    batch_limit: int = 4096      # 控制单次列数，防止极端情况下内存峰值
) -> Tuple[Dict[Tuple[str, int], np.ndarray], float]:
    """
    per-sample模式GPR去噪：谱分解一次，样本级噪声方差 σ_i^2 用谱域缩放
    返回 (denoised_dataset, total_time)

    支持两种数据格式：
    - 2016格式: dataset[key] 形状: (num_samples, 2, n)
    - 2018格式: dataset[key] 形状: (num_samples, n, 2)
    """

    total_samples = sum(len(v) for v in dataset.values())
    processed = 0

    t0 = time.time()
    denoised_dataset: Dict[Tuple[str, int], np.ndarray] = {}

    print(f'开始per-sample模式GPR去噪（分组仅按 SNR 和序列长度 n），总样本数: {total_samples}')

    # 检测数据格式
    first_key = next(iter(dataset.keys()))
    first_samples = dataset[first_key]
    if len(first_samples.shape) == 3:
        if first_samples.shape[1] == 2:  # (samples, 2, seq_len) - 2016格式
            data_format_2016 = True
            print("检测到2016数据格式: (samples, 2, seq_len)")
        elif first_samples.shape[2] == 2:  # (samples, seq_len, 2) - 2018格式
            data_format_2016 = False
            print("检测到2018数据格式: (samples, seq_len, 2)")
        else:
            raise ValueError(f"无法识别的数据格式: {first_samples.shape}")
    else:
        raise ValueError(f"期望3维数据，得到: {first_samples.shape}")

    # 仅按 (snr, n) 分组
    snr_groups = group_by_snr_and_length_adaptive(dataset, data_format_2016)

    for (snr_db, n), entries in sorted(snr_groups.items(), key=lambda x: x[0]):
        # 汇总该 SNR+n 组的样本（跨所有调制）
        counts = [len(s) for _, s in entries]
        total_in_group = int(sum(counts))
        if total_in_group == 0 or n == 0:
            # 直接回填空
            for key, samples in entries:
                denoised_dataset[key] = samples
            continue

        # 拼接为统一格式 (M, 2, n) 或 (M, n, 2)
        stacked = np.concatenate([s for _, s in entries], axis=0)
        ls = length_scale_from_snr(float(snr_db))
        K = rbf_kernel_same_grid(n, ls)

        print(f'- 处理 SNR={snr_db}dB, n={n}, 合计样本: {total_in_group}, 模块组合数: {len(entries)}')

        # per-sample：谱分解一次，样本级噪声方差 σ_i^2 用谱域缩放
        eigvals, eigvecs = np.linalg.eigh(K)  # SPD
        # 每个样本噪声
        M = total_in_group
        sigmas = np.empty((M,), dtype=np.float64)

        # 根据数据格式计算功率
        for i in range(M):
            if data_format_2016:
                # (samples, 2, seq_len)
                i_comp = stacked[i, 0, :]
                q_comp = stacked[i, 1, :]
            else:
                # (samples, seq_len, 2)
                i_comp = stacked[i, :, 0]
                q_comp = stacked[i, :, 1]

            pwr = float(np.mean(i_comp ** 2 + q_comp ** 2))
            sigmas[i] = estimate_noise_std(pwr, float(snr_db))
        noise_vars_samples = sigmas ** 2  # (M,)

        # 构造 Y 与每列噪声方差（实部/虚部共用同一 σ²）
        Y = np.empty((n, M * 2), dtype=np.float64)

        if data_format_2016:
            # (M, 2, n) -> Y[:, even] = I, Y[:, odd] = Q
            Y[:, 0::2] = stacked[:, 0, :].T
            Y[:, 1::2] = stacked[:, 1, :].T
        else:
            # (M, n, 2) -> Y[:, even] = I, Y[:, odd] = Q
            Y[:, 0::2] = stacked[:, :, 0].T
            Y[:, 1::2] = stacked[:, :, 1].T

        noise_vars_cols = np.empty((M * 2,), dtype=np.float64)
        noise_vars_cols[0::2] = noise_vars_samples
        noise_vars_cols[1::2] = noise_vars_samples

        denoised_cols = spectral_gp_denoise_same_inputs(eigvecs, eigvals, Y, noise_vars_cols)

        denoised_group = np.empty_like(stacked)

        if data_format_2016:
            # 转换回 (M, 2, n) 格式
            denoised_group[:, 0, :] = denoised_cols[:, 0::2].T
            denoised_group[:, 1, :] = denoised_cols[:, 1::2].T
        else:
            # 转换回 (M, n, 2) 格式
            denoised_group[:, :, 0] = denoised_cols[:, 0::2].T
            denoised_group[:, :, 1] = denoised_cols[:, 1::2].T

        # 将组结果拆回各 (mod, snr) 键，保持每键内相对顺序
        offset = 0
        for (key, samples) in entries:
            cnt = len(samples)
            if cnt == 0:
                denoised_dataset[key] = samples
                continue
            seg = denoised_group[offset:offset + cnt]
            denoised_dataset[key] = seg
            offset += cnt

        processed += total_in_group
        elapsed = time.time() - t0
        print(f'  已完成: {processed}/{total_samples} ({processed/max(total_samples,1)*100:.1f}%), 用时累计: {elapsed:.1f}s')

    total_time = time.time() - t0
    print(f'per-sample模式GPR去噪完成，总用时: {total_time:.2f} 秒')
    return denoised_dataset, total_time


def apply_efficient_gpr_denoising_per_sample(X_all, y_all, snr_values_all, mods=None):
    """
    Apply efficient GPR denoising using per-sample mode.
    This function reorganizes data by (modulation, SNR) and calls the per-sample GPR implementation.

    Args:
        X_all: Input data array
        y_all: Labels (integer indices)
        snr_values_all: SNR values for each sample
        mods: List of modulation names (if None, will create generic names)
    """
    # Get modulation types from y_all (assuming integer labels)
    unique_mod_indices = np.unique(y_all)

    # Use provided modulation names or create mock names
    if mods is not None and len(mods) > max(unique_mod_indices):
        mod_names = mods
    else:
        mod_names = [f"MOD{i:02d}" for i in range(max(unique_mod_indices) + 1)]

    # Reorganize data into the format expected by efficient GPR
    # dataset[key] where key is (mod_name, snr_value)
    dataset = {}

    for mod_idx in unique_mod_indices:
        mod_name = mod_names[mod_idx]
        mod_mask = (y_all == mod_idx)

        for snr_val in np.unique(snr_values_all[mod_mask]):
            combined_mask = mod_mask & (snr_values_all == snr_val)
            if np.sum(combined_mask) > 0:
                key = (mod_name, int(snr_val))
                dataset[key] = X_all[combined_mask]

    print(f"Reorganized data into {len(dataset)} (modulation, SNR) groups for per-sample GPR processing...")

    # Apply per-sample GPR denoising
    denoised_dataset, processing_time = apply_gpr_denoising_efficient_per_sample(
        dataset,
        batch_limit=4096
    )

    # Reorganize denoised data back to the original format
    denoised_X_all = np.zeros_like(X_all)

    for mod_idx in unique_mod_indices:
        mod_name = mod_names[mod_idx]
        mod_mask = (y_all == mod_idx)

        for snr_val in np.unique(snr_values_all[mod_mask]):
            combined_mask = mod_mask & (snr_values_all == snr_val)
            if np.sum(combined_mask) > 0:
                key = (mod_name, int(snr_val))
                if key in denoised_dataset:
                    denoised_X_all[combined_mask] = denoised_dataset[key]

    print(f"Efficient GPR denoising (per-sample mode) completed in {processing_time:.2f} seconds")
    return denoised_X_all