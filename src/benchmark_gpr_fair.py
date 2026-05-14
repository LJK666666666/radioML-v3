#!/usr/bin/env python3
"""Fair comparison: standard GPR (sklearn) vs spectral GPR, all optimization levels."""
import time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efficient_gpr_per_sample import (
    rbf_kernel_same_grid, length_scale_from_snr,
    spectral_gp_denoise_same_inputs,
    spectral_gp_denoise_same_inputs_cuda,
    estimate_noise_std,
)


def make_data(n_samples=220000, n=128, n_groups=20):
    snr_levels = np.arange(-20, 18 + 1, 2)[:n_groups]
    per_group = n_samples // n_groups
    data = np.random.randn(n_samples, 2, n).astype(np.float32)
    snr_all = np.repeat(snr_levels, per_group)
    return data, snr_all, snr_levels


# =================== Standard GPR (sklearn, matching original 884s baseline) ===================

def standard_gpr_sklearn(data, snr_all, snr_levels, sample_ratio=0.01):
    """Original sklearn GPR: GaussianProcessRegressor per sample.
    Matches src_backup/preprocess.py implementation."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    n = data.shape[2]
    X_grid = np.arange(n).reshape(-1, 1)
    total_samples = 0
    t0 = time.perf_counter()

    for snr_db in snr_levels:
        mask = snr_all == snr_db
        stacked = data[mask]
        M_full = len(stacked)
        M = max(1, int(M_full * sample_ratio))
        stacked = stacked[:M]

        ls = length_scale_from_snr(float(snr_db))
        snr_lin = 10.0 ** (float(snr_db) / 10.0)

        for i in range(M):
            pwr = float(np.mean(stacked[i, 0] ** 2 + stacked[i, 1] ** 2))
            noise_std = np.sqrt(pwr / (2.0 * (snr_lin + 1.0)))

            kernel = RBF(length_scale=ls, length_scale_bounds="fixed")

            gpr_real = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=True)
            gpr_real.fit(X_grid, stacked[i, 0])
            _ = gpr_real.predict(X_grid)

            gpr_imag = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=True)
            gpr_imag.fit(X_grid, stacked[i, 1])
            _ = gpr_imag.predict(X_grid)

        total_samples += M

    elapsed = time.perf_counter() - t0
    return elapsed / sample_ratio, total_samples


# =================== Standard GPR (GPU batched solve) ===================

def standard_gpr_gpu_solve(data, snr_all, snr_levels):
    """Optimized standard GPR: GPU batched torch.linalg.solve, float32, full dataset."""
    import torch
    n = data.shape[2]
    _ = torch.randn(2, 2, device='cuda') @ torch.randn(2, 2, device='cuda')
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for snr_db in snr_levels:
        mask = snr_all == snr_db
        stacked = data[mask]
        M = len(stacked)
        ls = length_scale_from_snr(float(snr_db))
        K = rbf_kernel_same_grid(n, ls).astype(np.float32)

        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        pwr = np.mean(stacked[:, 0] ** 2 + stacked[:, 1] ** 2, axis=1)
        noise_vars = (pwr / (2.0 * (snr_lin + 1.0))).astype(np.float32)

        K_gpu = torch.from_numpy(K).cuda()
        I_n = torch.eye(n, dtype=torch.float32, device='cuda')
        nv_gpu = torch.from_numpy(noise_vars).cuda()
        # (M, n, n) each = K + σ_i² I
        A_batch = K_gpu.unsqueeze(0) + nv_gpu[:, None, None] * I_n.unsqueeze(0)

        Y_gpu = torch.from_numpy(stacked).cuda()  # (M, 2, n)
        rhs = Y_gpu.permute(0, 2, 1)  # (M, n, 2)
        _ = torch.linalg.solve(A_batch, rhs)

    torch.cuda.synchronize()
    return time.perf_counter() - t0


# =================== Spectral GPR ===================

def spectral_gpr_cpu(data, snr_all, snr_levels):
    """Spectral GPR: vectorized + float32, CPU numpy."""
    n = data.shape[2]
    t0 = time.perf_counter()
    for snr_db in snr_levels:
        mask = snr_all == snr_db
        stacked = data[mask]
        M = len(stacked)
        ls = length_scale_from_snr(float(snr_db))
        K = rbf_kernel_same_grid(n, ls).astype(np.float32)
        eigvals, eigvecs = np.linalg.eigh(K)

        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        pwr = np.mean(stacked[:, 0] ** 2 + stacked[:, 1] ** 2, axis=1)
        noise_vars = (pwr / (2.0 * (snr_lin + 1.0))).astype(np.float32)

        Y = np.empty((n, M * 2), dtype=np.float32)
        Y[:, 0::2] = stacked[:, 0, :].T
        Y[:, 1::2] = stacked[:, 1, :].T
        nv_cols = np.empty(M * 2, dtype=np.float32)
        nv_cols[0::2] = noise_vars
        nv_cols[1::2] = noise_vars
        _ = spectral_gp_denoise_same_inputs(eigvecs, eigvals, Y, nv_cols)
    return time.perf_counter() - t0


def spectral_gpr_gpu(data, snr_all, snr_levels):
    """Spectral GPR: vectorized + float32 + CUDA matmul."""
    import torch
    n = data.shape[2]
    _ = torch.randn(2, 2, device='cuda') @ torch.randn(2, 2, device='cuda')
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for snr_db in snr_levels:
        mask = snr_all == snr_db
        stacked = data[mask]
        M = len(stacked)
        ls = length_scale_from_snr(float(snr_db))
        K = rbf_kernel_same_grid(n, ls).astype(np.float32)
        eigvals, eigvecs = np.linalg.eigh(K)

        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        pwr = np.mean(stacked[:, 0] ** 2 + stacked[:, 1] ** 2, axis=1)
        noise_vars = (pwr / (2.0 * (snr_lin + 1.0))).astype(np.float32)

        Y = np.empty((n, M * 2), dtype=np.float32)
        Y[:, 0::2] = stacked[:, 0, :].T
        Y[:, 1::2] = stacked[:, 1, :].T
        nv_cols = np.empty(M * 2, dtype=np.float32)
        nv_cols[0::2] = noise_vars
        nv_cols[1::2] = noise_vars

        eigvecs_gpu = torch.from_numpy(eigvecs).float().cuda()
        eigvals_gpu = torch.from_numpy(eigvals).float().cuda()
        _ = spectral_gp_denoise_same_inputs_cuda(eigvecs_gpu, eigvals_gpu, Y, nv_cols)

    torch.cuda.synchronize()
    return time.perf_counter() - t0


if __name__ == '__main__':
    N = 220000
    print(f"Generating {N:,} synthetic samples...\n")
    data, snr_all, snr_levels = make_data(N)

    print(f"{'='*70}")
    print(f"  Fair GPR Speedup Comparison ({N:,} samples, n=128, G=20)")
    print(f"{'='*70}")

    # 1. sklearn baseline (1% sample → extrapolate)
    ratio = 0.01
    print(f"\n[1/4] Standard GPR (sklearn, {ratio*100:.0f}% sample, extrapolated)...")
    t_sklearn, n_tested = standard_gpr_sklearn(data, snr_all, snr_levels, sample_ratio=ratio)
    print(f"  Tested {n_tested} samples, extrapolated to full: {t_sklearn:.1f}s  ({t_sklearn/N*1000:.4f} ms/sample)")

    # 2. Standard GPR, GPU batched (full)
    print(f"\n[2/4] Standard GPR (GPU batched solve, f32, full)...")
    t_std_gpu = standard_gpr_gpu_solve(data, snr_all, snr_levels)
    print(f"  Time: {t_std_gpu:.2f}s  ({t_std_gpu/N*1000:.4f} ms/sample)")

    # 3. Spectral GPR, CPU
    print(f"\n[3/4] Spectral GPR (CPU, f32, vectorized)...")
    t_spec_cpu = spectral_gpr_cpu(data, snr_all, snr_levels)
    print(f"  Time: {t_spec_cpu:.3f}s  ({t_spec_cpu/N*1000:.4f} ms/sample)")

    # 4. Spectral GPR, GPU
    print(f"\n[4/4] Spectral GPR (GPU, f32, vectorized)...")
    t_spec_gpu = spectral_gpr_gpu(data, snr_all, snr_levels)
    print(f"  Time: {t_spec_gpu:.3f}s  ({t_spec_gpu/N*1000:.4f} ms/sample)")

    print(f"\n{'='*70}")
    print(f"  {'Method':<42} {'Time(s)':>8} {'ms/samp':>8} {'vs sklearn':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*10}")
    for label, t in [
        (f'Standard GPR (sklearn, extrap from {ratio*100:.0f}%)', t_sklearn),
        ('Standard GPR (GPU batched solve)', t_std_gpu),
        ('Spectral GPR (CPU, f32)', t_spec_cpu),
        ('Spectral GPR (GPU, f32)', t_spec_gpu),
    ]:
        print(f"  {label:<42} {t:>8.2f} {t/N*1000:>8.4f} {f'{t_sklearn/t:.0f}x':>10}")

    print(f"\n  Algorithmic speedup (same hardware):")
    print(f"    GPU: Std solve / Spectral = {t_std_gpu/t_spec_gpu:.1f}x")
    print(f"  Overall: sklearn / Spectral GPU = {t_sklearn/t_spec_gpu:.0f}x")
    print(f"{'='*70}")
