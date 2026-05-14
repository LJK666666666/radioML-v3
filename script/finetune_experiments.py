#!/usr/bin/env python3
"""
实验脚本：
任务2 - SNR误差鲁棒性微调实验
  在无SNR误差GPR去噪权重基础上，用（无误差+有误差）合并训练数据微调，
  有误差去噪验证集做学习率调度和早停。
任务3 - 基线去噪方法公平对比
  在无去噪权重基础上，用移动平均/小波去噪数据微调后再测试。
"""
import os, sys, time, copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import uniform_filter1d
import pywt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import load_data, split_data_raw
from model.iqformer_torch_model import build_iqformer_model
from efficient_gpr_per_sample import (
    length_scale_from_snr, rbf_kernel_same_grid,
    spectral_gp_denoise_same_inputs,
)


# ============================================================================
# 去噪方法
# ============================================================================

def denoise_moving_average(X, window_size=5):
    out = np.empty_like(X)
    out[:, 0, :] = uniform_filter1d(X[:, 0, :], size=window_size, axis=1)
    out[:, 1, :] = uniform_filter1d(X[:, 1, :], size=window_size, axis=1)
    return out


def denoise_wavelet(X, wavelet='db4', level=2):
    out = np.empty_like(X)
    for ch in range(2):
        for i in tqdm(range(len(X)), desc=f"Wavelet ch{ch}", leave=False):
            coeffs = pywt.wavedec(X[i, ch, :], wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(len(X[i, ch, :])))
            new_coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode='soft') for c in coeffs[1:]]
            out[i, ch, :] = pywt.waverec(new_coeffs, wavelet)[:X.shape[2]]
    return out


def denoise_gpr_spectral(X, snr_values, snr_levels):
    out = np.empty_like(X)
    n = X.shape[2]
    for snr_db in tqdm(snr_levels, desc="GPR denoise"):
        mask = snr_values == snr_db
        stacked = X[mask].astype(np.float32)
        M = len(stacked)
        if M == 0:
            continue
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

        dn = spectral_gp_denoise_same_inputs(eigvecs, eigvals, Y, nv_cols)
        result = np.empty_like(stacked)
        result[:, 0, :] = dn[:, 0::2].T
        result[:, 1, :] = dn[:, 1::2].T
        out[mask] = result
    return out


def denoise_gpr_with_snr_error(X, snr_values, snr_levels, sigma_err):
    """GPR去噪，但SNR值加入高斯随机误差后snap到最近的离散SNR level"""
    noisy_snr = snr_values.astype(np.float64) + np.random.normal(0, sigma_err, size=len(snr_values))
    # snap到最近的离散SNR level
    snr_levels_arr = np.array(sorted(snr_levels), dtype=np.float64)
    idx = np.argmin(np.abs(noisy_snr[:, None] - snr_levels_arr[None, :]), axis=1)
    snapped_snr = snr_levels_arr[idx]
    return denoise_gpr_spectral(X, snapped_snr, snr_levels)


# ============================================================================
# 微调函数
# ============================================================================

def finetune_torch_model(
    model, X_train, y_train, X_val, y_val,
    save_path, epochs=50, batch_size=128,
    learning_rate=1e-4, patience_lr=3, patience_es=10, factor=0.5,
):
    """在预训练权重基础上微调PyTorch模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=factor, patience=patience_lr, min_lr=1e-7,
    )

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    last_path = save_path.replace(".pt", "_last.pt")

    print(f"Fine-tuning model, saving best to {save_path}")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)
            pbar.set_postfix(loss=train_loss/train_total, acc=train_correct/train_total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_acc = val_correct / max(1, val_total)
        train_acc = train_correct / max(1, train_total)
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss/train_total:.4f} - acc: {train_acc:.4f} "
              f"- val_loss: {val_loss/val_total:.4f} - val_acc: {val_acc:.4f}"
              + (f" - lr: {old_lr:.2e}->{new_lr:.2e}" if new_lr < old_lr else ""))

        torch.save(model.state_dict(), last_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_es:
            print(f"EarlyStopping at epoch {epoch+1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, save_path)
    print(f"Best val_acc: {best_val_acc:.4f}")
    return model, best_val_acc


def evaluate_iqformer(model, X_test, y_test_int, device, batch_size=256):
    model.eval()
    X_t = torch.from_numpy(X_test.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(all_preds)
    return np.mean(preds == y_test_int)


def load_iqformer_weights(weights_path, input_shape, num_classes, device):
    model = build_iqformer_model(input_shape, num_classes).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model


# ============================================================================
# 主实验
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[2, 3], required=True,
                        help='2: SNR error robustness finetune; 3: baseline denoising finetune comparison')
    parser.add_argument('--sigma_err', type=float, default=3.0,
                        help='SNR estimation error std (task 2)')
    parser.add_argument('--finetune_epochs', type=int, default=50)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    print("Loading RML2016.10a dataset...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'RML2016.10a_dict.pkl')
    dataset = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = \
        split_data_raw(dataset)
    num_classes = len(mods)
    input_shape = (2, 128)
    snr_levels = np.array(sorted(set(snr_train.tolist())))
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 模型权重路径
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'models')
    gpr_weights = os.path.join(weights_dir, 'iqformer_model_efficient_gpr_per_sample_stratified.pt')
    raw_weights = os.path.join(weights_dir, 'iqformer_model_stratified.pt')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'finetune_results')
    os.makedirs(results_dir, exist_ok=True)

    # ========================================================================
    # 任务2：SNR误差鲁棒性微调
    # ========================================================================
    if args.task == 2:
        sigma_err = args.sigma_err
        print(f"\n{'='*60}")
        print(f"Task 2: SNR Error Robustness Finetuning (sigma_err={sigma_err})")
        print(f"{'='*60}")

        # 1) 无误差GPR去噪训练/验证/测试集
        print("\n[Step 1] GPR denoising (no error) on train/val/test...")
        X_train_gpr = denoise_gpr_spectral(X_train, snr_train, snr_levels)
        X_val_gpr = denoise_gpr_spectral(X_val, snr_val, snr_levels)
        X_test_gpr = denoise_gpr_spectral(X_test, snr_test, snr_levels)

        # 2) 有误差GPR去噪训练/验证集
        print(f"\n[Step 2] GPR denoising with SNR error (sigma={sigma_err}) on train/val...")
        X_train_gpr_err = denoise_gpr_with_snr_error(X_train, snr_train, snr_levels, sigma_err)
        X_val_gpr_err = denoise_gpr_with_snr_error(X_val, snr_val, snr_levels, sigma_err)
        # 有误差去噪测试集（多个sigma_err）
        test_sigmas = [1, 2, 3, 4, 5, 6, 8, 10]

        # 3) 合并训练数据（无误差 + 有误差）
        print("\n[Step 3] Merging training data (no-error + with-error)...")
        X_train_merged = np.concatenate([X_train_gpr, X_train_gpr_err], axis=0)
        y_train_merged = np.concatenate([y_train, y_train], axis=0)

        # 4) 加载无误差GPR权重并微调
        print("\n[Step 4] Loading GPR pretrained weights and finetuning...")
        model = load_iqformer_weights(gpr_weights, input_shape, num_classes, device)
        save_path = os.path.join(results_dir, f'iqformer_finetuned_snr_err_sigma{sigma_err}.pt')
        model, best_val_acc = finetune_torch_model(
            model, X_train_merged, y_train_merged,
            X_val_gpr_err, y_val,  # 有误差验证集做调度和早停
            save_path=save_path,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.finetune_lr,
        )

        # 5) 测试：在不同sigma_err下评估
        print(f"\n[Step 5] Evaluating finetuned model under various SNR errors...")

        # 也加载原始GPR权重（无微调）做对比
        model_orig = load_iqformer_weights(gpr_weights, input_shape, num_classes, device)
        # 无去噪baseline
        model_raw = load_iqformer_weights(raw_weights, input_shape, num_classes, device)

        print(f"\n{'Method':<40} {'Accuracy':>10}")
        print(f"{'-'*40} {'-'*10}")

        # baseline (无去噪)
        acc_baseline = evaluate_iqformer(model_raw, X_test, y_test, device)
        print(f"{'No denoising (baseline)':<40} {acc_baseline*100:>9.2f}%")

        # 无误差GPR（原始权重）
        acc_gpr_orig = evaluate_iqformer(model_orig, X_test_gpr, y_test, device)
        print(f"{'GPR exact SNR (original weights)':<40} {acc_gpr_orig*100:>9.2f}%")

        # 无误差GPR（微调权重）
        acc_gpr_ft = evaluate_iqformer(model, X_test_gpr, y_test, device)
        print(f"{'GPR exact SNR (finetuned weights)':<40} {acc_gpr_ft*100:>9.2f}%")

        # 不同sigma_err下对比原始vs微调
        print(f"\n{'sigma_err':<12} {'Original':>12} {'Finetuned':>12} {'Diff':>10}")
        print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        for s in test_sigmas:
            X_test_err = denoise_gpr_with_snr_error(X_test, snr_test, snr_levels, s)
            acc_orig = evaluate_iqformer(model_orig, X_test_err, y_test, device)
            acc_ft = evaluate_iqformer(model, X_test_err, y_test, device)
            diff = acc_ft - acc_orig
            print(f"{s:<12} {acc_orig*100:>11.2f}% {acc_ft*100:>11.2f}% {diff*100:>+9.2f}%")

    # ========================================================================
    # 任务3：基线去噪方法公平对比（微调后测试）
    # ========================================================================
    elif args.task == 3:
        print(f"\n{'='*60}")
        print(f"Task 3: Baseline Denoising Fair Comparison (with finetuning)")
        print(f"{'='*60}")

        # 1) 对训练/验证/测试集应用各种去噪方法
        print("\n[Step 1] Applying denoising methods to train/val/test sets...")

        print("  Moving average (w=5)...")
        X_train_ma5 = denoise_moving_average(X_train, window_size=5)
        X_val_ma5 = denoise_moving_average(X_val, window_size=5)
        X_test_ma5 = denoise_moving_average(X_test, window_size=5)

        print("  Wavelet (db4, level=2)...")
        X_train_wv = denoise_wavelet(X_train, wavelet='db4', level=2)
        X_val_wv = denoise_wavelet(X_val, wavelet='db4', level=2)
        X_test_wv = denoise_wavelet(X_test, wavelet='db4', level=2)

        print("  GPR spectral...")
        X_train_gpr = denoise_gpr_spectral(X_train, snr_train, snr_levels)
        X_val_gpr = denoise_gpr_spectral(X_val, snr_val, snr_levels)
        X_test_gpr = denoise_gpr_spectral(X_test, snr_test, snr_levels)

        # 2) 从无去噪权重微调MA和Wavelet，从GPR权重测试GPR
        print("\n[Step 2] Finetuning from raw weights on MA/Wavelet denoised data...")

        # MA(w=5)微调
        print("\n--- Finetuning on Moving Average (w=5) ---")
        model_ma5 = load_iqformer_weights(raw_weights, input_shape, num_classes, device)
        save_ma5 = os.path.join(results_dir, 'iqformer_finetuned_ma5.pt')
        model_ma5, _ = finetune_torch_model(
            model_ma5, X_train_ma5, y_train, X_val_ma5, y_val,
            save_path=save_ma5,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.finetune_lr,
        )

        # Wavelet微调
        print("\n--- Finetuning on Wavelet (db4, L=2) ---")
        model_wv = load_iqformer_weights(raw_weights, input_shape, num_classes, device)
        save_wv = os.path.join(results_dir, 'iqformer_finetuned_wavelet.pt')
        model_wv, _ = finetune_torch_model(
            model_wv, X_train_wv, y_train, X_val_wv, y_val,
            save_path=save_wv,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.finetune_lr,
        )

        # GPR使用已有权重（已在GPR去噪数据上完整训练过）
        model_gpr = load_iqformer_weights(gpr_weights, input_shape, num_classes, device)

        # 无去噪baseline
        model_raw = load_iqformer_weights(raw_weights, input_shape, num_classes, device)

        # 3) 测试
        print(f"\n[Step 3] Evaluation Results")
        print(f"{'='*55}")
        print(f"  {'Method':<35} {'Accuracy':>10}")
        print(f"  {'-'*35} {'-'*10}")

        acc_baseline = evaluate_iqformer(model_raw, X_test, y_test, device)
        print(f"  {'None (baseline)':<35} {acc_baseline*100:>9.2f}%")

        acc_ma5 = evaluate_iqformer(model_ma5, X_test_ma5, y_test, device)
        print(f"  {'Moving Average (w=5, finetuned)':<35} {acc_ma5*100:>9.2f}%")

        acc_wv = evaluate_iqformer(model_wv, X_test_wv, y_test, device)
        print(f"  {'Wavelet (db4, L=2, finetuned)':<35} {acc_wv*100:>9.2f}%")

        acc_gpr = evaluate_iqformer(model_gpr, X_test_gpr, y_test, device)
        print(f"  {'GPR (proposed, full training)':<35} {acc_gpr*100:>9.2f}%")
        print(f"{'='*55}")

        # 保存结果
        result_file = os.path.join(results_dir, 'baseline_comparison_finetuned.txt')
        with open(result_file, 'w') as f:
            f.write("Baseline Denoising Fair Comparison (IQFormer on RML2016.10a)\n")
            f.write("MA and Wavelet: finetuned from raw-data weights\n")
            f.write("GPR: trained from scratch on GPR-denoised data\n\n")
            f.write(f"{'Method':<35} {'Accuracy':>10}\n")
            f.write(f"{'-'*45}\n")
            f.write(f"{'None (baseline)':<35} {acc_baseline*100:>9.2f}%\n")
            f.write(f"{'Moving Average (w=5, finetuned)':<35} {acc_ma5*100:>9.2f}%\n")
            f.write(f"{'Wavelet (db4, L=2, finetuned)':<35} {acc_wv*100:>9.2f}%\n")
            f.write(f"{'GPR (proposed, full training)':<35} {acc_gpr*100:>9.2f}%\n")
        print(f"\nResults saved to {result_file}")
