#!/usr/bin/env python3
"""Evaluate IQFormer on data denoised by baseline methods (moving average, wavelet)
vs GPR, vs no denoising. Uses the same test split and trained IQFormer weights."""
import os, sys, time
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
import pywt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import load_data, split_data_raw
from model.iqformer_torch_model import build_iqformer_model
from efficient_gpr_per_sample import (
    length_scale_from_snr, rbf_kernel_same_grid,
    spectral_gp_denoise_same_inputs,
)


def evaluate_iqformer(model, X_test, y_test_int, device, batch_size=256):
    """Evaluate IQFormer and return overall accuracy."""
    model.eval()
    X_t = torch.from_numpy(X_test.astype(np.float32))
    dataset = torch.utils.data.TensorDataset(X_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(all_preds)
    acc = np.mean(preds == y_test_int)
    return acc


def denoise_moving_average(X, window_size=5):
    """Apply moving average filter to I and Q channels independently."""
    out = np.empty_like(X)
    out[:, 0, :] = uniform_filter1d(X[:, 0, :], size=window_size, axis=1)
    out[:, 1, :] = uniform_filter1d(X[:, 1, :], size=window_size, axis=1)
    return out


def denoise_wavelet(X, wavelet='db4', level=2):
    """Apply wavelet soft-threshold denoising to I and Q channels."""
    out = np.empty_like(X)
    for ch in range(2):
        for i in range(len(X)):
            coeffs = pywt.wavedec(X[i, ch, :], wavelet, level=level)
            # Universal threshold
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(len(X[i, ch, :])))
            new_coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode='soft') for c in coeffs[1:]]
            out[i, ch, :] = pywt.waverec(new_coeffs, wavelet)[:X.shape[2]]
    return out


def denoise_gpr_spectral(X, snr_values, snr_levels):
    """Apply spectral GPR denoising (float32, vectorized)."""
    out = np.empty_like(X)
    n = X.shape[2]
    for snr_db in snr_levels:
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


if __name__ == '__main__':
    # Load data
    print("Loading RML2016.10a dataset...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'RML2016.10a_dict.pkl')
    dataset = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = \
        split_data_raw(dataset)
    num_classes = len(mods)
    snr_levels = np.array(sorted(set(snr_test.tolist())))
    print(f"Test set: {len(X_test)} samples, {num_classes} classes")

    # Load trained IQFormer (GPR-denoised weights = best model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_iqformer_model((2, 128), num_classes).to(device)
    weights_path = os.path.join(os.path.dirname(__file__), '..',
                                'output', 'models', 'iqformer_model_efficient_gpr_per_sample_stratified.pt')
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded IQFormer weights from {weights_path}")

    # Also load IQFormer trained on raw data (for no-denoising baseline)
    model_raw = build_iqformer_model((2, 128), num_classes).to(device)
    weights_raw = os.path.join(os.path.dirname(__file__), '..',
                               'output', 'models', 'iqformer_model_stratified.pt')
    ckpt_raw = torch.load(weights_raw, map_location=device)
    if isinstance(ckpt_raw, dict) and 'model_state_dict' in ckpt_raw:
        model_raw.load_state_dict(ckpt_raw['model_state_dict'])
    else:
        model_raw.load_state_dict(ckpt_raw)
    print(f"Loaded raw IQFormer weights from {weights_raw}")

    results = []

    # 1. No denoising (raw model on raw data)
    print("\n[1/5] No denoising (baseline)...")
    acc = evaluate_iqformer(model_raw, X_test, y_test, device)
    print(f"  Accuracy: {acc*100:.2f}%")
    results.append(('None (baseline)', acc))

    # 2. Moving average (window=5, matching L0=5)
    print("\n[2/5] Moving average (window=5)...")
    X_ma = denoise_moving_average(X_test, window_size=5)
    acc = evaluate_iqformer(model, X_ma, y_test, device)
    print(f"  Accuracy: {acc*100:.2f}%")
    results.append(('Moving Average (w=5)', acc))

    # 3. Moving average (window=9)
    print("\n[3/5] Moving average (window=9)...")
    X_ma9 = denoise_moving_average(X_test, window_size=9)
    acc = evaluate_iqformer(model, X_ma9, y_test, device)
    print(f"  Accuracy: {acc*100:.2f}%")
    results.append(('Moving Average (w=9)', acc))

    # 4. Wavelet denoising
    print("\n[4/5] Wavelet denoising (db4, level=2)...")
    X_wv = denoise_wavelet(X_test, wavelet='db4', level=2)
    acc = evaluate_iqformer(model, X_wv, y_test, device)
    print(f"  Accuracy: {acc*100:.2f}%")
    results.append(('Wavelet (db4, L=2)', acc))

    # 5. GPR denoising
    print("\n[5/5] GPR denoising (spectral, RBF)...")
    X_gpr = denoise_gpr_spectral(X_test, snr_test, snr_levels)
    acc = evaluate_iqformer(model, X_gpr, y_test, device)
    print(f"  Accuracy: {acc*100:.2f}%")
    results.append(('GPR (proposed)', acc))

    # Summary
    print(f"\n{'='*50}")
    print(f"  {'Method':<25} {'Accuracy':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for name, acc in results:
        print(f"  {name:<25} {acc*100:>9.2f}%")
    print(f"{'='*50}")
