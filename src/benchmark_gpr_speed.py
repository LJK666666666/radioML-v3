#!/usr/bin/env python3
"""Benchmark GPR denoising speed: original vs vectorized+float32 vs CUDA."""
import time, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from efficient_gpr_per_sample import (
    rbf_kernel_same_grid, length_scale_from_snr,
    spectral_gp_denoise_same_inputs,
    spectral_gp_denoise_same_inputs_cuda,
    estimate_noise_std,
)


def make_synthetic_data(n_samples=220000, n=128, n_snr_groups=20):
    """Create synthetic data mimicking RML2016.10a structure."""
    snr_levels = np.arange(-20, 18 + 1, 2)[:n_snr_groups]
    samples_per_group = n_samples // n_snr_groups
    data = np.random.randn(n_samples, 2, n).astype(np.float32)
    snr_all = np.repeat(snr_levels, samples_per_group)
    return data, snr_all, snr_levels


def benchmark_original_float64(data, snr_all, snr_levels):
    """Original: float64 + per-sample for loop."""
    n = data.shape[2]
    t0 = time.perf_counter()
    for snr_db in snr_levels:
        mask = snr_all == snr_db
        stacked = data[mask]
        M = len(stacked)
        ls = length_scale_from_snr(float(snr_db))
        K = rbf_kernel_same_grid(n, ls)
        eigvals, eigvecs = np.linalg.eigh(K)

        # Original: per-sample for loop + float64
        sigmas = np.empty(M, dtype=np.float64)
        for i in range(M):
            pwr = float(np.mean(stacked[i, 0] ** 2 + stacked[i, 1] ** 2))
            sigmas[i] = estimate_noise_std(pwr, float(snr_db))
        noise_vars = sigmas ** 2

        Y = np.empty((n, M * 2), dtype=np.float64)
        Y[:, 0::2] = stacked[:, 0, :].T
        Y[:, 1::2] = stacked[:, 1, :].T
        nv_cols = np.empty(M * 2, dtype=np.float64)
        nv_cols[0::2] = noise_vars
        nv_cols[1::2] = noise_vars

        _ = spectral_gp_denoise_same_inputs(eigvecs, eigvals, Y, nv_cols)
    return time.perf_counter() - t0


def benchmark_vectorized_float32(data, snr_all, snr_levels):
    """Optimized: vectorized power + float32."""
    n = data.shape[2]
    t0 = time.perf_counter()
    for snr_db in snr_levels:
        mask = snr_all == snr_db
        stacked = data[mask]
        M = len(stacked)
        ls = length_scale_from_snr(float(snr_db))
        K = rbf_kernel_same_grid(n, ls).astype(np.float32)
        eigvals, eigvecs = np.linalg.eigh(K)

        # Vectorized power + float32
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


def benchmark_cuda(data, snr_all, snr_levels):
    """CUDA: vectorized + float32 + GPU matmul."""
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return None

    n = data.shape[2]
    # Warmup
    dummy = torch.randn(128, 128, device='cuda')
    _ = dummy @ dummy
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
    print(f"Generating synthetic data: {N} samples, 128 length, 20 SNR groups...")
    data, snr_all, snr_levels = make_synthetic_data(N)

    print(f"\n{'='*55}")
    print(f"  GPR Denoising Speed Benchmark ({N:,} samples)")
    print(f"{'='*55}")

    # Original
    print("\n[1/3] Original (float64 + for loop)...")
    t1 = benchmark_original_float64(data, snr_all, snr_levels)
    print(f"  Time: {t1:.3f}s  ({t1/N*1000:.4f} ms/sample)")

    # Vectorized + float32
    print("\n[2/3] Vectorized + float32...")
    t2 = benchmark_vectorized_float32(data, snr_all, snr_levels)
    print(f"  Time: {t2:.3f}s  ({t2/N*1000:.4f} ms/sample)")
    print(f"  Speedup vs original: {t1/t2:.2f}x")

    # CUDA
    print("\n[3/3] CUDA (vectorized + float32 + GPU matmul)...")
    t3 = benchmark_cuda(data, snr_all, snr_levels)
    if t3 is not None:
        print(f"  Time: {t3:.3f}s  ({t3/N*1000:.4f} ms/sample)")
        print(f"  Speedup vs original: {t1/t3:.2f}x")
        print(f"  Speedup vs vec+f32:  {t2/t3:.2f}x")

    print(f"\n{'='*55}")
    print(f"  {'Method':<30} {'Time(s)':>8} {'ms/sample':>10} {'Speedup':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
    print(f"  {'Original (f64+loop)':<30} {t1:>8.3f} {t1/N*1000:>10.4f} {'1.00x':>8}")
    print(f"  {'Vectorized + float32':<30} {t2:>8.3f} {t2/N*1000:>10.4f} {f'{t1/t2:.2f}x':>8}")
    if t3 is not None:
        print(f"  {'CUDA (vec+f32+GPU)':<30} {t3:>8.3f} {t3/N*1000:>10.4f} {f'{t1/t3:.2f}x':>8}")
    print(f"{'='*55}")
