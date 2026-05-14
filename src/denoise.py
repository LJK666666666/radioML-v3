"""
Standalone denoising module.

Provides split-level denoising, augmentation, one-hot encoding, and caching.
Used by --mode denoise and --mode train/evaluate with --use_predicted_snr.
"""

import os
import pickle
import sys
import numpy as np

# Ensure gpr directory is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gpr'))

from preprocess import (
    calculate_power, estimate_noise_std, apply_gp_regression,
    augment_iq_data,
)


def denoise_split(X, y_int, snr_values, mods, denoising_method, split_name=""):
    """Denoise a single data split.

    Args:
        X: ndarray (N, 2, seq_len) - raw IQ data
        y_int: ndarray (N,) - integer class labels
        snr_values: ndarray (N,) - SNR dB values used for denoising (real or predicted)
        mods: list of modulation type names
        denoising_method: 'gpr', 'efficient_gpr_per_sample', or 'gpr_fft'
        split_name: label for progress messages

    Returns:
        X_denoised: ndarray (N, 2, seq_len)
    """
    X_denoised = X.copy()
    total_samples = X_denoised.shape[0]

    if total_samples == 0:
        return X_denoised

    prefix = f"[{split_name}] " if split_name else ""
    print(f"{prefix}Denoising {total_samples} samples with method={denoising_method}...")

    if denoising_method.lower() == 'gpr':
        progress_step = max(1, total_samples // 100)
        for i in range(total_samples):
            if i % progress_step == 0 or i == total_samples - 1:
                progress_percent = (i + 1) / total_samples * 100
                print(f"{prefix}Processing sample {i+1}/{total_samples} ({progress_percent:.1f}%)")

            current_snr = snr_values[i]
            i_component = X_denoised[i, 0, :]
            q_component = X_denoised[i, 1, :]
            complex_signal = i_component + 1j * q_component

            total_power = calculate_power(i_component, q_component)
            noise_std = estimate_noise_std(total_power, current_snr)
            length_scale_val = 5.0 if current_snr >= 0 else min(10, 5.0 - current_snr * 0.25)
            denoised_signal = apply_gp_regression(
                complex_signal, noise_std, kernel_name='rbf', length_scale=length_scale_val
            )

            X_denoised[i, 0, :] = np.real(denoised_signal)
            X_denoised[i, 1, :] = np.imag(denoised_signal)

    elif denoising_method.lower() == 'efficient_gpr_per_sample':
        from efficient_gpr_per_sample import apply_efficient_gpr_denoising_per_sample
        X_denoised = apply_efficient_gpr_denoising_per_sample(X_denoised, y_int, snr_values, mods)

    elif denoising_method.lower() == 'gpr_fft':
        from gpr_fft import apply_fft_gpr_denoising_per_sample
        X_denoised = apply_fft_gpr_denoising_per_sample(X_denoised, y_int, snr_values, mods)

    else:
        print(f"{prefix}Warning: Denoising method '{denoising_method}' not recognized. Returning original data.")
        return X_denoised

    print(f"{prefix}Denoising complete.")
    return X_denoised


def denoise_and_cache_splits(X_train, X_val, X_test,
                             y_train_int, y_val_int, y_test_int,
                             snr_train, snr_val, snr_test,
                             mods, denoising_method, cache_dir, cache_tag,
                             augment_data,
                             snr_val_for_denoise, snr_test_for_denoise):
    """Denoise train/val/test splits, augment train, one-hot encode, and cache.

    - train always uses snr_train (real SNR)
    - val uses snr_val_for_denoise (real or predicted)
    - test uses snr_test_for_denoise (real or predicted)
    - snr_val / snr_test stored in cache are the REAL SNR values
      (needed for evaluate_by_snr grouping)

    Returns:
        dict with keys:
            X_train, X_val, X_test (denoised, augmented)
            y_train, y_val, y_test (one-hot)
            snr_train, snr_val, snr_test (real values)
            mods
    """
    from keras.utils import to_categorical

    # Denoise each split
    if denoising_method.lower() != 'none':
        X_train = denoise_split(X_train, y_train_int, snr_train, mods,
                                denoising_method, split_name="train")
        X_val = denoise_split(X_val, y_val_int, snr_val_for_denoise, mods,
                              denoising_method, split_name="val")
        X_test = denoise_split(X_test, y_test_int, snr_test_for_denoise, mods,
                               denoising_method, split_name="test")

    # Augment training data
    if augment_data and X_train.shape[0] > 0:
        print("Starting data augmentation: 3 rotations at 90-degree increments.")
        X_original = X_train.copy()
        y_original = y_train_int.copy()
        snr_original = snr_train.copy()

        augmented_X = []
        augmented_y = []
        augmented_snr = []

        angle = 90
        num = 360 // angle - 1  # 3 rotations: 90, 180, 270 degrees
        for i in range(num):
            current_angle_deg = (i + 1) * angle
            print(f"Augmenting: rotation {i+1}/{num}, angle: {current_angle_deg} degrees.")
            theta_rad = np.deg2rad(current_angle_deg)
            augmented_X.append(augment_iq_data(X_original, theta_rad))
            augmented_y.append(y_original)
            augmented_snr.append(snr_original)

        if augmented_X:
            X_train = np.concatenate([X_train] + augmented_X, axis=0)
            y_train_int = np.concatenate([y_train_int] + augmented_y, axis=0)
            snr_train = np.concatenate([snr_train] + augmented_snr, axis=0)

        print(f"Train set: {X_original.shape[0]} -> {X_train.shape[0]} (augmented)")

    # One-hot encode labels
    num_classes = len(mods)
    y_train = to_categorical(y_train_int, num_classes) if y_train_int.size > 0 \
        else np.array([]).reshape(0, num_classes)
    y_val = to_categorical(y_val_int, num_classes) if y_val_int.size > 0 \
        else np.array([]).reshape(0, num_classes)
    y_test = to_categorical(y_test_int, num_classes) if y_test_int.size > 0 \
        else np.array([]).reshape(0, num_classes)

    result = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'snr_train': snr_train, 'snr_val': snr_val, 'snr_test': snr_test,
        'mods': mods,
    }

    # Cache to disk
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_tag}_denoised_splits.pkl")
    print(f"Saving denoised splits cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Cache saved ({X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test).")

    return result


def load_cached_splits(cache_dir, cache_tag):
    """Load cached denoised splits.

    Returns:
        dict or None (if cache file does not exist)
    """
    cache_path = os.path.join(cache_dir, f"{cache_tag}_denoised_splits.pkl")
    if not os.path.exists(cache_path):
        return None
    print(f"Loading cached denoised splits from {cache_path}...")
    with open(cache_path, 'rb') as f:
        result = pickle.load(f)
    print(f"Cache loaded: train={result['X_train'].shape}, "
          f"val={result['X_val'].shape}, test={result['X_test'].shape}")
    return result
