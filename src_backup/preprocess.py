import numpy as np
import pickle
import os # Added for os.path.exists
import hashlib # Added for creating unique filenames
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from scipy.stats import norm
from keras.utils import to_categorical # Changed from tensorflow.keras
from keras.models import load_model # Changed from tensorflow.keras
import pywt
import sys
import importlib


def _run_efficient_gpr_denoising(dataset_dict, mode: str):
    """
    调用 gpr.optimized_gpr 的高效GPR去噪。
    mode: 'per-key' 或 'per-sample'
    返回 denoised_dataset (同结构 dict) 与 total_time。
    """
    # 确保可以导入到项目根目录下的 gpr 包
    root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    optimized_gpr = importlib.import_module('gpr.optimized_gpr')
    denoised_dataset, total_time = optimized_gpr.apply_gpr_denoising_efficient(
        dataset_dict,
        noise_mode='per-key' if mode == 'per-key' else 'per-sample',
        batch_limit=4096,
    )
    print(f"Efficient GPR ({mode}) finished in {total_time:.2f}s")
    return denoised_dataset, total_time


# Helper functions for saving/loading denoised datasets
def create_denoised_filename(denoising_method, specific_snrs=None, ddae_model_path=None):
    """
    Create a unique filename for denoised dataset based on parameters.
    Args:
        denoising_method (str): The denoising method used
        specific_snrs (list): List of specific SNRs used
        ddae_model_path (str): Path to DDAE model if used
    Returns:
        str: Unique filename for the denoised dataset
    """
    # Create a string representing the configuration
    config_str = f"method_{denoising_method}"
    
    if specific_snrs is not None:
        snrs_str = "_".join(map(str, sorted(specific_snrs)))
        config_str += f"_snrs_{snrs_str}"
    else:
        config_str += "_snrs_all"
    
    if denoising_method.lower() == 'ddae' and ddae_model_path:
        # Add model path hash to distinguish different DDAE models
        model_hash = hashlib.md5(ddae_model_path.encode()).hexdigest()[:8]
        config_str += f"_model_{model_hash}"
    
    # Create hash of the configuration for shorter filename
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    return f"denoised_data_{denoising_method}_{config_hash}.pkl"

def save_denoised_dataset(X_all, y_all, snr_values_all, filename, denoised_dir):
    """
    Save denoised dataset to file.
    Args:
        X_all (np.ndarray): Denoised feature data
        y_all (np.ndarray): Labels
        snr_values_all (np.ndarray): SNR values
        filename (str): Filename to save to
        denoised_dir (str): Directory to save in
    """
    os.makedirs(denoised_dir, exist_ok=True)
    filepath = os.path.join(denoised_dir, filename)
    
    data_to_save = {
        'X_all': X_all,
        'y_all': y_all,
        'snr_values_all': snr_values_all
    }
    
    print(f"Saving denoised dataset to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Denoised dataset saved successfully.")

def load_denoised_dataset(filename, denoised_dir):
    """
    Load denoised dataset from file.
    Args:
        filename (str): Filename to load from
        denoised_dir (str): Directory to load from
    Returns:
        tuple: (X_all, y_all, snr_values_all) or None if file doesn't exist
    """
    filepath = os.path.join(denoised_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    print(f"Loading existing denoised dataset from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        X_all = data['X_all']
        y_all = data['y_all']
        snr_values_all = data['snr_values_all']
        
        print(f"Denoised dataset loaded successfully. Shape: {X_all.shape}")
        return X_all, y_all, snr_values_all
    
    except Exception as e:
        print(f"Error loading denoised dataset: {e}")
        return None


# Gaussian Process Regression Functions
def calculate_power(i_component, q_component):
    """Calculate the power of the signal from I and Q components."""
    return np.mean(i_component**2 + q_component**2)

def estimate_noise_std(signal_power, snr_db):
    """Estimate noise standard deviation from signal power and SNR."""
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / (snr_linear + 1)
    return np.sqrt(noise_power / 2)

def apply_wavelet_denoising(complex_signal, wavelet='db4', level=1, mode='soft'):
    """
    Apply Wavelet Denoising to a complex signal.
    Args:
        complex_signal (np.ndarray): Array of complex numbers representing the signal.
        wavelet (str): Type of wavelet to use.
        level (int): Decomposition level.
        mode (str): Type of thresholding ('soft' or 'hard').
    Returns:
        np.ndarray: Denoised complex signal.
    """
    real_part = complex_signal.real
    imag_part = complex_signal.imag

    def denoise_part(part):
        coeffs = pywt.wavedec(part, wavelet, level=level)
        # Estimate noise std from the first level detail coefficients
        sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
        # Universal threshold
        threshold = sigma * np.sqrt(2 * np.log(len(part)))
        
        denoised_coeffs = [coeffs[0]] # Keep approximation coefficients
        for i in range(1, len(coeffs)):
            denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=mode))
            
        return pywt.waverec(denoised_coeffs, wavelet)

    denoised_real = denoise_part(real_part)
    denoised_imag = denoise_part(imag_part)
    
    # Ensure the output length matches the input length
    output_len = len(complex_signal)
    if len(denoised_real) > output_len:
        denoised_real = denoised_real[:output_len]
    if len(denoised_imag) > output_len:
        denoised_imag = denoised_imag[:output_len]
        
    return denoised_real + 1j * denoised_imag

def apply_ddae_denoising(complex_signal, model_path='../model_weight_saved/ddae_model.keras'):
    """
    Apply Deep Denoising Autoencoder to denoise a complex signal.
    Args:
        complex_signal (np.ndarray): Array of complex numbers representing the signal.
        model_path (str): Path to the trained DDAE model.
                          Defaults to '../model_weight_saved/ddae_model.keras'.
    Returns:
        np.ndarray: Denoised complex signal, or original signal if model not found or error occurs.
    """
    if not os.path.exists(model_path):
        print(f"Warning: DDAE model not found at {model_path}. Returning original signal.")
        return complex_signal

    try:
        # Load the DAE model, compile=False for faster loading if only for inference
        dae_model = load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading DDAE model: {e}. Returning original signal.")
        return complex_signal

    # Preprocess Input Signal
    real_part = complex_signal.real
    imag_part = complex_signal.imag
    signal_2channel = np.stack((real_part, imag_part), axis=-1) # Shape: (sequence_length, 2)

    # Normalization
    instance_max_val = np.max(np.abs(signal_2channel))
    if instance_max_val == 0:
        # Avoid division by zero for zero-signals; return as is.
        # Or, could return np.zeros_like(complex_signal) if that's more appropriate.
        print("Warning: Max value of signal is 0. Returning original signal.")
        return complex_signal
    
    normalized_signal = signal_2channel / instance_max_val
    
    # Reshape for the model: (1, sequence_length, 2)
    model_input = np.expand_dims(normalized_signal, axis=0)

    # Predict with DAE
    try:
        denoised_output_normalized = dae_model.predict(model_input)
    except Exception as e:
        print(f"Error during DDAE model prediction: {e}. Returning original signal.")
        return complex_signal

    # Postprocess Output Signal
    # Remove batch dimension: (sequence_length, 2)
    denoised_squeezed = np.squeeze(denoised_output_normalized, axis=0)
    
    # Denormalization
    denoised_signal_2channel = denoised_squeezed * instance_max_val
    
    # Extract denoised real and imaginary parts
    denoised_real = denoised_signal_2channel[:, 0]
    denoised_imag = denoised_signal_2channel[:, 1]
    
    # Reconstruct complex signal
    final_denoised_signal = denoised_real + 1j * denoised_imag
    
    return final_denoised_signal

# def apply_gp_regression(complex_signal, noise_std, kernel_name='rbf', length_scale=1.0, matern_nu=1.5, rational_quadratic_alpha=1.0):
def apply_gp_regression(complex_signal, noise_std, kernel_name='rbf', length_scale=50, matern_nu=1.5, rational_quadratic_alpha=1.0):

    """
    Apply Gaussian Process Regression to denoise a complex signal.
    Args:
        complex_signal (np.ndarray): Array of complex numbers representing the signal.
        noise_std (float): Estimated standard deviation of the noise.
        kernel_name (str): Type of kernel ('rbf', 'matern', 'rational_quadratic').
        length_scale (float): Length scale parameter for RBF, Matern, RationalQuadratic kernels.
        matern_nu (float): Nu parameter for Matern kernel.
        rational_quadratic_alpha (float): Alpha parameter for RationalQuadratic kernel.
    Returns:
        np.ndarray: Denoised complex signal.
    """
    X = np.arange(len(complex_signal)).reshape(-1, 1)
    y_real = complex_signal.real
    y_imag = complex_signal.imag

    # Kernel selection
    if kernel_name.lower() == 'rbf':
        kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")
    elif kernel_name.lower() == 'matern':
        kernel = Matern(length_scale=length_scale, nu=matern_nu, length_scale_bounds="fixed")
    elif kernel_name.lower() == 'rational_quadratic':
        kernel = RationalQuadratic(length_scale=length_scale, alpha=rational_quadratic_alpha, length_scale_bounds="fixed")
    else:
        print(f"Warning: Kernel '{kernel_name}' not recognized. Using RBF kernel as default.")
        kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")

    # Apply GPR to the real part
    gpr_real = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=True)
    gpr_real.fit(X, y_real)
    y_real_denoised, _ = gpr_real.predict(X, return_std=True)

    # Apply GPR to the imaginary part
    gpr_imag = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=True)
    gpr_imag.fit(X, y_imag)
    y_imag_denoised, _ = gpr_imag.predict(X, return_std=True)
    
    return y_real_denoised + 1j * y_imag_denoised

def load_data(file_path):
    """Load RadioML dataset."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def augment_iq_data(X_data, theta_rad):
    """
    Augment I/Q data by rotating the I and Q channels.
    Args:
        X_data: Input data of shape (num_samples, 2, sequence_length)
        theta_rad: Rotation angle in radians
    Returns:
        Augmented data of the same shape as X_data
    """
    I_original = X_data[:, 0, :]
    Q_original = X_data[:, 1, :]
    
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    I_augmented = I_original * cos_theta - Q_original * sin_theta
    Q_augmented = I_original * sin_theta + Q_original * cos_theta
    
    X_augmented = np.stack((I_augmented, Q_augmented), axis=1)
    return X_augmented


def prepare_data(dataset, test_size=0.2, validation_split=0.1, snrs_filter=None, 
                 augment_data=False):
    """
    Prepare data for training and testing.
    
    Args:
        dataset: The loaded RadioML dataset
        test_size: Proportion of data to use for testing
        validation_split: Proportion of training data to use for validation
        snrs_filter: List of SNR values to include (None=all)
        augment_data: Boolean flag to enable/disable data augmentation on training set
        
    Returns:
        X_train, X_val, X_test: Training, validation and test data
        y_train, y_val, y_test: Training, validation and test labels
        classes: List of modulation types
    """
    # Get the list of modulation types
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    
    # Filter by SNR if specified
    if snrs_filter is None:
        snrs_to_process = sorted(list(set([k[1] for k in dataset.keys()])))
    else:
        snrs_to_process = snrs_filter
    
    # Create a mapping from modulation type to index
    mod_to_index = {mod: i for i, mod in enumerate(mods)}
    
    # Lists to hold the samples and labels
    X_list = []
    y_list = []
    
    # Collect all samples
    for mod in mods:
        for snr_val in snrs_to_process:
            key = (mod, snr_val)
            if key in dataset:
                X_list.append(dataset[key])
                y_list.append(np.ones(len(dataset[key])) * mod_to_index[mod])
    
    # Convert lists to numpy arrays
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list).astype(int)
    
    # Split data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42, stratify=y_all)
    
    # Further split training data into training and validation sets
    # Adjust validation_split calculation if test_size is 0
    if 1 - test_size == 0: # Avoid division by zero if test_size is 1.0
        val_size_adjusted = 0
    else:
        val_size_adjusted = validation_split / (1 - test_size)

    if val_size_adjusted >= 1.0: # Ensure val_size_adjusted is less than 1
        val_size_adjusted = 0.5 # Or some other sensible default if validation_split is too high for the remaining data
        print(f"Warning: validation_split too high for remaining data after test split. Adjusted val_size to {val_size_adjusted}")

    if val_size_adjusted > 0 and X_train_val.shape[0] > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val)
    else: # If val_size_adjusted is 0 or X_train_val is empty
        X_train, y_train = X_train_val, y_train_val
        # Initialize X_val, y_val as empty arrays with correct number of dimensions if X_train is not empty
        if X_train.ndim == 3: # For X data (num_samples, 2, sequence_length)
             X_val = np.array([]).reshape(0, X_train.shape[1], X_train.shape[2]) if X_train.size > 0 else np.array([]).reshape(0,2,0) # handle case where X_train could be (0,2,128)
        elif X_train.ndim == 2: # For X data (num_samples, features)
             X_val = np.array([]).reshape(0, X_train.shape[1]) if X_train.size > 0 else np.array([]).reshape(0,0)
        else: # Fallback for 1D or other
             X_val = np.array([])
        y_val = np.array([])


    # Data Augmentation for training set
    if augment_data and X_train.shape[0] > 0:
        print(f"Starting data augmentation: 11 rotations, each by 30 degree increments.")
        X_original_for_aug = X_train.copy()
        y_original_for_aug = y_train.copy()
        
        augmented_X_accumulated = []
        augmented_y_accumulated = []
        
        for i in range(11): # 0 to 10 for 11 augmentations
            current_angle_deg = (i + 1) * 30.0
            print(f"Augmenting training data: rotation {i+1}/11, angle: {current_angle_deg} degrees.")
            theta_rad = np.deg2rad(current_angle_deg)
            X_augmented_single = augment_iq_data(X_original_for_aug, theta_rad)
            
            augmented_X_accumulated.append(X_augmented_single)
            augmented_y_accumulated.append(y_original_for_aug) # Append original labels for this augmented set
            
        if augmented_X_accumulated:
            X_train = np.concatenate([X_train] + augmented_X_accumulated, axis=0)
            y_train = np.concatenate([y_train] + augmented_y_accumulated, axis=0)
        
        print(f"Size of training set before augmentation: {X_original_for_aug.shape[0]}")
        print(f"Number of augmentations performed: 11")
        print(f"Size of training set after augmentation: {X_train.shape[0]}")

    # Convert labels to one-hot encoding
    num_classes = len(mods)
    if y_train.size > 0: # Check if y_train is not empty before one-hot encoding
        y_train = to_categorical(y_train, num_classes)
    else: # Handle empty y_train
        y_train = np.array([]).reshape(0, num_classes)

    if y_val.size > 0: # Check if y_val is not empty
        y_val = to_categorical(y_val, num_classes)
    else: # Handle empty y_val
        y_val = np.array([]).reshape(0, num_classes)

    if y_test.size > 0: # Check if y_test is not empty
        y_test = to_categorical(y_test, num_classes)
    else: # Handle empty y_test
        y_test = np.array([]).reshape(0, num_classes)

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, mods


# def prepare_data_by_snr(dataset, test_size=0.2, validation_split=0.1, specific_snrs=None,
#                         augment_data=False, denoising_method='gpr', ddae_model_path=None,
#                         denoised_cache_dir='../denoised_datasets'):
def prepare_data_by_snr(dataset, test_size=0.2, validation_split=0.2, specific_snrs=None,
                        augment_data=False, denoising_method='gpr', ddae_model_path=None,
                        denoised_cache_dir='../denoised_datasets'):
    """
    Organize data for training and testing, keeping samples separated by SNR.
    Useful for evaluating performance across different SNRs.
    Can also augment training data.
    
    Args:
        dataset: The loaded RadioML dataset
        test_size: Proportion of data to use for testing
        validation_split: Proportion of training data to use for validation
        specific_snrs: List of SNR values to include (None=all)
        augment_data: Boolean flag to enable/disable data augmentation on training set
        denoising_method (str): Denoising method to apply ('gpr', 'wavelet', 'ddae', 'none'). Defaults to 'gpr'.
        ddae_model_path (str): Path to the trained DDAE model if using 'ddae' method.
        denoised_cache_dir (str): Directory to save/load denoised datasets. Defaults to '../denoised_datasets'.

    Returns:
        X_train, X_val, X_test: Training, validation and test data
        y_train, y_val, y_test: Training, validation and test labels
        snr_train, snr_val, snr_test: SNR values for each sample
        classes: List of modulation types
    """
    # Get the list of modulation types and SNRs
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    if specific_snrs is None:
        snrs_list = sorted(list(set([k[1] for k in dataset.keys()])))
    else:
        snrs_list = specific_snrs
        
    # Create a mapping from modulation type to index
    mod_to_index = {mod: i for i, mod in enumerate(mods)}
    
    # Lists to hold the samples, labels, and SNR values
    X_all_list = []
    y_all_list = []
    snr_values_all_list = []
    
    # Collect all samples
    for mod in mods:
        for snr_val in snrs_list:
            key = (mod, snr_val)
            if key in dataset:
                X_all_list.append(dataset[key])
                y_all_list.append(np.ones(len(dataset[key])) * mod_to_index[mod])
                snr_values_all_list.append(np.ones(len(dataset[key])) * snr_val)
    
    # Convert lists to numpy arrays (may be overridden by efficient modes below)
    X_all = np.vstack(X_all_list) if X_all_list else np.empty((0, 2, 0))
    y_all = np.hstack(y_all_list).astype(int) if y_all_list else np.array([], dtype=int)
    snr_values_all = np.hstack(snr_values_all_list) if snr_values_all_list else np.array([])

    # Check for cached denoised data and apply denoising if needed
    if denoising_method.lower() != 'none':
        # Generate filename for cached denoised data
        denoised_filename = create_denoised_filename(denoising_method, specific_snrs, ddae_model_path)
        print(f"Checking for cached denoised data: {denoised_filename}")
        
        # Try to load existing denoised data
        cached_data = load_denoised_dataset(denoised_filename, denoised_cache_dir)
        
        if cached_data is not None:
            print("Using cached denoised dataset.")
            X_all, y_all, snr_values_all = cached_data
        else:
            print(f"No cached data found. Applying {denoising_method} denoising to the dataset...")
            if X_all.shape[0] == 0:
                print("X_all is empty. Skipping denoising.")
            else:
                method = denoising_method.lower()
                if method in ('efficient_gpr', 'efficient_gpr_per_sample'):
                    # 过滤到所需 SNR 的子集再执行高效去噪
                    filtered_dataset = {k: v for k, v in dataset.items() if (k[0] in mods and k[1] in snrs_list)}
                    mode = 'per-key' if method == 'efficient_gpr' else 'per-sample'
                    denoised_ds, _ = _run_efficient_gpr_denoising(filtered_dataset, mode)

                    # 按与原来相同的顺序重建 X_all/y_all/snr_values_all
                    X_list_dn, y_list_dn, snr_list_dn = [], [], []
                    for mod in mods:
                        for snr_val in snrs_list:
                            key = (mod, snr_val)
                            if key in denoised_ds:
                                arr = denoised_ds[key]
                                if len(arr) == 0:
                                    continue
                                X_list_dn.append(arr)
                                y_list_dn.append(np.ones(len(arr)) * mod_to_index[mod])
                                snr_list_dn.append(np.ones(len(arr)) * snr_val)
                    X_all = np.vstack(X_list_dn) if X_list_dn else np.empty((0, 2, 0))
                    y_all = np.hstack(y_list_dn).astype(int) if y_list_dn else np.array([], dtype=int)
                    snr_values_all = np.hstack(snr_list_dn) if snr_list_dn else np.array([])

                    # 缓存保存
                    save_denoised_dataset(X_all, y_all, snr_values_all, denoised_filename, denoised_cache_dir)
                else:
                    # 原有逐样本处理流程（gpr/wavelet/ddae）
                    total_samples = X_all.shape[0]
                    progress_step = max(1, total_samples // 100)
                    print(f"Total samples to process: {total_samples}")

                    for i in range(X_all.shape[0]):
                        if i % progress_step == 0 or i == total_samples - 1:
                            progress_percent = (i + 1) / total_samples * 100
                            print(f"Processing sample {i+1}/{total_samples} ({progress_percent:.1f}% complete)")

                        current_snr = snr_values_all[i]
                        i_component = X_all[i, 0, :]
                        q_component = X_all[i, 1, :]
                        complex_signal = i_component + 1j * q_component

                        denoised_signal = complex_signal

                        if method == 'gpr':
                            total_power = calculate_power(i_component, q_component)
                            noise_std = estimate_noise_std(total_power, current_snr)
                            length_scale_val = 5.0 if current_snr >= 0 else min(10, 5.0 - current_snr * 0.25)
                            denoised_signal = apply_gp_regression(complex_signal, noise_std, kernel_name='rbf', length_scale=length_scale_val)
                        elif method == 'wavelet':
                            denoised_signal = apply_wavelet_denoising(complex_signal, wavelet='db4', level=2)
                        elif method == 'ddae':
                            denoised_signal = apply_ddae_denoising(complex_signal, model_path=ddae_model_path)
                        else:
                            if i == 0:
                                print(f"Warning: Denoising method '{denoising_method}' not recognized. Skipping denoising.")
                            denoised_signal = complex_signal

                        X_all[i, 0, :] = np.real(denoised_signal)
                        X_all[i, 1, :] = np.imag(denoised_signal)

                    print(f"{denoising_method} application complete.")
                    save_denoised_dataset(X_all, y_all, snr_values_all, denoised_filename, denoised_cache_dir)
    else:
        print("No denoising method applied.")
        
    # Split data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test, snr_train_val, snr_test = train_test_split(
        X_all, y_all, snr_values_all, test_size=test_size, random_state=42, stratify=y_all
    )
    
    # Further split training data into training and validation sets
    if 1 - test_size == 0: 
        val_size_adjusted = 0
    else:
        val_size_adjusted = validation_split / (1 - test_size)
    
    if val_size_adjusted >= 1.0: 
        val_size_adjusted = 0.5 
        print(f"Warning: validation_split too high for remaining data after test split. Adjusted val_size to {val_size_adjusted}")

    if val_size_adjusted > 0 and X_train_val.shape[0] > 0 :
        X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
            X_train_val, y_train_val, snr_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val
        )
    else: 
        X_train, y_train, snr_train = X_train_val, y_train_val, snr_train_val
        if X_train.ndim == 3:
             X_val = np.array([]).reshape(0, X_train.shape[1], X_train.shape[2]) if X_train.size > 0 else np.array([]).reshape(0,2,0)
        elif X_train.ndim == 2:
             X_val = np.array([]).reshape(0, X_train.shape[1]) if X_train.size > 0 else np.array([]).reshape(0,0)
        else: 
             X_val = np.array([])
        y_val = np.array([]) 
        snr_val = np.array([])


    # Data Augmentation for training set
    if augment_data and X_train.shape[0] > 0:
        print(f"Starting data augmentation for SNR-specific data: 11 rotations, each by 30 degree increments.")
        X_original_for_aug = X_train.copy()
        y_original_for_aug = y_train.copy()
        snr_original_for_aug = snr_train.copy()

        augmented_X_accumulated = []
        augmented_y_accumulated = []
        augmented_snr_accumulated = []

        angle = 90
        num = 360 // angle - 1
        for i in range(num): # 0 to 6 for 7 augmentations
            current_angle_deg = (i + 1) * angle
            print(f"Augmenting training data (SNR-specific): rotation {i+1}/{num}, angle: {current_angle_deg} degrees.")
            theta_rad = np.deg2rad(current_angle_deg)
            X_augmented_single = augment_iq_data(X_original_for_aug, theta_rad)
            
            augmented_X_accumulated.append(X_augmented_single)
            augmented_y_accumulated.append(y_original_for_aug)
            augmented_snr_accumulated.append(snr_original_for_aug)
            
        if augmented_X_accumulated:
            X_train = np.concatenate([X_train] + augmented_X_accumulated, axis=0)
            y_train = np.concatenate([y_train] + augmented_y_accumulated, axis=0)
            snr_train = np.concatenate([snr_train] + augmented_snr_accumulated, axis=0)

        print(f"Size of training set before augmentation (SNR-specific): {X_original_for_aug.shape[0]}")
        print(f"Number of augmentations performed: 7")
        print(f"Size of training set after augmentation (SNR-specific): {X_train.shape[0]}")

    # Convert labels to one-hot encoding
    num_classes = len(mods)
    if y_train.size > 0:
        y_train = to_categorical(y_train, num_classes)
    else:
        y_train = np.array([]).reshape(0, num_classes)

    if y_val.size > 0:
        y_val = to_categorical(y_val, num_classes)
    else: 
        y_val = np.array([]).reshape(0, num_classes)

    if y_test.size > 0:
        y_test = to_categorical(y_test, num_classes)
    else:
        y_test = np.array([]).reshape(0, num_classes)

    print(f"Training set: {X_train.shape}, {y_train.shape}, SNR array: {snr_train.shape if snr_train.size > 0 else 'empty'}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}, SNR array: {snr_val.shape if snr_val.size > 0 else 'empty'}")
    print(f"Test set: {X_test.shape}, {y_test.shape}, SNR array: {snr_test.shape if snr_test.size > 0 else 'empty'}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods


def prepare_data_by_snr_stratified(dataset, test_size=0.2, validation_split=0.1, specific_snrs=None,
                                   augment_data=False, denoising_method='gpr', ddae_model_path=None,
                                   denoised_cache_dir='../denoised_datasets'):
    """
    Organize data for training and testing with stratified splitting by both modulation type and SNR.
    This ensures balanced distribution of (modulation, SNR) combinations across train/val/test splits.
    
    Args:
        dataset: The loaded RadioML dataset
        test_size: Proportion of data to use for testing
        validation_split: Proportion of training data to use for validation
        specific_snrs: List of SNR values to include (None=all)
        augment_data: Boolean flag to enable/disable data augmentation on training set
        denoising_method (str): Denoising method to apply ('gpr', 'wavelet', 'ddae', 'none'). Defaults to 'gpr'.
        ddae_model_path (str): Path to the trained DDAE model if using 'ddae' method.
        denoised_cache_dir (str): Directory to save/load denoised datasets. Defaults to '../denoised_datasets'.

    Returns:
        X_train, X_val, X_test: Training, validation and test data
        y_train, y_val, y_test: Training, validation and test labels
        snr_train, snr_val, snr_test: SNR values for each sample
        classes: List of modulation types
    """
    # Get the list of modulation types and SNRs
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    if specific_snrs is None:
        snrs_list = sorted(list(set([k[1] for k in dataset.keys()])))
    else:
        snrs_list = specific_snrs
        
    # Create a mapping from modulation type to index
    mod_to_index = {mod: i for i, mod in enumerate(mods)}
    
    # Lists to hold the samples, labels, SNR values, and composite labels
    X_all_list = []
    y_all_list = []
    snr_values_all_list = []
    composite_labels_list = []
    
    # Collect all samples with composite labels
    for mod in mods:
        for snr_val in snrs_list:
            key = (mod, snr_val)
            if key in dataset:
                samples = dataset[key]
                num_samples = len(samples)
                
                X_all_list.append(samples)
                y_all_list.append(np.ones(num_samples) * mod_to_index[mod])
                snr_values_all_list.append(np.ones(num_samples) * snr_val)
                
                # Create composite labels for stratification: "modulation_snr"
                # Use string representation for stratification
                composite_label = f"{mod}_{snr_val}"
                composite_labels_list.extend([composite_label] * num_samples)
    
    # Convert lists to numpy arrays
    X_all = np.vstack(X_all_list)
    y_all = np.hstack(y_all_list).astype(int)
    snr_values_all = np.hstack(snr_values_all_list)
    composite_labels = np.array(composite_labels_list)
    
    print(f"Dataset loaded: {X_all.shape[0]} samples with {len(np.unique(composite_labels))} unique (modulation, SNR) combinations")

    # Check for cached denoised data and apply denoising if needed
    if denoising_method.lower() != 'none':
        # Generate filename for cached denoised data (same as regular method)
        denoised_filename = create_denoised_filename(denoising_method, specific_snrs, ddae_model_path)
        print(f"Checking for cached denoised data: {denoised_filename}")
        
        # Try to load existing denoised data
        cached_data = load_denoised_dataset(denoised_filename, denoised_cache_dir)
        
        if cached_data is not None:
            print("Using cached denoised dataset.")
            X_all, y_all, snr_values_all = cached_data
        else:
            print(f"No cached data found. Applying {denoising_method} denoising to the dataset...")
            if X_all.shape[0] == 0:
                print("X_all is empty. Skipping denoising.")
            else:
                method = denoising_method.lower()
                if method in ('efficient_gpr', 'efficient_gpr_per_sample'):
                    # 仅保留需要的 SNR 键进行高效去噪
                    filtered_dataset = {k: v for k, v in dataset.items() if (k[0] in mods and k[1] in snrs_list)}
                    mode = 'per-key' if method == 'efficient_gpr' else 'per-sample'
                    denoised_ds, _ = _run_efficient_gpr_denoising(filtered_dataset, mode)

                    # 以相同次序重建数组与复合标签
                    X_list_dn, y_list_dn, snr_list_dn, composite_labels_list = [], [], [], []
                    for mod in mods:
                        for snr_val in snrs_list:
                            key = (mod, snr_val)
                            if key in denoised_ds:
                                arr = denoised_ds[key]
                                if len(arr) == 0:
                                    continue
                                X_list_dn.append(arr)
                                y_list_dn.append(np.ones(len(arr)) * mod_to_index[mod])
                                snr_list_dn.append(np.ones(len(arr)) * snr_val)
                                composite_labels_list.extend([f"{mod}_{snr_val}"] * len(arr))
                    X_all = np.vstack(X_list_dn) if X_list_dn else np.empty((0, 2, 0))
                    y_all = np.hstack(y_list_dn).astype(int) if y_list_dn else np.array([], dtype=int)
                    snr_values_all = np.hstack(snr_list_dn) if snr_list_dn else np.array([])
                    composite_labels = np.array(composite_labels_list)

                    # 缓存保存
                    save_denoised_dataset(X_all, y_all, snr_values_all, denoised_filename, denoised_cache_dir)
                else:
                    # 原有逐样本处理流程
                    total_samples = X_all.shape[0]
                    progress_step = max(1, total_samples // 100)
                    print(f"Total samples to process: {total_samples}")

                    for i in range(X_all.shape[0]):
                        if i % progress_step == 0 or i == total_samples - 1:
                            progress_percent = (i + 1) / total_samples * 100
                            print(f"Processing sample {i+1}/{total_samples} ({progress_percent:.1f}% complete)")

                        current_snr = snr_values_all[i]
                        i_component = X_all[i, 0, :]
                        q_component = X_all[i, 1, :]
                        complex_signal = i_component + 1j * q_component

                        denoised_signal = complex_signal

                        if method == 'gpr':
                            total_power = calculate_power(i_component, q_component)
                            noise_std = estimate_noise_std(total_power, current_snr)
                            length_scale_val = 5.0 if current_snr >= 0 else min(10, 5.0 - current_snr * 0.25)
                            denoised_signal = apply_gp_regression(complex_signal, noise_std, kernel_name='rbf', length_scale=length_scale_val)
                        elif method == 'wavelet':
                            denoised_signal = apply_wavelet_denoising(complex_signal, wavelet='db4', level=2)
                        elif method == 'ddae':
                            denoised_signal = apply_ddae_denoising(complex_signal, model_path=ddae_model_path)
                        else:
                            if i == 0:
                                print(f"Warning: Denoising method '{denoising_method}' not recognized. Skipping denoising.")
                            denoised_signal = complex_signal

                        X_all[i, 0, :] = np.real(denoised_signal)
                        X_all[i, 1, :] = np.imag(denoised_signal)

                    print(f"{denoising_method} application complete.")
                    save_denoised_dataset(X_all, y_all, snr_values_all, denoised_filename, denoised_cache_dir)
    else:
        print("No denoising method applied.")
        
    # Stratified split by composite labels (modulation + SNR)
    print("Performing stratified split by (modulation, SNR) combinations...")
    try:
        X_train_val, X_test, y_train_val, y_test, snr_train_val, snr_test, comp_train_val, comp_test = train_test_split(
            X_all, y_all, snr_values_all, composite_labels, 
            test_size=test_size, random_state=42, stratify=composite_labels
        )
        print(f"Successfully stratified by {len(np.unique(composite_labels))} (modulation, SNR) combinations")
    except ValueError as e:
        print(f"Warning: Stratified split failed ({e}). Falling back to standard split by modulation only.")
        X_train_val, X_test, y_train_val, y_test, snr_train_val, snr_test = train_test_split(
            X_all, y_all, snr_values_all, test_size=test_size, random_state=42, stratify=y_all
        )
    
    # Further split training data into training and validation sets with stratification
    if 1 - test_size == 0: 
        val_size_adjusted = 0
    else:
        val_size_adjusted = validation_split / (1 - test_size)
    
    if val_size_adjusted >= 1.0: 
        val_size_adjusted = 0.5 
        print(f"Warning: validation_split too high for remaining data after test split. Adjusted val_size to {val_size_adjusted}")

    if val_size_adjusted > 0 and X_train_val.shape[0] > 0:
        try:
            # Create composite labels for the remaining training+validation data
            comp_train_val_remaining = []
            for i in range(len(y_train_val)):
                mod_idx = y_train_val[i]
                snr_val = snr_train_val[i]
                mod_name = mods[mod_idx]
                comp_train_val_remaining.append(f"{mod_name}_{snr_val}")
            
            X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
                X_train_val, y_train_val, snr_train_val, 
                test_size=val_size_adjusted, random_state=42, 
                stratify=comp_train_val_remaining
            )
            print("Successfully applied stratified validation split")
        except ValueError as e:
            print(f"Warning: Stratified validation split failed ({e}). Using standard split.")
            X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
                X_train_val, y_train_val, snr_train_val, 
                test_size=val_size_adjusted, random_state=42, stratify=y_train_val
            )
    else: 
        X_train, y_train, snr_train = X_train_val, y_train_val, snr_train_val
        if X_train.ndim == 3:
             X_val = np.array([]).reshape(0, X_train.shape[1], X_train.shape[2]) if X_train.size > 0 else np.array([]).reshape(0,2,0)
        elif X_train.ndim == 2:
             X_val = np.array([]).reshape(0, X_train.shape[1]) if X_train.size > 0 else np.array([]).reshape(0,0)
        else: 
             X_val = np.array([])
        y_val = np.array([]) 
        snr_val = np.array([])

    # Data Augmentation for training set
    if augment_data and X_train.shape[0] > 0:
        print(f"Starting data augmentation for stratified data: 3 rotations at 90-degree increments.")
        X_original_for_aug = X_train.copy()
        y_original_for_aug = y_train.copy()
        snr_original_for_aug = snr_train.copy()

        augmented_X_accumulated = []
        augmented_y_accumulated = []
        augmented_snr_accumulated = []

        angle = 90
        num = 360 // angle - 1  # 3 rotations: 90°, 180°, 270°
        for i in range(num):
            current_angle_deg = (i + 1) * angle
            print(f"Augmenting training data (stratified): rotation {i+1}/{num}, angle: {current_angle_deg} degrees.")
            theta_rad = np.deg2rad(current_angle_deg)
            X_augmented_single = augment_iq_data(X_original_for_aug, theta_rad)
            
            augmented_X_accumulated.append(X_augmented_single)
            augmented_y_accumulated.append(y_original_for_aug)
            augmented_snr_accumulated.append(snr_original_for_aug)
            
        if augmented_X_accumulated:
            X_train = np.concatenate([X_train] + augmented_X_accumulated, axis=0)
            y_train = np.concatenate([y_train] + augmented_y_accumulated, axis=0)
            snr_train = np.concatenate([snr_train] + augmented_snr_accumulated, axis=0)

        print(f"Size of training set before augmentation (stratified): {X_original_for_aug.shape[0]}")
        print(f"Number of augmentations performed: {num}")
        print(f"Size of training set after augmentation (stratified): {X_train.shape[0]}")

    # Convert labels to one-hot encoding
    num_classes = len(mods)
    if y_train.size > 0:
        y_train = to_categorical(y_train, num_classes)
    else:
        y_train = np.array([]).reshape(0, num_classes)

    if y_val.size > 0:
        y_val = to_categorical(y_val, num_classes)
    else: 
        y_val = np.array([]).reshape(0, num_classes)

    if y_test.size > 0:
        y_test = to_categorical(y_test, num_classes)
    else:
        y_test = np.array([]).reshape(0, num_classes)

    print(f"Stratified Training set: {X_train.shape}, {y_train.shape}, SNR array: {snr_train.shape if snr_train.size > 0 else 'empty'}")
    print(f"Stratified Validation set: {X_val.shape}, {y_val.shape}, SNR array: {snr_val.shape if snr_val.size > 0 else 'empty'}")
    print(f"Stratified Test set: {X_test.shape}, {y_test.shape}, SNR array: {snr_test.shape if snr_test.size > 0 else 'empty'}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods


if __name__ == "__main__":
    # This can be used for testing the preprocessing functions
    file_path = "../data/RML2016.10a_dict.pkl"

    try:
        dataset = load_data(file_path)
        print("Dataset loaded successfully for __main__ test.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path} when running preprocess.py directly.")
        print("Please ensure the path is correct relative to the script's execution directory.")
        dataset = None

    if dataset:
        # Basic preprocessing
        print("\nTesting prepare_data:")
        X_train, X_val, X_test, y_train, y_val, y_test, mods = prepare_data(dataset, augment_data=False)
        print("\nTesting prepare_data with augmentation:")
        X_train_aug, _, _, y_train_aug, _, _, _ = prepare_data(dataset, augment_data=True)
        
        # SNR-aware preprocessing
        print("\nTesting prepare_data_by_snr:")
        X_train_snr, X_val_snr, X_test_snr, y_train_snr, y_val_snr, y_test_snr, snr_train, snr_val, snr_test, mods_snr = prepare_data_by_snr(dataset, augment_data=False)
        print("\nTesting prepare_data_by_snr with augmentation:")
        X_train_snr_aug, _, _, y_train_snr_aug, _, _, snr_train_aug, _, _, _ = prepare_data_by_snr(dataset, augment_data=True)