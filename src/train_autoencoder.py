import argparse
import os
import numpy as np
import pickle
import tensorflow as tf
import keras # Added for keras.utils.set_random_seed
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint # Changed from tensorflow.keras.callbacks

# Add src directory to Python path to allow direct import of modules
# This might be needed if running the script directly and src is not in PYTHONPATH
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.autoencoder_model import build_dae_model, compile_dae_model, save_dae_model

def set_random_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed) # Retained as per instruction for backend seeding
    # For Keras specific random operations if any
    keras.utils.set_random_seed(seed) # Changed from tf.keras.utils
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def add_noise_to_signals(signals, target_snr_db):
    """
    Adds Gaussian noise to signals to achieve a target SNR.
    Args:
        signals (np.ndarray): Array of clean signals (num_samples, sequence_length, num_channels).
        target_snr_db (float): Target Signal-to-Noise Ratio in dB.
    Returns:
        np.ndarray: Array of noisy signals.
    """
    noisy_signals_list = []
    for i in range(signals.shape[0]):
        signal_sample = signals[i] # Shape (sequence_length, num_channels)
        
        if signal_sample.shape[1] != 2:
            raise ValueError(f"Expected signal_sample to have 2 channels (I and Q), but got {signal_sample.shape[1]}")

        I = signal_sample[:, 0]
        Q = signal_sample[:, 1]
        complex_signal = I + 1j * Q
        
        sig_power = np.mean(np.abs(complex_signal)**2)
        if sig_power == 0: # Handle zero-power signals to avoid division by zero
            # If signal power is zero, noise power also becomes undefined or zero depending on interpretation.
            # Adding zero noise or very small noise.
            noise_std = 1e-9 
        else:
            snr_linear = 10**(target_snr_db / 10)
            noise_power = sig_power / snr_linear
            noise_std = np.sqrt(noise_power / 2) # Divide by 2 for I and Q components

        noise_i = np.random.normal(0, noise_std, size=I.shape)
        noise_q = np.random.normal(0, noise_std, size=Q.shape)
        
        noisy_I = I + noise_i
        noisy_Q = Q + noise_q
        
        noisy_signal_channels = np.stack((noisy_I, noisy_Q), axis=-1)
        noisy_signals_list.append(noisy_signal_channels)
        
    return np.array(noisy_signals_list)

def load_and_prepare_data(dataset_path, high_snr_threshold=10, noisy_snr_target=0, val_split=0.2):
    """
    Loads data, selects high SNR signals as clean, generates noisy versions,
    normalizes, and splits into training/validation sets.
    Args:
        dataset_path (str): Path to the RadioML dataset.
        high_snr_threshold (int): Minimum SNR to consider a signal "clean".
        noisy_snr_target (int): Target SNR for the noisy signals.
        val_split (float): Proportion of data for validation set.
    Returns:
        tuple: (X_noisy_train, X_noisy_val, X_clean_train, X_clean_val)
    """
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')
    
    clean_signals_list = []
    print(f"Extracting clean signals with SNR >= {high_snr_threshold} dB...")
    for (mod, snr), data_samples in dataset.items():
        if snr >= high_snr_threshold:
            # Original shape is (N, 2, 128) -> I/Q components first
            # Transpose to (N, 128, 2) -> sequence_length, num_channels (I/Q)
            samples_transposed = np.transpose(data_samples, (0, 2, 1))
            clean_signals_list.append(samples_transposed)
    
    if not clean_signals_list:
        raise ValueError(f"No signals found with SNR >= {high_snr_threshold}. Check dataset or threshold.")
        
    X_clean = np.vstack(clean_signals_list)
    print(f"Total clean signals extracted: {X_clean.shape[0]}")
    
    print(f"Generating noisy signals with target SNR: {noisy_snr_target} dB...")
    X_noisy = add_noise_to_signals(X_clean, noisy_snr_target)
    
    print("Normalizing signals...")
    # Global normalization based on the absolute maximum value in both noisy and clean sets
    # This ensures that both input (noisy) and target (clean) are scaled consistently.
    # Important for autoencoders where output should match the scale of the target.
    temp_for_norm = np.vstack((X_noisy.reshape(-1, X_noisy.shape[-1]), 
                               X_clean.reshape(-1, X_clean.shape[-1])))
    max_val = np.max(np.abs(temp_for_norm))
    
    if max_val == 0:
        print("Warning: Max value for normalization is 0. Skipping normalization.")
    else:
        X_noisy = X_noisy / max_val
        X_clean = X_clean / max_val
        print(f"Signals normalized by global max value: {max_val:.4f}")

    print("Splitting data into training and validation sets...")
    X_noisy_train, X_noisy_val, X_clean_train, X_clean_val = train_test_split(
        X_noisy, X_clean, test_size=val_split, random_state=42
    )
    print(f"Training set: Noisy={X_noisy_train.shape}, Clean={X_clean_train.shape}")
    print(f"Validation set: Noisy={X_noisy_val.shape}, Clean={X_clean_val.shape}")
    
    return X_noisy_train, X_noisy_val, X_clean_train, X_clean_val

def main():
    parser = argparse.ArgumentParser(description='Train Denoising Autoencoder (DAE) for RadioML signals.')
    parser.add_argument('--dataset_path', type=str, default='../RML2016.10a_dict.pkl',
                        help='Path to the RadioML dataset file.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--high_snr_threshold', type=int, default=10,
                        help='Minimum SNR (dB) to consider signals as clean.')
    parser.add_argument('--noisy_snr_target', type=int, default=0,
                        help='Target SNR (dB) for generating noisy signals.')
    parser.add_argument('--output_model_dir', type=str, default='../model_weight_saved/',
                        help='Directory to save the trained DAE model.')
    parser.add_argument('--output_model_name', type=str, default='ddae_model.keras',
                        help='Filename for the saved DAE model.')
    parser.add_argument('--input_seq_length', type=int, default=128,
                        help='Sequence length of the input signals.')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='Number of channels in the input signals (e.g., 2 for I/Q).')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    
    args = parser.parse_args()

    set_random_seed(args.random_seed)

    os.makedirs(args.output_model_dir, exist_ok=True)
    output_model_path = os.path.join(args.output_model_dir, args.output_model_name)

    # Load and prepare data
    try:
        X_noisy_train, X_noisy_val, X_clean_train, X_clean_val = load_and_prepare_data(
            args.dataset_path,
            args.high_snr_threshold,
            args.noisy_snr_target
        )
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset_path}")
        print("Please ensure the path is correct. If running from src/, it might be '../RML2016.10a_dict.pkl'")
        return
    except ValueError as e:
        print(f"Error during data preparation: {e}")
        return

    # Build and compile model
    input_shape = (args.input_seq_length, args.num_channels)
    dae_model = build_dae_model(input_shape)
    compile_dae_model(dae_model, args.learning_rate)
    
    print("\nDAE Model Summary:")
    dae_model.summary()

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True, 
        verbose=1
    )
    model_checkpoint = ModelCheckpoint(
        filepath=output_model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )

    # Train model
    print("\nStarting DAE model training...")
    history = dae_model.fit(
        X_noisy_train, X_clean_train,
        validation_data=(X_noisy_val, X_clean_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )
    print("Training complete.")
    print(f"Best model saved at {output_model_path} by ModelCheckpoint.")

    # Optionally save the final model explicitly if ModelCheckpoint might not be the last epoch
    # For example, if training finishes by epochs rather than early stopping.
    final_model_path = os.path.join(args.output_model_dir, args.output_model_name.replace('.keras', '_final.keras'))
    save_dae_model(dae_model, final_model_path)
    print(f"Final model state (after all epochs or early stopping) saved to {final_model_path}")

if __name__ == '__main__':
    main()
