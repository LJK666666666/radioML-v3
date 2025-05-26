from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Activation
from keras.models import Model, load_model
from keras.optimizers import Adam

def build_dae_model(input_shape):
    """
    Builds a Denoising Autoencoder (DAE) model.
    Args:
        input_shape (tuple): Shape of the input data (sequence_length, num_channels).
    Returns:
        keras.models.Model: The constructed DAE model.
    """
    inputs = Input(shape=input_shape)

    # Encoder
    # Block 1
    x = Conv1D(filters=64, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)  # 128 -> 64

    # Block 2
    x = Conv1D(filters=32, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)  # 64 -> 32

    # Bottleneck (Encoded representation)
    encoded = Conv1D(filters=16, kernel_size=7, padding='same', activation='relu')(x)
    # No max pooling here, this is the bottleneck

    # Decoder
    # Block 1
    x = Conv1D(filters=32, kernel_size=7, padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling1D(size=2)(x)  # 32 -> 64

    # Block 2
    x = Conv1D(filters=64, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling1D(size=2)(x)  # 64 -> 128
    
    # Output layer
    # Reconstruct the original number of channels (e.g., 2 for I/Q)
    # Using 'linear' activation for reconstruction, as signal values can be positive or negative.
    # 'tanh' could also be an option if inputs are normalized to [-1, 1].
    decoded = Conv1D(filters=input_shape[-1], kernel_size=7, activation='linear', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    return autoencoder

def compile_dae_model(model, learning_rate=1e-3):
    """
    Compiles the DAE model.
    Args:
        model (keras.models.Model): The DAE model to compile.
        learning_rate (float): Learning rate for the Adam optimizer.
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

def save_dae_model(model, filepath):
    """
    Saves the DAE model to a file.
    Args:
        model (keras.models.Model): The DAE model to save.
        filepath (str): Path to save the model file.
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_dae_model(filepath):
    """
    Loads a DAE model from a file.
    Args:
        filepath (str): Path to the model file.
    Returns:
        keras.models.Model: The loaded DAE model.
    """
    print(f"Loading model from {filepath}...")
    # No custom objects needed for these standard layers
    model = load_model(filepath)
    print("Model loaded successfully.")
    return model

if __name__ == '__main__':
    # Example Usage
    seq_len = 128
    num_chan = 2  # For I and Q components
    shape = (seq_len, num_chan)
    
    print("Building DAE model...")
    dae = build_dae_model(input_shape=shape)
    
    print("\nCompiling DAE model...")
    compile_dae_model(dae, learning_rate=0.001)
    
    print("\nDAE Model Summary:")
    dae.summary()
    
    # Test save and load
    import tempfile
    import os

    # Create a temporary directory to ensure the file can be written
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_model_path = os.path.join(tmpdir, "temp_dae_model.keras")
        
        print(f"\nSaving model to temporary file: {temp_model_path}")
        save_dae_model(dae, temp_model_path)
        
        print("\nLoading model from temporary file...")
        loaded_dae = load_dae_model(temp_model_path)
        
        print("\nLoaded DAE Model Summary:")
        loaded_dae.summary()

    print("\nAutoencoder model functions created and tested successfully.")
