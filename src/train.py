import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model
import time

from preprocess import load_data, prepare_data
from models import build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model, build_transformer_model, get_callbacks


def train_model(model, X_train, y_train, X_val, y_val, model_path, batch_size=128, epochs=100):
    """
    Train a model and save it.
    
    Args:
        model: The model to train
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        model_path: Path to save the model
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        
    Returns:
        History object containing training history
    """
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Prepare callbacks
    callbacks = get_callbacks(model_path)
    
    # Train the model
    print(f"Training model, saving to {model_path}")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return history


def plot_training_history(history, output_path):
    """Plot and save training history (accuracy and loss)."""
    # Ensure the directory for the output plot exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path) # Save the figure
    plt.close() # Close the figure to free memory and prevent display if not intended
    
    print(f"Training history plot saved to {output_path}")


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Define paths
    dataset_path = "../RML2016.10a_dict.pkl"
    output_dir = "../models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_data(dataset_path)
    
    if not dataset:
        print("Failed to load dataset")
        return
    
    # Prepare data for training
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, mods = prepare_data(dataset)
    
    # Print dataset information
    print(f"Number of modulation types: {len(mods)}")
    print(f"Modulation types: {mods}")
    
    # Get input shape from the data
    input_shape = X_train.shape[1:]
    num_classes = len(mods)
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Train CNN1D model
    print("\n" + "="*50)
    print("Training CNN1D Model")
    print("="*50)
    cnn1d_model = build_cnn1d_model(input_shape, num_classes)
    cnn1d_model.summary()
    cnn1d_history = train_model(
        cnn1d_model, 
        X_train, y_train, 
        X_val, y_val, 
        os.path.join(output_dir, "cnn1d_model.keras")
    )
    plot_training_history(cnn1d_history, os.path.join(output_dir, "cnn1d_history.png"))
    
    # Reshape for CNN2D model
    X_train_2d = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val_2d = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    
    # Train CNN2D model
    print("\n" + "="*50)
    print("Training CNN2D Model")
    print("="*50)
    cnn2d_model = build_cnn2d_model(input_shape, num_classes)
    cnn2d_model.summary()
    cnn2d_history = train_model(
        cnn2d_model, 
        X_train_2d, y_train, 
        X_val_2d, y_val, 
        os.path.join(output_dir, "cnn2d_model.keras")
    )
    plot_training_history(cnn2d_history, os.path.join(output_dir, "cnn2d_history.png"))
    
    # Train ResNet model
    print("\n" + "="*50)
    print("Training ResNet Model")
    print("="*50)
    resnet_model = build_resnet_model(input_shape, num_classes)
    resnet_model.summary()
    resnet_history = train_model(
        resnet_model, 
        X_train, y_train, 
        X_val, y_val, 
        os.path.join(output_dir, "resnet_model.keras")
    )
    plot_training_history(resnet_history, os.path.join(output_dir, "resnet_history.png"))

    # Train ComplexNN model
    print("\n" + "="*50)
    print("Training ComplexNN Model")
    print("="*50)
    complex_nn_model = build_complex_nn_model(input_shape, num_classes)
    complex_nn_model.summary()
    complex_nn_history = train_model(
        complex_nn_model,
        X_train, y_train,
        X_val, y_val,
        os.path.join(output_dir, "complex_nn_model.keras")
    )
    plot_training_history(complex_nn_history, os.path.join(output_dir, "complex_nn_history.png"))

    # Train Transformer model
    print("\n" + "="*50)
    print("Training Transformer Model")
    print("="*50)
    transformer_model = build_transformer_model(input_shape, num_classes)
    transformer_model.summary()
    transformer_history = train_model(
        transformer_model,
        X_train, y_train,
        X_val, y_val,
        os.path.join(output_dir, "transformer_model.keras")
    )
    plot_training_history(transformer_history, os.path.join(output_dir, "transformer_history.png"))
    
    print("\nAll models trained successfully!")


if __name__ == "__main__":
    main()