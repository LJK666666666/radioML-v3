import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

from preprocess import load_data, prepare_data_by_snr
from models import build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model, build_transformer_model
# Import custom layers for model loading
from model.complex_nn_model import ComplexConv1D, ComplexBatchNormalization, ComplexDense, ComplexMagnitude, complex_relu


def load_trained_model(model_path):
    """Load a trained model from disk."""
    try:
        # Create custom objects dict for complex layers
        custom_objects = {
            'ComplexConv1D': ComplexConv1D,
            'ComplexBatchNormalization': ComplexBatchNormalization,
            'ComplexDense': ComplexDense,
            'ComplexMagnitude': ComplexMagnitude,
            'complex_relu': complex_relu
        }
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def evaluate_by_snr(model, X_test, y_test, snr_test, mods, output_dir):
    """
    Evaluate model performance by SNR values.
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels (one-hot encoded)
        snr_test: SNR value for each test example
        mods: List of modulation types
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert one-hot encoded labels back to class indices
    y_true = np.argmax(y_test, axis=1)
    
    # Get model predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate overall accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Get unique SNR values
    snrs = np.unique(snr_test)
    
    # Calculate accuracy by SNR
    snr_accuracies = []
    for snr in snrs:
        # Get indices for this SNR
        indices = np.where(snr_test == snr)[0]
        
        # Calculate accuracy for this SNR
        snr_y_true = y_true[indices]
        snr_y_pred = y_pred_classes[indices]
        snr_accuracy = np.mean(snr_y_pred == snr_y_true)
        
        snr_accuracies.append(snr_accuracy)
        print(f"SNR {snr} dB: Accuracy = {snr_accuracy:.4f}")
    
    # Plot accuracy vs. SNR
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, snr_accuracies, 'o-')
    plt.grid(True)
    plt.xlabel('Signal-to-Noise Ratio (dB)')
    plt.ylabel('Classification Accuracy')
    plt.title('Classification Accuracy vs. SNR')
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_snr.png'))
    plt.close()
    
    # Save accuracy by SNR to CSV
    df = pd.DataFrame({'SNR': snrs, 'Accuracy': snr_accuracies})
    df.to_csv(os.path.join(output_dir, 'accuracy_by_snr.csv'), index=False)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=mods, yticklabels=mods)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(y_true, y_pred_classes, target_names=mods)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Calculate accuracy by modulation type and SNR
    mod_snr_acc = np.zeros((len(mods), len(snrs)))
    
    for i, mod in enumerate(mods):
        for j, snr in enumerate(snrs):
            # Get indices for this modulation and SNR
            mod_indices = np.where(y_true == i)[0]
            snr_indices = np.where(snr_test == snr)[0]
            indices = np.intersect1d(mod_indices, snr_indices)
            
            if len(indices) > 0:
                # Calculate accuracy
                acc = np.mean(y_pred_classes[indices] == y_true[indices])
                mod_snr_acc[i, j] = acc
    
    # Plot accuracy heatmap by modulation and SNR
    plt.figure(figsize=(12, 10))
    sns.heatmap(mod_snr_acc, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=snrs, yticklabels=mods)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Modulation Type')
    plt.title('Classification Accuracy by Modulation Type and SNR')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_mod_snr.png'))
    plt.close()
    
    # Return overall accuracy
    return accuracy


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Define paths
    dataset_path = "../RML2016.10a_dict.pkl"
    models_dir = "../models"
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_data(dataset_path)
    
    if not dataset:
        print("Failed to load dataset")
        return
    
    # Prepare data for evaluation, keeping track of SNR values
    print("Preparing data with SNR tracking...")
    X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_data_by_snr(dataset)
    
    # Print dataset information
    print(f"Number of modulation types: {len(mods)}")
    print(f"Modulation types: {mods}")
    print(f"SNR values: {np.unique(snr_test)}")
    
    # Get input shape from the data
    input_shape = X_train.shape[1:]
    num_classes = len(mods)
    
    # Evaluate CNN1D model
    cnn1d_model_path = os.path.join(models_dir, "cnn1d_model.keras") # Changed to .keras
    if os.path.exists(cnn1d_model_path):
        print("\n" + "="*50)
        print("Evaluating CNN1D Model")
        print("="*50)
        cnn1d_model = load_trained_model(cnn1d_model_path)
        if cnn1d_model:
            cnn1d_accuracy = evaluate_by_snr(
                cnn1d_model, 
                X_test, 
                y_test, 
                snr_test, 
                mods, 
                os.path.join(results_dir, "cnn1d")
            )
    
    # Reshape for CNN2D model
    X_test_2d = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # Evaluate CNN2D model
    cnn2d_model_path = os.path.join(models_dir, "cnn2d_model.keras") # Changed to .keras
    if os.path.exists(cnn2d_model_path):
        print("\n" + "="*50)
        print("Evaluating CNN2D Model")
        print("="*50)
        cnn2d_model = load_trained_model(cnn2d_model_path)
        if cnn2d_model:
            # X_test is used directly as build_cnn2d_model handles reshape internally
            evaluate_by_snr(cnn2d_model, X_test, y_test, snr_test, mods, 
                            os.path.join(results_dir, "cnn2d_evaluation_results"))
    
    # Evaluate ResNet model
    resnet_model_path = os.path.join(models_dir, "resnet_model.keras") # Changed to .keras
    if os.path.exists(resnet_model_path):
        print("\n" + "="*50)
        print("Evaluating ResNet Model")
        print("="*50)
        resnet_model = load_trained_model(resnet_model_path)
        if resnet_model:
            evaluate_by_snr(resnet_model, X_test, y_test, snr_test, mods, 
                            os.path.join(results_dir, "resnet_evaluation_results"))

    # Evaluate ComplexNN model
    complex_nn_model_path = os.path.join(models_dir, "complex_nn_model.keras")
    if os.path.exists(complex_nn_model_path):
        print("\n" + "="*50)
        print("Evaluating ComplexNN Model")
        print("="*50)
        complex_nn_model = load_trained_model(complex_nn_model_path)
        if complex_nn_model:
            evaluate_by_snr(complex_nn_model, X_test, y_test, snr_test, mods, 
                            os.path.join(results_dir, "complex_nn_evaluation_results"))

    # Evaluate Transformer model
    transformer_model_path = os.path.join(models_dir, "transformer_model.keras")
    if os.path.exists(transformer_model_path):
        print("\n" + "="*50)
        print("Evaluating Transformer Model")
        print("="*50)
        transformer_model = load_trained_model(transformer_model_path)
        if transformer_model:
            evaluate_by_snr(transformer_model, X_test, y_test, snr_test, mods,
                            os.path.join(results_dir, "transformer_evaluation_results"))


if __name__ == "__main__":
    main()