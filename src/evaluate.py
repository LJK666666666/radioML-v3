import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

from preprocess import load_data, prepare_data_by_snr_stratified
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
    y_pred = model.predict(X_test, batch_size=128)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate overall accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Save overall accuracy to file
    with open(os.path.join(output_dir, 'overall_accuracy.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Correct predictions: {np.sum(y_pred_classes == y_true)}\n")
    
    print(f"Overall accuracy saved to {os.path.join(output_dir, 'overall_accuracy.txt')}")
    
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
    
    # Save comprehensive results summary
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Total Samples: {len(y_true)}\n")
        f.write(f"Correct Predictions: {np.sum(y_pred_classes == y_true)}\n")
        f.write(f"Number of SNR Values: {len(snrs)}\n")
        f.write(f"SNR Range: {snrs.min()} to {snrs.max()} dB\n\n")
        
        f.write("ACCURACY BY SNR:\n")
        f.write("-"*30 + "\n")
        for snr, acc in zip(snrs, snr_accuracies):
            f.write(f"SNR {snr:2.0f} dB: {acc:.4f}\n")
        f.write("\n")
        
        f.write(f"Best SNR Performance: {snrs[np.argmax(snr_accuracies)]} dB ({max(snr_accuracies):.4f})\n")
        f.write(f"Worst SNR Performance: {snrs[np.argmin(snr_accuracies)]} dB ({min(snr_accuracies):.4f})\n")
        f.write(f"Average SNR Accuracy: {np.mean(snr_accuracies):.4f}\n")
    
    print(f"Evaluation summary saved to {os.path.join(output_dir, 'evaluation_summary.txt')}")
    
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
    
    for i, _ in enumerate(mods):
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


def get_available_models():
    """Return list of available model types for evaluation"""
    return ['cnn1d', 'cnn2d', 'resnet', 'complex_nn']


def evaluate_single_model(model_name, models_dir, results_dir, X_test, y_test, snr_test, mods):
    """
    Evaluate a single model by name.

    Args:
        model_name: Name of the model to evaluate
        models_dir: Directory containing trained models
        results_dir: Directory to save evaluation results
        X_test, y_test, snr_test, mods: Test data and metadata

    Returns:
        accuracy: Overall accuracy of the model, or None if evaluation failed
    """
    model_path = os.path.join(models_dir, f"{model_name}_model.keras")

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    print(f"\n{'='*50}")
    print(f"Evaluating {model_name.upper()} Model")
    print(f"{'='*50}")

    # Load the model
    model = load_trained_model(model_path)
    if not model:
        print(f"Failed to load model: {model_name}")
        return None

    # Handle special data preparation for CNN2D
    if model_name == 'cnn2d':
        # For CNN2D, we use the original X_test (the model handles reshaping internally)
        test_data = X_test
    else:
        test_data = X_test

    # Evaluate the model
    try:
        accuracy = evaluate_by_snr(
            model,
            test_data,
            y_test,
            snr_test,
            mods,
            os.path.join(results_dir, f"{model_name}_evaluation_results")
        )
        print(f"Successfully evaluated {model_name} model")
        return accuracy
    except Exception as e:
        print(f"Error evaluating {model_name} model: {e}")
        return None


def evaluate_selected_models(model_names, models_dir, results_dir, X_test, y_test, snr_test, mods):
    """
    Evaluate multiple models by name.

    Args:
        model_names: List of model names to evaluate
        models_dir: Directory containing trained models
        results_dir: Directory to save evaluation results
        X_test, y_test, snr_test, mods: Test data and metadata

    Returns:
        results: Dictionary mapping model names to their accuracies
    """
    results = {}

    for model_name in model_names:
        accuracy = evaluate_single_model(
            model_name, models_dir, results_dir,
            X_test, y_test, snr_test, mods
        )
        if accuracy is not None:
            results[model_name] = accuracy

    return results


def generate_evaluation_summary(results, output_dir):
    """
    Generate a summary report of all model evaluations.

    Args:
        results: Dictionary mapping model names to their accuracies
        output_dir: Directory to save the summary
    """
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")

    with open(summary_path, 'w') as f:
        f.write("RadioML Signal Classification - Model Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of models evaluated: {len(results)}\n\n")

        if results:
            # Sort models by accuracy (descending)
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

            f.write("Model Performance Ranking:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Rank':<6} {'Model':<20} {'Accuracy':<10}\n")
            f.write("-" * 40 + "\n")

            for rank, (model_name, accuracy) in enumerate(sorted_results, 1):
                f.write(f"{rank:<6} {model_name:<20} {accuracy:<10.4f}\n")

            f.write("-" * 40 + "\n\n")

            best_model, best_accuracy = sorted_results[0]
            f.write(f"Best performing model: {best_model} ({best_accuracy:.4f})\n")

            if len(sorted_results) > 1:
                worst_model, worst_accuracy = sorted_results[-1]
                f.write(f"Worst performing model: {worst_model} ({worst_accuracy:.4f})\n")
                f.write(f"Performance gap: {best_accuracy - worst_accuracy:.4f}\n")

            avg_accuracy = sum(results.values()) / len(results)
            f.write(f"Average accuracy: {avg_accuracy:.4f}\n")
        else:
            f.write("No models were successfully evaluated.\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Evaluation summary saved to {summary_path}")


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define paths
    dataset_path = "../data/RML2016.10a_dict.pkl"
    models_dir = "../output/models"
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
    _, _, X_test, _, _, y_test, _, _, snr_test, mods = prepare_data_by_snr_stratified(dataset)

    # Print dataset information
    print(f"Number of modulation types: {len(mods)}")
    print(f"Modulation types: {mods}")
    print(f"SNR values: {np.unique(snr_test)}")

    # Get available models to evaluate
    available_models = get_available_models()
    print(f"Available models for evaluation: {available_models}")

    # Evaluate all available models
    print(f"\n{'='*60}")
    print("EVALUATING ALL AVAILABLE MODELS")
    print(f"{'='*60}")

    results = evaluate_selected_models(
        available_models, models_dir, results_dir,
        X_test, y_test, snr_test, mods
    )

    # Generate evaluation summary
    generate_evaluation_summary(results, results_dir)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED")
    print(f"{'='*60}")
    print(f"Successfully evaluated {len(results)} models")
    if results:
        best_model = max(results.items(), key=lambda x: x[1])
        print(f"Best model: {best_model[0]} (accuracy: {best_model[1]:.4f})")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()