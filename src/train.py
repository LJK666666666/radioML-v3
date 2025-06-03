import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model
import time

from preprocess import load_data, prepare_data
from models import build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model, build_transformer_model, get_callbacks, get_detailed_logging_callback


def load_adaboost_model(filepath):
    """
    Load an AdaBoost model from a pickle file.
    
    Args:
        filepath: Path to the AdaBoost .pkl file
        
    Returns:
        Loaded AdaBoost model or None if loading fails
    """
    try:
        from model.adaboost_model import AdaBoostClassifier, KerasAdaBoostWrapper
        
        # Create a temporary AdaBoost classifier to use the load method
        temp_model = AdaBoostClassifier(input_shape=(2, 128), num_classes=11)  # These will be overwritten
        temp_model.load(filepath)
        
        # Wrap it in the Keras-compatible wrapper
        return KerasAdaBoostWrapper(temp_model)
    except Exception as e:
        print(f"Error loading AdaBoost model from {filepath}: {e}")
        return None


def train_model(model, X_train, y_train, X_val, y_val, model_path, batch_size=128, epochs=100, detailed_logging=True):
    """
    Train a model and save it.
    
    Args:
        model: The model to train
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        model_path: Path to save the model
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        detailed_logging: Whether to enable detailed epoch-by-epoch logging
        
    Returns:
        History object containing training history
    """
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Check if this is an AdaBoost model
    is_adaboost = hasattr(model, 'adaboost_model') and hasattr(model, '_convert_history')
    
    if is_adaboost:
        # For AdaBoost, we don't use standard Keras callbacks
        print(f"Training AdaBoost model, will save to pickle format")
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the AdaBoost model (it will handle the .pkl conversion internally)
        model.save(model_path)
        
        # Also save with _last suffix for consistency
        last_model_path = model_path.replace('.keras', '_last.keras')
        model.save(last_model_path)
        
    else:
        # Standard Keras model training
        # Prepare callbacks
        callbacks = get_callbacks(model_path)
        
        # Add detailed logging callback if enabled
        if detailed_logging:
            # Extract model name from path for logging
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            log_dir = os.path.join(os.path.dirname(model_path), "logs")
            detailed_logger = get_detailed_logging_callback(log_dir, model_name)
            callbacks.append(detailed_logger)
        
        # Train the model
        print(f"Training model, saving best to {model_path}")
        start_time = time.time()
        
        # Prepare the last model path
        last_model_path = model_path.replace('.keras', '_last.keras')
        
        # Create a custom callback to save the model after each epoch
        # This ensures we capture the true last epoch before EarlyStopping restores weights
        class SaveLastEpochCallback(tf.keras.callbacks.Callback):
            def __init__(self, save_path):
                super().__init__()
                self.save_path = save_path
            
            def on_epoch_end(self, epoch, logs=None):
                # Save the model after each epoch (overwriting previous saves)
                # This way we always have the true last trained epoch
                self.model.save(self.save_path)
        
        save_last_callback = SaveLastEpochCallback(last_model_path)
        callbacks.append(save_last_callback)
        
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
        print(f"Last epoch model saved to {last_model_path}")
    
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


def save_training_summary(history, model_name, output_dir):
    """
    Save a comprehensive training summary with detailed metrics.
    
    Args:
        history: Keras training history object
        model_name: Name of the model
        output_dir: Directory to save the summary
    """
    summary_path = os.path.join(output_dir, f"{model_name}_training_summary.txt")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic information
        total_epochs = len(history.history['loss'])
        f.write(f"Total epochs trained: {total_epochs}\n")
        
        # Final metrics
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        f.write(f"Final training loss: {final_train_loss:.4f}\n")
        f.write(f"Final training accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final validation loss: {final_val_loss:.4f}\n")
        f.write(f"Final validation accuracy: {final_val_acc:.4f}\n\n")
        
        # Best metrics
        best_train_acc = max(history.history['accuracy'])
        best_train_acc_epoch = history.history['accuracy'].index(best_train_acc) + 1
        best_val_acc = max(history.history['val_accuracy'])
        best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        f.write(f"Best training accuracy: {best_train_acc:.4f} (epoch {best_train_acc_epoch})\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_acc_epoch})\n\n")
        
        # Loss information
        min_train_loss = min(history.history['loss'])
        min_train_loss_epoch = history.history['loss'].index(min_train_loss) + 1
        min_val_loss = min(history.history['val_loss'])
        min_val_loss_epoch = history.history['val_loss'].index(min_val_loss) + 1
        
        f.write(f"Minimum training loss: {min_train_loss:.4f} (epoch {min_train_loss_epoch})\n")
        f.write(f"Minimum validation loss: {min_val_loss:.4f} (epoch {min_val_loss_epoch})\n\n")
        
        # Overfitting analysis
        train_val_acc_diff = final_train_acc - final_val_acc
        f.write(f"Training vs Validation accuracy difference: {train_val_acc_diff:.4f}\n")
        if train_val_acc_diff > 0.1:
            f.write("Warning: Large gap between training and validation accuracy suggests overfitting.\n")
        elif train_val_acc_diff < 0:
            f.write("Note: Validation accuracy is higher than training accuracy.\n")
        else:
            f.write("Good: Training and validation accuracies are well aligned.\n")
        
        f.write("\n" + "=" * 50 + "\n")
    
    print(f"Training summary saved to {summary_path}")


def generate_comprehensive_training_report(model_histories, output_dir):
    """
    Generate a comprehensive report comparing all trained models.
    
    Args:
        model_histories: Dictionary of model names and their training histories
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, "comprehensive_training_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("RadioML Signal Classification - Comprehensive Training Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of models trained: {len(model_histories)}\n\n")
        
        # Summary table
        f.write("Model Performance Summary:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<20} {'Final Val Acc':<15} {'Best Val Acc':<15} {'Epochs':<10}\n")
        f.write("-" * 70 + "\n")
        
        best_model = None
        best_val_acc = 0
        
        for model_name, history in model_histories.items():
            final_val_acc = history.history['val_accuracy'][-1]
            best_val_acc_model = max(history.history['val_accuracy'])
            total_epochs = len(history.history['loss'])
            
            f.write(f"{model_name:<20} {final_val_acc:<15.4f} {best_val_acc_model:<15.4f} {total_epochs:<10}\n")
            
            if best_val_acc_model > best_val_acc:
                best_val_acc = best_val_acc_model
                best_model = model_name
        
        f.write("-" * 70 + "\n\n")
        f.write(f"Best performing model: {best_model} (Validation Accuracy: {best_val_acc:.4f})\n\n")
        
        # Detailed analysis for each model
        for model_name, history in model_histories.items():
            f.write(f"Detailed Analysis - {model_name}:\n")
            f.write("-" * 40 + "\n")
            
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            best_val_acc_model = max(history.history['val_accuracy'])
            best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc_model) + 1
            
            overfitting_gap = final_train_acc - final_val_acc
            
            f.write(f"  Final Training Accuracy: {final_train_acc:.4f}\n")
            f.write(f"  Final Validation Accuracy: {final_val_acc:.4f}\n")
            f.write(f"  Best Validation Accuracy: {best_val_acc_model:.4f} (Epoch {best_val_acc_epoch})\n")
            f.write(f"  Final Training Loss: {final_train_loss:.4f}\n")
            f.write(f"  Final Validation Loss: {final_val_loss:.4f}\n")
            f.write(f"  Overfitting Gap: {overfitting_gap:.4f}\n")
            
            if overfitting_gap > 0.1:
                f.write("  Status: Potential overfitting detected\n")
            elif overfitting_gap < -0.05:
                f.write("  Status: Underfitting or validation set advantage\n")
            else:
                f.write("  Status: Good generalization\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("Recommendations:\n")
        f.write("-" * 20 + "\n")
        f.write(f"1. Deploy {best_model} for best performance\n")
        
        # Find models with overfitting
        overfitting_models = []
        for model_name, history in model_histories.items():
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            if final_train_acc - final_val_acc > 0.1:
                overfitting_models.append(model_name)
        
        if overfitting_models:
            f.write(f"2. Consider regularization for: {', '.join(overfitting_models)}\n")
        
        f.write("3. Monitor training logs for detailed epoch-by-epoch analysis\n")
        f.write("4. Consider ensemble methods combining top-performing models\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Comprehensive training report saved to {report_path}")


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
    
    model_histories = {}  # To store histories of all models
    
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
    save_training_summary(cnn1d_history, "cnn1d_model", output_dir)
    model_histories["cnn1d_model"] = cnn1d_history  # Save history
    
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
    save_training_summary(cnn2d_history, "cnn2d_model", output_dir)
    model_histories["cnn2d_model"] = cnn2d_history  # Save history
    
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
    save_training_summary(resnet_history, "resnet_model", output_dir)
    model_histories["resnet_model"] = resnet_history  # Save history

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
    save_training_summary(complex_nn_history, "complex_nn_model", output_dir)
    model_histories["complex_nn_model"] = complex_nn_history  # Save history

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
    save_training_summary(transformer_history, "transformer_model", output_dir)
    model_histories["transformer_model"] = transformer_history  # Save history
    
    # Generate comprehensive training report
    generate_comprehensive_training_report(model_histories, output_dir)
    
    print("\nAll models trained successfully!")


if __name__ == "__main__":
    main()