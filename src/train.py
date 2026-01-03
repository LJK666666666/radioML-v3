import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import time

from preprocess import load_data, prepare_data_by_snr_stratified
from models import build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model, get_callbacks, get_detailed_logging_callback


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


def train_model_resume(model, X_train, y_train, X_val, y_val, best_model_path, last_model_path,
                      batch_size=128, epochs=100, initial_epoch=0, log_json_path=None,
                      log_csv_path=None, existing_log_data=None):
    """
    Resume training from a checkpoint with preserved logging.

    Args:
        model: The loaded model to resume training
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        best_model_path: Path to save the best model
        last_model_path: Path to save the last epoch model
        batch_size: Batch size for training
        epochs: Total number of epochs to train for
        initial_epoch: Epoch to resume from
        log_json_path: Path to JSON log file
        log_csv_path: Path to CSV log file
        existing_log_data: Existing log data to append to

    Returns:
        History object containing training history
    """
    import json
    from datetime import datetime

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(last_model_path), exist_ok=True)
    if log_json_path:
        os.makedirs(os.path.dirname(log_json_path), exist_ok=True)

    # Get best validation accuracy from existing logs
    best_val_acc = 0.0
    if existing_log_data and 'epochs' in existing_log_data:
        best_val_acc = max([epoch.get('val_accuracy', 0.0) for epoch in existing_log_data['epochs']])
        print(f"Best validation accuracy from previous training: {best_val_acc:.4f}")

    # Custom callback to save best model only if it's better than previous best
    class ResumeModelCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, best_path, last_path, best_val_acc=0.0):
            super().__init__()
            self.best_path = best_path
            self.last_path = last_path
            self.best_val_acc = best_val_acc

        def on_epoch_end(self, epoch, logs=None):
            # Always save last model
            self.model.save(self.last_path)

            # Only save best model if validation accuracy improved
            current_val_acc = logs.get('val_accuracy', 0.0)
            if current_val_acc > self.best_val_acc:
                print(f"\nValidation accuracy improved from {self.best_val_acc:.4f} to {current_val_acc:.4f}")
                print(f"Saving best model to {self.best_path}")
                self.model.save(self.best_path)
                self.best_val_acc = current_val_acc
            else:
                print(f"\nValidation accuracy {current_val_acc:.4f} did not improve from {self.best_val_acc:.4f}")

    # Custom callback for logging to continue existing logs
    class ResumeLoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, json_path, csv_path, existing_data, initial_epoch):
            super().__init__()
            self.json_path = json_path
            self.csv_path = csv_path
            self.existing_data = existing_data or {"epochs": []}
            self.initial_epoch = initial_epoch

        def on_epoch_end(self, epoch, logs=None):
            # epoch is the current epoch number from Keras (starts from initial_epoch)
            # We want to record the actual epoch number in our logs
            actual_epoch = epoch + 1  # Keras epochs are 0-based, we want 1-based
            epoch_data = {
                "epoch": actual_epoch,
                "train_loss": float(logs.get('loss', 0.0)),
                "train_accuracy": float(logs.get('accuracy', 0.0)),
                "val_loss": float(logs.get('val_loss', 0.0)),
                "val_accuracy": float(logs.get('val_accuracy', 0.0)),
                "learning_rate": float(self.model.optimizer.learning_rate.numpy()),
                "timestamp": datetime.now().isoformat()
            }

            # Append to existing data
            self.existing_data['epochs'].append(epoch_data)

            # Save JSON
            if self.json_path:
                with open(self.json_path, 'w') as f:
                    json.dump(self.existing_data, f, indent=2)

            # Append to CSV
            if self.csv_path:
                import pandas as pd
                df_new = pd.DataFrame([epoch_data])
                if os.path.exists(self.csv_path):
                    df_new.to_csv(self.csv_path, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(self.csv_path, index=False)

    # Prepare callbacks
    callbacks = []

    # Add checkpoint callback
    checkpoint_callback = ResumeModelCheckpoint(best_model_path, last_model_path, best_val_acc)
    callbacks.append(checkpoint_callback)

    # Add logging callback if paths provided
    if log_json_path or log_csv_path:
        logging_callback = ResumeLoggingCallback(log_json_path, log_csv_path, existing_log_data, initial_epoch)
        callbacks.append(logging_callback)

    # Add learning rate scheduler (optional)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7
    )
    callbacks.append(reduce_lr)

    # Train the model
    print(f"Resuming training from epoch {initial_epoch + 1} to {epochs}")
    print(f"Saving best model to {best_model_path}")
    print(f"Saving last model to {last_model_path}")

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"Resume training completed in {training_time:.2f} seconds")

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


def get_available_models():
    """Return list of available model types for training"""
    return ['cnn1d', 'cnn2d', 'resnet', 'complex_nn']


def build_model_by_name(model_name, input_shape, num_classes):
    """Build a model by name and return the model instance"""
    model_builders = {
        'cnn1d': build_cnn1d_model,
        'cnn2d': build_cnn2d_model,
        'resnet': build_resnet_model,
        'complex_nn': build_complex_nn_model,
    }

    if model_name in model_builders:
        return model_builders[model_name](input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_single_model(model_name, X_train, y_train, X_val, y_val, input_shape, num_classes, output_dir):
    """
    Train a single model by name.

    Args:
        model_name: Name of the model to train
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        input_shape: Input shape for the model
        num_classes: Number of output classes
        output_dir: Directory to save the model and results

    Returns:
        history: Training history object, or None if training failed
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name.upper()} Model")
    print(f"{'='*50}")

    try:
        # Handle special data preparation for CNN2D
        if model_name == 'cnn2d':
            # Reshape data for 2D model
            X_train_model = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
            X_val_model = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        else:
            X_train_model = X_train
            X_val_model = X_val

        # Build model
        model = build_model_by_name(model_name, input_shape, num_classes)

        print(f"Model architecture for {model_name}:")
        model.summary()

        # Train model
        history = train_model(
            model,
            X_train_model, y_train,
            X_val_model, y_val,
            os.path.join(output_dir, f"{model_name}_model.keras")
        )

        # Plot and save training history
        plot_training_history(
            history,
            os.path.join(output_dir, f"{model_name}_history.png")
        )

        # Save training summary
        save_training_summary(history, f"{model_name}_model", output_dir)

        print(f"Successfully completed training for {model_name}")
        return history

    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return None


def train_selected_models(model_names, X_train, y_train, X_val, y_val, input_shape, num_classes, output_dir):
    """
    Train multiple models by name.

    Args:
        model_names: List of model names to train
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        input_shape: Input shape for the models
        num_classes: Number of output classes
        output_dir: Directory to save the models and results

    Returns:
        model_histories: Dictionary mapping model names to their training histories
    """
    model_histories = {}

    for model_name in model_names:
        history = train_single_model(
            model_name, X_train, y_train, X_val, y_val,
            input_shape, num_classes, output_dir
        )
        if history is not None:
            model_histories[f"{model_name}_model"] = history

    return model_histories


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define paths
    dataset_path = "../data/RML2016.10a_dict.pkl"
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
    X_train, X_val, _, y_train, y_val, _, _, _, _, mods = prepare_data_by_snr_stratified(dataset)

    # Print dataset information
    print(f"Number of modulation types: {len(mods)}")
    print(f"Modulation types: {mods}")

    # Get input shape from the data
    input_shape = X_train.shape[1:]
    num_classes = len(mods)
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    # Get available models to train
    available_models = get_available_models()
    print(f"Available models for training: {available_models}")

    # Train all available models
    print(f"\n{'='*60}")
    print("TRAINING ALL AVAILABLE MODELS")
    print(f"{'='*60}")

    model_histories = train_selected_models(
        available_models, X_train, y_train, X_val, y_val,
        input_shape, num_classes, output_dir
    )

    # Generate comprehensive training report
    if model_histories:
        generate_comprehensive_training_report(model_histories, output_dir)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Successfully trained {len(model_histories)} models")
    if model_histories:
        best_model = max(model_histories.items(),
                        key=lambda x: max(x[1].history['val_accuracy']))
        best_val_acc = max(best_model[1].history['val_accuracy'])
        print(f"Best model: {best_model[0]} (best val accuracy: {best_val_acc:.4f})")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()