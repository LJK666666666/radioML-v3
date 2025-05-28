import os
import csv
import json
import time
from datetime import datetime
from keras.callbacks import Callback
import numpy as np


class DetailedLoggingCallback(Callback):
    """
    Custom Keras callback for detailed epoch-by-epoch logging.
    
    Logs training and validation metrics (accuracy, loss) and training time
    for each epoch to both console and CSV/JSON files.
    """
    
    def __init__(self, log_dir, model_name):
        """
        Initialize the detailed logging callback.
        
        Args:
            log_dir: Directory to save log files
            model_name: Name of the model (used for file naming)
        """
        super().__init__()
        self.log_dir = log_dir
        self.model_name = model_name
        self.epoch_start_time = None
        self.training_start_time = None
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files
        self.csv_log_path = os.path.join(log_dir, f"{model_name}_detailed_log.csv")
        self.json_log_path = os.path.join(log_dir, f"{model_name}_detailed_log.json")
        
        # Initialize CSV file with headers
        self._initialize_csv_log()
        
        # Initialize JSON log structure
        self.json_log = {
            'model_name': model_name,
            'training_start_time': None,
            'total_training_time': None,
            'epochs': []
        }
    
    def _initialize_csv_log(self):
        """Initialize CSV log file with headers."""
        with open(self.csv_log_path, 'w', newline='') as csvfile:
            fieldnames = [
                'epoch', 'train_loss', 'train_accuracy', 
                'val_loss', 'val_accuracy', 'epoch_time_seconds',
                'learning_rate', 'timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.training_start_time = time.time()
        self.json_log['training_start_time'] = datetime.now().isoformat()
        
        print(f"\n{'='*80}")
        print(f"Starting detailed logging for {self.model_name}")
        print(f"Log files will be saved to:")
        print(f"  CSV: {self.csv_log_path}")
        print(f"  JSON: {self.json_log_path}")
        print(f"{'='*80}")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1} - Starting...")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if self.epoch_start_time is None:
            return
        
        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time
        
        # Get current learning rate
        lr = float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer, 'learning_rate') else 'N/A'
        
        # Extract metrics from logs
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        # Create epoch data
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'epoch_time_seconds': round(epoch_time, 2),
            'learning_rate': lr,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print detailed epoch summary
        self._print_epoch_summary(epoch_data)
        
        # Save to CSV
        self._save_to_csv(epoch_data)
        
        # Add to JSON log
        self.json_log['epochs'].append(epoch_data)
        
        # Save JSON log (overwrite each time to keep it updated)
        self._save_json_log()
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.training_start_time is None:
            return
        
        total_time = time.time() - self.training_start_time
        self.json_log['total_training_time'] = round(total_time, 2)
        
        # Final save of JSON log
        self._save_json_log()
        
        # Print training summary
        self._print_training_summary(total_time)
    
    def _print_epoch_summary(self, epoch_data):
        """Print detailed epoch summary to console."""
        print(f"\nEpoch {epoch_data['epoch']} Results:")
        print(f"  Time: {epoch_data['epoch_time_seconds']:.2f}s")
        print(f"  Train - Loss: {epoch_data['train_loss']:.4f}, Accuracy: {epoch_data['train_accuracy']:.4f}")
        print(f"  Val   - Loss: {epoch_data['val_loss']:.4f}, Accuracy: {epoch_data['val_accuracy']:.4f}")
        print(f"  Learning Rate: {epoch_data['learning_rate']}")
        print(f"  {'-' * 60}")
    
    def _save_to_csv(self, epoch_data):
        """Save epoch data to CSV file."""
        try:
            with open(self.csv_log_path, 'a', newline='') as csvfile:
                fieldnames = [
                    'epoch', 'train_loss', 'train_accuracy', 
                    'val_loss', 'val_accuracy', 'epoch_time_seconds',
                    'learning_rate', 'timestamp'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(epoch_data)
        except Exception as e:
            print(f"Warning: Could not save to CSV log: {e}")
    
    def _save_json_log(self):
        """Save complete training log to JSON file."""
        try:
            with open(self.json_log_path, 'w') as jsonfile:
                json.dump(self.json_log, jsonfile, indent=2)
        except Exception as e:
            print(f"Warning: Could not save to JSON log: {e}")
    
    def _print_training_summary(self, total_time):
        """Print training summary."""
        print(f"\n{'='*80}")
        print(f"Training completed for {self.model_name}")
        print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Total epochs: {len(self.json_log['epochs'])}")
        
        if self.json_log['epochs']:
            # Calculate some statistics
            final_epoch = self.json_log['epochs'][-1]
            best_val_acc_epoch = max(self.json_log['epochs'], key=lambda x: x['val_accuracy'])
            avg_epoch_time = np.mean([e['epoch_time_seconds'] for e in self.json_log['epochs']])
            
            print(f"Final epoch - Train Acc: {final_epoch['train_accuracy']:.4f}, Val Acc: {final_epoch['val_accuracy']:.4f}")
            print(f"Best val accuracy: {best_val_acc_epoch['val_accuracy']:.4f} (Epoch {best_val_acc_epoch['epoch']})")
            print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        
        print(f"Detailed logs saved to:")
        print(f"  CSV: {self.csv_log_path}")
        print(f"  JSON: {self.json_log_path}")
        print(f"{'='*80}")


def get_detailed_logging_callback(log_dir, model_name):
    """
    Factory function to create a DetailedLoggingCallback.
    
    Args:
        log_dir: Directory to save log files
        model_name: Name of the model
        
    Returns:
        DetailedLoggingCallback instance
    """
    return DetailedLoggingCallback(log_dir, model_name)
