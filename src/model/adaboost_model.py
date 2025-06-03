"""
AdaBoost Model Implementation for RadioML Signal Classification

This module implements AdaBoost (Adaptive Boosting) algorithm with neural network base learners
for radio signal modulation classification. AdaBoost sequentially trains weak learners and
combines them into a strong classifier by adjusting sample weights based on previous errors.

Key Features:
- Neural network base learners (decision stumps implemented as simple NNs)
- Adaptive sample weight adjustment
- Sequential weak learner training
- Weighted prediction combination
- Compatible with existing RadioML training pipeline

Author: AI Assistant
Date: 2025
"""

import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os


class WeakLearner:
    """
    Simple neural network weak learner for AdaBoost.
    Implemented as a shallow neural network to act as a weak classifier.
    """
    
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        """
        Initialize weak learner.
        
        Args:
            input_shape: Shape of input data
            num_classes: Number of classification classes
            learning_rate: Learning rate for training
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.alpha = 0.0  # Weight of this weak learner in final prediction
        
    def _build_model(self):
        """Build a more robust neural network weak learner for multi-class problems."""
        model = Sequential([
            Input(shape=self.input_shape),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, sample_weights, epochs=100, batch_size=128, verbose=0):
        """
        Train the weak learner with weighted samples.
        
        Args:
            X: Training data
            y: Training labels (one-hot encoded)
            sample_weights: Weights for each training sample
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, min_delta=1e-4),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            sample_weight=sample_weights,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """Predict class probabilities."""
        return self.model.predict(X)
    
    def predict_classes(self, X):
        """Predict class labels."""
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)


class AdaBoostClassifier:
    """
    AdaBoost classifier with neural network weak learners.
    
    This implementation follows the AdaBoost.M1 algorithm adapted for multi-class
    classification with neural network base learners.
    """
    
    def __init__(self, input_shape, num_classes, n_estimators=50, learning_rate=1.0):
        """
        Initialize AdaBoost classifier.
        
        Args:
            input_shape: Shape of input data
            num_classes: Number of classification classes
            n_estimators: Number of weak learners to train
            learning_rate: Learning rate for weak learners
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weak_learners = []
        self.learner_weights = []
        self.classes_ = np.arange(num_classes)
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs_per_learner=100, 
            batch_size=128, verbose=1):
        """
        Train the AdaBoost classifier optimized for multi-class problems.
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs_per_learner: Number of epochs to train each weak learner
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Dictionary containing training history
        """
        n_samples = X_train.shape[0]
        
        # Convert one-hot to class indices for error calculation
        y_train_classes = np.argmax(y_train, axis=1)
        
        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        training_history = {
            'train_accuracy': [],
            'train_error': [],
            'val_accuracy': [],
            'learner_weights': []
        }
        
        for i in range(self.n_estimators):
            if verbose > 0:
                print(f"\nTraining weak learner {i+1}/{self.n_estimators}")
            
            # Create and train weak learner
            weak_learner = WeakLearner(
                self.input_shape, 
                self.num_classes,
                learning_rate=self.learning_rate * 0.1  # Smaller LR for weak learners
            )
            
            # Train with current sample weights
            history = weak_learner.train(
                X_train, y_train, 
                sample_weights, 
                epochs=epochs_per_learner,
                batch_size=batch_size,
                verbose=max(0, verbose-1)
            )
            
            # Make predictions
            predictions = weak_learner.predict_classes(X_train)
            
            # Calculate weighted error
            incorrect = predictions != y_train_classes
            error = np.average(incorrect, weights=sample_weights)
            
            # Prevent division by zero and ensure error is meaningful
            error = np.clip(error, 1e-10, 1 - 1e-10)
            # Calculate learner weight (alpha)
            # For multi-class: modified SAMME algorithm
            # More lenient for high-dimensional classification
            if error >= (1.0 - 1.0/self.num_classes):
                # If error is worse than random guessing, use very small alpha
                alpha = 0.1
            else:
                alpha = self.learning_rate * np.log((1 - error) / error) + np.log(self.num_classes - 1)
                alpha = max(0.1, alpha)  # Ensure minimum alpha
            
            weak_learner.alpha = alpha
            # Calculate learner weight (alpha)
            # For multi-class: alpha = ln((1-error)/error) + ln(K-1)
            alpha = 0.5 * np.log((1 - error) / error) + np.log(self.num_classes - 1)
            weak_learner.alpha = alpha
            
            # Add weak learner to ensemble
            self.weak_learners.append(weak_learner)
            self.learner_weights.append(alpha)
            
            # Update sample weights using SAMME algorithm for multi-class
            # More robust weight update for multi-class problems
            if error < 1e-10:
                # Perfect classifier, give small update to avoid numerical issues
                weight_update = 1.01
            else:
                # SAMME weight update: more gradual than original AdaBoost
                weight_update = np.exp(alpha * incorrect * (self.num_classes - 1) / self.num_classes)
            
            sample_weights *= weight_update
            sample_weights /= np.sum(sample_weights)  # Normalize
            
            # Prevent weight concentration on too few samples
            min_weight = 1.0 / (2 * n_samples)
            sample_weights = np.maximum(sample_weights, min_weight)
            sample_weights /= np.sum(sample_weights)  # Renormalize
            
            # Calculate current ensemble performance
            train_acc = self._calculate_accuracy(X_train, y_train)
            training_history['train_accuracy'].append(train_acc)
            training_history['train_error'].append(error)
            training_history['learner_weights'].append(alpha)
            
            if X_val is not None and y_val is not None:
                val_acc = self._calculate_accuracy(X_val, y_val)
                training_history['val_accuracy'].append(val_acc)
                
                if verbose > 0:
                    print(f"Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}")
            else:
                training_history['val_accuracy'].append(None)
                if verbose > 0:
                    print(f"Train accuracy: {train_acc:.4f}")
            
            # Early stopping if error is too high or too low
            # For 11-class problem, random accuracy is ~9%, so error of ~91%
            # We should allow training to continue unless error is extremely high
            if error > 0.95:  # Much more lenient threshold
                if verbose > 0:
                    print(f"Stopping early: error {error:.4f} > 0.95")
                break
            if error < 1e-10:
                if verbose > 0:
                    print(f"Stopping early: perfect weak learner")
                break
        
        return training_history
    
    def predict(self, X):
        """
        Make predictions using the ensemble of weak learners.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class probabilities
        """
        if not self.weak_learners:
            raise ValueError("AdaBoost classifier has not been trained yet.")
        
        # Initialize ensemble predictions
        ensemble_pred = np.zeros((X.shape[0], self.num_classes))
        
        # Combine predictions from all weak learners
        for learner, weight in zip(self.weak_learners, self.learner_weights):
            pred = learner.predict(X)
            ensemble_pred += weight * pred
        
        # Normalize to get probabilities
        ensemble_pred /= np.sum(self.learner_weights)
        
        # Apply softmax to ensure valid probabilities
        ensemble_pred = tf.nn.softmax(ensemble_pred).numpy()
        
        return ensemble_pred
    
    def predict_classes(self, X):
        """Predict class labels."""
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)
    
    def _calculate_accuracy(self, X, y):
        """Calculate accuracy on given data."""
        predictions = self.predict_classes(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(predictions == y_true)
    
    def save(self, filepath):
        """
        Save the AdaBoost model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model structure and weights
        model_data = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'learner_weights': self.learner_weights,
            'weak_learners_weights': []
        }
        
        # Save weights of each weak learner
        for i, learner in enumerate(self.weak_learners):
            learner_weights = learner.model.get_weights()
            model_data['weak_learners_weights'].append({
                'weights': learner_weights,
                'alpha': learner.alpha
            })
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"AdaBoost model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a pre-trained AdaBoost model.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model parameters
        self.input_shape = model_data['input_shape']
        self.num_classes = model_data['num_classes']
        self.n_estimators = model_data['n_estimators']
        self.learning_rate = model_data['learning_rate']
        self.learner_weights = model_data['learner_weights']
        
        # Recreate weak learners
        self.weak_learners = []
        for learner_data in model_data['weak_learners_weights']:
            weak_learner = WeakLearner(self.input_shape, self.num_classes, self.learning_rate)
            weak_learner.model.set_weights(learner_data['weights'])
            weak_learner.alpha = learner_data['alpha']
            self.weak_learners.append(weak_learner)
        
        print(f"AdaBoost model loaded from {filepath}")


def build_adaboost_model(input_shape, num_classes, n_estimators=50, learning_rate=0.8):
    """
    Build and return an AdaBoost classifier optimized for multi-class problems.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        n_estimators: Number of weak learners
        learning_rate: Learning rate for weak learners
        
    Returns:
        AdaBoost classifier instance
    """
    model = AdaBoostClassifier(
        input_shape=input_shape,
        num_classes=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    
    return model


def build_lightweight_adaboost_model(input_shape, num_classes, n_estimators=20, learning_rate=0.6):
    """
    Build a lightweight AdaBoost model with fewer estimators for faster training.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        n_estimators: Number of weak learners (reduced but reasonable for 11-class)
        learning_rate: Learning rate for weak learners
        
    Returns:
        Lightweight AdaBoost classifier instance
    """
    return build_adaboost_model(input_shape, num_classes, n_estimators, learning_rate)


# Wrapper class to make AdaBoost compatible with Keras-style interface
class KerasAdaBoostWrapper:
    """
    Wrapper to make AdaBoost compatible with Keras-style training interface.
    This allows AdaBoost to be used with the existing training pipeline.
    """
    
    def __init__(self, adaboost_model):
        """Initialize wrapper with AdaBoost model."""
        self.adaboost_model = adaboost_model
        self.history = None
    
    def fit(self, X_train, y_train, validation_data=None, epochs=100, batch_size=128, 
            verbose=1, callbacks=None):
        """
        Fit method compatible with Keras interface.
        
        Args:
            X_train: Training data
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of epochs (mapped to epochs per weak learner)
            batch_size: Batch size
            verbose: Verbosity level
            callbacks: Callbacks (ignored for AdaBoost)
            
        Returns:
            History-like object
        """
        X_val, y_val = validation_data if validation_data else (None, None)
        
        # Map epochs to reasonable values for weak learners
        epochs_per_learner = max(10, epochs // self.adaboost_model.n_estimators)
        
        training_history = self.adaboost_model.fit(
            X_train, y_train, X_val, y_val,
            epochs_per_learner=epochs_per_learner,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Convert to Keras-style history
        self.history = self._convert_history(training_history)
        return self.history
    
    def predict(self, X):
        """Predict method compatible with Keras interface."""
        return self.adaboost_model.predict(X)
    
    def save(self, filepath):
        """Save method compatible with Keras interface."""
        # Convert .keras extension to .pkl for AdaBoost
        if filepath.endswith('.keras'):
            filepath = filepath.replace('.keras', '.pkl')
        self.adaboost_model.save(filepath)
    
    def summary(self):
        """Print model summary."""
        print("AdaBoost Ensemble Classifier")
        print("="*50)
        print(f"Input shape: {self.adaboost_model.input_shape}")
        print(f"Number of classes: {self.adaboost_model.num_classes}")
        print(f"Number of weak learners: {self.adaboost_model.n_estimators}")
        print(f"Learning rate: {self.adaboost_model.learning_rate}")
        print("="*50)
        
        if self.adaboost_model.weak_learners:
            print(f"Trained weak learners: {len(self.adaboost_model.weak_learners)}")
            print("Weak learner architecture:")
            if self.adaboost_model.weak_learners:
                self.adaboost_model.weak_learners[0].model.summary()
    
    def _convert_history(self, training_history):
        """Convert AdaBoost history to Keras-style history."""
        class History:
            def __init__(self, history_dict):
                self.history = {
                    'accuracy': history_dict['train_accuracy'],
                    'loss': [1 - acc for acc in history_dict['train_accuracy']],  # Approximate loss
                }
                if history_dict['val_accuracy'][0] is not None:
                    self.history['val_accuracy'] = [acc for acc in history_dict['val_accuracy'] if acc is not None]
                    self.history['val_loss'] = [1 - acc for acc in self.history['val_accuracy']]
        
        return History(training_history)


def build_keras_adaboost_model(input_shape, num_classes, n_estimators=50, learning_rate=0.8):
    """
    Build AdaBoost model with Keras-compatible wrapper optimized for multi-class.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        n_estimators: Number of weak learners
        learning_rate: Learning rate for weak learners
        
    Returns:
        Keras-compatible AdaBoost model
    """
    adaboost_model = build_adaboost_model(input_shape, num_classes, n_estimators, learning_rate)
    return KerasAdaBoostWrapper(adaboost_model)
