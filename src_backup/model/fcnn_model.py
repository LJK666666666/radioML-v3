"""
Fully Connected Neural Network (FCNN) Models for RadioML Signal Classification

This module implements various fully connected neural network architectures
for radio signal modulation classification. FCNNs are simple yet effective
models that process flattened input features through multiple dense layers.

Key Features:
- Multiple architecture variants (standard, deep, lightweight, wide)
- Dropout regularization to prevent overfitting
- Batch normalization for stable training
- Configurable activation functions
- Suitable for RadioML I/Q signal classification

Architectures:
1. Standard FCNN: Balanced depth and width
2. Deep FCNN: More layers for complex pattern learning
3. Lightweight FCNN: Fewer parameters for fast inference
4. Wide FCNN: Wider layers for feature representation

Author: AI Assistant
Date: 2025
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.optimizers import Adam
from keras.regularizers import l2


def build_fcnn_model(input_shape, num_classes, 
                     hidden_units=[512, 256, 128], 
                     dropout_rate=0.5, 
                     activation='relu',
                     use_batch_norm=True,
                     l2_reg=1e-4,
                     learning_rate=0.001):
    """
    Build a standard fully connected neural network model.
    
    Args:
        input_shape: Shape of input data (e.g., (128, 2) for RadioML)
        num_classes: Number of classification classes
        hidden_units: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        activation: Activation function for hidden layers
        use_batch_norm: Whether to use batch normalization
        l2_reg: L2 regularization strength
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Input layer with flattening
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    
    # Hidden layers
    for i, units in enumerate(hidden_units):
        model.add(Dense(units, 
                       activation=activation,
                       kernel_regularizer=l2(l2_reg),
                       name=f'dense_{i+1}'))
        
        if use_batch_norm:
            model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
        
        model.add(Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_deep_fcnn_model(input_shape, num_classes):
    """
    Build a deep fully connected neural network with more layers.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        
    Returns:
        Compiled Keras model
    """
    return build_fcnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_units=[1024, 512, 256, 128, 64],
        dropout_rate=0.5,
        activation='relu',
        use_batch_norm=True,
        l2_reg=1e-4,
        learning_rate=0.001
    )


def build_lightweight_fcnn_model(input_shape, num_classes):
    """
    Build a lightweight fully connected neural network for fast inference.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        
    Returns:
        Compiled Keras model
    """
    return build_fcnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_units=[256, 128],
        dropout_rate=0.3,
        activation='relu',
        use_batch_norm=True,
        l2_reg=1e-5,
        learning_rate=0.001
    )


def build_wide_fcnn_model(input_shape, num_classes):
    """
    Build a wide fully connected neural network with larger layers.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        
    Returns:
        Compiled Keras model
    """
    return build_fcnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_units=[1024, 1024, 512],
        dropout_rate=0.6,
        activation='relu',
        use_batch_norm=True,
        l2_reg=1e-4,
        learning_rate=0.0005
    )


def build_shallow_fcnn_model(input_shape, num_classes):
    """
    Build a shallow fully connected neural network with fewer layers.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        
    Returns:
        Compiled Keras model
    """
    return build_fcnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_units=[512, 256],
        dropout_rate=0.4,
        activation='relu',
        use_batch_norm=True,
        l2_reg=1e-4,
        learning_rate=0.001
    )


def build_custom_fcnn_model(input_shape, num_classes, 
                           architecture_type='balanced',
                           activation='relu',
                           optimizer_lr=0.001):
    """
    Build a customizable fully connected neural network.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        architecture_type: Type of architecture ('balanced', 'deep', 'wide', 'light')
        activation: Activation function
        optimizer_lr: Learning rate
        
    Returns:
        Compiled Keras model
    """
    # Define architecture configurations
    architectures = {
        'balanced': {
            'hidden_units': [512, 256, 128],
            'dropout_rate': 0.5,
            'l2_reg': 1e-4
        },
        'deep': {
            'hidden_units': [1024, 512, 256, 128, 64, 32],
            'dropout_rate': 0.5,
            'l2_reg': 1e-4
        },
        'wide': {
            'hidden_units': [2048, 1024, 512],
            'dropout_rate': 0.6,
            'l2_reg': 1e-4
        },
        'light': {
            'hidden_units': [256, 128],
            'dropout_rate': 0.3,
            'l2_reg': 1e-5
        }
    }
    
    config = architectures.get(architecture_type, architectures['balanced'])
    
    return build_fcnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_units=config['hidden_units'],
        dropout_rate=config['dropout_rate'],
        activation=activation,
        use_batch_norm=True,
        l2_reg=config['l2_reg'],
        learning_rate=optimizer_lr
    )


def build_fcnn_variants(input_shape, num_classes):
    """
    Build multiple FCNN variants for comparison.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        
    Returns:
        Dictionary of compiled Keras models
    """
    models = {}
    
    # Standard FCNN
    models['standard_fcnn'] = build_fcnn_model(input_shape, num_classes)
    
    # Deep FCNN
    models['deep_fcnn'] = build_deep_fcnn_model(input_shape, num_classes)
    
    # Lightweight FCNN
    models['lightweight_fcnn'] = build_lightweight_fcnn_model(input_shape, num_classes)
    
    # Wide FCNN
    models['wide_fcnn'] = build_wide_fcnn_model(input_shape, num_classes)
    
    # Shallow FCNN
    models['shallow_fcnn'] = build_shallow_fcnn_model(input_shape, num_classes)
    
    return models


def build_fcnn_ensemble(input_shape, num_classes, n_models=3):
    """
    Build an ensemble of FCNN models with different architectures.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classification classes
        n_models: Number of models in ensemble
        
    Returns:
        List of compiled Keras models
    """
    ensemble = []
    
    # Different configurations for diversity
    configs = [
        {'hidden_units': [512, 256, 128], 'dropout_rate': 0.5, 'learning_rate': 0.001},
        {'hidden_units': [1024, 256, 64], 'dropout_rate': 0.4, 'learning_rate': 0.0008},
        {'hidden_units': [256, 512, 128], 'dropout_rate': 0.6, 'learning_rate': 0.0012},
        {'hidden_units': [768, 384, 192], 'dropout_rate': 0.5, 'learning_rate': 0.0009},
        {'hidden_units': [400, 200, 100], 'dropout_rate': 0.45, 'learning_rate': 0.0011}
    ]
    
    for i in range(min(n_models, len(configs))):
        config = configs[i]
        model = build_fcnn_model(
            input_shape=input_shape,
            num_classes=num_classes,
            **config
        )
        ensemble.append(model)
    
    return ensemble


def get_fcnn_model_info():
    """
    Get information about available FCNN models.
    
    Returns:
        Dictionary with model information
    """
    model_info = {
        'fcnn': {
            'name': 'Standard FCNN',
            'description': 'Balanced fully connected network with 3 hidden layers',
            'parameters': 'Medium (~500K)',
            'training_time': 'Fast',
            'use_case': 'General purpose classification'
        },
        'deep_fcnn': {
            'name': 'Deep FCNN',
            'description': 'Deep fully connected network with 5 hidden layers',
            'parameters': 'High (~1M)',
            'training_time': 'Medium',
            'use_case': 'Complex pattern learning'
        },
        'lightweight_fcnn': {
            'name': 'Lightweight FCNN',
            'description': 'Compact network with 2 hidden layers',
            'parameters': 'Low (~200K)',
            'training_time': 'Very Fast',
            'use_case': 'Fast inference, resource constraints'
        },
        'wide_fcnn': {
            'name': 'Wide FCNN',
            'description': 'Wide network with large hidden layers',
            'parameters': 'Very High (~2M)',
            'training_time': 'Slow',
            'use_case': 'Maximum representational capacity'
        },
        'shallow_fcnn': {
            'name': 'Shallow FCNN',
            'description': 'Simple network with 2 hidden layers',
            'parameters': 'Medium (~400K)',
            'training_time': 'Fast',
            'use_case': 'Simple pattern recognition'
        }
    }
    
    return model_info


# Utility functions for FCNN models
def print_fcnn_summary(input_shape=(128, 2), num_classes=11):
    """Print summary of all FCNN model variants."""
    print("📊 FCNN Model Variants Summary")
    print("=" * 60)
    
    model_info = get_fcnn_model_info()
    
    for model_key, info in model_info.items():
        print(f"\n🔹 {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Parameters: {info['parameters']}")
        print(f"   Training Time: {info['training_time']}")
        print(f"   Use Case: {info['use_case']}")
    
    print("\n" + "=" * 60)
    print("All models are optimized for RadioML I/Q signal classification")


def estimate_fcnn_parameters(input_shape, hidden_units, num_classes):
    """
    Estimate the number of parameters in an FCNN model.
    
    Args:
        input_shape: Input shape tuple
        hidden_units: List of hidden layer sizes
        num_classes: Number of output classes
        
    Returns:
        Estimated parameter count
    """
    # Calculate input size
    input_size = 1
    for dim in input_shape:
        input_size *= dim
    
    total_params = 0
    prev_size = input_size
    
    # Hidden layers
    for units in hidden_units:
        # Dense layer: weights + biases
        total_params += (prev_size * units) + units
        # BatchNorm: gamma + beta
        total_params += 2 * units
        prev_size = units
    
    # Output layer
    total_params += (prev_size * num_classes) + num_classes
    
    return total_params


if __name__ == "__main__":
    # Demo/test the FCNN models
    input_shape = (128, 2)  # RadioML I/Q shape
    num_classes = 11        # Number of modulation types
    
    print("🚀 FCNN Models Demo")
    print("=" * 50)
    
    # Print model information
    print_fcnn_summary()
    
    # Build and display a standard model
    print("\n🔧 Building Standard FCNN Model...")
    model = build_fcnn_model(input_shape, num_classes)
    model.summary()
    
    # Estimate parameters for different variants
    print("\n📈 Parameter Estimation:")
    variants = {
        'Standard': [512, 256, 128],
        'Deep': [1024, 512, 256, 128, 64],
        'Lightweight': [256, 128],
        'Wide': [1024, 1024, 512]
    }
    
    for name, hidden_units in variants.items():
        params = estimate_fcnn_parameters(input_shape, hidden_units, num_classes)
        print(f"   {name:<12}: ~{params:,} parameters")
    
    print("\n✅ FCNN models ready for training!")
