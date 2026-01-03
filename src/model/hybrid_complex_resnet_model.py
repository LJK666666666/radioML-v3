"""
Hybrid Complex-ResNet Model for Radio Signal Classification

This model combines the advantages of:
1. ComplexNN: Fast initial convergence and complex I/Q data processing
2. ResNet: Residual connections for better long-term learning and final performance

Key innovations:
- Complex-valued residual blocks for better I/Q signal processing
- Gradual transition from complex to real-valued processing
- Hybrid activation functions combining complex and traditional approaches
- Multi-scale feature extraction with residual connections
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, Activation, Permute
from keras.optimizers import Adam
import numpy as np
from keras.saving import register_keras_serializable

# Import complex layers from the existing complex_nn_model
from .complex_nn_model import (
    ComplexConv1D, ComplexBatchNormalization, ComplexActivation, 
    ComplexDense, ComplexMagnitude, ComplexPooling1D
)


@register_keras_serializable(package="HybridComplexResNet")
class ComplexResidualBlock(tf.keras.layers.Layer):
    """
    Complex-valued residual block that performs complex convolutions with skip connections.
    This combines the residual learning from ResNet with complex arithmetic from ComplexNN.
    """
    def __init__(self, filters, kernel_size=3, strides=1, activation_type='complex_leaky_relu', **kwargs):
        super(ComplexResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation_type = activation_type
        
    def build(self, input_shape):
        # Main path
        self.conv1 = ComplexConv1D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same'
        )
        self.bn1 = ComplexBatchNormalization()
        self.activation1 = ComplexActivation(self.activation_type)
        
        self.conv2 = ComplexConv1D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            padding='same'
        )
        self.bn2 = ComplexBatchNormalization()
        
        # Shortcut path
        input_filters = input_shape[-1] // 2  # Complex input has 2x channels
        if input_filters != self.filters or self.strides != 1:
            self.shortcut_conv = ComplexConv1D(
                filters=self.filters, 
                kernel_size=1, 
                strides=self.strides, 
                padding='same'
            )
            self.shortcut_bn = ComplexBatchNormalization()
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None
            
        self.final_activation = ComplexActivation(self.activation_type)
        
    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut path
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
            
        # Add residual connection (complex addition)
        x = self.complex_add(x, shortcut)
        x = self.final_activation(x)
        
        return x
    
    def complex_add(self, x, shortcut):
        """Complex addition for residual connections"""
        # Both x and shortcut have shape (batch, time, 2*filters)
        # where the last dimension alternates real and imaginary parts
        return tf.add(x, shortcut)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation_type': self.activation_type
        })
        return config


@register_keras_serializable(package="HybridComplexResNet")
class ComplexResidualBlockAdvanced(tf.keras.layers.Layer):
    """
    Advanced Complex Residual Block with enhanced complex processing capabilities.
    This block maintains complex arithmetic throughout the entire network.
    """
    def __init__(self, filters, kernel_size=3, strides=1, activation_type='complex_leaky_relu', 
                 use_attention=False, **kwargs):
        super(ComplexResidualBlockAdvanced, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation_type = activation_type
        self.use_attention = use_attention
        
    def build(self, input_shape):
        # Main path - deeper complex processing
        self.conv1 = ComplexConv1D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same'
        )
        self.bn1 = ComplexBatchNormalization()
        self.activation1 = ComplexActivation(self.activation_type)
        
        self.conv2 = ComplexConv1D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            padding='same'
        )
        self.bn2 = ComplexBatchNormalization()
        
        # Additional complex processing layer
        self.conv3 = ComplexConv1D(
            filters=self.filters, 
            kernel_size=1,  # 1x1 conv for feature refinement
            padding='same'
        )
        self.bn3 = ComplexBatchNormalization()
        
        # Shortcut path
        input_filters = input_shape[-1] // 2  # Complex input has 2x channels
        if input_filters != self.filters or self.strides != 1:
            self.shortcut_conv = ComplexConv1D(
                filters=self.filters, 
                kernel_size=1, 
                strides=self.strides, 
                padding='same'
            )
            self.shortcut_bn = ComplexBatchNormalization()
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None
            
        # Complex attention mechanism (optional)
        if self.use_attention:
            self.attention_conv = ComplexConv1D(filters=self.filters, kernel_size=1, padding='same')
            
        self.final_activation = ComplexActivation(self.activation_type)
        
    def call(self, inputs, training=None):
        # Main path with deeper complex processing
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation1(x)  # Additional activation
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        # Shortcut path
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
            
        # Complex attention (if enabled)
        if self.use_attention:
            attention_weights = self.attention_conv(x)
            # Apply complex sigmoid-like attention
            attention_weights = ComplexActivation('complex_tanh')(attention_weights)
            x = self.complex_multiply(x, attention_weights)
        
        # Add residual connection (complex addition)
        x = self.complex_add(x, shortcut)
        x = self.final_activation(x)
        
        return x
    
    def complex_add(self, x, shortcut):
        """Complex addition for residual connections"""
        return tf.add(x, shortcut)
    
    def complex_multiply(self, x, weights):
        """Complex multiplication for attention"""
        # Both inputs have shape (batch, time, 2*filters) with alternating real/imag
        input_dim = tf.shape(x)[-1] // 2
        
        x_real = x[..., :input_dim]
        x_imag = x[..., input_dim:]
        w_real = weights[..., :input_dim]
        w_imag = weights[..., input_dim:]
        
        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        output_real = x_real * w_real - x_imag * w_imag
        output_imag = x_real * w_imag + x_imag * w_real
        
        return tf.concat([output_real, output_imag], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation_type': self.activation_type,
            'use_attention': self.use_attention
        })
        return config


@register_keras_serializable(package="HybridComplexResNet")
class ComplexGlobalAveragePooling1D(tf.keras.layers.Layer):
    """
    Complex Global Average Pooling layer that performs global average pooling
    while maintaining complex structure.
    """
    def __init__(self, **kwargs):
        super(ComplexGlobalAveragePooling1D, self).__init__(**kwargs)
    
    def call(self, inputs):
        # inputs shape: (batch, time, 2*filters)
        # Perform global average pooling across the time dimension
        return tf.reduce_mean(inputs, axis=1)
    
    def compute_output_shape(self, input_shape):
        # Remove the time dimension, keep batch and channel dimensions
        return (input_shape[0], input_shape[2])
    
    def get_config(self):
        config = super().get_config()
        return config


def build_hybrid_complex_resnet_model(input_shape, num_classes, activation_type='complex_leaky_relu'):
    """
    Build a Pure Complex-Domain Hybrid ResNet model that processes entirely in complex domain.
    
    Architecture Overview:
    1. Complex input processing for fast initial convergence (ComplexNN advantage)
    2. Deep complex residual blocks throughout for better gradient flow (ResNet advantage)
    3. Complex processing maintained throughout the entire network
    4. Complex-to-real conversion only at the final classification layer
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        activation_type: Type of complex activation function to use
        
    Returns:
        A compiled Keras model combining ComplexNN and ResNet advantages in pure complex domain
    """
    
    inputs = Input(shape=input_shape)
    
    # Reshape input from (2, 128) to (128, 2) for complex processing
    x = Permute((2, 1))(inputs)  # (128, 2)
    
    # Stage 1: Initial Complex Feature Extraction (like ComplexNN for fast convergence)
    x = ComplexConv1D(filters=64, kernel_size=7, padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = ComplexActivation(activation_type)(x)
    x = ComplexPooling1D(pool_size=2)(x)  # (64, 128)
    
    # Stage 2: Complex Residual Blocks (combining ComplexNN + ResNet advantages)
    x = ComplexResidualBlock(filters=64, activation_type=activation_type)(x)
    x = ComplexResidualBlock(filters=64, activation_type=activation_type)(x)
    
    # Stage 3: Deeper Complex Residual Processing with downsampling
    x = ComplexResidualBlock(filters=128, strides=2, activation_type=activation_type)(x)
    x = ComplexResidualBlock(filters=128, activation_type=activation_type)(x)
    x = ComplexResidualBlock(filters=128, activation_type=activation_type)(x)
    
    # Stage 4: Advanced Complex Residual Processing
    x = ComplexResidualBlockAdvanced(filters=256, strides=2, activation_type=activation_type, use_attention=True)(x)
    x = ComplexResidualBlockAdvanced(filters=256, activation_type=activation_type, use_attention=False)(x)
    
    # Stage 5: High-level Complex Feature Processing
    x = ComplexResidualBlockAdvanced(filters=512, strides=2, activation_type=activation_type, use_attention=True)(x)
    x = ComplexResidualBlockAdvanced(filters=512, activation_type=activation_type, use_attention=False)(x)
    x = ComplexResidualBlockAdvanced(filters=512, activation_type=activation_type, use_attention=False)(x)
    
    # Stage 6: Complex Global Feature Extraction
    # Use custom complex global average pooling
    x = ComplexGlobalAveragePooling1D()(x)  # Global average pooling
    
    # Complex Dense Processing with residual connections
    x = ComplexDense(1024)(x)
    x = ComplexActivation(activation_type)(x)
    x = Dropout(0.5)(x)
    
    x = ComplexDense(512)(x)
    x = ComplexActivation(activation_type)(x)
    x = Dropout(0.3)(x)
    
    # Final complex to real conversion for classification
    # Extract magnitude and phase information
    x = ComplexMagnitude()(x)  # Convert to magnitude (real-valued)
    
    # Final real-valued classification layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a slightly higher learning rate for faster initial convergence
    # but with decay for stable final training
    initial_learning_rate = 0.002
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True
    )
    
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_lightweight_hybrid_model(input_shape, num_classes):
    """
    A lighter version of the pure complex hybrid model for faster training and comparison.
    Maintains complex processing throughout with fewer layers.
    """
    inputs = Input(shape=input_shape)
    x = Permute((2, 1))(inputs)  # (128, 2)
    
    # Simple complex start
    x = ComplexConv1D(filters=32, kernel_size=5, padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = ComplexActivation('complex_leaky_relu')(x)
    x = ComplexPooling1D(pool_size=2)(x)
    
    # Complex residual blocks
    x = ComplexResidualBlock(filters=64, activation_type='complex_leaky_relu')(x)
    x = ComplexResidualBlock(filters=128, strides=2, activation_type='complex_leaky_relu')(x)
    
    # Advanced complex processing
    x = ComplexResidualBlockAdvanced(filters=256, strides=2, activation_type='complex_leaky_relu', use_attention=False)(x)
    
    # Complex global pooling
    x = ComplexGlobalAveragePooling1D()(x)
    
    # Complex dense processing
    x = ComplexDense(512)(x)
    x = ComplexActivation('complex_leaky_relu')(x)
    x = Dropout(0.5)(x)
    
    # Convert to real for classification
    x = ComplexMagnitude()(x)  # Convert to magnitude (real-valued)
    
    # Final classification
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_ultra_lightweight_hybrid_model(input_shape, num_classes):
    """
    Ultra-lightweight hybrid complex model with ~10K parameters.
    Minimal complex processing with very small filter sizes.
    """
    inputs = Input(shape=input_shape)
    x = Permute((2, 1))(inputs)  # (128, 2)
    
    # Minimal complex processing
    x = ComplexConv1D(filters=8, kernel_size=3, padding='same')(x)  # Very small filter count
    x = ComplexBatchNormalization()(x)
    x = ComplexActivation('complex_leaky_relu')(x)
    x = ComplexPooling1D(pool_size=4)(x)  # Aggressive pooling to reduce dimensions
    
    # Single ultra-light complex residual block
    x = ComplexResidualBlock(filters=16, activation_type='complex_leaky_relu')(x)
    x = ComplexPooling1D(pool_size=4)(x)  # Further dimension reduction
    
    # Complex global pooling
    x = ComplexGlobalAveragePooling1D()(x)
    
    # Minimal complex dense processing
    x = ComplexDense(32)(x)  # Very small dense layer
    x = ComplexActivation('complex_leaky_relu')(x)
    x = Dropout(0.3)(x)
    
    # Convert to real for classification
    x = ComplexMagnitude()(x)  # Convert to magnitude (real-valued)
    
    # Minimal final classification layers
    x = Dense(16, activation='relu')(x)  # Very small hidden layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_micro_lightweight_hybrid_model(input_shape, num_classes):
    """
    Micro-lightweight hybrid complex model with ~100K parameters.
    Balanced between parameter efficiency and model capacity.
    """
    inputs = Input(shape=input_shape)
    x = Permute((2, 1))(inputs)  # (128, 2)
    
    # Light complex processing
    x = ComplexConv1D(filters=16, kernel_size=3, padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = ComplexActivation('complex_leaky_relu')(x)
    x = ComplexPooling1D(pool_size=2)(x)
    
    # Two light complex residual blocks
    x = ComplexResidualBlock(filters=32, activation_type='complex_leaky_relu')(x)
    x = ComplexResidualBlock(filters=64, strides=2, activation_type='complex_leaky_relu')(x)
    
    # One advanced block but with small filters
    x = ComplexResidualBlockAdvanced(filters=64, strides=2, activation_type='complex_leaky_relu', use_attention=False)(x)
    
    # Complex global pooling
    x = ComplexGlobalAveragePooling1D()(x)
    
    # Moderate complex dense processing
    x = ComplexDense(128)(x)
    x = ComplexActivation('complex_leaky_relu')(x)
    x = Dropout(0.4)(x)
    
    # Convert to real for classification
    x = ComplexMagnitude()(x)  # Convert to magnitude (real-valued)
    
    # Final classification layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model
