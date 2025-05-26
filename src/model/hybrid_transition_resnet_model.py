"""
Hybrid Complex-to-Real Transition ResNet Model for Radio Signal Classification

This model combines the advantages of ComplexNN and ResNet with gradual transition:
1. ComplexNN: Fast initial convergence and complex I/Q data processing
2. ResNet: Residual connections for better long-term learning and final performance
3. Hybrid Transition: Gradual transition from complex to real-valued processing

Key innovations:
- Complex-valued residual blocks for better I/Q signal processing
- HybridTransitionBlock for gradual transition from complex to real domain
- Traditional ResNet processing in final stages for robust classification
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


@register_keras_serializable(package="HybridTransitionResNet")
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


@register_keras_serializable(package="HybridTransitionResNet")
class HybridTransitionBlock(tf.keras.layers.Layer):
    """
    Hybrid transition block that gradually transitions from complex to real-valued processing.
    This allows the model to benefit from complex arithmetic early on while transitioning
    to traditional real-valued processing for final classification.
    """
    def __init__(self, filters, transition_ratio=0.5, activation_type='complex_leaky_relu', **kwargs):
        super(HybridTransitionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.transition_ratio = transition_ratio  # 0.0 = fully real, 1.0 = fully complex
        self.activation_type = activation_type
        
    def build(self, input_shape):
        # Complex branch
        self.complex_filters = int(self.filters * self.transition_ratio)
        self.real_filters = self.filters - self.complex_filters
        
        if self.complex_filters > 0:
            self.complex_conv = ComplexConv1D(
                filters=self.complex_filters, 
                kernel_size=3, 
                padding='same'
            )
            self.complex_bn = ComplexBatchNormalization()
            self.complex_activation = ComplexActivation(self.activation_type)
            
            # Convert complex to real for concatenation
            self.complex_to_real = ComplexMagnitude()
        
        if self.real_filters > 0:
            self.real_conv = Conv1D(
                filters=self.real_filters, 
                kernel_size=3, 
                padding='same'
            )
            self.real_bn = BatchNormalization()
            self.real_activation = Activation('relu')
            
    def call(self, inputs, training=None):
        outputs = []
        
        # Complex branch
        if self.complex_filters > 0:
            complex_out = self.complex_conv(inputs)
            complex_out = self.complex_bn(complex_out, training=training)
            complex_out = self.complex_activation(complex_out)
            # Convert to real values using magnitude
            complex_out = self.complex_to_real(complex_out)
            outputs.append(complex_out)
        
        # Real branch - convert complex input to real first
        if self.real_filters > 0:
            # Extract magnitude from complex input for real processing
            input_magnitude = ComplexMagnitude()(inputs)
            real_out = self.real_conv(input_magnitude)
            real_out = self.real_bn(real_out, training=training)
            real_out = self.real_activation(real_out)
            outputs.append(real_out)
        
        # Concatenate outputs
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tf.concat(outputs, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'transition_ratio': self.transition_ratio,
            'activation_type': self.activation_type
        })
        return config


def build_hybrid_transition_resnet_model(input_shape, num_classes, activation_type='complex_leaky_relu'):
    """
    Build a Hybrid Complex-to-Real Transition ResNet model that combines the best of both architectures.
    
    Architecture Overview:
    1. Complex input processing for fast initial convergence (ComplexNN advantage)
    2. Complex residual blocks for better gradient flow (ResNet + ComplexNN)
    3. Gradual transition from complex to real-valued processing (Hybrid innovation)
    4. Traditional ResNet-style final layers for robust classification (ResNet advantage)
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        activation_type: Type of complex activation function to use
        
    Returns:
        A compiled Keras model combining ComplexNN and ResNet advantages with gradual transition
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
    
    # Stage 3: Deeper Complex Residual Processing
    x = ComplexResidualBlock(filters=128, strides=2, activation_type=activation_type)(x)
    x = ComplexResidualBlock(filters=128, activation_type=activation_type)(x)
    
    # Stage 4: Hybrid Transition (gradually moving from complex to real)
    x = HybridTransitionBlock(filters=256, transition_ratio=0.7)(x)  # 70% complex, 30% real
    x = MaxPooling1D(pool_size=2)(x)
    
    x = HybridTransitionBlock(filters=256, transition_ratio=0.3)(x)  # 30% complex, 70% real
    
    # Stage 5: Traditional ResNet-style processing for final robustness
    # At this point we're working with real-valued data
    
    # Residual block 1 (real-valued)
    shortcut = Conv1D(512, 1, strides=2, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv1D(512, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    # Residual block 2 (real-valued)
    shortcut = x
    x = Conv1D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    # Stage 6: Global Pooling and Classification
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers with residual connections
    dense_input = x
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Skip connection for dense layers
    if dense_input.shape[-1] != 512:
        dense_skip = Dense(512, activation='linear')(dense_input)
    else:
        dense_skip = dense_input
    
    x = Add()([x, dense_skip])
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # Final classification layer
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


def build_lightweight_transition_model(input_shape, num_classes):
    """
    A lighter version of the hybrid transition model for faster training and comparison.
    Uses the gradual complex-to-real transition approach.
    """
    inputs = Input(shape=input_shape)
    x = Permute((2, 1))(inputs)  # (128, 2)
    
    # Simple complex start
    x = ComplexConv1D(filters=32, kernel_size=5, padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = ComplexActivation('complex_leaky_relu')(x)
    x = ComplexPooling1D(pool_size=2)(x)
    
    # One complex residual block
    x = ComplexResidualBlock(filters=64, activation_type='complex_leaky_relu')(x)
    
    # Transition to real
    x = HybridTransitionBlock(filters=128, transition_ratio=0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Simple real residual block
    shortcut = Conv1D(256, 1, strides=2, padding='same')(x)
    x = Conv1D(256, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    # Classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_comparison_models(input_shape, num_classes):
    """
    Build multiple variants for comparison:
    1. High transition ratio model (more complex processing)
    2. Medium transition ratio model (balanced)
    3. Low transition ratio model (more real processing)
    """
    
    def build_variant(transition_ratios, name_suffix):
        inputs = Input(shape=input_shape)
        x = Permute((2, 1))(inputs)
        
        # Initial complex processing
        x = ComplexConv1D(filters=64, kernel_size=7, padding='same')(x)
        x = ComplexBatchNormalization()(x)
        x = ComplexActivation('complex_leaky_relu')(x)
        x = ComplexPooling1D(pool_size=2)(x)
        
        # Complex residual blocks
        x = ComplexResidualBlock(filters=64, activation_type='complex_leaky_relu')(x)
        x = ComplexResidualBlock(filters=128, strides=2, activation_type='complex_leaky_relu')(x)
        
        # Transition blocks with different ratios
        for i, ratio in enumerate(transition_ratios):
            x = HybridTransitionBlock(filters=256, transition_ratio=ratio)(x)
            if i < len(transition_ratios) - 1:  # Don't pool after last transition
                x = MaxPooling1D(pool_size=2)(x)
        
        # Final real processing
        x = Conv1D(512, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Classification
        x = GlobalAveragePooling1D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        return model
    
    # Different transition strategies
    models = {
        'high_complex': build_variant([0.8, 0.6], 'high_complex'),
        'medium_complex': build_variant([0.6, 0.3], 'medium_complex'), 
        'low_complex': build_variant([0.4, 0.1], 'low_complex')
    }
    
    return models
