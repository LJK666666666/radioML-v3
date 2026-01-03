#!/usr/bin/env python3
"""
CGDNN Model Implementation

Based on: AMR-Benchmark/RML201610a/CGDNet/rmlmodels/CGDNN.py
CGDNN: Convolutional Gated Deep Neural Network
"""

import os
import tensorflow as tf
import math
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, concatenate, Flatten, Reshape,
    GaussianNoise, Activation, GaussianDropout, Conv2D, MaxPool2D, Lambda, Multiply, Add, Subtract,
    LeakyReLU, BatchNormalization
)

# Try to import CuDNN layers, fallback if not available
try:
    from keras.layers import CuDNNLSTM, CuDNNGRU
    CUDNN_AVAILABLE = True
except ImportError:
    from keras.layers import LSTM, GRU
    CUDNN_AVAILABLE = False
    print("Warning: CuDNN layers not available, using regular LSTM/GRU")


def build_cgdnn_model(input_shape, num_classes=11, weights=None):
    """
    Build CGDNN model exactly as specified in AMR-Benchmark

    Args:
        input_shape: Input shape (2, seq_len)
        num_classes: Number of classes
        weights: Path to weights file (optional)

    Returns:
        Compiled Keras model
    """
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.2  # dropout rate (%)

    # Note: Original uses different input shapes, we adapt to our format
    input_layer = Input(shape=(1, 2, input_shape[1]), name='cgdnn_input')

    # First conv layer
    x1 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='lecun_uniform',
                data_format="channels_first")(input_layer)
    x1 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same',
                   data_format="channels_first")(x1)
    x1 = GaussianDropout(dr)(x1)

    # Second conv layer
    x2 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='glorot_uniform',
                data_format="channels_first")(x1)
    x2 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same',
                   data_format="channels_first")(x2)
    x2 = GaussianDropout(dr)(x2)

    # Third conv layer
    x3 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='glorot_uniform',
                data_format="channels_first")(x2)
    x3 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same',
                   data_format="channels_first")(x3)
    x3 = GaussianDropout(dr)(x3)

    # Concatenate first and third layers
    x11 = concatenate([x1, x3], axis=3)

    # Reshape for GRU - calculate dimensions based on actual tensor shape
    # For 128 input: after 3 conv layers with (1,6) kernels: 128-5-5-5 = 113
    # With 100 channels (50+50) and 2 height dimension
    x4 = Reshape(target_shape=(100, 226), name='reshape4')(x11)  # 100 channels, 2*113 = 226

    # GRU layer
    if CUDNN_AVAILABLE:
        x4 = CuDNNGRU(units=50)(x4)
    else:
        x4 = GRU(units=50)(x4)
    x4 = GaussianDropout(dr)(x4)

    # Dense layers
    x = Dense(256, activation='relu', name='fc4', kernel_initializer='he_normal')(x4)
    x = GaussianDropout(dr)(x)
    x = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=x, name='CGDNN')

    # Load weights if provided
    if weights is not None:
        model.load_weights(weights)

    return model


def build_cgdnn_model_adapted(input_shape, num_classes=11):
    """
    Adapted CGDNN model for the main.py framework
    Converts from (2, seq_len) to (1, 2, seq_len) format expected by CGDNN
    """
    # Input in standard format
    inputs = Input(shape=input_shape, name='input_signals')

    # Reshape to CGDNN expected format: (batch, 2, seq_len) -> (batch, 1, 2, seq_len)
    x = Reshape((1, input_shape[0], input_shape[1]))(inputs)

    # Apply CGDNN core
    dr = 0.2

    # First conv layer
    x1 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='lecun_uniform',
                data_format="channels_first")(x)
    x1 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same',
                   data_format="channels_first")(x1)
    x1 = GaussianDropout(dr)(x1)

    # Second conv layer
    x2 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='glorot_uniform',
                data_format="channels_first")(x1)
    x2 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same',
                   data_format="channels_first")(x2)
    x2 = GaussianDropout(dr)(x2)

    # Third conv layer
    x3 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='glorot_uniform',
                data_format="channels_first")(x2)
    x3 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same',
                   data_format="channels_first")(x3)
    x3 = GaussianDropout(dr)(x3)

    # Concatenate first and third layers
    x11 = concatenate([x1, x3], axis=3)

    # Dynamic reshape based on actual tensor dimensions
    # Use Lambda layer to calculate reshape dimensions dynamically
    def dynamic_reshape(x):
        shape = tf.shape(x)
        batch_size = shape[0]
        channels = shape[1]  # Should be 100 (50+50)
        height = shape[2]    # Should be 2
        width = shape[3]     # Varies based on conv operations

        # Reshape to (batch, channels, height*width)
        reshaped = tf.reshape(x, [batch_size, channels, height * width])
        return reshaped

    x4 = Lambda(dynamic_reshape, name='dynamic_reshape')(x11)

    # GRU layer - input shape: (batch, timesteps=100, features=height*width)
    if CUDNN_AVAILABLE:
        x4 = CuDNNGRU(units=50)(x4)
    else:
        x4 = GRU(units=50)(x4)
    x4 = GaussianDropout(dr)(x4)

    # Dense layers
    x = Dense(256, activation='relu', name='fc4', kernel_initializer='he_normal')(x4)
    x = GaussianDropout(dr)(x)
    outputs = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CGDNN_Adapted')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    print("Testing CGDNN model...")

    # Test original format model
    try:
        model_original = build_cgdnn_model((2, 128), num_classes=11)
        print(f"Original CGDNN model parameters: {model_original.count_params():,}")
    except Exception as e:
        print(f"Original CGDNN model creation failed: {e}")

    # Test adapted format model
    try:
        model_adapted = build_cgdnn_model_adapted((2, 128), num_classes=11)
        print(f"Adapted CGDNN model parameters: {model_adapted.count_params():,}")
        model_adapted.summary()
    except Exception as e:
        print(f"Adapted CGDNN model creation failed: {e}")

    print("CGDNN model testing completed!")