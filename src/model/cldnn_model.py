#!/usr/bin/env python3
"""
CLDNN Model Implementation

Based on: AMR-Benchmark/RML201610a/CLDNN/rmlmodels/CLDNNLikeModel.py
CLDNN: Convolutional, Long Short-Term Memory, Deep Neural Network
"""

import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, concatenate, Conv2D,
    LSTM, Permute, Reshape, ZeroPadding2D, Activation, Lambda
)

# Try to import CuDNNLSTM, fallback to LSTM if not available
try:
    from keras.layers import CuDNNLSTM
    CUDNN_AVAILABLE = True
except ImportError:
    CUDNN_AVAILABLE = False
    print("Warning: CuDNNLSTM not available, using regular LSTM")


def build_cldnn_model(input_shape, num_classes=11, weights=None):
    """
    Build CLDNN model exactly as specified in AMR-Benchmark

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

    dr = 0.5  # Dropout rate

    # Input layer - note the original uses (1, 2, 128) format
    input_x = Input(shape=(1, 2, input_shape[1]), name='cldnn_input')

    # First conv block
    input_x_padding = ZeroPadding2D((0, 2), data_format="channels_first")(input_x)
    layer11 = Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv11",
                     kernel_initializer='glorot_uniform', data_format="channels_first")(input_x_padding)
    layer11 = Dropout(dr)(layer11)

    # Second conv block
    layer11_padding = ZeroPadding2D((0, 2), data_format="channels_first")(layer11)
    layer12 = Conv2D(50, (1, 8), padding="valid", activation="relu", name="conv12",
                     kernel_initializer='glorot_uniform', data_format="channels_first")(layer11_padding)
    layer12 = Dropout(dr)(layer12)

    # Third conv block
    layer12 = ZeroPadding2D((0, 2), data_format="channels_first")(layer12)
    layer13 = Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv13",
                     kernel_initializer='glorot_uniform', data_format="channels_first")(layer12)
    layer13 = Dropout(dr)(layer13)

    # Concatenate first and third conv layers
    concat = concatenate([layer11, layer13])
    concat_shape = concat.shape

    # Calculate dimensions for LSTM
    # Original calculation from the code
    input_dim = int(concat_shape[-1] * concat_shape[-2])
    timesteps = int(concat_shape[-3])

    # Reshape for LSTM
    concat = Reshape((timesteps, input_dim))(concat)

    # LSTM layer
    if CUDNN_AVAILABLE:
        lstm_out = CuDNNLSTM(units=50)(concat)
    else:
        lstm_out = LSTM(units=50)(concat)

    # Dense layers
    layer_dense1 = Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(lstm_out)
    layer_dropout = Dropout(dr)(layer_dense1)
    layer_dense2 = Dense(num_classes, kernel_initializer='he_normal', name="dense2")(layer_dropout)
    layer_softmax = Activation('softmax')(layer_dense2)
    output = Reshape([num_classes])(layer_softmax)

    model = Model(inputs=input_x, outputs=output, name='CLDNN')

    # Load weights if provided
    if weights is not None:
        model.load_weights(weights)

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_cldnn_model_adapted(input_shape, num_classes=11):
    """
    Adapted CLDNN model for the main.py framework
    Converts from (2, seq_len) to (1, 2, seq_len) format expected by CLDNN
    """
    # Input in standard format
    inputs = Input(shape=input_shape, name='input_signals')

    # Reshape to CLDNN expected format: (batch, 2, seq_len) -> (batch, 1, 2, seq_len)
    x = Reshape((1, input_shape[0], input_shape[1]))(inputs)

    # Build the core CLDNN model but as a submodel
    cldnn_input = Input(shape=(1, 2, input_shape[1]))
    cldnn_output = build_cldnn_core(cldnn_input, num_classes)
    cldnn_submodel = Model(inputs=cldnn_input, outputs=cldnn_output)

    # Apply the submodel
    outputs = cldnn_submodel(x)

    # Create the final model
    model = Model(inputs=inputs, outputs=outputs, name='CLDNN_Adapted')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_cldnn_core(input_x, num_classes):
    """Core CLDNN layers"""
    dr = 0.5

    # First conv block
    input_x_padding = ZeroPadding2D((0, 2), data_format="channels_first")(input_x)
    layer11 = Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv11",
                     kernel_initializer='glorot_uniform', data_format="channels_first")(input_x_padding)
    layer11 = Dropout(dr)(layer11)

    # Second conv block
    layer11_padding = ZeroPadding2D((0, 2), data_format="channels_first")(layer11)
    layer12 = Conv2D(50, (1, 8), padding="valid", activation="relu", name="conv12",
                     kernel_initializer='glorot_uniform', data_format="channels_first")(layer11_padding)
    layer12 = Dropout(dr)(layer12)

    # Third conv block
    layer12 = ZeroPadding2D((0, 2), data_format="channels_first")(layer12)
    layer13 = Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv13",
                     kernel_initializer='glorot_uniform', data_format="channels_first")(layer12)
    layer13 = Dropout(dr)(layer13)

    # Concatenate first and third conv layers
    concat = concatenate([layer11, layer13])

    # Dynamic reshape for LSTM - calculate dimensions at runtime
    def dynamic_reshape_cldnn(x):
        shape = tf.shape(x)
        batch_size = shape[0]
        channels = shape[1]  # Should be 100 (50+50)
        height = shape[2]    # Should be 2
        width = shape[3]     # Varies based on conv operations

        # Reshape to (batch, timesteps=channels, features=height*width)
        reshaped = tf.reshape(x, [batch_size, channels, height * width])
        return reshaped

    concat = Lambda(dynamic_reshape_cldnn, name='dynamic_reshape_cldnn')(concat)

    # LSTM layer
    if CUDNN_AVAILABLE:
        lstm_out = CuDNNLSTM(units=50)(concat)
    else:
        lstm_out = LSTM(units=50)(concat)

    # Dense layers
    layer_dense1 = Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(lstm_out)
    layer_dropout = Dropout(dr)(layer_dense1)
    layer_dense2 = Dense(num_classes, kernel_initializer='he_normal', name="dense2")(layer_dropout)
    layer_softmax = Activation('softmax')(layer_dense2)

    return layer_softmax


if __name__ == '__main__':
    print("Testing CLDNN model...")

    # Test original format model
    try:
        model_original = build_cldnn_model((2, 128), num_classes=11)
        print(f"Original CLDNN model parameters: {model_original.count_params():,}")
    except Exception as e:
        print(f"Original CLDNN model creation failed: {e}")

    # Test adapted format model
    try:
        model_adapted = build_cldnn_model_adapted((2, 128), num_classes=11)
        print(f"Adapted CLDNN model parameters: {model_adapted.count_params():,}")
        model_adapted.summary()
    except Exception as e:
        print(f"Adapted CLDNN model creation failed: {e}")

    print("CLDNN model creation successful!")