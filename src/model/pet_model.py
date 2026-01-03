#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PET (Phase Enhancement Transformer) Model

This module implements the PET model from the ULCNN project.
PET uses trigonometric transformations to enhance phase information
in I/Q signal data for improved classification performance.

This implementation exactly matches the original PETCGDNN architecture
from AMR papers (PETCGDNN2016.py and PETCGDNN2018.py).
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, Flatten, Reshape,
    Conv2D, GRU, Lambda, Concatenate, BatchNormalization,
    Activation, Multiply, Add, Subtract, concatenate
)
from keras.optimizers import Adam
import keras

# Custom functions for Lambda layers (needed for model serialization)
def cos_function(x):
    """Custom cosine function for Lambda layer"""
    return tf.keras.backend.cos(x)

def sin_function(x):
    """Custom sine function for Lambda layer"""
    return tf.keras.backend.sin(x)

def transpose_function(x):
    """Custom transpose function for Lambda layer"""
    return tf.transpose(x, perm=[0, 2, 1])

def extract_i_2016(x):
    """Extract I channel for 2016 format"""
    return x[:, 0, :]

def extract_q_2016(x):
    """Extract Q channel for 2016 format"""
    return x[:, 1, :]

def extract_i_2018(x):
    """Extract I channel for 2018 format"""
    return x[:, :, 0]

def extract_q_2018(x):
    """Extract Q channel for 2018 format"""
    return x[:, :, 1]

# Legacy lambda functions for backward compatibility with old saved models
def legacy_cos_lambda(x):
    """Legacy cosine lambda for compatibility"""
    return tf.keras.backend.cos(x)

def legacy_sin_lambda(x):
    """Legacy sine lambda for compatibility"""
    return tf.keras.backend.sin(x)

def legacy_transpose_lambda(x):
    """Legacy transpose lambda for compatibility"""
    return tf.transpose(x, perm=[0, 2, 1])

def legacy_extract_i_lambda(x):
    """Legacy I extraction lambda for compatibility"""
    return x[:, 0, :]

def legacy_extract_q_lambda(x):
    """Legacy Q extraction lambda for compatibility"""
    return x[:, 1, :]


def build_pet_model_main(input_shape, num_classes):
    """
    Main PET model builder for integration with the existing pipeline.

    This function creates a single-input model that internally splits the input
    to match the original PETCGDNN architecture while being compatible with
    the existing data pipeline.

    This implementation exactly matches the original PETCGDNN architecture from
    AMR papers and automatically adapts to different sequence lengths (128, 1024, etc.).

    Architecture (following PETCGDNN2016.py and PETCGDNN2018.py):
    1. Phase enhancement through trigonometric transformation
    2. Spatial feature extraction with 2D convolutions (Conv2D(75,(8,2)) → Conv2D(25,(5,1)))
    3. Temporal feature extraction with GRU(128)
    4. Final classification

    Args:
        input_shape: Input shape of the data (2, seq_len) - already converted externally
        num_classes: Number of classes to classify

    Returns:
        A compiled Keras model compatible with existing pipeline
    """

    seq_len = input_shape[1]  # Extract sequence length from (2, seq_len)

    # Single input for compatibility with existing pipeline
    inputs = Input(shape=input_shape, name='input')  # (2, seq_len)

    # Internal data preparation to match original PETCGDNN architecture
    # Convert (2, seq_len) to the three inputs needed by original architecture

    # Transpose and reshape for main input: (2, seq_len) -> (seq_len, 2) -> (seq_len, 2, 1)
    main_input = Lambda(transpose_function, output_shape=(seq_len, 2), name='transpose_main')(inputs)
    main_input = Reshape((seq_len, 2, 1), name='reshape_main')(main_input)

    # Extract I and Q channels: (2, seq_len) -> (seq_len,) each
    input_i = Lambda(extract_i_2016, output_shape=(seq_len,), name='extract_i')(inputs)  # I channel
    input_q = Lambda(extract_q_2016, output_shape=(seq_len,), name='extract_q')(inputs)  # Q channel

    # Phase enhancement transformation (exactly as in AMR)
    x1 = Flatten()(main_input)
    x1 = Dense(1, name='fc2')(x1)
    x1 = Activation('linear')(x1)

    # Trigonometric functions (exactly as in AMR)
    cos1 = Lambda(cos_function, output_shape=(1,), name='cos_lambda')(x1)
    sin1 = Lambda(sin_function, output_shape=(1,), name='sin_lambda')(x1)

    # Phase rotation (exactly as in AMR)
    x11 = Multiply()([input_i, cos1])
    x12 = Multiply()([input_q, sin1])
    x21 = Multiply()([input_q, cos1])
    x22 = Multiply()([input_i, sin1])
    y1 = Add()([x11, x12])
    y2 = Subtract()([x21, x22])

    # Reshape and concatenate (exactly as in AMR)
    y1 = Reshape(target_shape=(seq_len, 1), name='reshape1')(y1)
    y2 = Reshape(target_shape=(seq_len, 1), name='reshape2')(y2)
    x11 = concatenate([y1, y2])
    x3 = Reshape(target_shape=(seq_len, 2, 1), name='reshape3')(x11)

    # Spatial feature extraction (exactly as in AMR)
    x3 = Conv2D(75, (8, 2), padding='valid', activation="relu",
                name="conv1_1", kernel_initializer='glorot_uniform')(x3)
    x3 = Conv2D(25, (5, 1), padding='valid', activation="relu",
                name="conv1_2", kernel_initializer='glorot_uniform')(x3)

    # Calculate temporal dimension after convolutions
    temporal_dim = seq_len - 8 - 5 + 2  # seq_len - 11

    # Temporal feature extraction (exactly as in AMR)
    x4 = Reshape(target_shape=(temporal_dim, 25), name='reshape4')(x3)
    x4 = GRU(units=128)(x4)  # Using GRU instead of CuDNNGRU for compatibility

    # Classification (exactly as in AMR)
    x = Dense(num_classes, activation='softmax', name='softmax')(x4)

    # Create model with single input (compatible with existing pipeline)
    model = Model(inputs=inputs, outputs=x, name='PET_CGDNN')

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    # optimizer = Adam(learning_rate=0.01)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model