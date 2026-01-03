#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCLDNN (Multi-Channel LDNN) Model

This module implements the MCLDNN model from the ULCNN project.
MCLDNN combines multi-channel inputs with spatial and temporal
feature extraction using CNNs and LSTMs.

Original paper reference: ULCNN project
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, Flatten, Reshape, Lambda,
    Conv1D, Conv2D, LSTM, concatenate,
    BatchNormalization, Activation
)
from keras.optimizers import Adam


def build_mcldnn_model(input_shape, num_classes):
    """
    Build MCLDNN (Multi-Channel LDNN) model for radio signal classification.

    MCLDNN processes I/Q data through multiple channels:
    1. Combined I/Q channel processing with 2D convolutions
    2. Separate I and Q channel processing with 1D convolutions
    3. Feature fusion and temporal processing with LSTM
    4. Final classification

    Args:
        input_shape: Input shape of the data (2, seq_len) - already converted externally
        num_classes: Number of classes to classify

    Returns:
        A compiled Keras model
    """

    seq_len = input_shape[1]  # Extract sequence length from (2, seq_len)

    # Main input for I/Q data
    input_main = Input(shape=input_shape, name='input_main')

    # Extract I and Q channels for separate processing
    from .complexnn import ExtractChannelLayer, TransposeLayer
    input_i = ExtractChannelLayer(0, 1, name='extract_i_channel')(input_main)  # (batch, 1, seq_len)
    input_q = ExtractChannelLayer(1, 2, name='extract_q_channel')(input_main)  # (batch, 1, seq_len)

    # Reshape I and Q for 1D convolution: (batch, 1, seq_len) -> (batch, seq_len, 1)
    input_i_reshaped = TransposeLayer([0, 2, 1], name='reshape_i')(input_i)
    input_q_reshaped = TransposeLayer([0, 2, 1], name='reshape_q')(input_q)

    # Part A: Multi-channel Inputs and Spatial Characteristics Mapping Section

    # Path 1: Combined I/Q processing with 2D convolution
    # Reshape main input for 2D conv: (2, seq_len) -> (2, seq_len, 1)
    input_2d = Reshape([input_shape[0], input_shape[1], 1], input_shape=input_shape)(input_main)

    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
               name='conv2d_1', kernel_initializer='glorot_uniform')(input_2d)

    # Path 2: I channel processing with 1D convolution
    x2 = Conv1D(50, 8, padding='causal', activation='relu',
               name='conv1d_i', kernel_initializer='glorot_uniform')(input_i_reshaped)
    x2_reshaped = Reshape([1, seq_len, 50], name='reshape_i_conv')(x2)

    # Path 3: Q channel processing with 1D convolution
    x3 = Conv1D(50, 8, padding='causal', activation='relu',
               name='conv1d_q', kernel_initializer='glorot_uniform')(input_q_reshaped)
    x3_reshaped = Reshape([1, seq_len, 50], name='reshape_q_conv')(x3)

    # Concatenate I and Q processed features
    x_iq = concatenate([x2_reshaped, x3_reshaped], axis=1, name='concat_iq')

    # Additional convolution on concatenated I/Q features
    x_iq = Conv2D(50, (1, 8), padding='same', activation='relu',
                 name='conv2d_iq', kernel_initializer='glorot_uniform')(x_iq)

    # Concatenate all paths
    x = concatenate([x1, x_iq], name='concat_all_paths')

    # Final spatial convolution
    x = Conv2D(100, (2, 5), padding='valid', activation='relu',
              name='conv2d_final', kernel_initializer='glorot_uniform')(x)

    # Part B: Temporal Characteristics Extraction Section

    # Calculate temporal dimension after convolutions
    temporal_dim = seq_len - 5 + 1  # for 'valid' padding

    # Reshape for LSTM processing: extract temporal dimension
    x = Reshape(target_shape=(temporal_dim, 100), name='reshape_for_lstm')(x)

    # First LSTM layer
    x = LSTM(units=128, return_sequences=True, name='lstm_1')(x)

    # Second LSTM layer
    x = LSTM(units=128, return_sequences=False, name='lstm_2')(x)

    # Dense layers for classification
    x = Dense(128, activation='selu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(128, activation='selu', name='dense_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='softmax_output')(x)

    # Create model
    model = Model(inputs=input_main, outputs=outputs, name='MCLDNN')

    # Compile model
    # optimizer = Adam(learning_rate=0.001) # none
    # adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    optimizer = Adam(learning_rate=0.001) # efficient_gpr_per_sample
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model