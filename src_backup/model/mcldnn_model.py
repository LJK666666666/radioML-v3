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
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    
    # Main input for I/Q data
    input_main = Input(shape=input_shape, name='input_main')
    
    # Extract I and Q channels for separate processing
    from .complexnn import ExtractChannelLayer, TransposeLayer
    input_i = ExtractChannelLayer(0, 1, name='extract_i_channel')(input_main)  # (batch, 1, 128)
    input_q = ExtractChannelLayer(1, 2, name='extract_q_channel')(input_main)  # (batch, 1, 128)
    
    # Reshape I and Q for 1D convolution: (batch, 1, 128) -> (batch, 128, 1)
    input_i_reshaped = TransposeLayer([0, 2, 1], name='reshape_i')(input_i)
    input_q_reshaped = TransposeLayer([0, 2, 1], name='reshape_q')(input_q)
    
    # Part A: Multi-channel Inputs and Spatial Characteristics Mapping Section
    
    # Path 1: Combined I/Q processing with 2D convolution
    # Reshape main input for 2D conv: (2, 128) -> (2, 128, 1)
    input_2d = Reshape([input_shape[0], input_shape[1], 1], input_shape=input_shape)(input_main)
    
    x1 = Conv2D(50, (2, 8), padding='same', activation='relu', 
               name='conv2d_1', kernel_initializer='glorot_uniform')(input_2d)
    
    # Path 2: I channel processing with 1D convolution
    x2 = Conv1D(50, 8, padding='causal', activation='relu',
               name='conv1d_i', kernel_initializer='glorot_uniform')(input_i_reshaped)
    x2_reshaped = Reshape([-1, input_shape[1], 50], name='reshape_i_conv')(x2)
    
    # Path 3: Q channel processing with 1D convolution  
    x3 = Conv1D(50, 8, padding='causal', activation='relu',
               name='conv1d_q', kernel_initializer='glorot_uniform')(input_q_reshaped)
    x3_reshaped = Reshape([-1, input_shape[1], 50], name='reshape_q_conv')(x3)
    
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
    
    # Reshape for LSTM processing: extract temporal dimension
    x = Reshape(target_shape=(124, 100), name='reshape_for_lstm')(x)
    
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
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_mcldnn_model_multi_input(input_shape, num_classes):
    """
    Build MCLDNN model with multiple inputs (closer to original implementation).
    
    This version uses separate inputs for I/Q combined, I channel, and Q channel
    as in the original implementation.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model with multiple inputs
    """
    
    # Multiple inputs as in original
    input_iq = Input(shape=input_shape + (1,), name='input_iq')      # (2, 128, 1)
    input_i = Input(shape=(input_shape[1], 1), name='input_i')       # (128, 1)
    input_q = Input(shape=(input_shape[1], 1), name='input_q')       # (128, 1)
    
    # Part A: Multi-channel processing
    
    # Path 1: I/Q combined processing
    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
               name='conv2d_iq', kernel_initializer='glorot_uniform')(input_iq)
    
    # Path 2: I channel processing
    x2 = Conv1D(50, 8, padding='causal', activation='relu',
               name='conv1d_i', kernel_initializer='glorot_uniform')(input_i)
    x2_reshaped = Reshape([-1, input_shape[1], 50], name='reshape_i')(x2)
    
    # Path 3: Q channel processing
    x3 = Conv1D(50, 8, padding='causal', activation='relu',
               name='conv1d_q', kernel_initializer='glorot_uniform')(input_q)
    x3_reshaped = Reshape([-1, input_shape[1], 50], name='reshape_q')(x3)
    
    # Concatenate I and Q features
    x_combined = concatenate([x2_reshaped, x3_reshaped], axis=1, name='concat_iq')
    x_combined = Conv2D(50, (1, 8), padding='same', activation='relu',
                       name='conv2d_combined', kernel_initializer='glorot_uniform')(x_combined)
    
    # Concatenate all paths
    x = concatenate([x1, x_combined], name='concat_all')
    x = Conv2D(100, (2, 5), padding='valid', activation='relu',
              name='conv2d_final', kernel_initializer='glorot_uniform')(x)
    
    # Part B: Temporal processing
    x = Reshape(target_shape=(124, 100), name='reshape_temporal')(x)
    x = LSTM(units=128, return_sequences=True, name='lstm_1')(x)
    x = LSTM(units=128, return_sequences=False, name='lstm_2')(x)
    
    # Classification layers
    x = Dense(128, activation='selu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(128, activation='selu', name='dense_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model with multiple inputs
    model = Model(inputs=[input_iq, input_i, input_q], outputs=outputs, name='MCLDNN_MultiInput')
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def build_simplified_mcldnn_model(input_shape, num_classes):
    """
    Build a simplified version of MCLDNN with reduced complexity.
    
    Maintains the core multi-channel concept but with fewer layers.
    
    Args:
        input_shape: Input shape of the data
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    
    inputs = Input(shape=input_shape, name='input')
    
    # Extract I and Q channels
    from .complexnn import ExtractChannelLayer, TransposeLayer
    input_i = ExtractChannelLayer(0, 1, name='extract_i')(inputs)
    input_q = ExtractChannelLayer(1, 2, name='extract_q')(inputs)
    
    # Reshape for processing
    input_i = TransposeLayer([0, 2, 1], name='reshape_i')(input_i)
    input_q = TransposeLayer([0, 2, 1], name='reshape_q')(input_q)
    
    # Multi-channel processing
    input_2d = Reshape([input_shape[0], input_shape[1], 1], input_shape=input_shape)(inputs)
    x1 = Conv2D(32, (2, 8), padding='same', activation='relu', name='conv2d')(input_2d)
    
    x2 = Conv1D(32, 8, padding='same', activation='relu', name='conv1d_i')(input_i)
    x3 = Conv1D(32, 8, padding='same', activation='relu', name='conv1d_q')(input_q)
    
    # Combine features
    x2_reshaped = Reshape([-1, input_shape[1], 32], name='reshape_i_out')(x2)
    x3_reshaped = Reshape([-1, input_shape[1], 32], name='reshape_q_out')(x3)
    x_combined = concatenate([x2_reshaped, x3_reshaped], axis=1, name='concat_iq')
    
    x = concatenate([x1, x_combined], name='concat_all')
    
    # Temporal processing
    x = Reshape((-1, 32), name='reshape_temporal')(x)
    x = LSTM(64, name='lstm')(x)
    
    # Classification
    x = Dense(64, activation='relu', name='dense')(x)
    x = Dropout(0.3, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MCLDNN_Simplified')
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# Main function for integration
def build_mcldnn_model_main(input_shape, num_classes):
    """
    Main MCLDNN model builder for integration with existing pipeline.
    
    Uses single input version for compatibility.
    """
    return build_mcldnn_model(input_shape, num_classes)


# Legacy function name
def MCLDNN(input_shape, num_classes):
    """
    Legacy function name for MCLDNN model building.
    
    Args:
        input_shape: Input shape tuple
        num_classes: Number of output classes
        
    Returns:
        Compiled MCLDNN model
    """
    return build_mcldnn_model_main(input_shape, num_classes)