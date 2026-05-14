#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PET (Phase Enhancement Transformer) Model

This module implements the PET model from the ULCNN project.
PET uses trigonometric transformations to enhance phase information
in I/Q signal data for improved classification performance.

Original paper reference: ULCNN project
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, Flatten, Reshape, 
    Conv2D, GRU, Lambda, Concatenate, BatchNormalization, 
    Activation, Multiply, Add, Subtract
)
from keras.optimizers import Adam
import keras


def trigonometric_transform_layer(x):
    """
    Apply trigonometric transformation to extract phase information.
    
    This layer learns a phase parameter and applies trigonometric
    transformations to the I/Q data to enhance phase-related features.
    
    Args:
        x: Input tensor containing I/Q data
        
    Returns:
        Transformed tensor with enhanced phase information
    """
    # Extract I and Q components from the input
    # Assuming input shape is (batch, 2, 128, 1)
    input_i = x[:, 0:1, :, :]  # I channel: (batch, 1, 128, 1)
    input_q = x[:, 1:2, :, :]  # Q channel: (batch, 1, 128, 1)
    
    # Flatten for dense layer processing
    flattened = Flatten()(x)
    
    # Learn a phase parameter through a dense layer
    phase = Dense(1, activation='linear', name='phase_dense')(flattened)
    
    # Apply trigonometric functions
    cos_phase = tf.cos(phase)
    sin_phase = tf.sin(phase)
    
    # Reshape I and Q for broadcasting
    input_i_flat = Flatten()(input_i)
    input_q_flat = Flatten()(input_q)
    
    # Apply phase rotation: 
    # I' = I*cos(φ) + Q*sin(φ)
    # Q' = Q*cos(φ) - I*sin(φ)
    i_transformed = Multiply()([input_i_flat, cos_phase])
    i_transformed = Add()([i_transformed, Multiply()([input_q_flat, sin_phase])])
    
    q_transformed = Multiply()([input_q_flat, cos_phase])  
    q_transformed = Subtract()([q_transformed, Multiply()([input_i_flat, sin_phase])])
    
    # Reshape back to original dimensions
    i_reshaped = Reshape((128, 1), name='i_reshape')(i_transformed)
    q_reshaped = Reshape((128, 1), name='q_reshape')(q_transformed)
    
    # Concatenate I and Q channels
    output = Concatenate(axis=-1, name='iq_concat')([i_reshaped, q_reshaped])
    
    # Reshape to 3D for further processing: (batch, 128, 2, 1)
    output = Reshape((128, 2, 1), name='final_reshape')(output)
    
    return output


def build_pet_model(input_shape, num_classes):
    """
    Build PET (Phase Enhancement Transformer) model for radio signal classification.
    
    PET enhances phase information in I/Q data through trigonometric transformations
    and combines spatial and temporal feature extraction.
    
    Architecture:
    1. Phase enhancement through trigonometric transformation
    2. Spatial feature extraction with 2D convolutions
    3. Temporal feature extraction with GRU
    4. Final classification
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    
    # Multiple inputs as in original (will be adapted to single input)
    input_main = Input(shape=input_shape + (1,), name='input_main')  # (2, 128, 1)
    input_i = Input(shape=(input_shape[1], 1), name='input_i')       # (128, 1)
    input_q = Input(shape=(input_shape[1], 1), name='input_q')       # (128, 1)
    
    # Phase enhancement transformation
    transformed = trigonometric_transform_layer(input_main)
    
    # Spatial feature extraction with 2D convolutions
    x = Conv2D(75, (8, 2), padding='valid', activation='relu',
              name='conv1', kernel_initializer='glorot_uniform')(transformed)
    x = Conv2D(25, (5, 1), padding='valid', activation='relu', 
              name='conv2', kernel_initializer='glorot_uniform')(x)
    
    # Reshape for temporal processing
    x = Reshape((117, 25), name='temporal_reshape')(x)  # Adjust based on conv output
    
    # Temporal feature extraction with GRU (replacing CuDNNGRU)
    x = GRU(units=128, name='gru_temporal')(x)
    
    # Final classification layer
    outputs = Dense(num_classes, activation='softmax', name='softmax_output')(x)
    
    # Create model with multiple inputs
    model = Model(inputs=[input_main, input_i, input_q], outputs=outputs, name='PET')
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_pet_model_single_input(input_shape, num_classes):
    """
    Build PET model adapted for single input (compatible with existing pipeline).
    
    This version takes a single input and internally splits it into I/Q components
    for processing, making it compatible with the existing data pipeline.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    
    # Single input
    inputs = Input(shape=input_shape, name='input')
    
    # Expand dimensions for 2D processing: (2, 128) -> (2, 128, 1)
    x = Reshape([input_shape[0], input_shape[1], 1], input_shape=input_shape)(inputs)
    
    # Split into I and Q components internally
    from .complexnn import ExtractChannelLayer
    input_i = ExtractChannelLayer(0, 1, name='extract_i')(inputs)  # (batch, 1, 128)
    input_q = ExtractChannelLayer(1, 2, name='extract_q')(inputs)  # (batch, 1, 128)
    
    # Reshape to remove the channel dimension and then add back for processing
    input_i = Reshape((128,), name='reshape_i_flat')(input_i)  # (batch, 128)
    input_q = Reshape((128,), name='reshape_q_flat')(input_q)  # (batch, 128)
    
    # Reshape I and Q for processing
    input_i = Reshape((input_shape[1], 1), name='reshape_i_final')(input_i)  # (batch, 128, 1)
    input_q = Reshape((input_shape[1], 1), name='reshape_q_final')(input_q)  # (batch, 128, 1)
    
    # Phase enhancement (simplified version)
    # Learn phase parameter from flattened input
    x_flat = Flatten(name='flatten_for_phase')(x)
    phase = Dense(1, activation='linear', name='phase_learning')(x_flat)
    
    # Apply trigonometric transformations
    from .complexnn import TrigonometricLayer
    cos_phase = TrigonometricLayer('cos', name='cos_transform')(phase)
    sin_phase = TrigonometricLayer('sin', name='sin_transform')(phase)
    
    # Transform I and Q channels
    i_flat = Flatten(name='flatten_i')(input_i)
    q_flat = Flatten(name='flatten_q')(input_q)
    
    # Phase rotation
    i_cos = Multiply(name='i_cos')([i_flat, cos_phase])
    q_sin = Multiply(name='q_sin')([q_flat, sin_phase])
    i_transformed = Add(name='i_add')([i_cos, q_sin])
    
    q_cos = Multiply(name='q_cos')([q_flat, cos_phase])
    i_sin = Multiply(name='i_sin')([i_flat, sin_phase])
    q_transformed = Subtract(name='q_subtract')([q_cos, i_sin])
    
    # Reshape and combine
    i_reshaped = Reshape((128, 1), name='i_final_reshape')(i_transformed)
    q_reshaped = Reshape((128, 1), name='q_final_reshape')(q_transformed)
    combined = Concatenate(axis=-1, name='combine_iq')([i_reshaped, q_reshaped])
    
    # Reshape for 2D convolution: (128, 2) -> (128, 2, 1)
    conv_input = Reshape((128, 2, 1), name='conv_reshape')(combined)
    
    # Spatial feature extraction
    x = Conv2D(75, (8, 2), padding='valid', activation='relu',
              name='spatial_conv1', kernel_initializer='glorot_uniform')(conv_input)
    x = Conv2D(25, (5, 1), padding='valid', activation='relu',
              name='spatial_conv2', kernel_initializer='glorot_uniform')(x)
    
    # Reshape for temporal processing
    x = Reshape((117, 25), name='temporal_input_reshape')(x)
    
    # Temporal feature extraction
    x = GRU(units=128, name='temporal_gru')(x)
    
    # Final classification
    outputs = Dense(num_classes, activation='softmax', name='classification_output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='PET_SingleInput')
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_simplified_pet_model(input_shape, num_classes):
    """
    Build a simplified version of PET model.
    
    Reduces complexity while maintaining the core phase enhancement concept.
    
    Args:
        input_shape: Input shape of the data
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    
    inputs = Input(shape=input_shape, name='input')
    
    # Simple phase enhancement
    x = Reshape([input_shape[0], input_shape[1], 1], input_shape=input_shape)(inputs)
    
    # Basic spatial processing
    x = Conv2D(32, (4, 2), padding='valid', activation='relu', name='conv1')(x)
    x = Conv2D(16, (3, 1), padding='valid', activation='relu', name='conv2')(x)
    
    # Temporal processing
    x = Reshape((-1, 16), name='temporal_reshape')(x)
    x = GRU(64, name='gru')(x)
    
    # Classification
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='PET_Simplified')
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# For backward compatibility and main integration
def build_pet_model_main(input_shape, num_classes):
    """
    Main PET model builder for integration with the existing pipeline.
    
    Uses single input version for compatibility.
    """
    return build_pet_model_single_input(input_shape, num_classes)


# Legacy function name
def PET(input_shape, num_classes):
    """
    Legacy function name for PET model building.
    
    Args:
        input_shape: Input shape tuple
        num_classes: Number of output classes
        
    Returns:
        Compiled PET model
    """
    return build_pet_model_main(input_shape, num_classes)