#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SCNN (Separable CNN) Model

This module implements the SCNN model from the ULCNN project.
SCNN uses separable convolutions for efficient signal classification.

Original paper reference: ULCNN project
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape, Permute
from keras.layers import Conv1D, SeparableConv1D, BatchNormalization, Activation
from keras.optimizers import Adam


def build_scnn_model(input_shape, num_classes):
    """
    Build SCNN (Separable CNN) model for radio signal classification.
    
    SCNN uses separable convolutions which decompose standard convolution into
    depthwise convolution followed by pointwise convolution, making it more
    computationally efficient while maintaining good performance.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    
    # Input layer
    inputs = Input(shape=input_shape, name='input')
    
    # Reshape from (2, 128) to (128, 2) for temporal processing
    # This treats the data as 128 time steps with 2 features (I/Q) each
    x = Reshape([input_shape[1], input_shape[0]], input_shape=input_shape)(inputs)
    
    # First separable convolution block
    x = Conv1D(128, 16, activation='relu', padding='same', name='conv1d_1')(x)
    x = BatchNormalization(name='batch_norm_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    
    # Second separable convolution block
    x = SeparableConv1D(64, 8, activation='relu', padding='same', name='separable_conv1d_1')(x)
    x = BatchNormalization(name='batch_norm_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    
    # Flatten for dense layers
    x = Flatten(name='flatten')(x)
    
    # Output layer
    outputs = Dense(num_classes, name='dense_output')(x)
    outputs = Activation('softmax', name='softmax')(outputs)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='SCNN')
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_scnn_model_original_structure(input_shape, num_classes):
    """
    Build SCNN model with structure closer to the original implementation.
    
    This version follows the exact layer structure from the original ULCNN SCNN model.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)  
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    
    # Input layer - expecting (2, 128) shape
    inputs = Input(shape=input_shape, name='input')
    
    # Reshape to (128, 2) as in original implementation
    x = Reshape([input_shape[1], input_shape[0]], input_shape=input_shape)(inputs)
    
    # First convolution block - matches original conv1d with 128 filters, kernel 16
    x = Conv1D(128, 16, activation='relu', padding='same', name='conv1d_128_16')(x)
    x = BatchNormalization(name='batch_norm_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    
    # Separable convolution block - matches original separable conv with 64 filters, kernel 8
    x = SeparableConv1D(64, 8, activation='relu', padding='same', name='separable_conv1d_64_8')(x)
    x = BatchNormalization(name='batch_norm_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    
    # Flatten and output - matches original structure
    x = Flatten(name='flatten')(x)
    x = Dense(num_classes, name='dense_output')(x)
    outputs = Activation('softmax', name='softmax_output')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs, name='SCNN_Original')
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


# For backward compatibility and testing
def SCNN(input_shape, num_classes):
    """
    Legacy function name for SCNN model building.
    
    Args:
        input_shape: Input shape tuple
        num_classes: Number of output classes
        
    Returns:
        Compiled SCNN model
    """
    return build_scnn_model(input_shape, num_classes)