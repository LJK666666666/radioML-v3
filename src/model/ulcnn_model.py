#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ULCNN (Ultra-Lightweight Complex Neural Network) Model

This module implements the ULCNN model from the ULCNN project.
ULCNN uses complex convolutions with mobile units and channel attention
for efficient radio signal classification.

Original paper reference: ULCNN project
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Activation, Add, GlobalAveragePooling1D
from keras.optimizers import Adam
import numpy as np

# Import complex neural network components
from .complexnn import (
    ComplexConv1D, ComplexBatchNormalization,
    DWConvMobile, ChannelAttention
)


def build_ulcnn_model(input_shape, num_classes, n_neuron=16, n_mobileunit=6, kernel_size=5):
    """
    Build ULCNN (Ultra-Lightweight Complex Neural Network) model.

    ULCNN combines complex convolutions with mobile units and channel attention
    to create an efficient model for radio signal classification.

    Architecture:
    1. Initial complex convolution with batch normalization
    2. Multiple mobile units with channel attention
    3. Feature fusion from different stages
    4. Final classification layer

    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        n_neuron: Number of neurons/filters in the first layer (default: 16)
        n_mobileunit: Number of mobile units (default: 6)
        kernel_size: Convolution kernel size (default: 5)

    Returns:
        A compiled Keras model
    """

    # Input layer - expecting (2, 128) for I/Q data
    inputs = Input(shape=input_shape, name='input')

    # Reshape to (128, 2) for complex processing
    # This treats data as 128 time steps with complex values (I, Q)
    from .complexnn import TransposeLayer
    x = TransposeLayer(perm=[0, 2, 1], name='transpose_input')(inputs)  # (batch, 128, 2)

    # Initial complex convolution
    x = ComplexConv1D(n_neuron, kernel_size, padding='same', name='complex_conv_initial')(x)
    x = ComplexBatchNormalization(name='complex_bn_initial')(x)
    x = Activation('relu', name='activation_initial')(x)

    # Store intermediate features for fusion
    features = []

    # Mobile units with channel attention
    for i in range(n_mobileunit):
        x = DWConvMobile(n_neuron, kernel_size, name=f'mobile_unit_{i}')(x)
        x = ChannelAttention(name=f'channel_attention_{i}')(x)

        # Store features from later stages for fusion
        if i >= 3:  # Store features from mobile units 4, 5, 6 (0-indexed: 3, 4, 5)
            feature = GlobalAveragePooling1D(name=f'gap_feature_{i}')(x)
            features.append(feature)

    # Feature fusion - combine features from different stages
    if len(features) >= 3:
        # Add features from stages 4, 5, 6 as in original implementation
        f = Add(name='feature_fusion_1')([features[0], features[1]])  # f4 + f5
        f = Add(name='feature_fusion_2')([f, features[2]])           # (f4 + f5) + f6
    elif len(features) == 2:
        f = Add(name='feature_fusion_1')([features[0], features[1]])
    elif len(features) == 1:
        f = features[0]
    else:
        # Fallback: use global average pooling of final features
        f = GlobalAveragePooling1D(name='gap_final')(x)

    # Final classification layer
    f = Dense(num_classes, name='dense_output')(f)
    outputs = Activation('softmax', name='modulation')(f)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='ULCNN')

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model