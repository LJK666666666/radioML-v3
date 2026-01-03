#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCNet (Multi-scale CNN) Model

This module implements the MCNet model from the ULCNN project.
MCNet uses a complex multi-scale architecture with custom blocks,
skip connections, and multiple pooling strategies.

Original paper reference: ULCNN project
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, Flatten, Reshape,
    Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization,
    concatenate, Add, Activation
)
from keras.optimizers import Adam
import keras


def pre_block(x, conv1_size, conv2_size, pool_size, block_name):
    """
    Pre-processing block for MCNet.

    Applies two different convolutions with different pooling strategies
    and concatenates the results.

    Args:
        x: Input tensor
        conv1_size: Kernel size for first convolution
        conv2_size: Kernel size for second convolution
        pool_size: Pooling size
        block_name: Name prefix for layers

    Returns:
        Concatenated output tensor
    """
    base = x

    # First branch: convolution + average pooling
    x0 = Conv2D(32, conv1_size, padding='same', activation='relu',
                name=f'{block_name}_conv1', kernel_initializer='glorot_normal',
                data_format='channels_last')(base)
    x0 = AveragePooling2D(pool_size=pool_size, strides=pool_size,
                         padding='valid', data_format='channels_last',
                         name=f'{block_name}_avgpool')(x0)

    # Second branch: convolution + max pooling
    x1 = Conv2D(32, conv2_size, padding='same', activation='relu',
                name=f'{block_name}_conv2', kernel_initializer='glorot_normal',
                data_format='channels_last')(base)
    x1 = MaxPooling2D(pool_size=pool_size, strides=pool_size,
                     padding='valid', data_format='channels_last',
                     name=f'{block_name}_maxpool')(x1)

    # Concatenate branches
    output = concatenate([x0, x1], axis=-1, name=f'{block_name}_concat')

    return output


def m_block(x, filters_size01, filters_size02, filters_size03,
           conv0_size, conv1_size, conv2_size, conv3_size, block_name):
    """
    Multi-scale block (M-block) for MCNet.

    Applies multiple convolutions with different kernel sizes and
    concatenates the results for multi-scale feature extraction.

    Args:
        x: Input tensor
        filters_size01: Number of filters for base convolution
        filters_size02: Number of filters for branches 1 and 2
        filters_size03: Number of filters for branch 3
        conv0_size: Kernel size for base convolution
        conv1_size: Kernel size for first branch
        conv2_size: Kernel size for second branch
        conv3_size: Kernel size for third branch
        block_name: Name prefix for layers

    Returns:
        Concatenated multi-scale features
    """
    base = x

    # Base convolution
    base_x = Conv2D(filters_size01, conv0_size, padding='same', activation='relu',
                   name=f'{block_name}_conv0', kernel_initializer='glorot_normal',
                   data_format='channels_last')(base)

    # Multi-scale branches
    x0 = Conv2D(filters_size02, conv1_size, padding='same', activation='relu',
               name=f'{block_name}_conv1', kernel_initializer='glorot_normal',
               data_format='channels_last')(base_x)

    x1 = Conv2D(filters_size02, conv2_size, padding='same', activation='relu',
               name=f'{block_name}_conv2', kernel_initializer='glorot_normal',
               data_format='channels_last')(base_x)

    x2 = Conv2D(filters_size03, conv3_size, padding='same', activation='relu',
               name=f'{block_name}_conv3', kernel_initializer='glorot_normal',
               data_format='channels_last')(base_x)

    # Concatenate all branches
    output = concatenate([x0, x1], axis=-1, name=f'{block_name}_concat1')
    output = concatenate([output, x2], axis=-1, name=f'{block_name}_concat2')

    return output


def m_block_p(x, conv0_size, conv1_size, conv2_size, conv3_size, pool_size, block_name):
    """
    Multi-scale block with pooling (M-block-p) for MCNet.

    Similar to m_block but includes pooling operations for downsampling.

    Args:
        x: Input tensor
        conv0_size: Kernel size for base convolution
        conv1_size: Kernel size for first branch
        conv2_size: Kernel size for second branch
        conv3_size: Kernel size for third branch
        pool_size: Pooling size
        block_name: Name prefix for layers

    Returns:
        Concatenated multi-scale features with pooling
    """
    base = x

    # Base convolution
    base_x = Conv2D(32, conv0_size, padding='same', activation='relu',
                   name=f'{block_name}_conv0', kernel_initializer='glorot_normal',
                   data_format='channels_last')(base)

    # Multi-scale branches with pooling
    x0 = Conv2D(48, conv1_size, padding='same', activation='relu',
               name=f'{block_name}_conv1', kernel_initializer='glorot_normal',
               data_format='channels_last')(base_x)
    x0 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid',
                     data_format='channels_last', name=f'{block_name}_pool1')(x0)

    x1 = Conv2D(48, conv2_size, padding='same', activation='relu',
               name=f'{block_name}_conv2', kernel_initializer='glorot_normal',
               data_format='channels_last')(base_x)
    x1 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid',
                     data_format='channels_last', name=f'{block_name}_pool2')(x1)

    x2 = Conv2D(32, conv3_size, padding='same', activation='relu',
               name=f'{block_name}_conv3', kernel_initializer='glorot_normal',
               data_format='channels_last')(base_x)
    x2 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid',
                     data_format='channels_last', name=f'{block_name}_pool3')(x2)

    # Concatenate all branches
    output = concatenate([x0, x1], axis=-1, name=f'{block_name}_concat1')
    output = concatenate([output, x2], axis=-1, name=f'{block_name}_concat2')

    return output


def build_mcnet_model(input_shape, num_classes):
    """
    Build MCNet (Multi-scale CNN) model for radio signal classification.

    MCNet uses a complex multi-scale architecture with:
    - Pre-processing blocks for initial feature extraction
    - Multi-scale blocks (M-blocks) for feature learning
    - Skip connections for gradient flow
    - Multiple pooling strategies

    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify

    Returns:
        A compiled Keras model
    """

    # Input layer
    inputs = Input(shape=input_shape, name='input')

    # Reshape to 2D format: (2, 128) -> (2, 128, 1) for 2D convolutions
    x = Reshape([input_shape[0], input_shape[1], 1], input_shape=input_shape)(inputs)

    # Initial convolution and pooling
    x = Conv2D(64, kernel_size=(3, 7), strides=(1, 1), padding='same',
              activation='relu', name='conv0', kernel_initializer='glorot_normal',
              data_format='channels_last')(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid',
                    data_format='channels_last', name='initial_pool')(x)

    # Pre-processing block
    x = pre_block(x, conv1_size=(1, 3), conv2_size=(3, 1),
                 pool_size=(1, 2), block_name='pre_block_1')

    # First skip connection preparation
    x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same',
              name='skip_conv_1')(x)
    x_skip1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid',
                          data_format='channels_last', name='skip_pool_1')(x)
    x_skip1 = Reshape([2, 8, 128], name='skip_reshape_1')(x_skip1)

    # Continue main path
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid',
                    data_format='channels_last', name='main_pool_1')(x)

    # First M-block with pooling
    x = m_block_p(x, conv0_size=(1, 1), conv1_size=(3, 1), conv2_size=(1, 3),
                 conv3_size=(1, 1), pool_size=(1, 2), block_name='m_block_p_1')

    # Add skip connection
    x = Add(name='skip_add_1')([x, x_skip1])

    # Store for next skip connection
    x_skip2 = x

    # Second M-block (without pooling)
    x = m_block(x, filters_size01=32, filters_size02=48, filters_size03=32,
               conv0_size=(1, 1), conv1_size=(1, 3), conv2_size=(3, 1),
               conv3_size=(1, 1), block_name='m_block_1')

    # Add skip connection
    x = Add(name='skip_add_2')([x, x_skip2])

    # Prepare third skip connection
    x_skip3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid',
                          data_format='channels_last', name='skip_pool_2')(x)

    # Second M-block with pooling
    x = m_block_p(x, conv0_size=(1, 1), conv1_size=(1, 3), conv2_size=(3, 1),
                 conv3_size=(1, 1), pool_size=(1, 2), block_name='m_block_p_2')

    # Add skip connection
    x = Add(name='skip_add_3')([x, x_skip3])

    # Store for final skip connection
    x_skip4 = x

    # Third M-block (without pooling)
    x = m_block(x, filters_size01=32, filters_size02=48, filters_size03=32,
               conv0_size=(1, 1), conv1_size=(1, 3), conv2_size=(3, 1),
               conv3_size=(1, 3), block_name='m_block_2')

    # Add skip connection
    x = Add(name='skip_add_4')([x, x_skip4])

    # Final concatenation (as in original)
    x = concatenate([x, x_skip4], axis=-1, name='final_concat')

    # Final pooling and classification
    x = AveragePooling2D(pool_size=(2, 1), strides=(1, 2), padding='valid',
                        data_format='channels_last', name='final_pool')(x)
    x = BatchNormalization(name='final_bn')(x)
    x = Flatten(name='flatten')(x)

    # Output layer
    outputs = Dense(num_classes, kernel_initializer='glorot_normal',
                   name='dense_output', activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='MCNet')

    # Compile model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model