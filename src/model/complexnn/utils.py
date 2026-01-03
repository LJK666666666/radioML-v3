#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complex Neural Network Utilities

This module provides utility functions for complex neural networks,
including channel operations and attention mechanisms used in ULCNN models.
"""

import tensorflow as tf
from keras.layers import Layer, SeparableConv1D, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Add, Multiply, Reshape
from keras.saving import register_keras_serializable
import keras.backend as K


@register_keras_serializable(package="ULComplexNN")
class TransposeLayer(Layer):
    """
    Custom transpose layer to replace Lambda functions for serialization compatibility.
    """
    def __init__(self, perm, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.perm = perm
    
    def call(self, inputs):
        return tf.transpose(inputs, perm=self.perm)
    
    def get_config(self):
        config = super().get_config()
        config.update({'perm': self.perm})
        return config


@register_keras_serializable(package="ULComplexNN")
class ExtractChannelLayer(Layer):
    """
    Custom layer to extract specific channels from input tensor.
    """
    def __init__(self, channel_start, channel_end, **kwargs):
        super(ExtractChannelLayer, self).__init__(**kwargs)
        self.channel_start = channel_start
        self.channel_end = channel_end
    
    def call(self, inputs):
        return inputs[:, self.channel_start:self.channel_end, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channel_start': self.channel_start,
            'channel_end': self.channel_end
        })
        return config


@register_keras_serializable(package="ULComplexNN")
class TrigonometricLayer(Layer):
    """
    Custom layer for trigonometric transformations (cos, sin).
    """
    def __init__(self, function='cos', **kwargs):
        super(TrigonometricLayer, self).__init__(**kwargs)
        self.function = function
    
    def call(self, inputs):
        if self.function == 'cos':
            return tf.cos(inputs)
        elif self.function == 'sin':
            return tf.sin(inputs)
        else:
            raise ValueError(f"Unsupported function: {self.function}")
    
    def get_config(self):
        config = super().get_config()
        config.update({'function': self.function})
        return config


def channel_shuffle(x, groups=2):
    """
    Channel shuffle operation for efficient neural networks.
    
    Rearranges channels to enable information flow between channel groups.
    This is commonly used in ShuffleNet and similar architectures.
    
    Args:
        x: Input tensor with shape (batch, time_steps, channels)
        groups: Number of groups to shuffle (default: 2)
        
    Returns:
        Tensor with shuffled channels
    """
    batch_size = tf.shape(x)[0]
    time_steps = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    
    channels_per_group = channels // groups
    
    # Reshape to separate groups
    x = tf.reshape(x, [batch_size, time_steps, groups, channels_per_group])
    
    # Transpose to shuffle
    x = tf.transpose(x, [0, 1, 3, 2])
    
    # Reshape back to original format
    x = tf.reshape(x, [batch_size, time_steps, channels])
    
    return x


@register_keras_serializable(package="ULComplexNN")
class ChannelShuffle(Layer):
    """
    Channel Shuffle Layer
    
    A proper Keras layer implementation of channel shuffle operation.
    """
    def __init__(self, groups=2, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.groups = groups
    
    def call(self, inputs):
        return channel_shuffle(inputs, self.groups)
    
    def get_config(self):
        config = super().get_config()
        config.update({'groups': self.groups})
        return config


def dwconv_mobile(x, neurons, kernel_size=5):
    """
    Depthwise separable convolution mobile unit.
    
    This function implements a mobile convolution block using separable convolutions
    with channel shuffle for efficient processing.
    
    Args:
        x: Input tensor
        neurons: Number of output neurons (filters)
        kernel_size: Convolution kernel size (default: 5)
        
    Returns:
        Processed tensor after mobile convolution
    """
    # Depthwise separable convolution with stride 2 for downsampling
    x = SeparableConv1D(int(2 * neurons), kernel_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Apply channel shuffle
    x = ChannelShuffle(groups=2)(x)
    
    return x


@register_keras_serializable(package="ULComplexNN")
class DWConvMobile(Layer):
    """
    Depthwise Separable Mobile Convolution Layer
    
    A proper Keras layer implementation of the mobile convolution unit.
    """
    def __init__(self, neurons, kernel_size=5, **kwargs):
        super(DWConvMobile, self).__init__(**kwargs)
        self.neurons = neurons
        self.kernel_size = kernel_size
        
        # Create sublayers
        self.separable_conv = SeparableConv1D(
            int(2 * neurons), 
            kernel_size, 
            strides=2, 
            padding='same'
        )
        self.batch_norm = BatchNormalization()
        self.activation = Activation('relu')
        self.channel_shuffle = ChannelShuffle(groups=2)
    
    def call(self, inputs):
        x = self.separable_conv(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.channel_shuffle(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'neurons': self.neurons,
            'kernel_size': self.kernel_size
        })
        return config


def channelattention(x):
    """
    Channel attention mechanism.
    
    Implements a channel attention module that learns to emphasize
    important channels and suppress less useful ones.
    
    Args:
        x: Input tensor with shape (batch, time_steps, channels)
        
    Returns:
        Tensor with applied channel attention
    """
    # Global average pooling and global max pooling
    x_gap = GlobalAveragePooling1D()(x)
    x_gmp = GlobalMaxPooling1D()(x)
    
    # Get channel dimension
    channels = K.int_shape(x)[-1]
    if channels is None:
        channels = tf.shape(x)[-1]
    
    # Shared dense layers for attention computation
    # Reduction ratio of 16 is commonly used
    reduction_ratio = 16
    intermediate_dim = max(1, channels // reduction_ratio)
    
    # Shared MLP for both GAP and GMP
    def shared_mlp(input_tensor):
        x_mlp = Reshape((1, channels))(input_tensor)
        x_mlp = Dense(intermediate_dim, activation='relu')(x_mlp)
        x_mlp = Dense(channels)(x_mlp)
        return x_mlp
    
    # Apply shared MLP to both pooled features
    x_gap_mlp = shared_mlp(x_gap)
    x_gmp_mlp = shared_mlp(x_gmp)
    
    # Combine and apply sigmoid activation
    attention_weights = Add()([x_gap_mlp, x_gmp_mlp])
    attention_weights = Activation('sigmoid')(attention_weights)
    
    # Apply attention weights to input
    x_attended = Multiply()([x, attention_weights])
    
    return x_attended


@register_keras_serializable(package="ULComplexNN")
class ChannelAttention(Layer):
    """
    Channel Attention Layer
    
    A proper Keras layer implementation of channel attention mechanism.
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.intermediate_dim = max(1, self.channels // self.reduction_ratio)
        
        # Create sublayers
        self.gap = GlobalAveragePooling1D()
        self.gmp = GlobalMaxPooling1D()
        
        # Shared MLP layers
        self.dense1 = Dense(self.intermediate_dim, activation='relu')
        self.dense2 = Dense(self.channels)
        
        self.add_layer = Add()
        self.sigmoid = Activation('sigmoid')
        self.multiply = Multiply()
        
        super(ChannelAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Global pooling
        x_gap = self.gap(inputs)
        x_gmp = self.gmp(inputs)
        
        # Reshape for dense layers
        x_gap = Reshape((1, self.channels))(x_gap)
        x_gmp = Reshape((1, self.channels))(x_gmp)
        
        # Shared MLP
        x_gap_mlp = self.dense2(self.dense1(x_gap))
        x_gmp_mlp = self.dense2(self.dense1(x_gmp))
        
        # Combine and apply attention
        attention_weights = self.sigmoid(self.add_layer([x_gap_mlp, x_gmp_mlp]))
        output = self.multiply([inputs, attention_weights])
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


# Rotation matrix for data augmentation (used in original ULCNN training)
def rotate_matrix(theta):
    """
    Create a 2D rotation matrix for I/Q data augmentation.
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        2x2 rotation matrix
    """
    import numpy as np
    m = np.zeros((2, 2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    return m


def rotate_data_augmentation(x, y):
    """
    Apply rotation-based data augmentation to I/Q data.
    
    This function creates rotated versions of the input data at 90-degree intervals,
    which is equivalent to phase shifts in the complex domain.
    
    Args:
        x: Input data with shape (N, 2, L) where 2 represents I/Q channels
        y: Labels corresponding to input data
        
    Returns:
        Tuple of (augmented_x, augmented_y) with 4x more samples
    """
    import numpy as np
    
    N, L, C = np.shape(x)
    
    # Create rotated versions
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))      # 90 degrees
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))        # 180 degrees  
    x_rotate3 = np.matmul(x, rotate_matrix(3*np.pi/2))    # 270 degrees
    
    # Combine all rotations
    x_augmented = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))
    
    # Replicate labels for each rotation
    y_augmented = np.tile(y, (4, 1))
    y_augmented = y_augmented.T.reshape(-1)
    
    return x_augmented, y_augmented