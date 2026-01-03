#!/usr/bin/env python3
"""
AMC-Net Model Implementation in TensorFlow/Keras

Completely restructured to match PyTorch implementation at AMC-Net/models/model.py
Original paper: Adaptive Multi-scale Convolution Network for Automatic Modulation Classification

Key changes to match PyTorch exactly:
1. Input layout: (batch, 2, W) -> (batch, 2, W, 1) with I/Q in height dimension
2. MultiScaleModule: Direct (2, k) convolution with in_channels=1
3. AdaCorrModule: FFT on width dimension, not channel dimension
4. FeaFusionModule: Attention across channel dimension, not time dimension
5. Correct data flow: channels_first equivalent layout
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, Flatten, Reshape, Lambda,
    Conv2D, Conv1D, concatenate, Add, Multiply,
    BatchNormalization, Activation, GlobalAveragePooling1D,
    ZeroPadding2D, Permute, PReLU
)
from keras import backend as K
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
import numpy as np
import math


@register_keras_serializable(package="AMCNet")
class L2NormalizeWidthLayer(tf.keras.layers.Layer):
    """L2 normalization along width dimension"""
    def __init__(self, **kwargs):
        super(L2NormalizeWidthLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.l2_normalize(x, axis=2)


@register_keras_serializable(package="AMCNet")
class SqueezeHeightLayer(tf.keras.layers.Layer):
    """Squeeze height dimension (axis=1)"""
    def __init__(self, **kwargs):
        super(SqueezeHeightLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.squeeze(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3])


@register_keras_serializable(package="AMCNet")
class TransposeChannelWidthLayer(tf.keras.layers.Layer):
    """Transpose from (batch, W, C) to (batch, C, W)"""
    def __init__(self, **kwargs):
        super(TransposeChannelWidthLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.transpose(x, perm=[0, 2, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[1])


@register_keras_serializable(package="AMCNet")
class GlobalAvgPoolWidthLayer(tf.keras.layers.Layer):
    """Global average pooling along width dimension (last axis)"""
    def __init__(self, **kwargs):
        super(GlobalAvgPoolWidthLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.reduce_mean(x, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


@register_keras_serializable(package="AMCNet")
class Conv_Block(tf.keras.layers.Layer):
    """
    Exact replica of PyTorch Conv_Block:
    ZeroPad2d((1, 1, 0, 0)) + Conv2d(in_c, out_c, kernel_size=(1, 3)) + ReLU + BatchNorm2d
    """
    def __init__(self, in_channel, out_channel, **kwargs):
        super(Conv_Block, self).__init__(**kwargs)
        self.in_c = in_channel
        self.out_c = out_channel

        # ZeroPad2d((1, 1, 0, 0)) equivalent - padding left/right by 1
        self.zero_pad = ZeroPadding2D(padding=((0, 0), (1, 1)))
        self.conv = Conv2D(self.out_c, kernel_size=(1, 3), use_bias=True)
        self.relu = Activation('relu')
        self.bn = BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channel": self.in_c,
            "out_channel": self.out_c,
        })
        return config

    def call(self, x):
        x = self.zero_pad(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


@register_keras_serializable(package="AMCNet")
class MultiScaleModule(tf.keras.layers.Layer):
    """
    Exact replica of PyTorch MultiScaleModule:
    - conv_3: ZeroPad2d((1, 1, 0, 0)) + Conv2d(1, out_c//3, kernel_size=(2, 3))
    - conv_5: ZeroPad2d((2, 2, 0, 0)) + Conv2d(1, out_c//3, kernel_size=(2, 5))
    - conv_7: ZeroPad2d((3, 3, 0, 0)) + Conv2d(1, out_c//3, kernel_size=(2, 7))

    Input: (batch, 2, W, 1) where H=2 contains I/Q, in_channels=1
    """
    def __init__(self, out_channel, **kwargs):
        super(MultiScaleModule, self).__init__(**kwargs)
        self.out_c = out_channel

        # Path 1: kernel_size=(2, 3), in_channels=1, out_channels=out_c//3
        self.zero_pad_3 = ZeroPadding2D(padding=((0, 0), (1, 1)))
        self.conv_3 = Conv2D(self.out_c // 3, kernel_size=(2, 3), use_bias=True)
        self.relu_3 = Activation('relu')
        self.bn_3 = BatchNormalization()

        # Path 2: kernel_size=(2, 5), in_channels=1, out_channels=out_c//3
        self.zero_pad_5 = ZeroPadding2D(padding=((0, 0), (2, 2)))
        self.conv_5 = Conv2D(self.out_c // 3, kernel_size=(2, 5), use_bias=True)
        self.relu_5 = Activation('relu')
        self.bn_5 = BatchNormalization()

        # Path 3: kernel_size=(2, 7), in_channels=1, out_channels=out_c//3
        self.zero_pad_7 = ZeroPadding2D(padding=((0, 0), (3, 3)))
        self.conv_7 = Conv2D(self.out_c // 3, kernel_size=(2, 7), use_bias=True)
        self.relu_7 = Activation('relu')
        self.bn_7 = BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channel": self.out_c,
        })
        return config

    def call(self, x):
        # Input: (batch, 2, W, 1) - I/Q in height=2, channels=1

        # Path 1: (2, 3) convolution
        y1 = self.zero_pad_3(x)
        y1 = self.conv_3(y1)
        y1 = self.relu_3(y1)
        y1 = self.bn_3(y1)

        # Path 2: (2, 5) convolution
        y2 = self.zero_pad_5(x)
        y2 = self.conv_5(y2)
        y2 = self.relu_5(y2)
        y2 = self.bn_5(y2)

        # Path 3: (2, 7) convolution
        y3 = self.zero_pad_7(x)
        y3 = self.conv_7(y3)
        y3 = self.relu_7(y3)
        y3 = self.bn_7(y3)

        # Concatenate along channel dimension (axis=-1 in channels_last)
        x = concatenate([y1, y2, y3], axis=-1)
        return x


@register_keras_serializable(package="AMCNet")
class TinyMLP(tf.keras.layers.Layer):
    """
    Exact replica of PyTorch TinyMLP:
    Linear(N, N//4) + ReLU + Linear(N//4, N) + Tanh
    """
    def __init__(self, N, **kwargs):
        super(TinyMLP, self).__init__(**kwargs)
        self.N = N

        self.linear1 = Dense(self.N // 4, use_bias=True)
        self.relu = Activation('relu')
        self.linear2 = Dense(self.N, use_bias=True)
        self.tanh = Activation('tanh')

    def get_config(self):
        config = super().get_config()
        config.update({
            "N": self.N,
        })
        return config

    def call(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.tanh(x)
        return x


@register_keras_serializable(package="AMCNet")
class AdaCorrModule(tf.keras.layers.Layer):
    """
    Exact replica of PyTorch AdaCorrModule using FFT operations
    Fixed: FFT on width dimension, MLP operates on width dimension
    """
    def __init__(self, N, **kwargs):
        super(AdaCorrModule, self).__init__(**kwargs)
        self.N = N  # sig_len (width dimension)
        self.Im = TinyMLP(N)
        self.Re = TinyMLP(N)

    def get_config(self):
        config = super().get_config()
        config.update({
            "N": self.N,
        })
        return config

    def call(self, x):
        # x: [batch, 2, W, 1] in TF channels_last format
        x_init = x

        # Apply FFT along the width dimension (axis=2)
        x = tf.cast(x, tf.complex64)
        x = tf.signal.fft(tf.squeeze(x, axis=-1))  # Remove channel dim, FFT on width
        x = tf.expand_dims(x, axis=-1)  # Restore channel dim

        X_re = tf.math.real(x)
        X_im = tf.math.imag(x)

        # Reshape for MLP processing: apply MLP on width dimension
        # (batch, 2, W, 1) -> (batch, 2, 1, W) for MLP input
        X_re_for_mlp = tf.transpose(X_re, perm=[0, 1, 3, 2])
        X_im_for_mlp = tf.transpose(X_im, perm=[0, 1, 3, 2])

        h_re = self.Re(X_re_for_mlp)
        h_im = self.Im(X_im_for_mlp)

        # Transpose back: (batch, 2, 1, W) -> (batch, 2, W, 1)
        h_re = tf.transpose(h_re, perm=[0, 1, 3, 2])
        h_im = tf.transpose(h_im, perm=[0, 1, 3, 2])

        # Complex multiplication: h_re * X_re + 1j * h_im * X_im
        x = tf.complex(tf.multiply(h_re, X_re), tf.multiply(h_im, X_im))
        x = tf.math.real(tf.signal.ifft(tf.squeeze(x, axis=-1)))
        x = tf.expand_dims(x, axis=-1)

        # Residual connection
        x = Add()([x, x_init])

        return x


@register_keras_serializable(package="AMCNet")
class FeaFusionModule(tf.keras.layers.Layer):
    """
    Exact replica of PyTorch FeaFusionModule:
    - Input: (batch, C, W) where attention is across channel dimension C
    - Linear operations on width dimension W
    - Multi-head attention across channels
    """
    def __init__(self, num_attention_heads, input_size, hidden_size, **kwargs):
        super(FeaFusionModule, self).__init__(**kwargs)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads %d"
                % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.input_size = input_size  # sig_len (width dimension)
        self.hidden_size = hidden_size  # also sig_len

        # Linear layers operate on the width dimension (last dim after transpose)
        self.key_layer = Dense(hidden_size, use_bias=True)
        self.query_layer = Dense(hidden_size, use_bias=True)
        self.value_layer = Dense(hidden_size, use_bias=True)
        self.dropout = Dropout(0.5)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_attention_heads": self.num_attention_heads,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
        })
        return config

    def trans_to_multiple_heads(self, x):
        # x shape: [batch_size, num_channels, hidden_size]
        batch_size = tf.shape(x)[0]
        num_channels = tf.shape(x)[1]

        # Reshape to [batch_size, num_channels, num_heads, head_size]
        x = tf.reshape(x, [batch_size, num_channels, self.num_attention_heads, self.attention_head_size])

        # Transpose to [batch_size, num_heads, num_channels, head_size]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, x):
        # Input: (batch, num_channels, width) - attention across channels
        # Linear operations on width dimension
        key = self.key_layer(x)      # (batch, C, width) -> (batch, C, hidden_size)
        query = self.query_layer(x)  # (batch, C, width) -> (batch, C, hidden_size)
        value = self.value_layer(x)  # (batch, C, width) -> (batch, C, hidden_size)

        key_heads = self.trans_to_multiple_heads(key)      # (batch, heads, C, head_size)
        query_heads = self.trans_to_multiple_heads(query)  # (batch, heads, C, head_size)
        value_heads = self.trans_to_multiple_heads(value)  # (batch, heads, C, head_size)

        # Attention scores across channel dimension
        attention_scores = tf.matmul(query_heads, key_heads, transpose_b=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        context = tf.matmul(attention_probs, value_heads)

        # Reshape back: [batch_size, num_heads, num_channels, head_size] -> [batch_size, num_heads*num_channels, head_size]
        # This matches PyTorch: context.contiguous().view(shape[0], -1, shape[-1])
        batch_size = tf.shape(context)[0]
        num_channels = tf.shape(context)[2]
        head_size = tf.shape(context)[3]
        # Flatten heads*channels: (batch, heads, C, head_size) -> (batch, heads*C, head_size)
        context = tf.reshape(context, [batch_size, self.num_attention_heads * num_channels, head_size])

        return context


def build_amcnet_model(input_shape, num_classes=11, sig_len=128, extend_channel=36,
                      latent_dim=512, num_heads=2, conv_chan_list=None):
    """
    Build AMC-Net model exactly matching PyTorch implementation.

    Args:
        input_shape: Input shape (2, seq_len)
        num_classes: Number of modulation classes (default: 11)
        sig_len: Signal length (default: 128)
        extend_channel: Extended channel number (default: 36)
        latent_dim: Latent dimension (default: 512)
        num_heads: Number of attention heads (default: 2)
        conv_chan_list: Channel list for conv layers (default: [36, 64, 128, 256])

    Returns:
        Compiled Keras model
    """

    if conv_chan_list is None:
        conv_chan_list = [36, 64, 128, 256]

    # Infer sig_len from input_shape if not provided
    if sig_len != input_shape[1]:
        sig_len = input_shape[1]

    # Input: (batch, 2, W) exactly like PyTorch
    inputs = Input(shape=input_shape, name='input_signals')

    # PyTorch: x.unsqueeze(1) -> (batch, 1, 2, W)
    # TensorFlow equivalent: (batch, 2, W) -> (batch, 2, W, 1)
    x = Reshape((input_shape[0], input_shape[1], 1))(inputs)  # (batch, 2, W, 1)

    # AdaCorrModule - operates on (batch, 2, W, 1)
    x = AdaCorrModule(sig_len)(x)

    # L2 normalization along width dimension (matching PyTorch commented line)
    # PyTorch: x = x / x.norm(p=2, dim=-1, keepdim=True)  # dim=-1 is width
    # x = L2NormalizeWidthLayer()(x)  # axis=2 is width dimension

    # MultiScaleModule - input: (batch, 2, W, 1), treats 2 as height, 1 as in_channels
    x = MultiScaleModule(extend_channel)(x)  # output: (batch, 1, W, 36)

    # Conv stem layers
    stem_layers_num = len(conv_chan_list) - 1
    for t in range(stem_layers_num):
        x = Conv_Block(conv_chan_list[t], conv_chan_list[t + 1])(x)

    # After conv stem: (batch, 1, W, 256)
    # Remove height dimension: (batch, 1, W, 256) -> (batch, W, 256)
    x = SqueezeHeightLayer()(x)

    # Transpose to match PyTorch: (batch, W, 256) -> (batch, 256, W)
    # This matches PyTorch input to FFM: (N, C=256, W)
    x = TransposeChannelWidthLayer()(x)

    # Feature Fusion Module - operates on (batch, C=256, W)
    # Output: (batch, heads*C, head_size=W/heads) = (batch, 2*256, 128/2) = (batch, 512, 64)
    x = FeaFusionModule(num_heads, sig_len, sig_len)(x)

    # Global Average Pooling on head_size dimension: (batch, heads*C, head_size) -> (batch, heads*C)
    # This matches PyTorch AdaptiveAvgPool1d(1) followed by squeeze
    x = GlobalAvgPoolWidthLayer()(x)  # (batch, 512, 64) -> (batch, 512)

    # Now x is (batch, 512) matching PyTorch exactly

    # Classifier - matching PyTorch version exactly
    # PyTorch: Linear(512, 512) + Dropout(0.5) + PReLU + Linear(512, num_classes)
    x = Dense(latent_dim, use_bias=True)(x)        # 512 -> 512 (latent_dim should be 512)
    x = Dropout(0.5)(x)
    x = PReLU()(x)                                 # PReLU activation
    outputs = Dense(num_classes, use_bias=True)(x) # 512 -> num_classes

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='AMC_Net')

    # Compile model - use default learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


# Export the main build function for integration
build_amcnet = build_amcnet_model


if __name__ == "__main__":
    # Test model creation
    print("Testing AMC-Net model creation...")

    # Test with default parameters (RML2016 format)
    print("\nTesting RML2016 format (2, 128):")
    model_2016 = build_amcnet_model((2, 128), num_classes=11, sig_len=128)
    model_2016.summary()
    print(f"Total parameters: {model_2016.count_params():,}")

    # Test with RML2018 format
    print("\nTesting RML2018 format (2, 1024):")
    model_2018 = build_amcnet_model((2, 1024), num_classes=24, sig_len=1024)
    model_2018.summary()
    print(f"Total parameters: {model_2018.count_params():,}")

    print("\n✅ AMC-Net model created successfully!")