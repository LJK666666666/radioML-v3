#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complex Convolution Layers

This module implements complex-valued convolution layers for neural networks.
Adapted from the original complexnn library for compatibility with current TensorFlow/Keras versions.
"""

import tensorflow as tf
from keras.layers import Layer
from keras import initializers, regularizers, constraints, activations
from keras.saving import register_keras_serializable
import keras.backend as K
import numpy as np


@register_keras_serializable(package="ULComplexNN")
class ComplexConv1D(Layer):
    """
    Complex 1D Convolution Layer
    
    This layer performs complex convolution on complex-valued input data.
    The input is expected to have real and imaginary parts concatenated along the last axis.
    
    For complex numbers z1 = a + bi and z2 = c + di:
    z1 * z2 = (ac - bd) + (ad + bc)i
    
    Args:
        filters: Integer, the dimensionality of the output space (number of complex filters)
        kernel_size: Integer, specifying the length of the 1D convolution window
        strides: Integer, specifying the stride length of the convolution
        padding: One of "valid" or "same" (case-insensitive)
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector
        kernel_initializer: Initializer for the kernel weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to the kernel weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
        kernel_constraint: Constraint function applied to the kernel matrix
        bias_constraint: Constraint function applied to the bias vector
    """
    
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ComplexConv1D, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
    def build(self, input_shape):
        # Input shape: (batch, time_steps, 2*input_dim) where input_dim is number of complex channels
        input_dim = input_shape[-1] // 2
        
        # Complex kernel weights: W = W_real + j*W_imag
        self.kernel_real = self.add_weight(
            name='kernel_real',
            shape=(self.kernel_size, input_dim, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )
        
        self.kernel_imag = self.add_weight(
            name='kernel_imag', 
            shape=(self.kernel_size, input_dim, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )
        
        if self.use_bias:
            self.bias_real = self.add_weight(
                name='bias_real',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
            )
            
            self.bias_imag = self.add_weight(
                name='bias_imag',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
            )
        
        super(ComplexConv1D, self).build(input_shape)
        
    def call(self, inputs):
        # Split input into real and imaginary parts
        input_dim = tf.shape(inputs)[-1] // 2
        input_real = inputs[..., :input_dim]
        input_imag = inputs[..., input_dim:]
        
        # Complex convolution: (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
        conv_rr = tf.nn.conv1d(input_real, self.kernel_real, stride=self.strides, padding=self.padding)
        conv_ri = tf.nn.conv1d(input_real, self.kernel_imag, stride=self.strides, padding=self.padding)
        conv_ir = tf.nn.conv1d(input_imag, self.kernel_real, stride=self.strides, padding=self.padding)
        conv_ii = tf.nn.conv1d(input_imag, self.kernel_imag, stride=self.strides, padding=self.padding)
        
        # Complex multiplication result
        output_real = conv_rr - conv_ii  # Real part: ac - bd
        output_imag = conv_ri + conv_ir  # Imaginary part: ad + bc
        
        # Add bias if used
        if self.use_bias:
            output_real = tf.nn.bias_add(output_real, self.bias_real)
            output_imag = tf.nn.bias_add(output_imag, self.bias_imag)
        
        # Concatenate real and imaginary parts
        output = tf.concat([output_real, output_imag], axis=-1)
        
        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            out_length = (input_shape[1] - self.kernel_size) // self.strides + 1
        else:  # 'SAME'
            out_length = (input_shape[1] + self.strides - 1) // self.strides
        
        return (input_shape[0], out_length, 2 * self.filters)
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding.lower(),
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ComplexConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))