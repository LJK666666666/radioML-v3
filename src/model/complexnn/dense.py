#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complex Dense Layer

This module implements complex-valued dense (fully connected) layers for neural networks.
Adapted from the original complexnn library for compatibility with current TensorFlow/Keras versions.
"""

import tensorflow as tf
from keras.layers import Layer
from keras import initializers, regularizers, constraints, activations
from keras.saving import register_keras_serializable
import keras.backend as K
import numpy as np
from numpy.random import RandomState


@register_keras_serializable(package="ULComplexNN")
class ComplexDense(Layer):
    """
    Complex Dense (Fully Connected) Layer
    
    Implements a complex-valued dense layer that performs complex matrix multiplication.
    For complex input z = x + iy and complex weights W = W_r + iW_i:
    
    output_real = x * W_r - y * W_i + b_r
    output_imag = x * W_i + y * W_r + b_i
    
    Args:
        units: Positive integer, dimensionality of the output space (number of complex units)
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector
        init_criterion: Weight initialization criterion ('he' or 'glorot')
        kernel_initializer: Initializer for the kernel weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to the kernel weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
        activity_regularizer: Regularizer function applied to the output
        kernel_constraint: Constraint function applied to the kernel matrix
        bias_constraint: Constraint function applied to the bias vector
        seed: Random seed for weight initialization
    """
    
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        
        # Handle complex initialization
        if kernel_initializer == 'complex':
            self.kernel_initializer = 'complex'
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
            
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
            
        self.supports_masking = True
        
    def build(self, input_shape):
        assert len(input_shape) == 2, "ComplexDense layer expects 2D input"
        assert input_shape[-1] % 2 == 0, "Input dimension must be even (real and imaginary parts)"
        
        input_dim = input_shape[-1] // 2  # Number of complex input features
        kernel_shape = (input_dim, self.units)
        
        # Compute fan_in and fan_out for proper initialization
        fan_in = input_dim
        fan_out = self.units
        
        # Initialize scaling factor based on criterion
        if self.init_criterion == 'he':
            s = np.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = np.sqrt(1. / (fan_in + fan_out))
        else:
            s = np.sqrt(1. / fan_in)  # Default to He initialization
        
        # Create random number generator with fixed seed
        rng = RandomState(seed=self.seed)
        
        # Complex weight initialization
        if self.kernel_initializer == 'complex':
            # Initialize complex weights using Gaussian distribution
            def init_real_kernel(shape, dtype=None):
                return rng.normal(size=kernel_shape, loc=0, scale=s).astype(dtype or tf.float32)
            
            def init_imag_kernel(shape, dtype=None):
                return rng.normal(size=kernel_shape, loc=0, scale=s).astype(dtype or tf.float32)
            
            real_init = init_real_kernel
            imag_init = init_imag_kernel
        else:
            # Use standard initializer for both real and imaginary parts
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer
        
        # Create weight matrices
        self.kernel_real = self.add_weight(
            name='kernel_real',
            shape=kernel_shape,
            initializer=real_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )
        
        self.kernel_imag = self.add_weight(
            name='kernel_imag',
            shape=kernel_shape,
            initializer=imag_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )
        
        # Create bias vectors if needed
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(2 * self.units,),  # Real and imaginary bias
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
            )
        
        super(ComplexDense, self).build(input_shape)
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        input_dim = input_shape[-1] // 2
        
        # Split input into real and imaginary parts
        input_real = inputs[..., :input_dim]
        input_imag = inputs[..., input_dim:]
        
        # Complex matrix multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        output_real = tf.matmul(input_real, self.kernel_real) - tf.matmul(input_imag, self.kernel_imag)
        output_imag = tf.matmul(input_real, self.kernel_imag) + tf.matmul(input_imag, self.kernel_real)
        
        # Concatenate real and imaginary parts
        output = tf.concat([output_real, output_imag], axis=-1)
        
        # Add bias if used
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        
        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.units  # Real and imaginary parts
        return tuple(output_shape)
    
    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': 'complex' if self.kernel_initializer == 'complex' else initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seed': self.seed
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))