#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complex Batch Normalization

This module implements complex-valued batch normalization for neural networks.
Adapted from the original complexnn library for compatibility with current TensorFlow/Keras versions.
"""

import tensorflow as tf
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras.saving import register_keras_serializable
import keras.backend as K
import numpy as np


@register_keras_serializable(package="ULComplexNN")
def sqrt_init(shape, dtype=None):
    """
    Square root initialization for complex batch normalization parameters.
    Returns 1/sqrt(2) for proper complex number scaling.
    """
    value = (1 / np.sqrt(2)) * tf.ones(shape, dtype=dtype)
    return value


@register_keras_serializable(package="ULComplexNN")
class ComplexBatchNormalization(Layer):
    """
    Complex Batch Normalization Layer
    
    Normalizes complex-valued inputs by computing complex statistics and applying
    a complex whitening transformation followed by scaling and shifting.
    
    For complex input z = x + iy, the layer computes:
    1. Complex mean: μ = E[z]
    2. Covariance matrix: V = [[Vrr, Vri], [Vri, Vii]]
       where Vrr = Var(x), Vii = Var(y), Vri = Cov(x,y)
    3. Whitening: z_norm = W * (z - μ) where W = V^(-1/2)
    4. Scaling and shifting: output = Γ * z_norm + β
    
    Args:
        axis: Integer, the axis that should be normalized (typically the features axis)
        momentum: Momentum for the moving average
        epsilon: Small float added to variance to avoid dividing by zero
        center: If True, add offset of beta to normalized tensor
        scale: If True, multiply by gamma
        beta_initializer: Initializer for the beta weight
        gamma_diag_initializer: Initializer for the diagonal gamma elements
        gamma_off_initializer: Initializer for the off-diagonal gamma elements
        moving_mean_initializer: Initializer for the moving mean
        moving_variance_initializer: Initializer for the moving variance
        moving_covariance_initializer: Initializer for the moving covariance
    """
    
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer=sqrt_init,
                 gamma_off_initializer='zeros',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer=sqrt_init,
                 moving_covariance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 beta_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_diag_initializer = gamma_diag_initializer if callable(gamma_diag_initializer) else initializers.get(gamma_diag_initializer)
        self.gamma_off_initializer = initializers.get(gamma_off_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = moving_variance_initializer if callable(moving_variance_initializer) else initializers.get(moving_variance_initializer)
        self.moving_covariance_initializer = initializers.get(moving_covariance_initializer)
        
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)
        
    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(f'Axis {self.axis} of input tensor should have a defined dimension '
                           f'but the layer received an input with shape {input_shape}.')
        
        # Number of complex features (input has real and imaginary parts concatenated)
        param_shape = (input_shape[self.axis] // 2,)
        
        if self.scale:
            # Gamma matrix elements: [[γ_rr, γ_ri], [γ_ri, γ_ii]]
            self.gamma_rr = self.add_weight(
                shape=param_shape,
                name='gamma_rr',
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint,
                trainable=True
            )
            
            self.gamma_ii = self.add_weight(
                shape=param_shape,
                name='gamma_ii', 
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint,
                trainable=True
            )
            
            self.gamma_ri = self.add_weight(
                shape=param_shape,
                name='gamma_ri',
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint,
                trainable=True
            )
            
            # Moving statistics for inference
            self.moving_Vrr = self.add_weight(
                shape=param_shape,
                name='moving_Vrr',
                initializer=self.moving_variance_initializer,
                trainable=False
            )
            
            self.moving_Vii = self.add_weight(
                shape=param_shape,
                name='moving_Vii',
                initializer=self.moving_variance_initializer,
                trainable=False
            )
            
            self.moving_Vri = self.add_weight(
                shape=param_shape,
                name='moving_Vri',
                initializer=self.moving_covariance_initializer,
                trainable=False
            )
        
        if self.center:
            self.beta = self.add_weight(
                shape=(input_shape[self.axis],),
                name='beta',
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True
            )
            
            self.moving_mean = self.add_weight(
                shape=(input_shape[self.axis],),
                name='moving_mean',
                initializer=self.moving_mean_initializer,
                trainable=False
            )
        
        super(ComplexBatchNormalization, self).build(input_shape)
        
    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        ndim = len(inputs.shape)
        
        # Compute reduction axes (all except batch and feature axes)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        if 0 in reduction_axes:
            reduction_axes.remove(0)  # Don't reduce over batch dimension
        
        input_dim = input_shape[self.axis] // 2
        
        # Split input into real and imaginary parts
        if self.axis == -1:
            input_real = inputs[..., :input_dim]
            input_imag = inputs[..., input_dim:]
        else:
            # Handle other axis positions if needed
            input_real = tf.slice(inputs, [0] * ndim, 
                                [-1 if i != self.axis else input_dim for i in range(ndim)])
            input_imag = tf.slice(inputs, [0 if i != self.axis else input_dim for i in range(ndim)],
                                [-1] * ndim)
        
        # Compute complex mean
        mu_real = tf.reduce_mean(input_real, axis=reduction_axes, keepdims=True)
        mu_imag = tf.reduce_mean(input_imag, axis=reduction_axes, keepdims=True)
        
        # Center the inputs
        if self.center:
            input_real_centered = input_real - mu_real
            input_imag_centered = input_imag - mu_imag
        else:
            input_real_centered = input_real
            input_imag_centered = input_imag
        
        if self.scale:
            # Compute covariance matrix elements
            Vrr = tf.reduce_mean(input_real_centered ** 2, axis=reduction_axes, keepdims=True) + self.epsilon
            Vii = tf.reduce_mean(input_imag_centered ** 2, axis=reduction_axes, keepdims=True) + self.epsilon
            Vri = tf.reduce_mean(input_real_centered * input_imag_centered, axis=reduction_axes, keepdims=True)
            
            # Complex standardization (whitening transformation)
            # Compute inverse square root of covariance matrix
            tau = Vrr + Vii  # Trace
            delta = Vrr * Vii - Vri ** 2  # Determinant
            
            s = tf.sqrt(delta + self.epsilon)  # Square root of determinant
            t = tf.sqrt(tau + 2 * s + self.epsilon)  # Normalization factor
            
            inverse_st = 1.0 / (s * t + self.epsilon)
            Wrr = (Vii + s) * inverse_st
            Wii = (Vrr + s) * inverse_st
            Wri = -Vri * inverse_st
            
            # Apply whitening transformation
            normalized_real = Wrr * input_real_centered + Wri * input_imag_centered
            normalized_imag = Wri * input_real_centered + Wii * input_imag_centered
            
            # Apply complex scaling (gamma transformation)
            output_real = self.gamma_rr * normalized_real - self.gamma_ri * normalized_imag
            output_imag = self.gamma_ri * normalized_real + self.gamma_ii * normalized_imag
            
        else:
            output_real = input_real_centered
            output_imag = input_imag_centered
        
        # Apply shifting (beta)
        if self.center:
            beta_real = self.beta[:input_dim]
            beta_imag = self.beta[input_dim:]
            output_real = output_real + beta_real
            output_imag = output_imag + beta_imag
        
        # Concatenate real and imaginary parts
        output = tf.concat([output_real, output_imag], axis=self.axis)
        
        # Skip moving average updates for now to avoid shape issues
        # TODO: Implement proper moving average updates for inference mode
        
        return output
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_diag_initializer': 'sqrt_init' if self.gamma_diag_initializer == sqrt_init else initializers.serialize(self.gamma_diag_initializer),
            'gamma_off_initializer': initializers.serialize(self.gamma_off_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': 'sqrt_init' if self.moving_variance_initializer == sqrt_init else initializers.serialize(self.moving_variance_initializer),
            'moving_covariance_initializer': initializers.serialize(self.moving_covariance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint)
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))