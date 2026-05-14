#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complex Neural Network Module

This module provides complex-valued neural network layers and utilities
for processing I/Q signal data in radio frequency applications.

Authors: Adapted from ULCNN project for integration with RadioML classification framework
"""

from .conv import ComplexConv1D
from .bn import ComplexBatchNormalization, sqrt_init
from .dense import ComplexDense
from .utils import (
    channel_shuffle, dwconv_mobile, channelattention,
    ChannelShuffle, DWConvMobile, ChannelAttention,
    TransposeLayer, ExtractChannelLayer, TrigonometricLayer,
    rotate_matrix, rotate_data_augmentation
)

__all__ = [
    # Core complex layers
    'ComplexConv1D',
    'ComplexBatchNormalization', 
    'ComplexDense',
    
    # Initialization functions
    'sqrt_init',
    
    # Utility functions
    'channel_shuffle',
    'dwconv_mobile', 
    'channelattention',
    
    # Utility layers
    'ChannelShuffle',
    'DWConvMobile',
    'ChannelAttention',
    'TransposeLayer',
    'ExtractChannelLayer',
    'TrigonometricLayer',
    
    # Data augmentation utilities
    'rotate_matrix',
    'rotate_data_augmentation'
]