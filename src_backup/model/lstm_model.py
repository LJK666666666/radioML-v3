"""
LSTM Model for Radio Signal Classification

This module provides LSTM-based neural network architectures for radio signal classification.
LSTM (Long Short-Term Memory) networks are well-suited for sequential data and can capture
temporal dependencies in radio signals.
"""

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Dropout, LSTM, Bidirectional, BatchNormalization,
    Permute, GlobalMaxPooling1D, Concatenate, TimeDistributed,
    Activation, Add, LayerNormalization
)
from keras.optimizers import Adam
from keras.regularizers import l2


def build_lstm_model(input_shape, num_classes, lstm_units=128, num_layers=2, 
                     dropout_rate=0.3, recurrent_dropout=0.2, bidirectional=True, 
                     use_batch_norm=True, use_residual=False):
    """
    Build a standard LSTM model for radio signal classification.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length), e.g., (2, 128)
        num_classes: Number of classes to classify
        lstm_units: Number of LSTM units in each layer
        num_layers: Number of LSTM layers
        dropout_rate: Dropout rate for regularization
        recurrent_dropout: Recurrent dropout rate for LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        use_batch_norm: Whether to use batch normalization
        use_residual: Whether to add residual connections
        
    Returns:
        A compiled Keras model
    """
    model = Sequential()
    
    # Permute to (sequence_length, channels) for LSTM input
    # e.g., from (2, 128) to (128, 2)
    model.add(Permute((2, 1), input_shape=input_shape))
    
    # Build LSTM layers
    for i in range(num_layers):
        return_sequences = True if i < num_layers - 1 else False
        
        if bidirectional:
            model.add(Bidirectional(
                LSTM(lstm_units, 
                     return_sequences=return_sequences,
                     dropout=dropout_rate,
                     recurrent_dropout=recurrent_dropout,
                     kernel_regularizer=l2(0.01))
            ))
        else:
            model.add(LSTM(lstm_units, 
                          return_sequences=return_sequences,
                          dropout=dropout_rate,
                          recurrent_dropout=recurrent_dropout,
                          kernel_regularizer=l2(0.01)))
        
        if use_batch_norm and return_sequences:
            model.add(BatchNormalization())
    
    # Note: The last LSTM layer has return_sequences=False, so no pooling is needed
    # The output is already 2D (batch_size, features)
    
    # Dense layers for classification
    model.add(Dense(256, activation='relu'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


def build_advanced_lstm_model(input_shape, num_classes, lstm_units=128, 
                              num_layers=3, dropout_rate=0.3, 
                              recurrent_dropout=0.2, attention=True):
    """
    Build an advanced LSTM model with attention mechanism and residual connections.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        lstm_units: Number of LSTM units in each layer
        num_layers: Number of LSTM layers
        dropout_rate: Dropout rate for regularization
        recurrent_dropout: Recurrent dropout rate for LSTM layers
        attention: Whether to use attention mechanism
        
    Returns:
        A compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # Permute to (sequence_length, channels) for LSTM input
    x = Permute((2, 1))(inputs)
    
    # First LSTM layer
    x = Bidirectional(LSTM(lstm_units, 
                          return_sequences=True,
                          dropout=dropout_rate,
                          recurrent_dropout=recurrent_dropout,
                          kernel_regularizer=l2(0.01)))(x)
    x = BatchNormalization()(x)
    
    # Store for residual connection
    residual = x
    
    # Additional LSTM layers with residual connections
    for i in range(1, num_layers):
        return_sequences = True if i < num_layers - 1 else True  # Always True for attention
        
        lstm_out = Bidirectional(LSTM(lstm_units, 
                                     return_sequences=return_sequences,
                                     dropout=dropout_rate,
                                     recurrent_dropout=recurrent_dropout,
                                     kernel_regularizer=l2(0.01)))(x)
        
        if return_sequences and lstm_out.shape[-1] == residual.shape[-1]:
            x = Add()([lstm_out, residual])
            x = LayerNormalization()(x)
        else:
            x = lstm_out
            
        x = BatchNormalization()(x)
        residual = x
    
    # Attention mechanism
    if attention:
        # Self-attention
        attention_weights = Dense(lstm_units * 2, activation='tanh')(x)
        attention_weights = Dense(1)(attention_weights)  # Remove softmax here
        attention_weights = tf.squeeze(attention_weights, axis=-1)  # Shape: (batch_size, sequence_length)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)  # Apply softmax on sequence dimension
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # Shape: (batch_size, sequence_length, 1)
        x = tf.keras.layers.Multiply()([x, attention_weights])
    
    # Global pooling
    x = GlobalMaxPooling1D()(x)
    
    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


def build_multi_scale_lstm_model(input_shape, num_classes, lstm_units=128):
    """
    Build a multi-scale LSTM model that processes signals at different temporal scales.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        lstm_units: Number of LSTM units in each branch
        
    Returns:
        A compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # Permute to (sequence_length, channels) for LSTM input
    x = Permute((2, 1))(inputs)
    
    # Branch 1: Full resolution
    branch1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    branch1 = GlobalMaxPooling1D()(branch1)
    
    # Branch 2: Downsampled by 2
    x_down2 = x[:, ::2, :]  # Downsample by factor of 2
    branch2 = Bidirectional(LSTM(lstm_units // 2, return_sequences=True))(x_down2)
    branch2 = GlobalMaxPooling1D()(branch2)
    
    # Branch 3: Downsampled by 4
    x_down4 = x[:, ::4, :]  # Downsample by factor of 4
    branch3 = Bidirectional(LSTM(lstm_units // 4, return_sequences=True))(x_down4)
    branch3 = GlobalMaxPooling1D()(branch3)
    
    # Concatenate all branches
    x = Concatenate()([branch1, branch2, branch3])
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


def build_lightweight_lstm_model(input_shape, num_classes, lstm_units=64):
    """
    Build a lightweight LSTM model for faster training and inference.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        lstm_units: Number of LSTM units
        
    Returns:
        A compiled Keras model
    """
    model = Sequential()
    
    # Permute to (sequence_length, channels) for LSTM input
    model.add(Permute((2, 1), input_shape=input_shape))
    
    # Single bidirectional LSTM layer
    model.add(Bidirectional(LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.2)))
    
    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model
