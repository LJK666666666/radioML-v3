import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Layer
from keras.layers import MultiHeadAttention, LayerNormalization, Add, Permute
from keras.optimizers import Adam


class RotaryPositionalEncoding(Layer):
    """
    Rotary Positional Encoding (RoPE) layer for transformer models.
    Supports two modes:
    1. 'sequential': Standard sequential position encoding
    2. 'phase': Position based on I/Q complex phase angle
    """

    def __init__(self, embed_dim, max_seq_len=128, mode='sequential', **kwargs):
        super(RotaryPositionalEncoding, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.mode = mode

        # Ensure embed_dim is even for RoPE
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for Rotary Positional Encoding")

    def build(self, input_shape):
        super(RotaryPositionalEncoding, self).build(input_shape)

    def get_angles(self, positions, embed_dim):
        """Calculate the angles for rotary encoding"""
        # Create frequency bands
        inv_freq = 1.0 / (10000 ** (tf.range(0, embed_dim, 2, dtype=tf.float32) / embed_dim))

        # Calculate angles: position * inv_freq
        angles = tf.einsum('i,j->ij', positions, inv_freq)

        return angles

    def apply_rotary_encoding(self, x, angles):
        """Apply rotary encoding to input tensor"""
        # Split x into pairs for rotation
        x_pairs = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1, 2])

        # Get cos and sin values
        cos_angles = tf.cos(angles)
        sin_angles = tf.sin(angles)

        # Expand dimensions to match x_pairs
        cos_angles = tf.expand_dims(cos_angles, -1)  # [seq_len, embed_dim//2, 1]
        sin_angles = tf.expand_dims(sin_angles, -1)  # [seq_len, embed_dim//2, 1]

        # Apply rotation
        x1, x2 = x_pairs[..., 0:1], x_pairs[..., 1:2]
        rotated_x1 = x1 * cos_angles - x2 * sin_angles
        rotated_x2 = x1 * sin_angles + x2 * cos_angles

        # Concatenate and reshape back
        rotated_pairs = tf.concat([rotated_x1, rotated_x2], axis=-1)
        rotated_x = tf.reshape(rotated_pairs, tf.shape(x))

        return rotated_x

    def call(self, inputs):
        """
        Apply rotary positional encoding

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, embed_dim]
                   For mode='phase', expects the original I/Q data as additional input
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        if self.mode == 'sequential':
            # Standard sequential positions
            positions = tf.cast(tf.range(seq_len), tf.float32)

        elif self.mode == 'phase':
            # Extract I/Q components and calculate phase angles
            # Assume inputs contain both embedded features and original I/Q data
            # For simplicity, we'll use sequential positions here and modify in the model
            positions = tf.cast(tf.range(seq_len), tf.float32)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Calculate rotation angles
        angles = self.get_angles(positions, self.embed_dim)

        # Apply rotary encoding
        encoded_inputs = self.apply_rotary_encoding(inputs, angles)

        return encoded_inputs

    def get_config(self):
        config = super(RotaryPositionalEncoding, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'max_seq_len': self.max_seq_len,
            'mode': self.mode
        })
        return config


class PhaseBasedPositionalEncoding(Layer):
    """
    Phase-based positional encoding using I/Q complex phase angles
    """

    def __init__(self, embed_dim, **kwargs):
        super(PhaseBasedPositionalEncoding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        super(PhaseBasedPositionalEncoding, self).build(input_shape)

    def call(self, inputs, iq_data):
        """
        Apply phase-based positional encoding

        Args:
            inputs: Embedded input tensor [batch_size, seq_len, embed_dim]
            iq_data: Original I/Q data [batch_size, seq_len, 2]
        """
        # Extract I and Q components
        i_component = iq_data[..., 0]  # [batch_size, seq_len]
        q_component = iq_data[..., 1]  # [batch_size, seq_len]

        # Calculate phase angles
        phase_angles = tf.atan2(q_component, i_component)  # [batch_size, seq_len]

        # Normalize phase angles to [0, 1]
        normalized_phases = (phase_angles + np.pi) / (2 * np.pi)

        # Create frequency bands for encoding
        inv_freq = 1.0 / (10000 ** (tf.range(0, self.embed_dim, 2, dtype=tf.float32) / self.embed_dim))

        # Calculate angles using normalized phases
        angles = tf.einsum('bi,j->bij', normalized_phases, inv_freq)

        # Create positional encoding
        pos_encoding_sin = tf.sin(angles)
        pos_encoding_cos = tf.cos(angles)

        # Interleave sin and cos
        pos_encoding = tf.stack([pos_encoding_sin, pos_encoding_cos], axis=-1)
        pos_encoding = tf.reshape(pos_encoding, [tf.shape(inputs)[0], tf.shape(inputs)[1], self.embed_dim])

        # Add positional encoding to inputs
        return inputs + pos_encoding

    def get_config(self):
        config = super(PhaseBasedPositionalEncoding, self).get_config()
        config.update({
            'embed_dim': self.embed_dim
        })
        return config


def build_transformer_model(input_shape, num_classes, num_heads=4, ff_dim=64, num_transformer_blocks=3, embed_dim=64, dropout_rate=0.1):
    """
    Build a Transformer-based model for radio signal classification.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed forward network inside transformer
        num_transformer_blocks: Number of transformer blocks
        embed_dim: Embedding dimension for the transformer
        dropout_rate: Dropout rate
        
    Returns:
        A compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # Permute to (sequence_length, channels) e.g., (128, 2)
    x = Permute((2, 1))(inputs) 
    
    # Project to embedding dimension
    x = Dense(embed_dim)(x)

    for _ in range(num_transformer_blocks):
        # Multi-Head Attention block
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Forward Network block
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dense(embed_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
    # Pooling and Output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x) # Increased dropout before final dense layers
    x = Dense(128, activation="relu")(x) # Intermediate dense layer
    x = Dropout(0.3)(x) # Increased dropout
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # optimizer = Adam(learning_rate=0.001)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


# def build_transformer_rope_sequential_model(input_shape, num_classes, num_heads=4, ff_dim=64, num_transformer_blocks=3, embed_dim=64, dropout_rate=0.1):
def build_transformer_rope_sequential_model(input_shape, num_classes, num_heads=4, ff_dim=256, num_transformer_blocks=4, embed_dim=256, dropout_rate=0.3):

    """
    Build a Transformer model with Rotary Positional Encoding using sequential positions.

    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed forward network inside transformer
        num_transformer_blocks: Number of transformer blocks
        embed_dim: Embedding dimension for the transformer
        dropout_rate: Dropout rate

    Returns:
        A compiled Keras model with RoPE sequential encoding
    """
    inputs = Input(shape=input_shape)

    # Permute to (sequence_length, channels) e.g., (128, 2)
    x = Permute((2, 1))(inputs)

    # Project to embedding dimension
    x = Dense(embed_dim)(x)

    # Apply Rotary Positional Encoding with sequential positions
    x = RotaryPositionalEncoding(embed_dim=embed_dim, mode='sequential')(x)

    for _ in range(num_transformer_blocks):
        # Multi-Head Attention block
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed Forward Network block
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dense(embed_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    # Pooling and Output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def build_transformer_rope_phase_model(input_shape, num_classes, num_heads=4, ff_dim=64, num_transformer_blocks=3, embed_dim=64, dropout_rate=0.1):
    """
    Build a Transformer model with Phase-based Positional Encoding using I/Q complex phase angles.

    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed forward network inside transformer
        num_transformer_blocks: Number of transformer blocks
        embed_dim: Embedding dimension for the transformer
        dropout_rate: Dropout rate

    Returns:
        A compiled Keras model with phase-based encoding
    """
    inputs = Input(shape=input_shape)

    # Permute to (sequence_length, channels) e.g., (128, 2)
    iq_data = Permute((2, 1))(inputs)

    # Project to embedding dimension
    x = Dense(embed_dim)(iq_data)

    # Apply Phase-based Positional Encoding using I/Q phase angles
    x = PhaseBasedPositionalEncoding(embed_dim=embed_dim)(x, iq_data)

    for _ in range(num_transformer_blocks):
        # Multi-Head Attention block
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed Forward Network block
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dense(embed_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    # Pooling and Output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
