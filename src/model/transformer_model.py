import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import MultiHeadAttention, LayerNormalization, Add, Permute
from keras.optimizers import Adam
 
def build_transformer_model(input_shape, num_classes, num_heads=4, ff_dim=64, num_transformer_blocks=1, embed_dim=64, dropout_rate=0.1):
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
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
