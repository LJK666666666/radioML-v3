import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization, Activation, Permute
from keras.optimizers import Adam

def build_resnet_model(input_shape, num_classes):
    """
    Build a ResNet-style model for radio signal classification.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Transpose to have sequence_length first, features second
    x = Permute((2, 1))(inputs)
    
    # First convolutional block
    x = Conv1D(64, 7, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Residual block 1
    shortcut = x
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    
    # Residual block 2
    shortcut = Conv1D(128, 1, strides=2, padding='same')(x)
    x = Conv1D(128, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    
    # Residual block 3
    shortcut = Conv1D(256, 1, strides=2, padding='same')(x)
    x = Conv1D(256, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    
    # Global pooling and dense layers
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Added dropout for regularization
    outputs = Dense(num_classes, activation='softmax')(x) # Added output layer
    
    model = Model(inputs=inputs, outputs=outputs) # Defined model
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model
