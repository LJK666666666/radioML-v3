from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Reshape
from keras.optimizers import Adam

def build_cnn2d_model(input_shape, num_classes):
    """
    Build a 2D CNN model for radio signal classification.
    This treats the I/Q components as a 2-channel image.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    # For RadioML, each example has shape (2, 128) for I/Q channels
    # Reshape to (128, 2, 1) to treat as an image with 2 spatial dimensions
    model = Sequential()
    
    # Reshape and reorder dimensions for 2D convolution
    model.add(Reshape((input_shape[1], input_shape[0], 1), input_shape=input_shape))
    
    # First convolutional block
    model.add(Conv2D(64, (3, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))  # Only pool along time dimension
    
    # Second convolutional block
    model.add(Conv2D(128, (3, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    
    # Third convolutional block
    model.add(Conv2D(256, (3, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    
    # Flatten the output and feed it into dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model
