from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization, Activation, Permute
from keras.optimizers import Adam

def build_cnn1d_model(input_shape, num_classes):
    """
    Build a 1D CNN model for radio signal classification.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    model = Sequential()
    
    # For RadioML, we typically have input shape (2, 128) where 2 represents I/Q channels
    # We need to treat each sample as having 128 time steps with 2 features each
    model.add(Permute((2, 1), input_shape=input_shape))
    
    # First convolutional block - now operating on time dimension
    model.add(Conv1D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Second convolutional block
    model.add(Conv1D(128, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Third convolutional block
    model.add(Conv1D(256, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    
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
