# common

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def get_callbacks(checkpoint_path, patience_lr=2, patience_es=30):
    """
    Prepare callbacks for model training.
    
    Args:
        checkpoint_path: Path to save the best model.
        patience_lr: Patience for ReduceLROnPlateau.
        patience_es: Patience for EarlyStopping.
        
    Returns:
        A list of Keras callbacks.
    """
    # checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=patience_lr, verbose=1, mode='min', min_lr=1e-7)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='min', restore_best_weights=True)
    
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience= patience_lr, verbose=1, mode='max', min_lr=1e-7)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience_es, verbose=1, mode='max', restore_best_weights=True)

    return [checkpoint, reduce_lr, early_stopping]



# ulcnn

# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# def get_callbacks(checkpoint_path, patience_lr=10, patience_es=50):
#     """
#     Prepare callbacks for model training.
    
#     Args:
#         checkpoint_path: Path to save the best model.
#         patience_lr: Patience for ReduceLROnPlateau.
#         patience_es: Patience for EarlyStopping.
        
#     Returns:
#         A list of Keras callbacks.
#     """
#     checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=patience_lr, verbose=1, mode='min', min_lr=1e-7)
#     early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='min', restore_best_weights=True)
    
#     # checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#     # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=patience_lr, verbose=1, mode='max', min_lr=1e-6)
#     # early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience_es, verbose=1, mode='max', restore_best_weights=True)

#     return [checkpoint, reduce_lr, early_stopping]

