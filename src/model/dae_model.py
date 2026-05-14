"""
DAE (Denoising Autoencoder) Model for RadioML
Based on: AMR-Benchmark/RML201610a/DAE/rmlmodels/DAE.py

Architecture parity with legacy script LSTMDAE/201610A/DAELSTM.py:
- Input to LSTM is (128, 2) i.e., time steps x features
- Two stacked LSTMs with 32 units each (return_sequences=True, return_state=True)
- Classification head from final hidden state: 32 -> BN -> Dropout(0) -> 16 -> BN -> Dropout(0) -> num_classes -> Softmax
- Decoder head from sequence output: TimeDistributed(Dense(2))
- Output order is [decoder(reconstruction), softmax(classification)] to match DAELSTM.py
- Compile config mirrors DAELSTM.py defaults: Adam(lr=1e-2), losses [mse, categorical_crossentropy],
  loss_weights [0.9(recon), 0.1(cls)] with metrics on classification

Note on preprocessing:
The legacy script converts I/Q -> [amplitude, phase] and L2-normalizes the amplitude channel per sample
before feeding the model. You can enable the same preprocessing inside the model by setting
`integrate_preprocessing=True` in `build_dae_model`; or use the helper `iq_to_amp_phase` below in your
input pipeline if you prefer external preprocessing.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization,
    TimeDistributed, Lambda
)

import numpy as np


def iq_to_amp_phase(x: np.ndarray) -> np.ndarray:
    """
    Convert I/Q input to amplitude/phase with amplitude L2-normalization per sample.

    Args:
        x: numpy array of shape (N, 2, 128) where x[:, 0, :] = I, x[:, 1, :] = Q

    Returns:
        numpy array of shape (N, 128, 2) where [:, :, 0] = normalized amplitude,
        [:, :, 1] = phase/π, ready to feed LSTM expecting (time, features)
    """
    assert x.ndim == 3 and x.shape[1] == 2, "Input must be (N, 2, 128)"
    signal_len = x.shape[2]
    cmplx = x[:, 0, :] + 1j * x[:, 1, :]
    amp = np.abs(cmplx)
    ang = np.arctan2(x[:, 1, :], x[:, 0, :]) / np.pi
    amp = amp.reshape(-1, 1, signal_len)
    ang = ang.reshape(-1, 1, signal_len)
    out = np.concatenate([amp, ang], axis=1)  # (N, 2, T)
    out = np.transpose(out, (0, 2, 1))        # (N, T, 2)
    # L2-normalize amplitude channel per sample
    for i in range(out.shape[0]):
        norm = np.linalg.norm(out[i, :, 0], ord=2)
        if norm > 0:
            out[i, :, 0] = out[i, :, 0] / norm
    return out.astype(np.float32)


def build_dae_model(input_shape=(2, 128), num_classes=11, use_gpu_lstm=True, integrate_preprocessing=False):
    """
    Build DAE model for automatic modulation classification.

    Args:
        input_shape: Input shape (channels, time_steps) = (2, 128) for I/Q data
        num_classes: Number of modulation classes
        use_gpu_lstm: Whether to use GPU-optimized LSTM (CuDNNLSTM-like behavior)
                     If False, uses standard LSTM

    Returns:
        Keras Model with dual outputs: (reconstruction, classification)

    Note:
        The original model uses CuDNNLSTM which is deprecated. We use standard LSTM
        with activation='tanh', recurrent_activation='sigmoid' to match CuDNNLSTM behavior.
    """
    # Input shape and optional integrated preprocessing (I/Q -> amp/phase + L2-normalize amplitude)
    inputs = Input(shape=input_shape, name='input')
    if integrate_preprocessing:
        eps = 1e-8
        if input_shape == (2, 128):
            # channels-first over time: (B, 2, 128)
            def _preproc(t):
                i = t[:, 0, :]  # (B, 128)
                q = t[:, 1, :]
                amp = tf.sqrt(tf.square(i) + tf.square(q))
                ang = tf.math.atan2(q, i) / tf.constant(np.pi, dtype=t.dtype)
                norm = tf.sqrt(tf.reduce_sum(tf.square(amp), axis=-1, keepdims=True)) + eps
                amp = amp / norm
                out = tf.stack([amp, ang], axis=-1)  # (B, 128, 2)
                return out
            x = Lambda(_preproc, name='iq_to_amp_phase_layer')(inputs)
        elif input_shape == (128, 2):
            # time-major, channels-last: (B, 128, 2)
            def _preproc(t):
                i = t[:, :, 0]
                q = t[:, :, 1]
                amp = tf.sqrt(tf.square(i) + tf.square(q))
                ang = tf.math.atan2(q, i) / tf.constant(np.pi, dtype=t.dtype)
                norm = tf.sqrt(tf.reduce_sum(tf.square(amp), axis=-1, keepdims=True)) + eps
                amp = amp / norm
                out = tf.stack([amp, ang], axis=-1)  # (B, 128, 2)
                return out
            x = Lambda(_preproc, name='iq_to_amp_phase_layer')(inputs)
        else:
            # Fallback: assume already (time, features)
            x = inputs
    else:
        # No preprocessing: ensure shape (time, features) for LSTM
        if input_shape == (2, 128):
            x = tf.keras.layers.Permute((2, 1), name='to_time_major')(inputs)  # (B, 128, 2)
        else:
            x = inputs  # already (128, 2)

    # Dropout rate
    dr = 0.0  # Set to 0 as in original

    # LSTM Unit 1: 32 units, return sequences and states
    # CuDNNLSTM equivalent: use activation='tanh', recurrent_activation='sigmoid'
    lstm_kwargs = {
        'units': 32,
        'return_state': True,
        'return_sequences': True
    }

    if use_gpu_lstm:
        # Standard LSTM with settings that match CuDNNLSTM behavior
        lstm_kwargs.update({
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'use_bias': True,
            'recurrent_dropout': 0.0
        })

    x, s, c = LSTM(**lstm_kwargs, name='encoder_1')(x)
    x = Dropout(dr, name='drop_1')(x)

    # LSTM Unit 2: 32 units, return sequences and states
    x, s1, c1 = LSTM(**lstm_kwargs, name='encoder_2')(x)

    # Classifier branch (uses final state s1)
    xc = Dense(32, activation='relu', name='clf_dense_1')(s1)
    xc = BatchNormalization(name='bn_1')(xc)
    xc = Dropout(dr, name='clf_drop_1')(xc)
    xc = Dense(16, activation='relu', name='clf_dense_2')(xc)
    xc = BatchNormalization(name='bn_2')(xc)
    xc = Dropout(dr, name='clf_drop_2')(xc)
    softmax = Dense(num_classes, activation='softmax', name='softmax')(xc)

    # Decoder branch (uses full sequence x)
    decoder = TimeDistributed(Dense(2), name='decoder')(x)

    # Create model with dual outputs (order aligned with legacy: [decoder, softmax])
    model = Model(inputs=inputs, outputs=[decoder, softmax], name='DAE')

    # Compile model (mirror DAELSTM.py): lr=1e-2, losses [mse, categorical_crossentropy],
    # loss_weights = [0.9, 0.1] with metrics on classification only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss={
            'decoder': 'mse',
            'softmax': 'categorical_crossentropy'
        },
        loss_weights={'decoder': 0.9, 'softmax': 0.1},
        metrics={'softmax': 'accuracy'}
    )

    return model


def build_dae_model_classifier_only(input_shape=(2, 128), num_classes=11, use_gpu_lstm=True, integrate_preprocessing=False):
    """
    Build DAE model for classification only (no reconstruction output).

    This version only outputs the classification predictions, which is more
    convenient for standard classification workflows.

    Args:
        input_shape: Input shape (channels, time_steps) = (2, 128) for I/Q data
        num_classes: Number of modulation classes
        use_gpu_lstm: Whether to use GPU-optimized LSTM
        integrate_preprocessing: If True, apply I/Q -> (amp, phase/pi) conversion
            with per-sample L2 normalization on amplitude inside the model.

    Returns:
        Keras Model with single classification output
    """
    # Input shape and optional integrated preprocessing
    inputs = Input(shape=input_shape, name='input')
    if integrate_preprocessing:
        eps = 1e-8
        if input_shape == (2, 128):
            # (B, 2, 128) channels-first over time
            def _preproc(t):
                i = t[:, 0, :]
                q = t[:, 1, :]
                amp = tf.sqrt(tf.square(i) + tf.square(q))
                ang = tf.math.atan2(q, i) / tf.constant(np.pi, dtype=t.dtype)
                norm = tf.sqrt(tf.reduce_sum(tf.square(amp), axis=-1, keepdims=True)) + eps
                amp = amp / norm
                out = tf.stack([amp, ang], axis=-1)  # (B, 128, 2)
                return out
            x = tf.keras.layers.Lambda(_preproc, name='iq_to_amp_phase_layer')(inputs)
        elif input_shape == (128, 2):
            # (B, 128, 2) time-major, channels-last
            def _preproc(t):
                i = t[:, :, 0]
                q = t[:, :, 1]
                amp = tf.sqrt(tf.square(i) + tf.square(q))
                ang = tf.math.atan2(q, i) / tf.constant(np.pi, dtype=t.dtype)
                norm = tf.sqrt(tf.reduce_sum(tf.square(amp), axis=-1, keepdims=True)) + eps
                amp = amp / norm
                out = tf.stack([amp, ang], axis=-1)  # (B, 128, 2)
                return out
            x = tf.keras.layers.Lambda(_preproc, name='iq_to_amp_phase_layer')(inputs)
        else:
            # Fallback: assume already (time, features)
            x = inputs
    else:
        # No preprocessing: ensure shape (time, features) for LSTM
        if input_shape == (2, 128):
            x = tf.keras.layers.Permute((2, 1), name='to_time_major')(inputs)  # (B, 128, 2)
        else:
            x = inputs

    # Dropout rate
    dr = 0.0  # Set to 0 as in original

    # LSTM Unit 1: 32 units, return sequences and states
    # CuDNNLSTM equivalent: use activation='tanh', recurrent_activation='sigmoid'
    lstm_kwargs = {
        'units': 32,
        'return_state': True,
        'return_sequences': True
    }

    if use_gpu_lstm:
        # Standard LSTM with settings that match CuDNNLSTM behavior
        lstm_kwargs.update({
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'use_bias': True,
            'recurrent_dropout': 0.0
        })

    x, s, c = LSTM(**lstm_kwargs, name='encoder_1')(x)
    x = Dropout(dr, name='drop_1')(x)

    # LSTM Unit 2: 32 units, return sequences and states
    x, s1, c1 = LSTM(**lstm_kwargs, name='encoder_2')(x)

    # Classifier branch (uses final state s1)
    xc = Dense(32, activation='relu', name='clf_dense_1')(s1)
    xc = BatchNormalization(name='bn_1')(xc)
    xc = Dropout(dr, name='clf_drop_1')(xc)
    xc = Dense(16, activation='relu', name='clf_dense_2')(xc)
    xc = BatchNormalization(name='bn_2')(xc)
    xc = Dropout(dr, name='clf_drop_2')(xc)
    outputs = Dense(num_classes, activation='softmax', name='softmax')(xc)

    # Create model with single classification output
    model = Model(inputs=inputs, outputs=outputs, name='DAE_Classifier')

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    # Test model building
    print("Testing DAE model construction...")

    # Test with input shape (2, 128)
    print("\n1. Testing with input_shape=(2, 128)...")
    model = build_dae_model(input_shape=(2, 128), num_classes=11)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    # Test with sample input
    import numpy as np
    x = np.random.randn(4, 2, 128).astype(np.float32)
    print(f"\nInput shape: {x.shape}")
    outputs = model.predict(x, verbose=0)
    print(f"Reconstruction output shape: {outputs[0].shape}")
    print(f"Classification output shape: {outputs[1].shape}")

    # Test classifier-only version
    print("\n2. Testing classifier-only version...")
    classifier_model = build_dae_model_classifier_only(input_shape=(2, 128), num_classes=11)
    classifier_model.summary()
    y = classifier_model.predict(x, verbose=0)
    print(f"\nClassification output shape: {y.shape}")

    print("\n✓ All tests passed!")
