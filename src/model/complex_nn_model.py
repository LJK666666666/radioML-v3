import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization, Activation, Permute
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from keras.saving import register_keras_serializable


@register_keras_serializable(package="ComplexNN")
class ComplexConv1D(tf.keras.layers.Layer):
    """
    Complex 1D Convolution Layer that performs true complex convolution.
    This layer processes complex inputs (I/Q data) using complex arithmetic.
    """
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None, use_bias=True, **kwargs):
        super(ComplexConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        
    def build(self, input_shape):
        # input_shape: (batch, time_steps, channels) where channels = 2 * num_complex_channels
        # For the first layer: (batch, time_steps, 2) - real I/Q data
        # For subsequent layers: (batch, time_steps, 2*filters) - complex filter outputs
        input_dim = input_shape[-1] // 2  # Number of complex input channels
        
        # Complex kernel: W = W_real + j*W_imag
        self.W_real = self.add_weight(
            shape=(self.kernel_size, input_dim, self.filters),
            initializer='glorot_uniform',
            name='W_real'
        )
        self.W_imag = self.add_weight(
            shape=(self.kernel_size, input_dim, self.filters),
            initializer='glorot_uniform', 
            name='W_imag'
        )
        
        if self.use_bias:
            self.b_real = self.add_weight(
                shape=(self.filters,),
                initializer='zeros',
                name='b_real'
            )
            self.b_imag = self.add_weight(
                shape=(self.filters,),
                initializer='zeros',
                name='b_imag'
            )
        
        super(ComplexConv1D, self).build(input_shape)
        
    def call(self, inputs):
        # Split input into real and imaginary parts
        # input shape: (batch, time_steps, 2*num_complex_channels)
        input_dim = tf.shape(inputs)[-1] // 2
        input_real = inputs[:, :, :input_dim]  # (batch, time_steps, input_dim)
        input_imag = inputs[:, :, input_dim:]  # (batch, time_steps, input_dim)
        
        # Complex convolution: (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
        # where a,b are input real/imag and c,d are weight real/imag
        
        conv_rr = tf.nn.conv1d(input_real, self.W_real, stride=self.strides, padding=self.padding)
        conv_ri = tf.nn.conv1d(input_real, self.W_imag, stride=self.strides, padding=self.padding)
        conv_ir = tf.nn.conv1d(input_imag, self.W_real, stride=self.strides, padding=self.padding)
        conv_ii = tf.nn.conv1d(input_imag, self.W_imag, stride=self.strides, padding=self.padding)
        
        # Complex multiplication
        output_real = conv_rr - conv_ii  # ac - bd
        output_imag = conv_ri + conv_ir  # ad + bc
        
        if self.use_bias:
            output_real = tf.nn.bias_add(output_real, self.b_real)
            output_imag = tf.nn.bias_add(output_imag, self.b_imag)
            
        # Concatenate real and imaginary parts
        output = tf.concat([output_real, output_imag], axis=-1)
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            out_length = input_shape[1] - self.kernel_size + 1
        else:  # 'SAME'
            out_length = input_shape[1]
        return (input_shape[0], out_length, 2 * self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding.lower(),
            'activation': tf.keras.utils.serialize_keras_object(self.activation),
            'use_bias': self.use_bias
        })
        return config


@register_keras_serializable(package="ComplexNN")
class ComplexBatchNormalization(tf.keras.layers.Layer):
    """
    复数值数据的复数批归一化。
    
    复数批归一化将标准批归一化扩展到复数值神经网络。
    与独立归一化实值特征的标准批归一化不同，
    复数批归一化考虑实部和虚部之间的统计依赖关系。
    
    该方法遵循以下关键步骤：
    1. 计算复数均值（中心化）
    2. 计算实部和虚部的2x2协方差矩阵
    3. 使用矩阵平方根应用白化变换
    4. 使用复数参数应用可学习的缩放和平移
    
    数学公式：
    - 输入: z = x + iy（复数表示为连接的[x, y]）
    - 中心化: z_centered = z - μ_z
    - 协方差矩阵: V = [[Vrr, Vri], [Vri, Vii]]
    - 白化: z_whitened = V^(-1/2) * z_centered
    - 输出: z_out = Γ * z_whitened + β
    
    其中 Γ 是2x2复数缩放矩阵，β 是复数偏置。
    """
    def __init__(self, **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        
    def build(self, input_shape):
        dim = input_shape[-1] // 2
        
        # Parameters for complex batch normalization
        self.gamma_rr = self.add_weight(shape=(dim,), initializer='ones', name='gamma_rr')
        self.gamma_ri = self.add_weight(shape=(dim,), initializer='zeros', name='gamma_ri')
        self.gamma_ii = self.add_weight(shape=(dim,), initializer='ones', name='gamma_ii')
        
        self.beta_real = self.add_weight(shape=(dim,), initializer='zeros', name='beta_real')
        self.beta_imag = self.add_weight(shape=(dim,), initializer='zeros', name='beta_imag')
        
        super(ComplexBatchNormalization, self).build(input_shape)
        
    def call(self, inputs, training=None):
        input_dim = tf.shape(inputs)[-1] // 2
        
        # Split into real and imaginary parts
        input_real = inputs[..., :input_dim]
        input_imag = inputs[..., input_dim:]
        
        # Compute complex statistics
        mu_real = tf.reduce_mean(input_real, axis=[0, 1], keepdims=True)
        mu_imag = tf.reduce_mean(input_imag, axis=[0, 1], keepdims=True)
        
        # Center the inputs
        input_real_centered = input_real - mu_real
        input_imag_centered = input_imag - mu_imag
        
        # Compute covariance matrix elements
        Vrr = tf.reduce_mean(input_real_centered**2, axis=[0, 1], keepdims=True) + 1e-5
        Vii = tf.reduce_mean(input_imag_centered**2, axis=[0, 1], keepdims=True) + 1e-5
        Vri = tf.reduce_mean(input_real_centered * input_imag_centered, axis=[0, 1], keepdims=True)
        
        # Compute normalization factors
        det = Vrr * Vii - Vri**2
        s = tf.sqrt(det)
        t = tf.sqrt(Vii + Vrr + 2 * s)
        
        inverse_st = 1.0 / (s * t)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st
        
        # Apply whitening transformation
        normalized_real = Wrr * input_real_centered + Wri * input_imag_centered
        normalized_imag = Wri * input_real_centered + Wii * input_imag_centered
        
        # Apply scaling and shifting
        output_real = self.gamma_rr * normalized_real - self.gamma_ri * normalized_imag + self.beta_real
        output_imag = self.gamma_ri * normalized_real + self.gamma_ii * normalized_imag + self.beta_imag
        
        return tf.concat([output_real, output_imag], axis=-1)

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="ComplexNN")
def complex_relu(x):
    """
    Complex ReLU activation function.
    Applies ReLU to both real and imaginary parts.
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    real_activated = tf.nn.relu(real_part)
    imag_activated = tf.nn.relu(imag_part)
    
    return tf.concat([real_activated, imag_activated], axis=-1)


@register_keras_serializable(package="ComplexNN")
def mod_relu(x, bias=0.5):
    """
    modReLU activation function for complex numbers.
    
    modReLU(z) = ReLU(|z| + bias) * (z / |z|)
    
    This activation preserves the phase information while applying
    ReLU to the magnitude with a learnable bias term.
    
    Args:
        x: Complex input tensor (concatenated real and imaginary parts)
        bias: Bias term for the magnitude (default: 0.5)
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    # Compute magnitude
    magnitude = tf.sqrt(real_part**2 + imag_part**2 + 1e-8)
    
    # Apply ReLU to (magnitude + bias)
    activated_magnitude = tf.nn.relu(magnitude + bias)
    
    # Normalize to get unit vector (preserve phase)
    normalized_real = real_part / (magnitude + 1e-8)
    normalized_imag = imag_part / (magnitude + 1e-8)
    
    # Scale by activated magnitude
    output_real = activated_magnitude * normalized_real
    output_imag = activated_magnitude * normalized_imag
    
    return tf.concat([output_real, output_imag], axis=-1)


@register_keras_serializable(package="ComplexNN")
def zrelu(x):
    """
    zReLU activation function for complex numbers.
    
    zReLU(z) = z if Re(z) >= 0 and Im(z) >= 0, else 0
    
    This activation only allows complex numbers in the first quadrant.
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    # Create mask for first quadrant (both real and imag >= 0)
    mask = tf.cast(tf.logical_and(real_part >= 0, imag_part >= 0), tf.float32)
    
    output_real = real_part * mask
    output_imag = imag_part * mask
    
    return tf.concat([output_real, output_imag], axis=-1)


@register_keras_serializable(package="ComplexNN")
def crelu(x):
    """
    CReLU (Complex ReLU) activation function.
    
    CReLU(z) = ReLU(Re(z)) + i * ReLU(Im(z))
    
    This is similar to the original complex_relu but with a clear name.
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    real_activated = tf.nn.relu(real_part)
    imag_activated = tf.nn.relu(imag_part)
    
    return tf.concat([real_activated, imag_activated], axis=-1)


@register_keras_serializable(package="ComplexNN")
def cardioid(x):
    """
    Cardioid activation function for complex numbers.
    
    Cardioid(z) = 0.5 * (1 + cos(arg(z))) * z
    
    This activation function has a heart-shaped acceptance region
    and preserves both magnitude and phase information.
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    # Compute phase (argument) of complex number
    phase = tf.atan2(imag_part, real_part)
    
    # Cardioid function: 0.5 * (1 + cos(phase))
    cardioid_factor = 0.5 * (1.0 + tf.cos(phase))
    
    output_real = cardioid_factor * real_part
    output_imag = cardioid_factor * imag_part
    
    return tf.concat([output_real, output_imag], axis=-1)


@register_keras_serializable(package="ComplexNN") 
def complex_tanh(x):
    """
    Complex tanh activation function.
    
    tanh(a + bi) = (tanh(a) + i*tan(b)) / (1 + i*tanh(a)*tan(b))
    
    Simplified version: applies tanh to both real and imaginary parts.
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    real_activated = tf.nn.tanh(real_part)
    imag_activated = tf.nn.tanh(imag_part)
    
    return tf.concat([real_activated, imag_activated], axis=-1)


@register_keras_serializable(package="ComplexNN")
def phase_amplitude_activation(x, phase_activation='linear', amplitude_activation='relu'):
    """
    Phase-Amplitude based activation function.
    
    Separates complex number into phase and amplitude, applies different
    activations to each, then reconstructs the complex number.
    
    Args:
        x: Complex input tensor
        phase_activation: Activation for phase ('linear', 'tanh', 'sigmoid')
        amplitude_activation: Activation for amplitude ('relu', 'elu', 'linear')
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    # Convert to polar form
    amplitude = tf.sqrt(real_part**2 + imag_part**2 + 1e-8)
    phase = tf.atan2(imag_part, real_part)
    
    # Apply activations
    if amplitude_activation == 'relu':
        activated_amplitude = tf.nn.relu(amplitude)
    elif amplitude_activation == 'elu':
        activated_amplitude = tf.nn.elu(amplitude)
    else:  # linear
        activated_amplitude = amplitude
        
    if phase_activation == 'tanh':
        activated_phase = tf.nn.tanh(phase)
    elif phase_activation == 'sigmoid':
        activated_phase = tf.nn.sigmoid(phase) * 2 * np.pi - np.pi  # Scale to [-π, π]
    else:  # linear
        activated_phase = phase
    
    # Convert back to cartesian form
    output_real = activated_amplitude * tf.cos(activated_phase)
    output_imag = activated_amplitude * tf.sin(activated_phase)
    
    return tf.concat([output_real, output_imag], axis=-1)


@register_keras_serializable(package="ComplexNN")
def complex_elu(x, alpha=1.0):
    """
    Complex ELU activation function (original style).
    Applies ELU to both real and imaginary parts separately.
    
    Args:
        x: Complex input tensor
        alpha: ELU parameter
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    real_activated = tf.nn.elu(real_part, alpha)
    imag_activated = tf.nn.elu(imag_part, alpha)
    
    return tf.concat([real_activated, imag_activated], axis=-1)


@register_keras_serializable(package="ComplexNN")
def complex_leaky_relu(x, alpha=0.2):
    """
    Complex Leaky ReLU activation function (original style).
    Applies Leaky ReLU to both real and imaginary parts separately.
    
    Args:
        x: Complex input tensor
        alpha: Negative slope coefficient
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    real_activated = tf.nn.leaky_relu(real_part, alpha)
    imag_activated = tf.nn.leaky_relu(imag_part, alpha)
    
    return tf.concat([real_activated, imag_activated], axis=-1)


@register_keras_serializable(package="ComplexNN")
def complex_swish(x):
    """
    Complex Swish activation function (original style).
    Applies Swish (x * sigmoid(x)) to both real and imaginary parts separately.
    
    Args:
        x: Complex input tensor
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    real_activated = real_part * tf.nn.sigmoid(real_part)
    imag_activated = imag_part * tf.nn.sigmoid(imag_part)
    
    return tf.concat([real_activated, imag_activated], axis=-1)


@register_keras_serializable(package="ComplexNN")
def real_imag_mixed_relu(x):
    """
    Mixed ReLU activation function (original style variant).
    Applies ReLU to real part and keeps imaginary part linear.
    This is one of the original approaches for complex activation.
    
    Args:
        x: Complex input tensor
    """
    input_dim = tf.shape(x)[-1] // 2
    real_part = x[..., :input_dim]
    imag_part = x[..., input_dim:]
    
    real_activated = tf.nn.relu(real_part)
    imag_activated = imag_part  # Keep imaginary part linear
    
    return tf.concat([real_activated, imag_activated], axis=-1)


@register_keras_serializable(package="ComplexNN")
class ComplexPooling1D(tf.keras.layers.Layer):
    """
    Complex-aware pooling layer that pools while maintaining complex structure.
    This replaces Lambda layers for pooling operations.
    """
    def __init__(self, pool_size=2, **kwargs):
        super(ComplexPooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size
    
    def call(self, inputs):
        # Simple strided sampling - every pool_size-th element
        return inputs[:, ::self.pool_size, :]
    
    def compute_output_shape(self, input_shape):
        new_length = input_shape[1] // self.pool_size
        return (input_shape[0], new_length, input_shape[2])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size
        })
        return config


@register_keras_serializable(package="ComplexNN")
class ComplexActivation(tf.keras.layers.Layer):
    """
    A proper layer for complex activation functions.
    This replaces Lambda layers to avoid serialization issues.
    """
    def __init__(self, activation_type='mod_relu', **kwargs):
        super(ComplexActivation, self).__init__(**kwargs)
        self.activation_type = activation_type
        
        # Map activation types to functions
        self.activation_map = {
            'complex_relu': complex_relu,
            'mod_relu': mod_relu,
            'zrelu': zrelu,
            'crelu': crelu,
            'cardioid': cardioid,
            'complex_tanh': complex_tanh,
            'phase_amplitude': lambda x: phase_amplitude_activation(x, 'linear', 'relu'),
            # Original style activations (separate real/imag processing)
            'complex_elu': complex_elu,
            'complex_leaky_relu': complex_leaky_relu,
            'complex_swish': complex_swish,
            'real_imag_mixed_relu': real_imag_mixed_relu
        }
        
        if activation_type not in self.activation_map:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
        self.activation_fn = self.activation_map[activation_type]
    
    def call(self, inputs):
        return self.activation_fn(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'activation_type': self.activation_type
        })
        return config


@register_keras_serializable(package="ComplexNN")
class ComplexMagnitude(tf.keras.layers.Layer):
    """
    Layer to extract magnitude from complex numbers.
    Converts complex representation to magnitude for classification.
    """
    def __init__(self, **kwargs):
        super(ComplexMagnitude, self).__init__(**kwargs)
        
    def call(self, inputs):
        input_dim = tf.shape(inputs)[-1] // 2
        real_part = inputs[..., :input_dim]
        imag_part = inputs[..., input_dim:]
        
        magnitude = tf.sqrt(real_part**2 + imag_part**2 + 1e-8)
        return magnitude
    
    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] // 2)

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="ComplexNN")
class ComplexDense(tf.keras.layers.Layer):
    """
    Complex Dense (Fully Connected) Layer.
    """
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        
    def build(self, input_shape):
        input_dim = input_shape[-1] // 2
        
        self.W_real = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            name='W_real'
        )
        self.W_imag = self.add_weight(
            shape=(input_dim, self.units), 
            initializer='glorot_uniform',
            name='W_imag'
        )
        
        if self.use_bias:
            self.b_real = self.add_weight(shape=(self.units,), initializer='zeros', name='b_real')
            self.b_imag = self.add_weight(shape=(self.units,), initializer='zeros', name='b_imag')
            
        super(ComplexDense, self).build(input_shape)
        
    def call(self, inputs):
        input_dim = tf.shape(inputs)[-1] // 2
        input_real = inputs[..., :input_dim]
        input_imag = inputs[..., input_dim:]
        
        # Complex matrix multiplication
        output_real = tf.matmul(input_real, self.W_real) - tf.matmul(input_imag, self.W_imag)
        output_imag = tf.matmul(input_real, self.W_imag) + tf.matmul(input_imag, self.W_real)
        
        if self.use_bias:
            output_real = tf.nn.bias_add(output_real, self.b_real)
            output_imag = tf.nn.bias_add(output_imag, self.b_imag)
            
        output = tf.concat([output_real, output_imag], axis=-1)
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.utils.serialize_keras_object(self.activation),
            'use_bias': self.use_bias
        })
        return config


# def build_complex_nn_model(input_shape, num_classes, activation_type='mod_relu'):
# def build_complex_nn_model(input_shape, num_classes, activation_type='complex_leaky_relu'):
def build_complex_nn_model(input_shape, num_classes, activation_type='complex_relu'):
    """
    Build a TRUE Complex Neural Network model for radio signal classification.
    This model performs actual complex arithmetic operations on I/Q data.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length) = (2, 128)
        num_classes: Number of classes to classify
        activation_type: Type of complex activation function to use
                        Options: 'complex_relu', 'mod_relu', 'zrelu', 'crelu', 
                                'cardioid', 'complex_tanh', 'phase_amplitude',
                                'complex_elu', 'complex_leaky_relu', 'complex_swish',
                                'real_imag_mixed_relu' (original style activations)
        
    Returns:
        A compiled Keras model that uses complex arithmetic
    """
    
    # Select activation function
    activation_map = {
        'complex_relu': complex_relu,
        'mod_relu': mod_relu,
        'zrelu': zrelu,
        'crelu': crelu,
        'cardioid': cardioid,
        'complex_tanh': complex_tanh,
        'phase_amplitude': lambda x: phase_amplitude_activation(x, 'linear', 'relu'),
        # Original style activations (separate real/imag processing)
        'complex_elu': complex_elu,
        'complex_leaky_relu': complex_leaky_relu,
        'complex_swish': complex_swish,
        'real_imag_mixed_relu': real_imag_mixed_relu
    }
    
    if activation_type not in activation_map:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    
    complex_activation = activation_map[activation_type]
    
    inputs = Input(shape=input_shape)
    
    # Reshape input from (2, 128) to (128, 2) for complex processing
    # This treats the data as 128 time steps with complex values (real, imag)
    x = Permute((2, 1))(inputs)  # (128, 2)
    
    # First Complex Convolutional Block
    x = ComplexConv1D(filters=64, kernel_size=5, padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = ComplexActivation(activation_type)(x)
    x = ComplexPooling1D(pool_size=2)(x)  # Complex-aware pooling
    
    # Second Complex Convolutional Block  
    x = ComplexConv1D(filters=128, kernel_size=5, padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = ComplexActivation(activation_type)(x)
    x = ComplexPooling1D(pool_size=2)(x)  # Complex-aware pooling
    
    # Third Complex Convolutional Block
    x = ComplexConv1D(filters=256, kernel_size=3, padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = ComplexActivation(activation_type)(x)
    
    # Global Average Pooling for complex data
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Complex Dense Layers
    x = ComplexDense(units=512)(x)
    x = ComplexActivation(activation_type)(x)
    x = Dropout(0.5)(x)
    
    x = ComplexDense(units=256)(x)
    x = ComplexActivation(activation_type)(x)
    x = Dropout(0.3)(x)
    
    # For classification, we need to convert complex output to real
    # Take the magnitude of complex features
    x_magnitude = ComplexMagnitude()(x)
    
    # Final classification layer (real-valued)
    outputs = Dense(num_classes, activation='softmax')(x_magnitude)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


# def build_simple_complex_nn_model(input_shape, num_classes, activation_type='mod_relu'):
def build_simple_complex_nn_model(input_shape, num_classes, activation_type='complex_leaky_relu'):
    """
    A simpler version of complex neural network for comparison.
    Uses basic complex operations without all the advanced features.
    
    Args:
        input_shape: Input shape of the data
        num_classes: Number of classes to classify
        activation_type: Type of complex activation function to use
    """
    # Select activation function
    activation_map = {
        'complex_relu': complex_relu,
        'mod_relu': mod_relu,
        'zrelu': zrelu,
        'crelu': crelu,
        'cardioid': cardioid,
        'complex_tanh': complex_tanh,
        'phase_amplitude': lambda x: phase_amplitude_activation(x, 'linear', 'relu'),
        # Original style activations (separate real/imag processing)
        'complex_elu': complex_elu,
        'complex_leaky_relu': complex_leaky_relu,
        'complex_swish': complex_swish,
        'real_imag_mixed_relu': real_imag_mixed_relu
    }
    
    if activation_type not in activation_map:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    
    complex_activation = activation_map[activation_type]
    
    inputs = Input(shape=input_shape)
    x = Permute((2, 1))(inputs)  # (128, 2)
    
    # Basic complex convolution blocks
    x = ComplexConv1D(filters=128, kernel_size=5, padding='same')(x)
    x = ComplexActivation(activation_type)(x)
    x = ComplexPooling1D(pool_size=2)(x)
    
    x = ComplexConv1D(filters=256, kernel_size=5, padding='same')(x)
    x = ComplexActivation(activation_type)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Extract magnitude for classification
    x_magnitude = ComplexMagnitude()(x)
    
    outputs = Dense(num_classes, activation='softmax')(x_magnitude)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
