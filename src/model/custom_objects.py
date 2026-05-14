"""
Custom objects registry for model loading/deserialization.

Centralizes all custom layer and function registrations needed to load saved models.
This avoids duplicating ~140 lines of custom objects code across multiple entry points.
"""

from model.complex_nn_model import (
    ComplexConv1D, ComplexBatchNormalization, ComplexDense, ComplexMagnitude,
    ComplexActivation, ComplexPooling1D,
    complex_relu, mod_relu, zrelu, crelu, cardioid, complex_tanh, phase_amplitude_activation,
    complex_elu, complex_leaky_relu, complex_swish, real_imag_mixed_relu
)
from model.hybrid_complex_resnet_model import (
    ComplexResidualBlock, ComplexResidualBlockAdvanced, ComplexGlobalAveragePooling1D
)

# Import ULCNN complex layers for model loading
try:
    from model.complexnn import (
        ComplexConv1D as ULCNNComplexConv1D,
        ComplexBatchNormalization as ULCNNComplexBatchNormalization,
        ComplexDense as ULCNNComplexDense,
        ChannelShuffle, DWConvMobile, ChannelAttention,
        TransposeLayer, ExtractChannelLayer, TrigonometricLayer, sqrt_init
    )
    ULCNN_LAYERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ULCNN complex layers not available: {e}")
    ULCNN_LAYERS_AVAILABLE = False


def get_all_custom_objects():
    """Get dictionary of ALL custom objects for generic model loading.

    Use this when the model type is unknown and you need to try loading
    with all possible custom objects (e.g., model_benchmark.py).
    """
    custom_objects = _get_complex_layer_objects()

    # Add AMC-Net objects
    try:
        from model.amcnet_model import (
            Conv_Block, MultiScaleModule, TinyMLP, AdaCorrModule, FeaFusionModule,
            L2NormalizeWidthLayer, SqueezeHeightLayer, TransposeChannelWidthLayer, GlobalAvgPoolWidthLayer
        )
        custom_objects.update({
            'Conv_Block': Conv_Block,
            'MultiScaleModule': MultiScaleModule,
            'TinyMLP': TinyMLP,
            'AdaCorrModule': AdaCorrModule,
            'FeaFusionModule': FeaFusionModule,
            'L2NormalizeWidthLayer': L2NormalizeWidthLayer,
            'SqueezeHeightLayer': SqueezeHeightLayer,
            'TransposeChannelWidthLayer': TransposeChannelWidthLayer,
            'GlobalAvgPoolWidthLayer': GlobalAvgPoolWidthLayer,
        })
    except ImportError:
        pass

    # Add benchmark model reshape functions
    custom_objects.update(_get_benchmark_reshape_objects())

    return custom_objects


def get_custom_objects_for_model(model_name):
    """Get custom objects dict for specific model types that need them.

    Args:
        model_name: Name of the model (e.g., 'complex_nn', 'amcnet', 'cldnn')

    Returns:
        dict or None: Custom objects dict, or None if the model uses only standard layers.
    """
    complex_models = ['complex_nn', 'hybrid_complex_resnet', 'lightweight_hybrid',
                      'ultra_lightweight_hybrid', 'micro_lightweight_hybrid']
    ulcnn_complex_models = ['ulcnn', 'mcldnn', 'pet']
    amcnet_models = ['amcnet']
    benchmark_models = ['cldnn', 'cgdnn']

    if model_name in complex_models or model_name in ulcnn_complex_models:
        custom_objects = _get_complex_layer_objects()

        if model_name in ulcnn_complex_models:
            custom_objects.update(_get_pet_lambda_objects())

        return custom_objects

    elif model_name in amcnet_models:
        from model.amcnet_model import (
            Conv_Block, MultiScaleModule, TinyMLP, AdaCorrModule, FeaFusionModule,
            L2NormalizeWidthLayer, SqueezeHeightLayer, TransposeChannelWidthLayer, GlobalAvgPoolWidthLayer
        )
        return {
            'Conv_Block': Conv_Block,
            'MultiScaleModule': MultiScaleModule,
            'TinyMLP': TinyMLP,
            'AdaCorrModule': AdaCorrModule,
            'FeaFusionModule': FeaFusionModule,
            'L2NormalizeWidthLayer': L2NormalizeWidthLayer,
            'SqueezeHeightLayer': SqueezeHeightLayer,
            'TransposeChannelWidthLayer': TransposeChannelWidthLayer,
            'GlobalAvgPoolWidthLayer': GlobalAvgPoolWidthLayer,
        }

    elif model_name in benchmark_models:
        return _get_benchmark_reshape_objects()

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_complex_layer_objects():
    """Return custom objects dict for complex-valued layers."""
    custom_objects = {
        'ComplexConv1D': ComplexConv1D,
        'ComplexBatchNormalization': ComplexBatchNormalization,
        'ComplexDense': ComplexDense,
        'ComplexMagnitude': ComplexMagnitude,
        'ComplexActivation': ComplexActivation,
        'ComplexPooling1D': ComplexPooling1D,
        'ComplexResidualBlock': ComplexResidualBlock,
        'ComplexResidualBlockAdvanced': ComplexResidualBlockAdvanced,
        'ComplexGlobalAveragePooling1D': ComplexGlobalAveragePooling1D,
        'complex_relu': complex_relu,
        'mod_relu': mod_relu,
        'zrelu': zrelu,
        'crelu': crelu,
        'cardioid': cardioid,
        'complex_tanh': complex_tanh,
        'phase_amplitude_activation': phase_amplitude_activation,
        'complex_elu': complex_elu,
        'complex_leaky_relu': complex_leaky_relu,
        'complex_swish': complex_swish,
        'real_imag_mixed_relu': real_imag_mixed_relu
    }

    if ULCNN_LAYERS_AVAILABLE:
        custom_objects.update({
            'ULCNNComplexConv1D': ULCNNComplexConv1D,
            'ULCNNComplexBatchNormalization': ULCNNComplexBatchNormalization,
            'ULCNNComplexDense': ULCNNComplexDense,
            'ChannelShuffle': ChannelShuffle,
            'DWConvMobile': DWConvMobile,
            'ChannelAttention': ChannelAttention,
            'TransposeLayer': TransposeLayer,
            'ExtractChannelLayer': ExtractChannelLayer,
            'TrigonometricLayer': TrigonometricLayer,
            'sqrt_init': sqrt_init
        })

    return custom_objects


def _get_pet_lambda_objects():
    """Return custom objects dict for PET model lambda functions."""
    from model.pet_model import (
        cos_function,
        sin_function,
        transpose_function,
        extract_i_2016,
        extract_q_2016,
        extract_i_2018,
        extract_q_2018,
        legacy_cos_lambda,
        legacy_sin_lambda,
        legacy_transpose_lambda,
        legacy_extract_i_lambda,
        legacy_extract_q_lambda,
    )
    return {
        'cos_function': cos_function,
        'sin_function': sin_function,
        'transpose_function': transpose_function,
        'extract_i_2016': extract_i_2016,
        'extract_q_2016': extract_q_2016,
        'extract_i_2018': extract_i_2018,
        'extract_q_2018': extract_q_2018,
        'legacy_cos_lambda': legacy_cos_lambda,
        'legacy_sin_lambda': legacy_sin_lambda,
        'legacy_transpose_lambda': legacy_transpose_lambda,
        'legacy_extract_i_lambda': legacy_extract_i_lambda,
        'legacy_extract_q_lambda': legacy_extract_q_lambda,
    }


def _get_benchmark_reshape_objects():
    """Return custom objects dict for benchmark model Lambda reshape functions."""
    def dynamic_reshape(x):
        """Dynamic reshape function for CGDNN"""
        import tensorflow as tf
        shape = tf.shape(x)
        batch_size = shape[0]
        channels = shape[1]
        height = shape[2]
        width = shape[3]
        return tf.reshape(x, [batch_size, channels, height * width])

    def dynamic_reshape_cldnn(x):
        """Dynamic reshape function for CLDNN"""
        import tensorflow as tf
        shape = tf.shape(x)
        batch_size = shape[0]
        channels = shape[1]
        height = shape[2]
        width = shape[3]
        return tf.reshape(x, [batch_size, channels, height * width])

    return {
        'dynamic_reshape': dynamic_reshape,
        'dynamic_reshape_cldnn': dynamic_reshape_cldnn,
    }
