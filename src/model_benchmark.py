#!/usr/bin/env python3
"""
Model Performance Benchmark Tool

This script evaluates model performance metrics including:
- Parameters count (trainable and non-trainable)
- FLOPs (Floating Point Operations)
- Runtime/Inference speed
- Memory usage

Usage examples:
python model_benchmark.py --model_path ../output/models/resnet_model.keras
python model_benchmark.py --model_path ../output/models/complex_nn_model_gpr_augment.keras --batch_size 64

To suppress TensorFlow warnings (recommended):
python model_benchmark.py --model_path <model_path> 2>/dev/null
"""

import os
import sys
import argparse
import time
import pickle
import psutil
import numpy as np
import contextlib
import subprocess

# Suppress all warnings and logs before TensorFlow import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Additional environment variables to suppress CUDA warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
import warnings
warnings.filterwarnings('ignore')

# More aggressive stderr suppression
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass
    def close(self):
        pass

# Redirect stderr to devnull temporarily
original_stderr = sys.stderr

# Context manager to completely suppress stderr
@contextlib.contextmanager
def suppress_stderr():
    try:
        sys.stderr = DevNull()
        yield
    finally:
        sys.stderr = original_stderr

# Import TensorFlow with complete stderr suppression
with suppress_stderr():
    import tensorflow as tf
    # Configure TensorFlow immediately after import
    tf.config.set_soft_device_placement(True)
    try:
        # Disable GPU memory growth logs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus[:1]:  # Only use first GPU
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
# Suppress all other common loggers without disabling them completely
for logger_name in ['tensorflow', 'absl']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)

import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom layers for model loading
from model.complex_nn_model import (
    ComplexConv1D, ComplexBatchNormalization, ComplexDense, ComplexMagnitude, 
    ComplexActivation, ComplexPooling1D,
    complex_relu, mod_relu, zrelu, crelu, cardioid, complex_tanh, phase_amplitude_activation,
    complex_elu, complex_leaky_relu, complex_swish, real_imag_mixed_relu
)
from model.hybrid_complex_resnet_model import (
    ComplexResidualBlock, ComplexResidualBlockAdvanced, ComplexGlobalAveragePooling1D
)

# Import ULCNN complex layers if available
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


def get_custom_objects_dict():
    """Get dictionary of all custom objects for model loading"""
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
    
    # Add ULCNN-specific layers if available
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


def load_model_safely(model_path):
    """
    Safely load a model with custom objects support
    
    Args:
        model_path: Path to the model file (.keras or .pkl)
        
    Returns:
        model: Loaded model object
        model_type: 'keras'
    """
    if model_path.endswith('.keras') or model_path.endswith('.h5'):
        # Load Keras model
        try:
            # First try without custom objects
            model = tf.keras.models.load_model(model_path)
            return model, 'keras'
        except Exception as e1:
            print(f"Standard loading failed: {e1}")
            try:
                # Try with custom objects
                tf.keras.config.enable_unsafe_deserialization()
                custom_objects = get_custom_objects_dict()
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                return model, 'keras'
            except Exception as e2:
                print(f"Custom objects loading failed: {e2}")
                return None, None
    
    else:
        print(f"Unsupported model format: {model_path}")
        return None, None


def count_parameters(model, model_type):
    """
    Count model parameters
    
    Args:
        model: Model object
        model_type: 'keras'
        
    Returns:
        dict: Parameter counts
    """
    if model_type == 'keras':
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params)
        }
    
    elif model_type == 'sklearn':
        # For sklearn models, count parameters in each estimator
        total_params = 0
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if hasattr(estimator, 'coef_'):
                    total_params += int(np.prod(estimator.coef_.shape))
                if hasattr(estimator, 'intercept_'):
                    total_params += int(np.prod(estimator.intercept_.shape))
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(total_params),  # All sklearn params are "trainable"
            'non_trainable_parameters': 0
        }
    
    return {'total_parameters': 0, 'trainable_parameters': 0, 'non_trainable_parameters': 0}


def calculate_flops(model, input_shape, model_type):
    """
    Calculate FLOPs (Floating Point Operations)
    
    Args:
        model: Model object
        input_shape: Input tensor shape (batch_size, ...)
        model_type: 'keras'
        
    Returns:
        int: Estimated FLOPs count
    """
    if model_type == 'keras':
        try:
            # Safer approach: estimate FLOPs based on layer types and parameters
            # without using the problematic tf.profiler
            
            total_flops = 0
            
            # Create a sample input to get layer output shapes
            sample_input = tf.random.normal(input_shape)
            
            # Get intermediate outputs for shape calculation
            x = sample_input
            
            for layer in model.layers:
                try:
                    x = layer(x)
                    # Estimate FLOPs based on layer type
                    if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv1D, tf.keras.layers.Conv2D)):
                        if hasattr(layer, 'kernel'):
                            kernel_params = tf.keras.backend.count_params(layer.kernel)
                            # Assume MACs ~ output elements * kernel spatial size
                            output_elems = int(tf.reduce_prod(x.shape[1:]).numpy()) if None not in x.shape[1:] else 0
                            kernel_spatial = 1
                            if hasattr(layer, 'kernel_size'):
                                if isinstance(layer.kernel_size, tuple):
                                    ks = layer.kernel_size
                                else:
                                    ks = (layer.kernel_size,)
                                for v in ks:
                                    kernel_spatial *= int(v)
                            # Rough estimate: output_elems * kernel_spatial * in_channels
                            in_ch = int(layer.kernel.shape[-2]) if len(layer.kernel.shape) >= 2 else 1
                            layer_flops = output_elems * kernel_spatial * in_ch
                            total_flops += int(layer_flops)
                    elif hasattr(layer, 'weights') and len(layer.weights) > 0:
                        # For other layers with weights, estimate 2 FLOPs per parameter
                        layer_params = sum([tf.keras.backend.count_params(w) for w in layer.weights])
                        total_flops += int(layer_params) * 2
                        
                except Exception:
                    # Skip problematic layers
                    continue
            
            # Fallback if estimate failed
            if total_flops == 0:
                total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
                total_flops = int(total_params) * 2
                
            return int(total_flops)
            
        except Exception as e:
            print(f"FLOPs calculation failed, using fallback estimation: {e}")
            total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
            return int(total_params) * 2
    
    elif model_type == 'sklearn':
        # For sklearn models, estimate based on model complexity
        if hasattr(model, 'estimators_'):
            # AdaBoost-like ensemble
            n_estimators = len(model.estimators_)
            # Assume each estimator does ~100 operations per prediction
            estimated_flops = n_estimators * 100 * input_shape[0]  # batch_size factor
            return int(estimated_flops)
        else:
            # Simple model
            return int(1000 * input_shape[0])  # Basic estimation
    
    return 0


def measure_inference_time(model, test_data, model_type, batch_size, num_runs=1):
    """
    Measure model inference time over the whole dataset in batches.

    Args:
        model: Model object
        test_data: Full input dataset for inference (shape: [num_samples, ...])
        model_type: 'keras'
        batch_size: Batch size used for batched inference
        num_runs: Number of full passes over the dataset for averaging

    Returns:
        dict: Timing statistics
    """
    num_samples = int(test_data.shape[0])
    num_batches = int(math.ceil(num_samples / batch_size)) if batch_size > 0 else 0

    # Warm-up on the first batch
    if num_samples > 0:
        first_batch = test_data[:min(batch_size, num_samples)]
        for _ in range(5):
            with suppress_stderr():
                if model_type == 'keras':
                    _ = model.predict(first_batch, verbose=0)
                elif model_type == 'sklearn':
                    _ = model.predict(first_batch.reshape(first_batch.shape[0], -1))

    # Actual timing
    batch_times = []
    total_samples_processed = 0

    for _ in range(num_runs):
        for i in range(num_batches):
            batch = test_data[i * batch_size: (i + 1) * batch_size]
            if batch.shape[0] == 0:
                continue

            start_time = time.time()
            # Suppress stderr during model inference to avoid GPU timer warnings
            with suppress_stderr():
                if model_type == 'keras':
                    _ = model.predict(batch, verbose=0)
                elif model_type == 'sklearn':
                    _ = model.predict(batch.reshape(batch.shape[0], -1))
            end_time = time.time()

            batch_times.append(end_time - start_time)
            total_samples_processed += int(batch.shape[0])

    times = np.array(batch_times, dtype=np.float64)
    total_time = float(np.sum(times)) if times.size else 0.0

    return {
        'mean_time': float(np.mean(times)) if times.size else 0.0,          # mean per-batch time (seconds)
        'std_time': float(np.std(times)) if times.size else 0.0,
        'min_time': float(np.min(times)) if times.size else 0.0,
        'max_time': float(np.max(times)) if times.size else 0.0,
        'median_time': float(np.median(times)) if times.size else 0.0,
        'throughput_samples_per_second': (total_samples_processed / total_time) if total_time > 0 else 0.0,
        'num_runs': int(num_runs),
        'num_batches_per_run': int(num_batches),
        'total_batches': int(num_batches * num_runs),
        'batch_size': int(batch_size),
        'total_samples_processed': int(total_samples_processed),
        'total_time_seconds': float(total_time)
    }


def measure_memory_usage(model, test_data, model_type):
    """
    Measure memory usage during inference
    
    Args:
        model: Model object
        test_data: Input data for inference
        model_type: 'keras'
        
    Returns:
        dict: Memory usage statistics
    """
    process = psutil.Process()
    
    # Measure baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Perform inference and measure peak memory
    with suppress_stderr():
        if model_type == 'keras':
            _ = model.predict(test_data, verbose=0)
        elif model_type == 'sklearn':
            _ = model.predict(test_data.reshape(test_data.shape[0], -1))
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'baseline_memory_mb': float(baseline_memory),
        'peak_memory_mb': float(peak_memory),
        'memory_increase_mb': float(peak_memory - baseline_memory)
    }


def load_denoised_dataset(dataset_path, num_samples=None):
    """
    Load denoised dataset from pickle file
    
    Args:
        dataset_path: Path to the denoised dataset pickle file
        num_samples: Number of samples to use (None for all samples)
        
    Returns:
        tuple: (X_data, y_data, snr_data) or None if loading fails
    """
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        X_all = data['X_all']
        y_all = data['y_all'] 
        snr_all = data['snr_values_all']
        
        if num_samples is not None and num_samples < len(X_all):
            # Use first num_samples samples
            X_all = X_all[:num_samples]
            y_all = y_all[:num_samples]
            snr_all = snr_all[:num_samples]
        
        print(f"Loaded dataset: {X_all.shape[0]} samples, shape {X_all.shape[1:]}")
        print(f"Data type: {X_all.dtype}, Labels: {len(np.unique(y_all))} classes")
        
        return X_all, y_all, snr_all
        
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        return None, None, None


def generate_test_data(input_shape, num_samples):
    """
    Generate random test data (fallback when no dataset provided)
    
    Args:
        input_shape: Shape of input data (excluding batch dimension)
        num_samples: Number of samples to generate
        
    Returns:
        np.ndarray: Random test data
    """
    full_shape = (num_samples,) + input_shape
    return np.random.normal(0, 1, full_shape).astype(np.float32)


def print_benchmark_results(model_path, model_type, params, flops, timing, memory, input_shape):
    """Print formatted benchmark results"""
    
    print("=" * 80)
    print("MODEL PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model Path: {model_path}")
    print(f"Model Type: {model_type}")
    print(f"Input Shape: {input_shape}")
    print()
    
    # Parameters
    print("PARAMETERS:")
    print(f"  Total Parameters:        {params['total_parameters']:,}")
    print(f"  Trainable Parameters:    {params['trainable_parameters']:,}")
    print(f"  Non-trainable Parameters: {params['non_trainable_parameters']:,}")
    print()
    
    # FLOPs
    print("COMPUTATIONAL COMPLEXITY:")
    print(f"  Estimated FLOPs:         {flops:,}")
    print(f"  FLOPs per Parameter:     {flops / max(params['total_parameters'], 1):.2f}")
    print()
    
    # Timing
    print("INFERENCE PERFORMANCE:")
    print(f"  Mean Inference Time:     {timing['mean_time']*1000:.2f} ms  (per batch)")
    print(f"  Std Inference Time:      {timing['std_time']*1000:.2f} ms")
    print(f"  Min Inference Time:      {timing['min_time']*1000:.2f} ms")
    print(f"  Max Inference Time:      {timing['max_time']*1000:.2f} ms")
    print(f"  Median Inference Time:   {timing['median_time']*1000:.2f} ms")
    print(f"  Throughput:              {timing['throughput_samples_per_second']:.2f} samples/sec")
    if 'total_samples_processed' in timing:
        print(f"  Total Samples Measured:  {timing['total_samples_processed']}")
        print(f"  Total Batches:           {timing['total_batches']} (batch_size={timing['batch_size']})")
        print(f"  Total Time:              {timing['total_time_seconds']:.3f} s over {timing['num_runs']} run(s)")
    print()
    
    # Memory
    print("MEMORY USAGE:")
    print(f"  Baseline Memory:         {memory['baseline_memory_mb']:.2f} MB")
    print(f"  Peak Memory:             {memory['peak_memory_mb']:.2f} MB")
    print(f"  Memory Increase:         {memory['memory_increase_mb']:.2f} MB")
    print()
    
    # Efficiency metrics
    print("EFFICIENCY METRICS:")
    params_mb = params['total_parameters'] * 4 / (1024 * 1024)  # Assume float32
    print(f"  Model Size (est.):       {params_mb:.2f} MB")
    print(f"  FLOPs per MB:            {flops / max(params_mb, 0.001):,.0f}")
    print(f"  Params per ms:           {params['total_parameters'] / max(timing['mean_time']*1000, 0.001):,.0f}")
    print("=" * 80)


def save_benchmark_results(results, output_path):
    """Save benchmark results to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Model Performance Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            if isinstance(value, dict):
                f.write(f"{key.upper()}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
                f.write("\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Benchmark results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Model Performance Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model_path ../output/models/resnet_model.keras
  %(prog)s --model_path ../output/models/complex_nn_model_gpr_augment.keras --batch_size 64
  %(prog)s --model_path ../output/models/ultra_lightweight_hybrid_model.keras --batch_size 128
  %(prog)s --model_path ../output/models/micro_lightweight_hybrid_model.keras --save_results
  %(prog)s --model_path ../output/models/resnet_model.keras --dataset_path ../denoised_datasets/denoised_data_efficient_gpr_per_sample_c869b840739e.pkl --num_samples 10000
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model file (.keras, .h5, or .pkl)')
    
    parser.add_argument('--input_shape', type=int, nargs='+', default=[2, 128],
                        help='Input shape (excluding batch dimension). Default: [2, 128]')
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for testing. Default: 32')
    
    parser.add_argument('--num_samples', type=int, default=22000,
                        help='Number of test samples to use (when using dataset) or generate (when using random data). Default: 220000')
    
    parser.add_argument('--dataset_path', type=str, default='../denoised_datasets/denoised_data_efficient_gpr_per_sample_c869b840739e.pkl',
                        help='Path to denoised dataset pickle file. If not provided or file not found, will generate random test data. Default: ../denoised_datasets/denoised_data_efficient_gpr_per_sample_c869b840739e.pkl')
    
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of full passes over the test dataset for timing. Each pass iterates all batches once. Default: 100')
    
    parser.add_argument('--output_dir', type=str, default='../output/benchmark_results',
                        help='Directory to save benchmark results')
    
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save detailed results to file (default: True)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    print(f"Loading model from: {args.model_path}")
    
    # Load model
    model, model_type = load_model_safely(args.model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    print(f"Successfully loaded {model_type} model")
    
    # Load test data (from dataset or generate random)
    input_shape = tuple(args.input_shape)
    test_data = None
    y_data = None
    snr_data = None
    
    # Try to load dataset first
    if args.dataset_path and os.path.exists(args.dataset_path):
        print(f"Loading test data from: {args.dataset_path}")
        test_data, y_data, snr_data = load_denoised_dataset(args.dataset_path, args.num_samples)
        
        if test_data is not None:
            # Verify input shape matches
            expected_shape = test_data.shape[1:]
            if expected_shape != input_shape:
                print(f"Warning: Dataset shape {expected_shape} doesn't match expected {input_shape}")
                print(f"Using dataset shape: {expected_shape}")
                input_shape = expected_shape
    
    # Fallback to random data if dataset loading failed
    if test_data is None:
        print(f"Dataset not found or failed to load. Generating random test data...")
        test_data = generate_test_data(input_shape, args.num_samples)
        print(f"Generated random test data: {test_data.shape}")
    
    test_batch = test_data[:args.batch_size]
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    
    # 1. Count parameters
    print("1. Counting parameters...")
    params = count_parameters(model, model_type)
    
    # 2. Calculate FLOPs
    print("2. Calculating FLOPs...")
    full_input_shape = (args.batch_size,) + input_shape
    flops = calculate_flops(model, full_input_shape, model_type)
    
    # 3. Measure inference time
    print("3. Measuring inference time...")
    timing = measure_inference_time(model, test_data, model_type, args.batch_size, args.num_runs)
    
    # 4. Measure memory usage
    print("4. Measuring memory usage...")
    memory = measure_memory_usage(model, test_batch, model_type)
    
    # Print results
    print_benchmark_results(args.model_path, model_type, params, flops, timing, memory, input_shape)
    
    # Save results if requested
    if args.save_results:
        results = {
            'model_path': args.model_path,
            'model_type': model_type,
            'input_shape': input_shape,
            'batch_size': args.batch_size,
            'parameters': params,
            'flops': flops,
            'timing': timing,
            'memory': memory
        }
        
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        output_path = os.path.join(args.output_dir, f"{model_name}_benchmark.txt")
        save_benchmark_results(results, output_path)


if __name__ == "__main__":
    main()