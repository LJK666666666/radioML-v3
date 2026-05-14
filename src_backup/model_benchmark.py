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
python model_benchmark.py --model_path ../output/models/adaboost_model.pkl --num_samples 1000
"""

import os
import sys
import argparse
import time
import pickle
import psutil
import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def configure_gpu():
    """
    Configure GPU settings and check availability
    
    Returns:
        bool: True if GPU is available and configured, False otherwise
    """
    # Check for GPU availability
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    if not gpu_devices:
        print("No GPU devices found. Running on CPU.")
        return False
    
    print(f"Found {len(gpu_devices)} GPU device(s):")
    for i, gpu in enumerate(gpu_devices):
        print(f"  GPU {i}: {gpu.name}")
    
    try:
        # Configure memory growth to avoid allocating all GPU memory at once
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set mixed precision policy for better performance on supported GPUs
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Enabled mixed precision (float16) for better GPU performance")
        
        # Verify GPU is being used
        with tf.device('/GPU:0'):
            # Create a small tensor to test GPU functionality
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = tf.matmul(test_tensor, test_tensor)
            print(f"GPU test successful: {result.device}")
        
        return True
        
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        print("Falling back to CPU")
        return False


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
from model.hybrid_transition_resnet_model import (
    HybridTransitionBlock
)
from train import load_adaboost_model


def get_custom_objects_dict():
    """Get dictionary of all custom objects for model loading"""
    return {
        'ComplexConv1D': ComplexConv1D,
        'ComplexBatchNormalization': ComplexBatchNormalization,
        'ComplexDense': ComplexDense,
        'ComplexMagnitude': ComplexMagnitude,
        'ComplexActivation': ComplexActivation,
        'ComplexPooling1D': ComplexPooling1D,
        'ComplexResidualBlock': ComplexResidualBlock,
        'ComplexResidualBlockAdvanced': ComplexResidualBlockAdvanced,
        'ComplexGlobalAveragePooling1D': ComplexGlobalAveragePooling1D,
        'HybridTransitionBlock': HybridTransitionBlock,
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


def load_model_safely(model_path):
    """
    Safely load a model with custom objects support
    
    Args:
        model_path: Path to the model file (.keras or .pkl)
        
    Returns:
        model: Loaded model object
        model_type: 'keras' or 'sklearn'
    """
    if model_path.endswith('.pkl'):
        # Load AdaBoost/sklearn model
        try:
            model = load_adaboost_model(model_path)
            if model is None:
                raise ValueError("Failed to load AdaBoost model")
            return model, 'sklearn'
        except Exception as e:
            print(f"Error loading AdaBoost model: {e}")
            return None, None
    
    elif model_path.endswith('.keras') or model_path.endswith('.h5'):
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
        model_type: 'keras' or 'sklearn'
        
    Returns:
        dict: Parameter counts
    """
    if model_type == 'keras':
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params
        }
    
    elif model_type == 'sklearn':
        # For sklearn models, count parameters in each estimator
        total_params = 0
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if hasattr(estimator, 'coef_'):
                    total_params += np.prod(estimator.coef_.shape)
                if hasattr(estimator, 'intercept_'):
                    total_params += np.prod(estimator.intercept_.shape)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': total_params,  # All sklearn params are "trainable"
            'non_trainable_parameters': 0
        }
    
    return {'total_parameters': 0, 'trainable_parameters': 0, 'non_trainable_parameters': 0}


def calculate_flops(model, input_shape, model_type):
    """
    Calculate FLOPs (Floating Point Operations)
    
    Args:
        model: Model object
        input_shape: Input tensor shape (batch_size, ...)
        model_type: 'keras' or 'sklearn'
        
    Returns:
        int: Estimated FLOPs count
    """
    if model_type == 'keras':
        try:
            # Use tf.profiler to get FLOPs (requires TensorFlow 2.x)
            concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
                tf.TensorSpec(shape=input_shape, dtype=tf.float32)
            )
            
            # Create profiler options
            options = tf.profiler.experimental.ProfilerOptions()
            
            # Run profiler
            with tf.profiler.experimental.Profile('.', options=options):
                _ = concrete_func(tf.random.normal(input_shape))
            
            # Note: This is a simplified approach
            # For more accurate FLOPs counting, consider using specialized libraries
            # like tensorflow-model-optimization or custom implementations
            
            # Rough estimation based on model size and operations
            total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
            # Rough estimate: each parameter involves ~2 FLOPs (multiply-accumulate)
            estimated_flops = total_params * 2
            
            return estimated_flops
            
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
            # Fallback estimation
            total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
            return total_params * 2
    
    elif model_type == 'sklearn':
        # For sklearn models, estimate based on model complexity
        if hasattr(model, 'estimators_'):
            # AdaBoost-like ensemble
            n_estimators = len(model.estimators_)
            # Assume each estimator does ~100 operations per prediction
            estimated_flops = n_estimators * 100 * input_shape[0]  # batch_size factor
            return estimated_flops
        else:
            # Simple model
            return 1000 * input_shape[0]  # Basic estimation
    
    return 0


def measure_inference_time(model, test_data, model_type, num_runs=100):
    """
    Measure model inference time
    
    Args:
        model: Model object
        test_data: Input data for inference
        model_type: 'keras' or 'sklearn'
        num_runs: Number of inference runs for averaging
        
    Returns:
        dict: Timing statistics
    """
    times = []
    
    # Check if GPU is available for accurate timing
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    
    # Warm-up runs
    for _ in range(5):
        if model_type == 'keras':
            _ = model.predict(test_data, verbose=0)
        elif model_type == 'sklearn':
            _ = model.predict(test_data.reshape(test_data.shape[0], -1))
    
    # Actual timing runs
    for _ in range(num_runs):
        start_time = time.time()
        
        if model_type == 'keras':
            _ = model.predict(test_data, verbose=0)
        elif model_type == 'sklearn':
            _ = model.predict(test_data.reshape(test_data.shape[0], -1))
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'throughput_samples_per_second': test_data.shape[0] / np.mean(times)
    }


def measure_memory_usage(model, test_data, model_type):
    """
    Measure memory usage during inference
    
    Args:
        model: Model object
        test_data: Input data for inference
        model_type: 'keras' or 'sklearn'
        
    Returns:
        dict: Memory usage statistics
    """
    process = psutil.Process()
    
    # Measure baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Perform inference and measure peak memory
    if model_type == 'keras':
        _ = model.predict(test_data, verbose=0)
    elif model_type == 'sklearn':
        _ = model.predict(test_data.reshape(test_data.shape[0], -1))
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'baseline_memory_mb': baseline_memory,
        'peak_memory_mb': peak_memory,
        'memory_increase_mb': peak_memory - baseline_memory
    }


def generate_test_data(input_shape, num_samples):
    """
    Generate random test data
    
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
    print(f"  Mean Inference Time:     {timing['mean_time']*1000:.2f} ms")
    print(f"  Std Inference Time:      {timing['std_time']*1000:.2f} ms")
    print(f"  Min Inference Time:      {timing['min_time']*1000:.2f} ms")
    print(f"  Max Inference Time:      {timing['max_time']*1000:.2f} ms")
    print(f"  Median Inference Time:   {timing['median_time']*1000:.2f} ms")
    print(f"  Throughput:              {timing['throughput_samples_per_second']:.2f} samples/sec")
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
  %(prog)s --model_path ../output/models/adaboost_model.pkl --num_samples 1000
  %(prog)s --model_path ../output/models/transformer_model.keras --input_shape 2 128 --num_runs 50
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model file (.keras, .h5, or .pkl)')
    
    parser.add_argument('--input_shape', type=int, nargs='+', default=[2, 128],
                        help='Input shape (excluding batch dimension). Default: [2, 128]')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing. Default: 32')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of test samples to generate. Default: 1000')
    
    parser.add_argument('--num_runs', type=int, default=100,
                        help='Number of inference runs for timing. Default: 100')
    
    parser.add_argument('--output_dir', type=str, default='../output/benchmark_results',
                        help='Directory to save benchmark results')
    
    parser.add_argument('--save_results', action='store_true',
                        help='Save detailed results to file')
    
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU if available (default: True)')
    
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Configure GPU/CPU usage
    print("=" * 80)
    print("DEVICE CONFIGURATION")
    print("=" * 80)
    
    if args.force_cpu:
        print("Forcing CPU usage as requested")
        tf.config.set_visible_devices([], 'GPU')
        gpu_available = False
    elif args.use_gpu:
        gpu_available = configure_gpu()
    else:
        print("GPU usage disabled by user")
        gpu_available = False
    
    print(f"Using device: {'GPU' if gpu_available else 'CPU'}")
    print("=" * 80)
    
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
    
    # Generate test data
    input_shape = tuple(args.input_shape)
    test_data = generate_test_data(input_shape, args.num_samples)
    test_batch = test_data[:args.batch_size]
    
    print(f"Generated test data: {test_data.shape}")
    
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
    timing = measure_inference_time(model, test_batch, model_type, args.num_runs)
    
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