#!/usr/bin/env python3
"""
Model Performance Benchmark Tool

Evaluates model performance metrics: parameters, FLOPs, inference speed.
Supports both Keras (.keras) and PyTorch (.pt) models.

Usage examples:
  python model_benchmark.py --model_path ../output/models/amcnet_model_stratified.keras
  python model_benchmark.py --model_name iqformer --num_classes 11
  python model_benchmark.py --batch_all
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
import contextlib
import math

# Suppress all warnings and logs before TensorFlow import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
import warnings
warnings.filterwarnings('ignore')


class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass
    def close(self):
        pass


original_stderr = sys.stderr


@contextlib.contextmanager
def suppress_stderr():
    try:
        sys.stderr = DevNull()
        yield
    finally:
        sys.stderr = original_stderr


# Import TensorFlow
with suppress_stderr():
    import tensorflow as tf
    tf.config.set_soft_device_placement(True)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus[:1]:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Import PyTorch
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.custom_objects import get_all_custom_objects as get_custom_objects_dict
from main import build_model_by_name, is_torch_model_name


# ======================== Model Loading ========================

def load_model_safely(model_path):
    """Load a Keras model with custom objects support."""
    if model_path.endswith('.keras') or model_path.endswith('.h5'):
        try:
            model = tf.keras.models.load_model(model_path)
            return model, 'keras'
        except Exception:
            try:
                tf.keras.config.enable_unsafe_deserialization()
                custom_objects = get_custom_objects_dict()
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                return model, 'keras'
            except Exception as e2:
                print(f"Keras loading failed: {e2}")
                return None, None
    elif model_path.endswith('.pt'):
        print("For .pt files, use --model_name to specify the architecture.")
        return None, None
    else:
        print(f"Unsupported model format: {model_path}")
        return None, None


def build_model_from_name(model_name, input_shape, num_classes):
    """Build model from name (no weights needed for benchmarking)."""
    model = build_model_by_name(model_name, input_shape, num_classes)
    if is_torch_model_name(model_name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        return model, 'torch'
    else:
        return model, 'keras'


# ======================== Parameter Counting ========================

def count_parameters(model, model_type):
    """Count model parameters."""
    if model_type == 'keras':
        trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
        return {
            'total_parameters': int(trainable + non_trainable),
            'trainable_parameters': int(trainable),
            'non_trainable_parameters': int(non_trainable)
        }
    elif model_type == 'torch':
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return {
            'total_parameters': int(trainable + non_trainable),
            'trainable_parameters': int(trainable),
            'non_trainable_parameters': int(non_trainable)
        }
    return {'total_parameters': 0, 'trainable_parameters': 0, 'non_trainable_parameters': 0}


# ======================== FLOPs Calculation ========================

def calculate_flops_keras(model, input_shape):
    """Estimate FLOPs for Keras model."""
    total_flops = 0
    try:
        sample_input = tf.random.normal(input_shape)
        x = sample_input
        for layer in model.layers:
            try:
                x = layer(x)
                if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv1D, tf.keras.layers.Conv2D)):
                    if hasattr(layer, 'kernel'):
                        output_elems = int(tf.reduce_prod(x.shape[1:]).numpy()) if None not in x.shape[1:] else 0
                        kernel_spatial = 1
                        if hasattr(layer, 'kernel_size'):
                            ks = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,)
                            for v in ks:
                                kernel_spatial *= int(v)
                        in_ch = int(layer.kernel.shape[-2]) if len(layer.kernel.shape) >= 2 else 1
                        total_flops += output_elems * kernel_spatial * in_ch
                elif hasattr(layer, 'weights') and len(layer.weights) > 0:
                    layer_params = sum(tf.keras.backend.count_params(w) for w in layer.weights)
                    total_flops += int(layer_params) * 2
            except Exception:
                continue
    except Exception:
        pass
    if total_flops == 0:
        total_flops = sum(tf.keras.backend.count_params(w) for w in model.weights) * 2
    return int(total_flops)


def calculate_flops_torch(model, input_shape):
    """Estimate FLOPs for PyTorch model using thop if available, else fallback."""
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape).to(device)
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return int(flops)
    except ImportError:
        pass
    # Fallback: 2x total parameters (rough estimate)
    total_params = sum(p.numel() for p in model.parameters())
    return int(total_params * 2)


def calculate_flops(model, input_shape, model_type):
    """Calculate FLOPs."""
    if model_type == 'keras':
        return calculate_flops_keras(model, input_shape)
    elif model_type == 'torch':
        return calculate_flops_torch(model, input_shape)
    return 0


# ======================== Inference Timing ========================

def measure_inference_time(model, test_data, model_type, batch_size, num_runs=1):
    """Measure model inference time over the whole dataset in batches."""
    num_samples = int(test_data.shape[0])
    num_batches = int(math.ceil(num_samples / batch_size)) if batch_size > 0 else 0

    if model_type == 'torch':
        device = next(model.parameters()).device
        # Warm-up
        if num_samples > 0:
            first_batch = torch.from_numpy(test_data[:min(batch_size, num_samples)]).float().to(device)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(first_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        batch_times = []
        total_samples_processed = 0
        for _ in range(num_runs):
            for i in range(num_batches):
                batch_np = test_data[i * batch_size: (i + 1) * batch_size]
                if batch_np.shape[0] == 0:
                    continue
                batch_t = torch.from_numpy(batch_np).float().to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    _ = model(batch_t)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                batch_times.append(end_time - start_time)
                total_samples_processed += int(batch_np.shape[0])

    else:  # keras — use @tf.function compiled forward for fair comparison with PyTorch
        @tf.function(jit_compile=False)
        def _keras_infer(x):
            return model(x, training=False)

        # Warm-up: trigger tf.function tracing + compilation
        if num_samples > 0:
            first_batch = tf.constant(test_data[:min(batch_size, num_samples)])
            for _ in range(10):
                _ = _keras_infer(first_batch)

        batch_times = []
        total_samples_processed = 0
        for _ in range(num_runs):
            for i in range(num_batches):
                batch_np = test_data[i * batch_size: (i + 1) * batch_size]
                if batch_np.shape[0] == 0:
                    continue
                batch_tf = tf.constant(batch_np)
                start_time = time.time()
                _ = _keras_infer(batch_tf)
                end_time = time.time()
                batch_times.append(end_time - start_time)
                total_samples_processed += int(batch_np.shape[0])

    times = np.array(batch_times, dtype=np.float64)
    total_time = float(np.sum(times)) if times.size else 0.0

    return {
        'mean_time': float(np.mean(times)) if times.size else 0.0,
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


# ======================== Output ========================

def print_benchmark_results(model_name, model_type, params, flops, timing, input_shape):
    """Print formatted benchmark results."""
    per_sample_ms = (timing['total_time_seconds'] / timing['total_samples_processed'] * 1000) \
        if timing['total_samples_processed'] > 0 else 0.0

    print("=" * 70)
    print(f"  {model_name}  ({model_type})")
    print("=" * 70)
    print(f"  Input Shape:             {input_shape}")
    print(f"  Total Parameters:        {params['total_parameters']:,}")
    print(f"  Trainable Parameters:    {params['trainable_parameters']:,}")
    print(f"  Estimated FLOPs:         {flops:,}")
    print(f"  Per-Sample Time:         {per_sample_ms:.4f} ms")
    print(f"  Throughput:              {timing['throughput_samples_per_second']:.0f} samples/sec")
    print(f"  Total Time:              {timing['total_time_seconds']:.3f} s  "
          f"({timing['total_samples_processed']} samples, "
          f"batch_size={timing['batch_size']}, runs={timing['num_runs']})")
    print("=" * 70)
    return per_sample_ms


def save_benchmark_results(results, output_path):
    """Save benchmark results to file."""
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
    print(f"Results saved to: {output_path}")


# ======================== Batch Benchmark ========================

def run_batch_benchmark(batch_size=256, num_samples=22000, num_runs=1, input_shape=(2, 128),
                        num_classes=11, output_dir='../output/benchmark_results'):
    """Benchmark all paper models and output a summary table."""
    # Models to benchmark (in order for the paper table)
    paper_models = [
        'iqformer', 'amcnet', 'fea_t', 'mcldnn', 'pet', 'ulcnn',
    ]
    display_names = {
        'iqformer': 'IQFormer',
        'amcnet': 'AMC-Net',
        'fea_t': 'FEA-T',
        'mcldnn': 'MCLDNN',
        'pet': 'PETCGDNN',
        'ulcnn': 'ULCNN',
    }

    test_data = np.random.normal(0, 1, (num_samples,) + input_shape).astype(np.float32)

    results_list = []
    print(f"\n{'='*70}")
    print(f"  BATCH BENCHMARK: {len(paper_models)} models")
    print(f"  batch_size={batch_size}, num_samples={num_samples}, num_runs={num_runs}")
    print(f"  input_shape={input_shape}, num_classes={num_classes}")
    print(f"{'='*70}\n")

    for model_name in paper_models:
        disp = display_names.get(model_name, model_name)
        print(f"--- Benchmarking {disp} ({model_name}) ---")
        try:
            model, model_type = build_model_from_name(model_name, input_shape, num_classes)
        except Exception as e:
            print(f"  ERROR building model: {e}\n")
            results_list.append({
                'name': disp, 'params': 'ERROR', 'flops': 'ERROR', 'per_sample_ms': 'ERROR'
            })
            continue

        params = count_parameters(model, model_type)
        full_input_shape = (batch_size,) + input_shape
        flops = calculate_flops(model, full_input_shape, model_type)
        timing = measure_inference_time(model, test_data, model_type, batch_size, num_runs)

        per_sample_ms = print_benchmark_results(disp, model_type, params, flops, timing, input_shape)

        results_list.append({
            'name': disp,
            'model_name': model_name,
            'model_type': model_type,
            'params': params['total_parameters'],
            'trainable_params': params['trainable_parameters'],
            'flops': flops,
            'per_sample_ms': per_sample_ms,
            'throughput': timing['throughput_samples_per_second'],
        })

        # Clean up GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print()

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE (for paper)")
    print("=" * 70)
    print(f"  {'Model':<12} {'Parameters':>12} {'FLOPs':>12} {'Per-Sample (ms)':>16}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*16}")
    for r in results_list:
        if r['params'] == 'ERROR':
            print(f"  {r['name']:<12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>16}")
        else:
            print(f"  {r['name']:<12} {r['params']:>12,} {r['flops']:>12,} {r['per_sample_ms']:>16.4f}")
    print(f"  {'GPR Denoise':<12} {'--':>12} {'--':>12} {'0.0125':>16}")
    print("=" * 70)

    # Save summary to file
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'batch_benchmark_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model Performance Benchmark Summary\n")
        f.write(f"batch_size={batch_size}, num_samples={num_samples}, num_runs={num_runs}\n")
        f.write(f"input_shape={input_shape}, num_classes={num_classes}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")
        f.write(f"{'Model':<12} {'Parameters':>12} {'FLOPs':>12} {'Per-Sample(ms)':>16}\n")
        f.write(f"{'-'*52}\n")
        for r in results_list:
            if r['params'] == 'ERROR':
                f.write(f"{r['name']:<12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>16}\n")
            else:
                f.write(f"{r['name']:<12} {r['params']:>12,} {r['flops']:>12,} {r['per_sample_ms']:>16.4f}\n")
        f.write(f"{'GPR Denoise':<12} {'--':>12} {'--':>12} {'0.0125':>16}\n")
    print(f"\nSummary saved to: {summary_path}")

    return results_list


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(
        description='Model Performance Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --batch_all
  %(prog)s --model_name iqformer --num_classes 11
  %(prog)s --model_path ../output/models/amcnet_model_stratified.keras
        """
    )

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model file (.keras or .h5)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name to build from scratch (e.g., iqformer, fea_t, amcnet, ulcnn, mcldnn, pet)')
    parser.add_argument('--batch_all', action='store_true',
                        help='Benchmark all paper models in batch mode')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[2, 128],
                        help='Input shape (excluding batch). Default: [2, 128]')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='Number of classes. Default: 11')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size. Default: 256')
    parser.add_argument('--num_samples', type=int, default=22000,
                        help='Number of test samples. Default: 22000')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of full passes for timing. Default: 1')
    parser.add_argument('--output_dir', type=str, default='../output/benchmark_results',
                        help='Directory to save results')

    args = parser.parse_args()
    input_shape = tuple(args.input_shape)

    # Batch mode
    if args.batch_all:
        run_batch_benchmark(
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            num_runs=args.num_runs,
            input_shape=input_shape,
            num_classes=args.num_classes,
            output_dir=args.output_dir,
        )
        return

    # Single model mode
    if args.model_name:
        print(f"Building model from name: {args.model_name}")
        model, model_type = build_model_from_name(args.model_name, input_shape, args.num_classes)
    elif args.model_path:
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found: {args.model_path}")
            return
        print(f"Loading model from: {args.model_path}")
        model, model_type = load_model_safely(args.model_path)
    else:
        print("Error: Provide --model_name, --model_path, or --batch_all")
        return

    if model is None:
        print("Failed to load/build model.")
        return

    print(f"Model type: {model_type}")

    # Generate test data
    test_data = np.random.normal(0, 1, (args.num_samples,) + input_shape).astype(np.float32)

    # Run benchmarks
    print("\nRunning benchmarks...")
    params = count_parameters(model, model_type)
    full_input_shape = (args.batch_size,) + input_shape
    flops = calculate_flops(model, full_input_shape, model_type)
    timing = measure_inference_time(model, test_data, model_type, args.batch_size, args.num_runs)

    name = args.model_name or os.path.basename(args.model_path or 'unknown')
    per_sample_ms = print_benchmark_results(name, model_type, params, flops, timing, input_shape)

    # Save results
    results = {
        'model_name': name,
        'model_type': model_type,
        'input_shape': input_shape,
        'batch_size': args.batch_size,
        'parameters': params,
        'flops': flops,
        'per_sample_ms': per_sample_ms,
        'timing': timing,
    }
    output_path = os.path.join(args.output_dir, f"{name}_benchmark.txt")
    save_benchmark_results(results, output_path)


if __name__ == "__main__":
    main()
