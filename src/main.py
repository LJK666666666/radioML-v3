#!/usr/bin/env python3
"""
Usage examples:
python main.py --models resnet cnn1d --mode train
python main.py --models hybrid_complex_resnet --mode evaluate
python main.py --models cnn2d complex_nn --mode all --epochs 100
"""

import os
import argparse
import time
import random
import numpy as np
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all logs, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
tf.get_logger().setLevel('ERROR')

# Import project modules
from explore_dataset import load_radioml_data, explore_dataset, plot_signal_examples
from preprocess import prepare_data_by_snr_stratified
from train import train_model, plot_training_history
from models import (
    build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model,
    get_callbacks,
    build_hybrid_complex_resnet_model, build_lightweight_hybrid_model,
    # AMC-Net model
    build_amcnet_model,
    # Benchmark models
    build_cldnn_model, build_cgdnn_model,
    # ULCNN models
    build_mcldnn_model, build_scnn_model, build_mcnet_model, build_pet_model, build_ulcnn_model,
    # New ultra-lightweight hybrid models
    build_ultra_lightweight_hybrid_model, build_micro_lightweight_hybrid_model
)
from evaluate import evaluate_by_snr

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

# Import ULCNN complex layers for model loading
try:
    from model.complexnn import (
        ComplexConv1D as ULCNNComplexConv1D,
        ComplexBatchNormalization as ULCNNComplexBatchNormalization,
        ComplexDense as ULCNNComplexDense,
        ChannelShuffle, DWConvMobile, ChannelAttention
    )
    ULCNN_LAYERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ULCNN complex layers not available: {e}")
    ULCNN_LAYERS_AVAILABLE = False


def set_random_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print(f"Random seed set to {seed}")


def configure_gpu(gpu_id=None):
    """Configure GPU devices for training"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPUs found. Using CPU for training.")
        return
    
    print(f"Available GPUs: {[gpu.name for gpu in gpus]}")
    
    if gpu_id is None:
        # Use all available GPUs
        print("Using all available GPUs")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        # Use specific GPU(s)
        try:
            gpu_ids = [int(x.strip()) for x in gpu_id.split(',')]
            selected_gpus = [gpus[i] for i in gpu_ids if i < len(gpus)]
            
            if not selected_gpus:
                print(f"Invalid GPU ID(s): {gpu_id}. Using all available GPUs.")
                selected_gpus = gpus
            
            # Set visible devices
            tf.config.set_visible_devices(selected_gpus, 'GPU')
            print(f"Using GPU(s): {[gpu.name for gpu in selected_gpus]}")
            
            # Enable memory growth for selected GPUs
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
        except (ValueError, IndexError) as e:
            print(f"Error parsing GPU ID '{gpu_id}': {e}")
            print("Using all available GPUs")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            print("Note: GPU configuration must be set before creating any operations")


def get_file_suffix(denoising_method, augment_data):
    """Generate suffix for file names based on denoising method and data augmentation"""
    suffix = ""
    if denoising_method and denoising_method != 'none':
        suffix += f"_{denoising_method}"
    if augment_data:
        suffix += "_augment"
    return suffix


def get_custom_objects_for_model(model_name):
    """Get custom objects dict for specific model types that need them"""
    # Models that need complex layer custom objects
    complex_models = ['complex_nn', 'hybrid_complex_resnet', 'lightweight_hybrid',
                     'ultra_lightweight_hybrid', 'micro_lightweight_hybrid']

    # ULCNN models that need complex layer custom objects
    ulcnn_complex_models = ['ulcnn', 'mcldnn', 'pet']  # mcnet and scnn use standard layers

    # AMC-Net models that need custom objects
    amcnet_models = ['amcnet']

    # Benchmark models that need custom objects (for Lambda functions)
    benchmark_models = ['cldnn', 'cgdnn']

    if model_name in complex_models or model_name in ulcnn_complex_models:
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

        if model_name in ulcnn_complex_models:
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

            custom_objects.update({
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
            })

        # Add ULCNN-specific layers if available
        if ULCNN_LAYERS_AVAILABLE:
            from model.complexnn import (
                TransposeLayer, ExtractChannelLayer, TrigonometricLayer, sqrt_init
            )
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
    elif model_name in amcnet_models:
        # AMC-Net custom objects
        from model.amcnet_model import (
            Conv_Block, MultiScaleModule, TinyMLP, AdaCorrModule, FeaFusionModule,
            L2NormalizeWidthLayer, SqueezeHeightLayer, TransposeChannelWidthLayer, GlobalAvgPoolWidthLayer
        )
        custom_objects = {
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
        return custom_objects
    elif model_name in benchmark_models:
        # Benchmark models custom objects (for Lambda functions)
        def dynamic_reshape(x):
            """Dynamic reshape function for CGDNN"""
            import tensorflow as tf
            shape = tf.shape(x)
            batch_size = shape[0]
            channels = shape[1]
            height = shape[2]
            width = shape[3]
            reshaped = tf.reshape(x, [batch_size, channels, height * width])
            return reshaped

        def dynamic_reshape_cldnn(x):
            """Dynamic reshape function for CLDNN"""
            import tensorflow as tf
            shape = tf.shape(x)
            batch_size = shape[0]
            channels = shape[1]
            height = shape[2]
            width = shape[3]
            reshaped = tf.reshape(x, [batch_size, channels, height * width])
            return reshaped

        custom_objects = {
            'dynamic_reshape': dynamic_reshape,
            'dynamic_reshape_cldnn': dynamic_reshape_cldnn,
        }
        return custom_objects
    return None


def evaluate_model_variants(model_name, model_base_path, X_test, y_test, snr_test, mods, results_dir, suffix="", custom_objects=None):
    """
    Evaluate both the best model and the last epoch model for a given model type.

    Args:
        model_name: Name of the model type (e.g., 'cnn1d', 'resnet')
        model_base_path: Base path without extension (e.g., '/path/models/cnn1d_model')
        X_test, y_test, snr_test, mods: Test data and metadata
        results_dir: Directory to save evaluation results
        suffix: File suffix for distinguishing denoising method and augmentation (e.g., '_gpr_augment')
        custom_objects: Custom objects needed for model loading (optional)
    """
    # Standard Keras model evaluation
    best_model_path = model_base_path + ".keras"
    if os.path.exists(best_model_path):
        print(f"\nEvaluating {model_name} Model (Best)...")
        try:
            if custom_objects:
                best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)
            else:
                best_model = tf.keras.models.load_model(best_model_path)
            print(f"Successfully loaded best model from {best_model_path}")
            evaluate_by_snr(
                best_model,
                X_test, y_test, snr_test, mods,
                os.path.join(results_dir, f'{model_name}_evaluation_results{suffix}')
            )
        except Exception as e:
            print(f"Error loading or evaluating best model {best_model_path}: {e}")
    else:
        print(f"Best model {best_model_path} not found for evaluation.")

    # Evaluate last epoch model
    last_model_path = model_base_path + "_last.keras"
    if os.path.exists(last_model_path):
        print(f"\nEvaluating {model_name} Model (Last Epoch)...")
        try:
            if custom_objects:
                last_model = tf.keras.models.load_model(last_model_path, custom_objects=custom_objects)
            else:
                last_model = tf.keras.models.load_model(last_model_path)
            print(f"Successfully loaded last epoch model from {last_model_path}")
            evaluate_by_snr(
                last_model,
                X_test, y_test, snr_test, mods,
                os.path.join(results_dir, f'{model_name}_evaluation_results_last{suffix}')
            )
        except Exception as e:
            print(f"Error loading or evaluating last model {last_model_path}: {e}")
    else:
        print(f"Last model {last_model_path} not found for evaluation.")


def get_available_models():
    """Return list of all available model types"""
    return [
        'cnn1d', 'cnn2d', 'resnet', 'complex_nn',
        'hybrid_complex_resnet', 'lightweight_hybrid',
        # AMC-Net model
        'amcnet',
        # Benchmark models
        'cldnn', 'cgdnn',
        # ULCNN models
        'mcldnn', 'scnn', 'mcnet', 'pet', 'ulcnn',
        # New ultra-lightweight hybrid models
        'ultra_lightweight_hybrid', 'micro_lightweight_hybrid'
    ]


def expand_model_selection(selected_models):
    """
    Expand model selection to handle special cases
    Returns a list of individual model names to process
    """
    expanded_models = []

    for model in selected_models:
        expanded_models.append(model)

    return expanded_models


def build_model_by_name(model_name, input_shape, num_classes):
    """Build a model by name and return the model instance"""
    model_builders = {
        'cnn1d': build_cnn1d_model,
        'cnn2d': build_cnn2d_model,
        'resnet': build_resnet_model,
        'complex_nn': build_complex_nn_model,
        'hybrid_complex_resnet': build_hybrid_complex_resnet_model,
        'lightweight_hybrid': build_lightweight_hybrid_model,
        # AMC-Net model
        'amcnet': build_amcnet_model,
        # Benchmark models
        'cldnn': build_cldnn_model,
        'cgdnn': build_cgdnn_model,
        # ULCNN models
        'mcldnn': build_mcldnn_model,
        'scnn': build_scnn_model,
        'mcnet': build_mcnet_model,
        'pet': build_pet_model,
        'ulcnn': build_ulcnn_model,
        # New ultra-lightweight hybrid models
        'ultra_lightweight_hybrid': build_ultra_lightweight_hybrid_model,
        'micro_lightweight_hybrid': build_micro_lightweight_hybrid_model,
    }

    # Handle standard models
    if model_name in model_builders:
        return model_builders[model_name](input_shape, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_selected_models(selected_models, X_train, y_train, X_val, y_val, input_shape, num_classes, 
                         models_dir, plots_dir, suffix, batch_size, epochs):
    """Train all selected models"""
    
    for model_name in selected_models:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Model...")
        print(f"{'='*60}")
        
        try:
            # Handle special data preparation for CNN2D
            if model_name == 'cnn2d':
                # Reshape data for 2D model
                X_train_model = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
                X_val_model = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
            else:
                X_train_model = X_train
                X_val_model = X_val
            
            # Build model
            model = build_model_by_name(model_name, input_shape, num_classes)
            
            print(f"Model architecture for {model_name}:")
            model.summary()
            
            # Train model
            history = train_model(
                model,
                X_train_model, y_train,
                X_val_model, y_val,
                os.path.join(models_dir, f"{model_name}_model{suffix}.keras"),
                batch_size=batch_size,
                epochs=epochs
            )
            
            # Plot and save training history
            plot_training_history(
                history,
                os.path.join(plots_dir, f"{model_name}_training_history{suffix}.png")
            )
            
            print(f"Successfully completed training for {model_name}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue


def evaluate_selected_models(selected_models, X_test, y_test, snr_test, mods, 
                           models_dir, results_dir, suffix):
    """Evaluate all selected models"""
    
    # Enable unsafe deserialization for models with custom layers
    tf.keras.config.enable_unsafe_deserialization()
    
    for model_name in selected_models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} Model...")
        print(f"{'='*60}")
        
        try:
            # Get custom objects if needed
            custom_objects = get_custom_objects_for_model(model_name)
            
            # Get model base path
            model_base_path = os.path.join(models_dir, f"{model_name}_model{suffix}")
            
            # Evaluate model variants (best and last)
            evaluate_model_variants(
                model_name, model_base_path, X_test, y_test, snr_test, mods, 
                results_dir, suffix, custom_objects
            )
            
            print(f"Successfully completed evaluation for {model_name}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue


def validate_model_selection(selected_models):
    """Validate that all selected models are available"""
    available_models = get_available_models()
    invalid_models = []
    
    for model in selected_models:
        if model not in available_models:
            invalid_models.append(model)
    
    if invalid_models:
        print(f"Error: Invalid model(s) selected: {invalid_models}")
        print(f"Available models: {available_models}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='RadioML Signal Classification with Flexible Model Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --models resnet cnn1d --mode train
  %(prog)s --models hybrid_complex_resnet --mode evaluate
  %(prog)s --models cnn2d complex_nn --mode all --epochs 100
  %(prog)s --models resnet --mode train  # Use stratified splitting by (modulation, SNR)
  %(prog)s --models cnn1d resnet --mode train --gpu_id 0  # Use GPU 0 only
  %(prog)s --models resnet --mode train --gpu_id 0,1  # Use GPU 0 and 1
        """
    )
    
    parser.add_argument('--mode', type=str, default='all',
                        choices=['explore', 'train', 'evaluate', 'all'],
                        help='Mode of operation')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=get_available_models(),
                        required=True,
                        help='Model architectures to use (select multiple)')
    
    parser.add_argument('--dataset_path', type=str, default='../data/RML2016.10a_dict.pkl',
                        help='Path to the RadioML dataset')
    
    parser.add_argument('--output_dir', type=str, default='../output',
                        help='Directory for outputs')
    
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--augment_data', action='store_true',
                        help='Enable data augmentation for training data (11 rotations, 30 deg increments)')
    
    parser.add_argument('--denoising_method', type=str, default='efficient_gpr_per_sample',
                        choices=['gpr', 'efficient_gpr_per_sample', 'gpr_fft', 'none'],
                        help='Denoising method to apply to the signals (gpr, efficient_gpr_per_sample, gpr_fft, none)')
    
    
    parser.add_argument('--denoised_cache_dir', type=str, default='../denoised_datasets',
                        help='Directory to save/load cached denoised datasets')
    
    
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU device ID to use for training (e.g., "0", "1", "0,1" for multiple GPUs). If not specified, uses all available GPUs.')
    
    args = parser.parse_args()
    
    # Validate model selection
    if not validate_model_selection(args.models):
        return
    
    # Expand model selection
    selected_models = expand_model_selection(args.models)
    
    print(f"Selected models for processing: {selected_models}")
    print(f"Total models selected: {len(selected_models)}")
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Configure GPU devices
    configure_gpu(args.gpu_id)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'training_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load dataset
    print(f"\n{'='*60}")
    print("LOADING DATASET")
    print(f"{'='*60}")
    print(f"Loading dataset from {args.dataset_path}...")
    start_time = time.time()
    dataset = load_radioml_data(args.dataset_path)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    
    # Explore dataset
    if args.mode in ['explore', 'all']:
        print(f"\n{'='*60}")
        print("EXPLORING DATASET")
        print(f"{'='*60}")
        mods, snrs = explore_dataset(dataset)
        plot_signal_examples(dataset, mods, os.path.join(args.output_dir, 'exploration'))
    
    # Prepare data
    if args.mode in ['train', 'evaluate', 'all']:
        print(f"\n{'='*60}")
        print("PREPARING DATA")
        print(f"{'='*60}")
        
        print("Using stratified splitting by (modulation type, SNR) combinations...")
        X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_data_by_snr_stratified(
            dataset,
            augment_data=args.augment_data,
            denoising_method=args.denoising_method,
            denoised_cache_dir=args.denoised_cache_dir
        )
    
    # Training
    if args.mode in ['train', 'all']:
        print(f"\n{'='*60}")
        print("TRAINING SELECTED MODELS")
        print(f"{'='*60}")
        
        # Generate file suffix based on denoising method and data augmentation
        suffix = get_file_suffix(args.denoising_method, args.augment_data)
        suffix += "_stratified"
        
        input_shape = X_train.shape[1:]
        num_classes = len(mods)
        
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Models to train: {selected_models}")
        
        # Train selected models
        train_selected_models(
            selected_models, X_train, y_train, X_val, y_val, 
            input_shape, num_classes, models_dir, plots_dir, 
            suffix, args.batch_size, args.epochs
        )
    
    # Evaluation
    if args.mode in ['evaluate', 'all']:
        print(f"\n{'='*60}")
        print("EVALUATING SELECTED MODELS")
        print(f"{'='*60}")
        
        # Generate file suffix for evaluation (always includes stratified)
        suffix = get_file_suffix(args.denoising_method, args.augment_data)
        suffix += "_stratified"
        
        print(f"Models to evaluate: {selected_models}")
        
        # Evaluate selected models
        evaluate_selected_models(
            selected_models, X_test, y_test, snr_test, mods,
            models_dir, results_dir, suffix
        )
    
    print(f"\n{'='*60}")
    print("ALL OPERATIONS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Processed models: {selected_models}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
