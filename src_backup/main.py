#!/usr/bin/env python3
"""
Flexible RadioML Signal Classification with Multiple Model Selection

This script allows for flexible selection of multiple models for training and evaluation,
rather than being limited to a single model type or all models.

Usage examples:
python flexible_main.py --models resnet cnn1d transformer --mode train
python flexible_main.py --models hybrid_complex_resnet lightweight_lstm fcnn --mode evaluate
python flexible_main.py --models cnn2d complex_nn adaboost --mode all --epochs 100
"""

import os
import argparse
import time
import random
import numpy as np
import tensorflow as tf

# Import project modules
from explore_dataset import load_radioml_data, explore_dataset, plot_signal_examples
from preprocess import prepare_data, prepare_data_by_snr, prepare_data_by_snr_stratified
from train import train_model, plot_training_history, load_adaboost_model
from models import (
    build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model,
    build_transformer_model, build_transformer_rope_sequential_model, build_transformer_rope_phase_model,
    build_lstm_model, build_advanced_lstm_model,
    build_multi_scale_lstm_model, build_lightweight_lstm_model, build_hybrid_complex_resnet_model,
    build_lightweight_hybrid_model, build_hybrid_transition_resnet_model, build_lightweight_transition_model,
    build_comparison_models, build_keras_adaboost_model, build_lightweight_adaboost_model,
    build_fcnn_model, build_deep_fcnn_model, build_lightweight_fcnn_model,
    build_wide_fcnn_model, build_shallow_fcnn_model, build_custom_fcnn_model, get_callbacks,
    # ULCNN models
    build_mcldnn_model, build_scnn_model, build_mcnet_model, build_pet_model, build_ulcnn_model
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
from model.hybrid_transition_resnet_model import (
    HybridTransitionBlock
)
from model.transformer_model import (
    RotaryPositionalEncoding, PhaseBasedPositionalEncoding
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
                     'hybrid_transition_resnet', 'lightweight_transition']

    # ULCNN models that need complex layer custom objects
    ulcnn_complex_models = ['ulcnn', 'mcldnn', 'pet']  # mcnet and scnn use standard layers

    # Models that need transformer custom objects
    transformer_models = ['transformer_rope_sequential', 'transformer_rope_phase']

    # Models from comparison_models that might need custom objects
    comparison_models = ['high_complex', 'medium_complex', 'low_complex']

    if model_name in complex_models or model_name in comparison_models or model_name in ulcnn_complex_models:
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
    elif model_name in transformer_models:
        return {
            'RotaryPositionalEncoding': RotaryPositionalEncoding,
            'PhaseBasedPositionalEncoding': PhaseBasedPositionalEncoding
        }
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
    # Check if this is an AdaBoost model
    is_adaboost = 'adaboost' in model_name.lower()
    
    if is_adaboost:
        # For AdaBoost models, look for .pkl files instead of .keras
        print(f"\nEvaluating AdaBoost {model_name} Model...")
        
        best_model_path = model_base_path + ".pkl"
        last_model_path = model_base_path + "_last.pkl"
        
        # Evaluate best model
        if os.path.exists(best_model_path):
            print(f"\nEvaluating {model_name} Model (Best)...")
            try:
                best_model = load_adaboost_model(best_model_path)
                if best_model:
                    print(f"Successfully loaded best AdaBoost model from {best_model_path}")
                    evaluate_by_snr(
                        best_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, f'{model_name}_evaluation_results{suffix}')
                    )
                else:
                    print(f"Failed to load best AdaBoost model from {best_model_path}")
            except Exception as e:
                print(f"Error loading or evaluating best AdaBoost model {best_model_path}: {e}")
        else:
            print(f"Best AdaBoost model {best_model_path} not found for evaluation.")
        
        # Evaluate last epoch model
        if os.path.exists(last_model_path):
            print(f"\nEvaluating {model_name} Model (Last Epoch)...")
            try:
                last_model = load_adaboost_model(last_model_path)
                if last_model:
                    print(f"Successfully loaded last epoch AdaBoost model from {last_model_path}")
                    evaluate_by_snr(
                        last_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, f'{model_name}_evaluation_results_last{suffix}')
                    )
                else:
                    print(f"Failed to load last epoch AdaBoost model from {last_model_path}")
            except Exception as e:
                print(f"Error loading or evaluating last AdaBoost model {last_model_path}: {e}")
        else:
            print(f"Last AdaBoost model {last_model_path} not found for evaluation.")
    
    else:
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
        'cnn1d', 'cnn2d', 'resnet', 'complex_nn', 'transformer',
        'transformer_rope_sequential', 'transformer_rope_phase',
        'lstm', 'advanced_lstm', 'multi_scale_lstm', 'lightweight_lstm',
        'hybrid_complex_resnet', 'lightweight_hybrid',
        'hybrid_transition_resnet', 'lightweight_transition',
        'comparison_models', 'adaboost', 'lightweight_adaboost',
        'fcnn', 'deep_fcnn', 'lightweight_fcnn', 'wide_fcnn', 'shallow_fcnn',
        # ULCNN models
        'mcldnn', 'scnn', 'mcnet', 'pet', 'ulcnn'
    ]


def expand_model_selection(selected_models):
    """
    Expand model selection to handle special cases like 'comparison_models'
    Returns a list of individual model names to process
    """
    expanded_models = []
    
    for model in selected_models:
        if model == 'comparison_models':
            # Expand comparison_models to its constituent models
            expanded_models.extend(['high_complex', 'medium_complex', 'low_complex'])
        else:
            expanded_models.append(model)
    
    return expanded_models


def build_model_by_name(model_name, input_shape, num_classes):
    """Build a model by name and return the model instance"""
    model_builders = {
        'cnn1d': build_cnn1d_model,
        'cnn2d': build_cnn2d_model,
        'resnet': build_resnet_model,
        'complex_nn': build_complex_nn_model,
        'transformer': build_transformer_model,
        'transformer_rope_sequential': build_transformer_rope_sequential_model,
        'transformer_rope_phase': build_transformer_rope_phase_model,
        'lstm': build_lstm_model,
        'advanced_lstm': build_advanced_lstm_model,
        'multi_scale_lstm': build_multi_scale_lstm_model,
        'lightweight_lstm': build_lightweight_lstm_model,
        'hybrid_complex_resnet': build_hybrid_complex_resnet_model,
        'lightweight_hybrid': build_lightweight_hybrid_model,
        'hybrid_transition_resnet': build_hybrid_transition_resnet_model,
        'lightweight_transition': build_lightweight_transition_model,
        'fcnn': build_fcnn_model,
        'deep_fcnn': build_deep_fcnn_model,
        'lightweight_fcnn': build_lightweight_fcnn_model,
        'wide_fcnn': build_wide_fcnn_model,
        'shallow_fcnn': build_shallow_fcnn_model,
        # ULCNN models
        'mcldnn': build_mcldnn_model,
        'scnn': build_scnn_model,
        'mcnet': build_mcnet_model,
        'pet': build_pet_model,
        'ulcnn': build_ulcnn_model,
    }
    
    # Handle AdaBoost models
    if model_name == 'adaboost':
        return build_keras_adaboost_model(input_shape, num_classes, n_estimators=10, learning_rate=1.0)
    elif model_name == 'lightweight_adaboost':
        return build_keras_adaboost_model(input_shape, num_classes, n_estimators=5, learning_rate=0.5)
    
    # Handle comparison models
    elif model_name in ['high_complex', 'medium_complex', 'low_complex']:
        comparison_models = build_comparison_models(input_shape, num_classes)
        return comparison_models[model_name]
    
    # Handle standard models
    elif model_name in model_builders:
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
  %(prog)s --models resnet cnn1d transformer --mode train
  %(prog)s --models hybrid_complex_resnet lightweight_lstm fcnn --mode evaluate  
  %(prog)s --models cnn2d complex_nn adaboost --mode all --epochs 100
  %(prog)s --models comparison_models --mode train  # Will train all comparison models
  %(prog)s --models resnet --mode train --stratified_split  # Use stratified splitting by (modulation, SNR)
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
    
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--augment_data', action='store_true',
                        help='Enable data augmentation for training data (11 rotations, 30 deg increments)')
    
    parser.add_argument('--denoising_method', type=str, default='gpr',
                        choices=['gpr', 'wavelet', 'ddae', 'none', 'efficient_gpr', 'efficient_gpr_per_sample'],
                        help='Denoising method: gpr, wavelet, ddae, none, efficient_gpr (SNR-grouped), or efficient_gpr_per_sample (eigen-optimized)')
    
    parser.add_argument('--ddae_model_path', type=str, default='model_weight_saved/ddae_model.keras',
                        help='Path to the trained Denoising Autoencoder model (.keras file). Assumes path from project root.')
    
    parser.add_argument('--denoised_cache_dir', type=str, default='../denoised_datasets',
                        help='Directory to save/load cached denoised datasets')
    
    parser.add_argument('--stratified_split', action='store_true',
                        help='Use stratified splitting by both modulation type and SNR (ensures balanced distribution)')
    
    args = parser.parse_args()
    
    # Validate model selection
    if not validate_model_selection(args.models):
        return
    
    # Expand model selection (handle comparison_models, etc.)
    selected_models = expand_model_selection(args.models)
    
    print(f"Selected models for processing: {selected_models}")
    print(f"Total models selected: {len(selected_models)}")
    
    # Set random seed
    set_random_seed(args.random_seed)
    
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
        
        # Choose data preparation method based on stratified_split flag
        if args.stratified_split:
            print("Using stratified splitting by (modulation type, SNR) combinations...")
            X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_data_by_snr_stratified(
                dataset, 
                augment_data=args.augment_data,
                denoising_method=args.denoising_method,
                ddae_model_path=args.ddae_model_path,
                denoised_cache_dir=args.denoised_cache_dir
            )
        else:
            print("Using standard splitting by modulation type only...")
            X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_data_by_snr(
                dataset, 
                augment_data=args.augment_data,
                denoising_method=args.denoising_method,
                ddae_model_path=args.ddae_model_path,
                denoised_cache_dir=args.denoised_cache_dir
            )
    
    # Training
    if args.mode in ['train', 'all']:
        print(f"\n{'='*60}")
        print("TRAINING SELECTED MODELS")
        print(f"{'='*60}")
        
        # Generate file suffix based on denoising method, data augmentation, and stratification
        suffix = get_file_suffix(args.denoising_method, args.augment_data)
        if args.stratified_split:
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
        
        # Generate file suffix for evaluation (same as training)
        suffix = get_file_suffix(args.denoising_method, args.augment_data)
        if args.stratified_split:
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
