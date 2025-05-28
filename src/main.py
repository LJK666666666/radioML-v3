import os
import argparse
import time
import random
import numpy as np
import tensorflow as tf # Added import for tf.keras.models.load_model

# Import project modules
from explore_dataset import load_radioml_data, explore_dataset, plot_signal_examples
from preprocess import prepare_data, prepare_data_by_snr
from train import train_model, plot_training_history # MODIFIED: Added plot_training_history
from models import build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model, build_transformer_model, build_lstm_model, build_advanced_lstm_model, build_multi_scale_lstm_model, build_lightweight_lstm_model, build_hybrid_complex_resnet_model, build_lightweight_hybrid_model, build_hybrid_transition_resnet_model, build_lightweight_transition_model, build_comparison_models, build_keras_adaboost_model, build_lightweight_adaboost_model, build_fcnn_model, build_deep_fcnn_model, build_lightweight_fcnn_model, build_wide_fcnn_model, build_shallow_fcnn_model, build_custom_fcnn_model, get_callbacks
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

# 设置随机种子以确保实验可重复性
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)  # Add this line
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


def main():
    parser = argparse.ArgumentParser(description='RadioML Signal Classification')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['explore', 'train', 'evaluate', 'all'],
                        help='Mode of operation')
    parser.add_argument('--model_type', type=str, default='resnet',
                        choices=['cnn1d', 'cnn2d', 'resnet', 'complex_nn', 'transformer', 
                                'lstm', 'advanced_lstm', 'multi_scale_lstm', 'lightweight_lstm',
                                'hybrid_complex_resnet', 'lightweight_hybrid', 
                                'hybrid_transition_resnet', 'lightweight_transition', 
                                'comparison_models', 'adaboost', 'lightweight_adaboost',
                                'fcnn', 'deep_fcnn', 'lightweight_fcnn', 'wide_fcnn', 'shallow_fcnn', 'all'],
                        help='Model architecture to use')
    parser.add_argument('--dataset_path', type=str, default='../RML2016.10a_dict.pkl',
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
                        choices=['gpr', 'wavelet', 'ddae', 'none'],
                        help='Denoising method to apply to the signals (gpr, wavelet, ddae, none)')
    parser.add_argument('--ddae_model_path', type=str, default='model_weight_saved/ddae_model.keras',
                        help='Path to the trained Denoising Autoencoder model (.keras file). Assumes path from project root.')
    parser.add_argument('--denoised_cache_dir', type=str, default='../denoised_datasets',
                        help='Directory to save/load cached denoised datasets')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.random_seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'training_plots') # Create a new directory for plots
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    start_time = time.time()
    dataset = load_radioml_data(args.dataset_path)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    
    # Explore dataset
    if args.mode in ['explore', 'all']:
        print("\n" + "="*50)
        print("Exploring Dataset")
        print("="*50)
        mods, snrs = explore_dataset(dataset)
        plot_signal_examples(dataset, mods, os.path.join(args.output_dir, 'exploration'))
    
    # Prepare data
    if args.mode in ['train', 'evaluate', 'all']:
        print("\n" + "="*50)
        print("Preparing Data")
        print("="*50)
        X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_data_by_snr(
            dataset, 
            augment_data=args.augment_data,
            denoising_method=args.denoising_method,
            ddae_model_path=args.ddae_model_path,
            denoised_cache_dir=args.denoised_cache_dir
        )
    
    # Training
    if args.mode in ['train', 'all']:
        print("\n" + "="*50)
        print("Training Models")
        print("="*50)
        
        # Generate file suffix based on denoising method and data augmentation
        suffix = get_file_suffix(args.denoising_method, args.augment_data)
        
        input_shape = X_train.shape[1:]
        num_classes = len(mods)
        
        # Train the selected model(s)
        if args.model_type in ['cnn1d', 'all']:
            print("\nTraining CNN1D Model...")
            cnn1d_model = build_cnn1d_model(input_shape, num_classes)
            cnn1d_model.summary()
            history_cnn1d = train_model( # Capture history
                cnn1d_model, 
                X_train, y_train, 
                X_val, y_val, 
                os.path.join(models_dir, f"cnn1d_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_cnn1d,
                os.path.join(plots_dir, f"cnn1d_training_history{suffix}.png")
            )
        
        if args.model_type in ['cnn2d', 'all']:
            print("\nTraining CNN2D Model...")
            # Reshape data for 2D model
            X_train_2d = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
            X_val_2d = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
            
            cnn2d_model = build_cnn2d_model(input_shape, num_classes)
            cnn2d_model.summary()
            history_cnn2d = train_model( # Capture history
                cnn2d_model, 
                X_train_2d, y_train, 
                X_val_2d, y_val, 
                os.path.join(models_dir, f"cnn2d_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_cnn2d,
                os.path.join(plots_dir, f"cnn2d_training_history{suffix}.png")
            )
        
        if args.model_type in ['resnet', 'all']:
            print("\nTraining ResNet Model...")
            resnet_model = build_resnet_model(input_shape, num_classes)
            resnet_model.summary()
            history_resnet = train_model( # Capture history
                resnet_model, 
                X_train, y_train, 
                X_val, y_val, 
                os.path.join(models_dir, f"resnet_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_resnet,
                os.path.join(plots_dir, f"resnet_training_history{suffix}.png")
            )
        
        if args.model_type in ['complex_nn', 'all']:
            print("\nTraining ComplexNN Model...")
            complex_nn_model = build_complex_nn_model(input_shape, num_classes)
            complex_nn_model.summary()
            history_complex_nn = train_model( # Capture history
                complex_nn_model, 
                X_train, y_train, 
                X_val, y_val, 
                os.path.join(models_dir, f"complex_nn_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_complex_nn,
                os.path.join(plots_dir, f"complex_nn_training_history{suffix}.png")
            )

        if args.model_type in ['transformer', 'all']:
            print("\nTraining Transformer Model...")
            transformer_model = build_transformer_model(input_shape, num_classes)
            transformer_model.summary()
            history_transformer = train_model( # Capture history
                transformer_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"transformer_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_transformer,
                os.path.join(plots_dir, f"transformer_training_history{suffix}.png")
            )

        if args.model_type in ['lstm', 'all']:
            print("\nTraining LSTM Model...")
            lstm_model = build_lstm_model(input_shape, num_classes)
            lstm_model.summary()
            history_lstm = train_model( # Capture history
                lstm_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"lstm_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_lstm,
                os.path.join(plots_dir, f"lstm_training_history{suffix}.png")
            )

        if args.model_type in ['advanced_lstm', 'all']:
            print("\nTraining Advanced LSTM Model...")
            advanced_lstm_model = build_advanced_lstm_model(input_shape, num_classes)
            advanced_lstm_model.summary()
            history_advanced_lstm = train_model( # Capture history
                advanced_lstm_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"advanced_lstm_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_advanced_lstm,
                os.path.join(plots_dir, f"advanced_lstm_training_history{suffix}.png")
            )

        if args.model_type in ['multi_scale_lstm', 'all']:
            print("\nTraining Multi-Scale LSTM Model...")
            multi_scale_lstm_model = build_multi_scale_lstm_model(input_shape, num_classes)
            multi_scale_lstm_model.summary()
            history_multi_scale_lstm = train_model( # Capture history
                multi_scale_lstm_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"multi_scale_lstm_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_multi_scale_lstm,
                os.path.join(plots_dir, f"multi_scale_lstm_training_history{suffix}.png")
            )

        if args.model_type in ['lightweight_lstm', 'all']:
            print("\nTraining Lightweight LSTM Model...")
            lightweight_lstm_model = build_lightweight_lstm_model(input_shape, num_classes)
            lightweight_lstm_model.summary()
            history_lightweight_lstm = train_model( # Capture history
                lightweight_lstm_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"lightweight_lstm_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_lightweight_lstm,
                os.path.join(plots_dir, f"lightweight_lstm_training_history{suffix}.png")
            )
    
        if args.model_type in ['hybrid_complex_resnet', 'all']:
            print("\nTraining Hybrid Complex ResNet Model...")
            hybrid_complex_resnet_model = build_hybrid_complex_resnet_model(input_shape, num_classes)
            hybrid_complex_resnet_model.summary()
            history_hybrid_complex_resnet = train_model( # Capture history
                hybrid_complex_resnet_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"hybrid_complex_resnet_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_hybrid_complex_resnet,
                os.path.join(plots_dir, f"hybrid_complex_resnet_training_history{suffix}.png")
            )

        if args.model_type in ['lightweight_hybrid', 'all']:
            print("\nTraining Lightweight Hybrid Model...")
            lightweight_hybrid_model = build_lightweight_hybrid_model(input_shape, num_classes)
            lightweight_hybrid_model.summary()
            history_lightweight_hybrid = train_model( # Capture history
                lightweight_hybrid_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"lightweight_hybrid_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_lightweight_hybrid,
                os.path.join(plots_dir, f"lightweight_hybrid_training_history{suffix}.png")
            )
    
        if args.model_type in ['hybrid_transition_resnet', 'all']:
            print("\nTraining Hybrid Transition ResNet Model...")
            hybrid_transition_resnet_model = build_hybrid_transition_resnet_model(input_shape, num_classes)
            hybrid_transition_resnet_model.summary()
            history_hybrid_transition_resnet = train_model( # Capture history
                hybrid_transition_resnet_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"hybrid_transition_resnet_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_hybrid_transition_resnet,
                os.path.join(plots_dir, f"hybrid_transition_resnet_training_history{suffix}.png")
            )

        if args.model_type in ['lightweight_transition', 'all']:
            print("\nTraining Lightweight Transition Model...")
            lightweight_transition_model = build_lightweight_transition_model(input_shape, num_classes)
            lightweight_transition_model.summary()
            history_lightweight_transition = train_model( # Capture history
                lightweight_transition_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"lightweight_transition_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_lightweight_transition,
                os.path.join(plots_dir, f"lightweight_transition_training_history{suffix}.png")
            )

        if args.model_type in ['comparison_models', 'all']:
            print("\nTraining Comparison Models...")
            comparison_models = build_comparison_models(input_shape, num_classes)
            
            for model_name, model in comparison_models.items():
                print(f"\nTraining {model_name} model...")
                model.summary()
                history = train_model( # Capture history
                    model,
                    X_train, y_train,
                    X_val, y_val,
                    os.path.join(models_dir, f"{model_name}_model{suffix}.keras"),
                    batch_size=args.batch_size,
                    epochs=args.epochs
                )
                plot_training_history( # Plot and save history
                    history,
                    os.path.join(plots_dir, f"{model_name}_training_history{suffix}.png")
                )

        if args.model_type in ['adaboost', 'all']:
            print("\nTraining AdaBoost Model...")
            adaboost_model = build_keras_adaboost_model(input_shape, num_classes, n_estimators=10, learning_rate=1.0)
            adaboost_model.summary()
            history_adaboost = train_model( # Capture history
                adaboost_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"adaboost_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_adaboost,
                os.path.join(plots_dir, f"adaboost_training_history{suffix}.png")
            )

        if args.model_type in ['lightweight_adaboost', 'all']:
            print("\nTraining Lightweight AdaBoost Model...")
            lightweight_adaboost_model = build_keras_adaboost_model(input_shape, num_classes, n_estimators=5, learning_rate=0.5)
            lightweight_adaboost_model.summary()
            history_lightweight_adaboost = train_model( # Capture history
                lightweight_adaboost_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"lightweight_adaboost_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_lightweight_adaboost,
                os.path.join(plots_dir, f"lightweight_adaboost_training_history{suffix}.png")
            )

        # FCNN Models Training
        if args.model_type in ['fcnn', 'all']:
            print("\nTraining FCNN Model...")
            fcnn_model = build_fcnn_model(input_shape, num_classes)
            fcnn_model.summary()
            history_fcnn = train_model( # Capture history
                fcnn_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"fcnn_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_fcnn,
                os.path.join(plots_dir, f"fcnn_training_history{suffix}.png")
            )

        if args.model_type in ['deep_fcnn', 'all']:
            print("\nTraining Deep FCNN Model...")
            deep_fcnn_model = build_deep_fcnn_model(input_shape, num_classes)
            deep_fcnn_model.summary()
            history_deep_fcnn = train_model( # Capture history
                deep_fcnn_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, f"deep_fcnn_model{suffix}.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_deep_fcnn,
                os.path.join(plots_dir, f"deep_fcnn_training_history{suffix}.png")
            )

        if args.model_type in ['lightweight_fcnn', 'all']:
            print("\nTraining Lightweight FCNN Model...")
            lightweight_fcnn_model = build_lightweight_fcnn_model(input_shape, num_classes)
            lightweight_fcnn_model.summary()
            history_lightweight_fcnn = train_model( # Capture history
                lightweight_fcnn_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, "lightweight_fcnn_model.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_lightweight_fcnn,
                os.path.join(plots_dir, "lightweight_fcnn_training_history.png")
            )

        if args.model_type in ['wide_fcnn', 'all']:
            print("\nTraining Wide FCNN Model...")
            wide_fcnn_model = build_wide_fcnn_model(input_shape, num_classes)
            wide_fcnn_model.summary()
            history_wide_fcnn = train_model( # Capture history
                wide_fcnn_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, "wide_fcnn_model.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_wide_fcnn,
                os.path.join(plots_dir, "wide_fcnn_training_history.png")
            )

        if args.model_type in ['shallow_fcnn', 'all']:
            print("\nTraining Shallow FCNN Model...")
            shallow_fcnn_model = build_shallow_fcnn_model(input_shape, num_classes)
            shallow_fcnn_model.summary()
            history_shallow_fcnn = train_model( # Capture history
                shallow_fcnn_model,
                X_train, y_train,
                X_val, y_val,
                os.path.join(models_dir, "shallow_fcnn_model.keras"),
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            plot_training_history( # Plot and save history
                history_shallow_fcnn,
                os.path.join(plots_dir, "shallow_fcnn_training_history.png")
            )
    
    # Evaluation
    if args.mode in ['evaluate', 'all']:
        print("\n" + "="*50)
        print("Evaluating Models")
        print("="*50)
        
        # Generate file suffix for evaluation (same as training)
        suffix = get_file_suffix(args.denoising_method, args.augment_data)
        
        # Ensure X_test, y_test, snr_test, mods are available
        # They are prepared if mode is 'train', 'evaluate', or 'all'
        
        if args.model_type in ['cnn1d', 'all']:
            model_path = os.path.join(models_dir, f"cnn1d_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating CNN1D Model...")
                try:
                    cnn1d_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        cnn1d_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'cnn1d_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")
        
        if args.model_type in ['cnn2d', 'all']:
            model_path = os.path.join(models_dir, f"cnn2d_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating CNN2D Model...")
                # X_test is used directly as build_cnn2d_model handles reshape internally
                try:
                    cnn2d_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        cnn2d_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'cnn2d_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")
        
        if args.model_type in ['resnet', 'all']:
            model_path = os.path.join(models_dir, f"resnet_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating ResNet Model...")
                try:
                    resnet_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        resnet_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'resnet_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['complex_nn', 'all']:
            model_path = os.path.join(models_dir, f"complex_nn_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating ComplexNN Model...")
                try:
                    # Enable unsafe deserialization to load models with Lambda layers
                    tf.keras.config.enable_unsafe_deserialization()
                    
                    # Create custom objects dict for complex layers
                    custom_objects = {
                        'ComplexConv1D': ComplexConv1D,
                        'ComplexBatchNormalization': ComplexBatchNormalization,
                        'ComplexDense': ComplexDense,
                        'ComplexMagnitude': ComplexMagnitude,
                        'ComplexActivation': ComplexActivation,
                        'ComplexPooling1D': ComplexPooling1D,
                        'complex_relu': complex_relu,
                        'mod_relu': mod_relu,
                        'zrelu': zrelu,
                        'crelu': crelu,
                        'cardioid': cardioid,
                        'complex_tanh': complex_tanh,
                        'phase_amplitude_activation': phase_amplitude_activation,
                        # Original style activations
                        'complex_elu': complex_elu,
                        'complex_leaky_relu': complex_leaky_relu,
                        'complex_swish': complex_swish,
                        'real_imag_mixed_relu': real_imag_mixed_relu
                    }
                    
                    complex_nn_eval_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        complex_nn_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'complex_nn_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['transformer', 'all']:
            model_path = os.path.join(models_dir, f"transformer_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Transformer Model...")
                try:
                    transformer_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        transformer_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'transformer_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['hybrid_complex_resnet', 'all']:
            model_path = os.path.join(models_dir, f"hybrid_complex_resnet_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Hybrid Complex ResNet Model...")
                try:
                    # Enable unsafe deserialization to load models with custom layers
                    tf.keras.config.enable_unsafe_deserialization()
                    
                    # Create custom objects dict for hybrid complex layers
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
                    
                    hybrid_complex_resnet_eval_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        hybrid_complex_resnet_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'hybrid_complex_resnet_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['lightweight_hybrid', 'all']:
            model_path = os.path.join(models_dir, f"lightweight_hybrid_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Lightweight Hybrid Model...")
                try:
                    # Enable unsafe deserialization to load models with custom layers
                    tf.keras.config.enable_unsafe_deserialization()
                    
                    # Create custom objects dict for hybrid layers
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
                    
                    lightweight_hybrid_eval_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        lightweight_hybrid_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'lightweight_hybrid_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")
    
        if args.model_type in ['hybrid_transition_resnet', 'all']:
            model_path = os.path.join(models_dir, f"hybrid_transition_resnet_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Hybrid Transition ResNet Model...")
                try:
                    # Enable unsafe deserialization to load models with custom layers
                    tf.keras.config.enable_unsafe_deserialization()
                    
                    # Create custom objects dict for hybrid transition layers
                    custom_objects = {
                        'ComplexConv1D': ComplexConv1D,
                        'ComplexBatchNormalization': ComplexBatchNormalization,
                        'ComplexDense': ComplexDense,
                        'ComplexMagnitude': ComplexMagnitude,
                        'ComplexActivation': ComplexActivation,
                        'ComplexPooling1D': ComplexPooling1D,
                        'ComplexResidualBlock': ComplexResidualBlock,
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
                    
                    hybrid_transition_resnet_eval_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        hybrid_transition_resnet_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'hybrid_transition_resnet_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['lightweight_transition', 'all']:
            model_path = os.path.join(models_dir, f"lightweight_transition_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Lightweight Transition Model...")
                try:
                    # Enable unsafe deserialization to load models with custom layers
                    tf.keras.config.enable_unsafe_deserialization()
                    
                    # Create custom objects dict for hybrid transition layers
                    custom_objects = {
                        'ComplexConv1D': ComplexConv1D,
                        'ComplexBatchNormalization': ComplexBatchNormalization,
                        'ComplexDense': ComplexDense,
                        'ComplexMagnitude': ComplexMagnitude,
                        'ComplexActivation': ComplexActivation,
                        'ComplexPooling1D': ComplexPooling1D,
                        'ComplexResidualBlock': ComplexResidualBlock,
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
                    
                    lightweight_transition_eval_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        lightweight_transition_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'lightweight_transition_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        # LSTM Models Evaluation
        if args.model_type in ['lstm', 'all']:
            model_path = os.path.join(models_dir, f"lstm_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating LSTM Model...")
                try:
                    lstm_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        lstm_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'lstm_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['advanced_lstm', 'all']:
            model_path = os.path.join(models_dir, f"advanced_lstm_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Advanced LSTM Model...")
                try:
                    advanced_lstm_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        advanced_lstm_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'advanced_lstm_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['multi_scale_lstm', 'all']:
            model_path = os.path.join(models_dir, f"multi_scale_lstm_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Multi-Scale LSTM Model...")
                try:
                    multi_scale_lstm_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        multi_scale_lstm_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'multi_scale_lstm_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['lightweight_lstm', 'all']:
            model_path = os.path.join(models_dir, f"lightweight_lstm_model{suffix}.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Lightweight LSTM Model...")
                try:
                    lightweight_lstm_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        lightweight_lstm_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'lightweight_lstm_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['comparison_models', 'all']:
            print("\nEvaluating Comparison Models...")
            comparison_model_names = ['high_complex', 'medium_complex', 'low_complex']
            
            for model_name in comparison_model_names:
                model_path = os.path.join(models_dir, f"{model_name}_model{suffix}.keras")
                if os.path.exists(model_path):
                    print(f"\nEvaluating {model_name} Model...")
                    try:
                        # Enable unsafe deserialization to load models with custom layers
                        tf.keras.config.enable_unsafe_deserialization()
                        
                        # Create custom objects dict for hybrid transition layers
                        custom_objects = {
                            'ComplexConv1D': ComplexConv1D,
                            'ComplexBatchNormalization': ComplexBatchNormalization,
                            'ComplexDense': ComplexDense,
                            'ComplexMagnitude': ComplexMagnitude,
                            'ComplexActivation': ComplexActivation,
                            'ComplexPooling1D': ComplexPooling1D,
                            'ComplexResidualBlock': ComplexResidualBlock,
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
                        
                        comparison_eval_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                        print(f"Successfully loaded model from {model_path}")
                        evaluate_by_snr(
                            comparison_eval_model,
                            X_test, y_test, snr_test, mods,
                            os.path.join(results_dir, f'{model_name}_evaluation_results')
                        )
                    except Exception as e:
                        print(f"Error loading or evaluating model {model_path}: {e}")
                else:
                    print(f"Model {model_path} not found for evaluation.")

        # AdaBoost Models Evaluation (Special handling for pickle files)
        if args.model_type in ['adaboost', 'all']:
            model_path = os.path.join(models_dir, "adaboost_model_adaboost.pkl")
            if os.path.exists(model_path):
                print("\nEvaluating AdaBoost Model...")
                try:
                    from model.adaboost_model import AdaBoostClassifier
                    adaboost_eval_model = AdaBoostClassifier(input_shape, num_classes)
                    adaboost_eval_model.load(model_path)
                    print(f"Successfully loaded AdaBoost model from {model_path}")
                    
                    # Create a wrapper for compatibility with evaluate_by_snr
                    class AdaBoostWrapper:
                        def __init__(self, adaboost_model):
                            self.model = adaboost_model
                        
                        def predict(self, X):
                            return self.model.predict(X)
                    
                    wrapper = AdaBoostWrapper(adaboost_eval_model)
                    evaluate_by_snr(
                        wrapper,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'adaboost_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating AdaBoost model {model_path}: {e}")
            else:
                print(f"AdaBoost model {model_path} not found for evaluation.")

        if args.model_type in ['lightweight_adaboost', 'all']:
            model_path = os.path.join(models_dir, "lightweight_adaboost_model_adaboost.pkl")
            if os.path.exists(model_path):
                print("\nEvaluating Lightweight AdaBoost Model...")
                try:
                    from model.adaboost_model import AdaBoostClassifier
                    lightweight_adaboost_eval_model = AdaBoostClassifier(input_shape, num_classes)
                    lightweight_adaboost_eval_model.load(model_path)
                    print(f"Successfully loaded Lightweight AdaBoost model from {model_path}")
                    
                    # Create a wrapper for compatibility with evaluate_by_snr
                    class AdaBoostWrapper:
                        def __init__(self, adaboost_model):
                            self.model = adaboost_model
                        
                        def predict(self, X):
                            return self.model.predict(X)
                    
                    wrapper = AdaBoostWrapper(lightweight_adaboost_eval_model)
                    evaluate_by_snr(
                        wrapper,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'lightweight_adaboost_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating Lightweight AdaBoost model {model_path}: {e}")
            else:
                print(f"Lightweight AdaBoost model {model_path} not found for evaluation.")

        # FCNN Models Evaluation
        if args.model_type in ['fcnn', 'all']:
            model_path = os.path.join(models_dir, "fcnn_model.keras")
            if os.path.exists(model_path):
                print("\nEvaluating FCNN Model...")
                try:
                    fcnn_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        fcnn_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'fcnn_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['deep_fcnn', 'all']:
            model_path = os.path.join(models_dir, "deep_fcnn_model.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Deep FCNN Model...")
                try:
                    deep_fcnn_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        deep_fcnn_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'deep_fcnn_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['lightweight_fcnn', 'all']:
            model_path = os.path.join(models_dir, "lightweight_fcnn_model.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Lightweight FCNN Model...")
                try:
                    lightweight_fcnn_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        lightweight_fcnn_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'lightweight_fcnn_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['wide_fcnn', 'all']:
            model_path = os.path.join(models_dir, "wide_fcnn_model.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Wide FCNN Model...")
                try:
                    wide_fcnn_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        wide_fcnn_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'wide_fcnn_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")

        if args.model_type in ['shallow_fcnn', 'all']:
            model_path = os.path.join(models_dir, "shallow_fcnn_model.keras")
            if os.path.exists(model_path):
                print("\nEvaluating Shallow FCNN Model...")
                try:
                    shallow_fcnn_eval_model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded model from {model_path}")
                    evaluate_by_snr(
                        shallow_fcnn_eval_model,
                        X_test, y_test, snr_test, mods,
                        os.path.join(results_dir, 'shallow_fcnn_evaluation_results')
                    )
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
            else:
                print(f"Model {model_path} not found for evaluation.")
    
    print("\nAll operations completed successfully!")


if __name__ == "__main__":

    set_random_seed(42)

    main()