# -*- coding: utf-8 -*-
"""
SNR Distribution Analysis Script

This script analyzes the distribution of SNR values across training, validation, 
and test sets to understand how the data splitting affects different SNR levels.

Author: Generated for RadioML project analysis
Date: 2025-06-04
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# Add the src directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explore_dataset import load_radioml_data
from preprocess import prepare_data_by_snr


def analyze_snr_distribution(dataset_path='../RML2016.10a_dict.pkl', 
                            test_size=0.2, 
                            validation_split=0.1,
                            denoising_method='none',
                            augment_data=False,
                            output_dir='../output/analysis'):
    """
    Analyze the distribution of SNR values across train/val/test splits.
    
    Args:
        dataset_path (str): Path to the RadioML dataset
        test_size (float): Proportion of data for testing
        validation_split (float): Proportion of training data for validation
        denoising_method (str): Denoising method ('gpr', 'wavelet', 'ddae', 'none')
        augment_data (bool): Whether to apply data augmentation
        output_dir (str): Directory to save analysis results
    
    Returns:
        dict: Analysis results containing distribution statistics
    """
    
    print("="*60)
    print("SNR Distribution Analysis")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_radioml_data(dataset_path)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return None
    
    print("Dataset loaded successfully.")
    
    # Get original dataset statistics
    print("\nAnalyzing original dataset...")
    original_stats = analyze_original_dataset(dataset)
    
    # Prepare data using the same method as in main.py
    print(f"\nPreparing data with:")
    print(f"  - Test size: {test_size}")
    print(f"  - Validation split: {validation_split}")
    print(f"  - Denoising method: {denoising_method}")
    print(f"  - Data augmentation: {augment_data}")
    
    try:
        (X_train, X_val, X_test, 
         y_train, y_val, y_test, 
         snr_train, snr_val, snr_test, mods) = prepare_data_by_snr(
            dataset, 
            test_size=test_size,
            validation_split=validation_split,
            augment_data=augment_data,
            denoising_method=denoising_method
        )
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return None
    
    # Analyze splits
    print("\nAnalyzing data splits...")
    split_stats = analyze_splits(snr_train, snr_val, snr_test, mods, y_train, y_val, y_test)
    
    # Generate detailed report
    results = {
        'original_stats': original_stats,
        'split_stats': split_stats,
        'parameters': {
            'test_size': test_size,
            'validation_split': validation_split,
            'denoising_method': denoising_method,
            'augment_data': augment_data
        }
    }
    
    # Save results
    save_analysis_results(results, output_dir)
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    
    return results


def analyze_original_dataset(dataset):
    """Analyze the original dataset structure."""
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    snrs = sorted(list(set([k[1] for k in dataset.keys()])))
    
    total_samples = 0
    mod_counts = defaultdict(int)
    snr_counts = defaultdict(int)
    mod_snr_counts = defaultdict(lambda: defaultdict(int))
    
    for (mod, snr), data in dataset.items():
        count = len(data)
        total_samples += count
        mod_counts[mod] += count
        snr_counts[snr] += count
        mod_snr_counts[mod][snr] = count
    
    print(f"Original dataset statistics:")
    print(f"  - Total samples: {total_samples:,}")
    print(f"  - Number of modulations: {len(mods)}")
    print(f"  - Number of SNR levels: {len(snrs)}")
    print(f"  - SNR range: {min(snrs)} to {max(snrs)} dB")
    print(f"  - Samples per (modulation, SNR): {total_samples // (len(mods) * len(snrs))}")
    
    return {
        'total_samples': total_samples,
        'mods': mods,
        'snrs': snrs,
        'mod_counts': dict(mod_counts),
        'snr_counts': dict(snr_counts),
        'mod_snr_counts': dict(mod_snr_counts)
    }


def analyze_splits(snr_train, snr_val, snr_test, mods, y_train, y_val, y_test):
    """Analyze the distribution across train/val/test splits."""
    
    # Convert one-hot encoded labels back to class indices
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_idx = np.argmax(y_train, axis=1)
    else:
        y_train_idx = y_train
        
    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
        y_val_idx = np.argmax(y_val, axis=1)
    else:
        y_val_idx = y_val
        
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_idx = np.argmax(y_test, axis=1)
    else:
        y_test_idx = y_test
    
    # Get unique SNR values
    all_snrs = sorted(list(set(np.concatenate([snr_train, snr_val, snr_test]))))
    
    # Count samples per SNR in each split
    snr_distribution = {}
    
    for snr in all_snrs:
        train_count = np.sum(snr_train == snr)
        val_count = np.sum(snr_val == snr)
        test_count = np.sum(snr_test == snr)
        total_count = train_count + val_count + test_count
        
        if total_count > 0:
            train_ratio = train_count / total_count
            val_ratio = val_count / total_count
            test_ratio = test_count / total_count
        else:
            train_ratio = val_ratio = test_ratio = 0
        
        snr_distribution[snr] = {
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'total_count': total_count,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio
        }
    
    # Analyze modulation distribution within each split
    mod_distribution = {}
    
    for i, mod in enumerate(mods):
        # Train set
        train_mod_mask = y_train_idx == i
        train_mod_snrs = snr_train[train_mod_mask] if np.any(train_mod_mask) else np.array([])
        
        # Val set
        val_mod_mask = y_val_idx == i
        val_mod_snrs = snr_val[val_mod_mask] if np.any(val_mod_mask) else np.array([])
        
        # Test set
        test_mod_mask = y_test_idx == i
        test_mod_snrs = snr_test[test_mod_mask] if np.any(test_mod_mask) else np.array([])
        
        mod_distribution[mod] = {
            'train_snr_counts': Counter(train_mod_snrs.astype(int)),
            'val_snr_counts': Counter(val_mod_snrs.astype(int)),
            'test_snr_counts': Counter(test_mod_snrs.astype(int))
        }
    
    # Print summary statistics
    print(f"\nSplit Summary:")
    print(f"  - Training samples: {len(snr_train):,}")
    print(f"  - Validation samples: {len(snr_val):,}")
    print(f"  - Test samples: {len(snr_test):,}")
    print(f"  - Total samples: {len(snr_train) + len(snr_val) + len(snr_test):,}")
    
    # Print SNR distribution table
    print(f"\nSNR Distribution Across Splits:")
    print(f"{'SNR':>4} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8} {'Train%':>8} {'Val%':>6} {'Test%':>7}")
    print("-" * 60)
    
    for snr in all_snrs:
        stats = snr_distribution[snr]
        print(f"{snr:>4} {stats['train_count']:>8} {stats['val_count']:>8} "
              f"{stats['test_count']:>8} {stats['total_count']:>8} "
              f"{stats['train_ratio']*100:>7.1f} {stats['val_ratio']*100:>5.1f} "
              f"{stats['test_ratio']*100:>6.1f}")
    
    return {
        'snr_distribution': snr_distribution,
        'mod_distribution': mod_distribution,
        'all_snrs': all_snrs,
        'split_sizes': {
            'train': len(snr_train),
            'val': len(snr_val),
            'test': len(snr_test),
            'total': len(snr_train) + len(snr_val) + len(snr_test)
        }
    }


def save_analysis_results(results, output_dir):
    """Save analysis results to CSV files."""
    
    # Save SNR distribution
    snr_data = []
    for snr, stats in results['split_stats']['snr_distribution'].items():
        snr_data.append({
            'SNR': snr,
            'Train_Count': stats['train_count'],
            'Val_Count': stats['val_count'],
            'Test_Count': stats['test_count'],
            'Total_Count': stats['total_count'],
            'Train_Ratio': stats['train_ratio'],
            'Val_Ratio': stats['val_ratio'],
            'Test_Ratio': stats['test_ratio']
        })
    
    snr_df = pd.DataFrame(snr_data)
    snr_df.to_csv(os.path.join(output_dir, 'snr_distribution.csv'), index=False)
    print(f"SNR distribution saved to: {os.path.join(output_dir, 'snr_distribution.csv')}")
    
    # Save modulation-SNR distribution
    mod_snr_data = []
    for mod, stats in results['split_stats']['mod_distribution'].items():
        for snr in results['split_stats']['all_snrs']:
            train_count = stats['train_snr_counts'].get(snr, 0)
            val_count = stats['val_snr_counts'].get(snr, 0)
            test_count = stats['test_snr_counts'].get(snr, 0)
            total_count = train_count + val_count + test_count
            
            mod_snr_data.append({
                'Modulation': mod,
                'SNR': snr,
                'Train_Count': train_count,
                'Val_Count': val_count,
                'Test_Count': test_count,
                'Total_Count': total_count
            })
    
    mod_snr_df = pd.DataFrame(mod_snr_data)
    mod_snr_df.to_csv(os.path.join(output_dir, 'modulation_snr_distribution.csv'), index=False)
    print(f"Modulation-SNR distribution saved to: {os.path.join(output_dir, 'modulation_snr_distribution.csv')}")


def create_visualizations(results, output_dir):
    """Create visualization plots for the analysis."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. SNR distribution across splits
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    snrs = results['split_stats']['all_snrs']
    train_counts = [results['split_stats']['snr_distribution'][snr]['train_count'] for snr in snrs]
    val_counts = [results['split_stats']['snr_distribution'][snr]['val_count'] for snr in snrs]
    test_counts = [results['split_stats']['snr_distribution'][snr]['test_count'] for snr in snrs]
    
    # Absolute counts
    x = np.arange(len(snrs))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Train', alpha=0.8)
    ax1.bar(x, val_counts, width, label='Validation', alpha=0.8)
    ax1.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('Sample Counts by SNR Across Splits')
    ax1.set_xticks(x)
    ax1.set_xticklabels(snrs, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ratio plots
    train_ratios = [results['split_stats']['snr_distribution'][snr]['train_ratio'] for snr in snrs]
    val_ratios = [results['split_stats']['snr_distribution'][snr]['val_ratio'] for snr in snrs]
    test_ratios = [results['split_stats']['snr_distribution'][snr]['test_ratio'] for snr in snrs]
    
    ax2.bar(x - width, train_ratios, width, label='Train', alpha=0.8)
    ax2.bar(x, val_ratios, width, label='Validation', alpha=0.8)
    ax2.bar(x + width, test_ratios, width, label='Test', alpha=0.8)
    
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Proportion')
    ax2.set_title('Proportion of Samples by SNR Across Splits')
    ax2.set_xticks(x)
    ax2.set_xticklabels(snrs, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of modulation-SNR distribution
    mod_names = list(results['original_stats']['mods'])
    
    # Create matrices for each split
    train_matrix = np.zeros((len(mod_names), len(snrs)))
    val_matrix = np.zeros((len(mod_names), len(snrs)))
    test_matrix = np.zeros((len(mod_names), len(snrs)))
    
    for i, mod in enumerate(mod_names):
        for j, snr in enumerate(snrs):
            if mod in results['split_stats']['mod_distribution']:
                train_matrix[i, j] = results['split_stats']['mod_distribution'][mod]['train_snr_counts'].get(snr, 0)
                val_matrix[i, j] = results['split_stats']['mod_distribution'][mod]['val_snr_counts'].get(snr, 0)
                test_matrix[i, j] = results['split_stats']['mod_distribution'][mod]['test_snr_counts'].get(snr, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Train set heatmap
    sns.heatmap(train_matrix, xticklabels=snrs, yticklabels=mod_names, 
                annot=True, fmt='g', cmap='Blues', ax=axes[0])
    axes[0].set_title('Training Set: Samples per (Modulation, SNR)')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('Modulation')
    
    # Validation set heatmap
    sns.heatmap(val_matrix, xticklabels=snrs, yticklabels=mod_names, 
                annot=True, fmt='g', cmap='Greens', ax=axes[1])
    axes[1].set_title('Validation Set: Samples per (Modulation, SNR)')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Modulation')
    
    # Test set heatmap
    sns.heatmap(test_matrix, xticklabels=snrs, yticklabels=mod_names, 
                annot=True, fmt='g', cmap='Reds', ax=axes[2])
    axes[2].set_title('Test Set: Samples per (Modulation, SNR)')
    axes[2].set_xlabel('SNR (dB)')
    axes[2].set_ylabel('Modulation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'modulation_snr_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def main():
    """Main function to run the analysis."""
    
    # Default parameters (matching main.py defaults)
    dataset_path = '../RML2016.10a_dict.pkl'
    test_size = 0.2
    validation_split = 0.1
    denoising_method = 'none'  # Use 'none' for faster analysis
    augment_data = False
    output_dir = '../output/snr_analysis'
    
    # Run analysis
    results = analyze_snr_distribution(
        dataset_path=dataset_path,
        test_size=test_size,
        validation_split=validation_split,
        denoising_method=denoising_method,
        augment_data=augment_data,
        output_dir=output_dir
    )
    
    if results:
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        # Check if SNR distribution is balanced
        snr_stats = results['split_stats']['snr_distribution']
        train_ratios = [stats['train_ratio'] for stats in snr_stats.values()]
        val_ratios = [stats['val_ratio'] for stats in snr_stats.values()]
        test_ratios = [stats['test_ratio'] for stats in snr_stats.values()]
        
        print(f"Train ratio range: {min(train_ratios):.3f} - {max(train_ratios):.3f}")
        print(f"Val ratio range: {min(val_ratios):.3f} - {max(val_ratios):.3f}")
        print(f"Test ratio range: {min(test_ratios):.3f} - {max(test_ratios):.3f}")
        
        # Check if ratios are close to expected values
        expected_train = 1 - test_size - (1 - test_size) * validation_split
        expected_val = (1 - test_size) * validation_split
        expected_test = test_size
        
        print(f"\nExpected ratios: Train={expected_train:.3f}, Val={expected_val:.3f}, Test={expected_test:.3f}")
        
        train_deviation = max(abs(r - expected_train) for r in train_ratios)
        val_deviation = max(abs(r - expected_val) for r in val_ratios)
        test_deviation = max(abs(r - expected_test) for r in test_ratios)
        
        print(f"Maximum deviations: Train={train_deviation:.3f}, Val={val_deviation:.3f}, Test={test_deviation:.3f}")
        
        if max(train_deviation, val_deviation, test_deviation) > 0.05:
            print("\n⚠️  WARNING: Significant deviation from expected ratios detected!")
            print("   This indicates that SNR values are not evenly distributed across splits.")
        else:
            print("\n✅ SNR distribution appears reasonably balanced across splits.")


if __name__ == "__main__":
    main()
