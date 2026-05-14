#!/usr/bin/env python3
"""
SNR Split Analysis Tool

This script analyzes the distribution of SNR values across train/validation/test splits
to evaluate whether the current data splitting strategy maintains balanced SNR distribution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from preprocess import load_data, prepare_data_by_snr


def analyze_snr_distribution(snr_train, snr_val, snr_test, y_train, y_val, y_test, mods):
    """
    Analyze SNR distribution across train/validation/test splits.
    
    Args:
        snr_train, snr_val, snr_test: SNR arrays for each split
        y_train, y_val, y_test: Label arrays for each split (one-hot encoded)
        mods: List of modulation types
        
    Returns:
        dict: Analysis results
    """
    # Convert one-hot back to class indices
    y_train_idx = np.argmax(y_train, axis=1) if y_train.size > 0 else np.array([])
    y_val_idx = np.argmax(y_val, axis=1) if y_val.size > 0 else np.array([])
    y_test_idx = np.argmax(y_test, axis=1) if y_test.size > 0 else np.array([])
    
    # Get unique SNR values
    all_snrs = np.concatenate([snr_train, snr_val, snr_test]) if snr_val.size > 0 else np.concatenate([snr_train, snr_test])
    unique_snrs = sorted(np.unique(all_snrs))
    
    # Initialize results
    results = {
        'snr_distribution': defaultdict(dict),
        'modulation_snr_distribution': defaultdict(lambda: defaultdict(dict)),
        'summary_stats': {},
        'unique_snrs': unique_snrs
    }
    
    # Count samples for each SNR in each split
    for snr_value in unique_snrs:
        train_count = np.sum(snr_train == snr_value)
        val_count = np.sum(snr_val == snr_value) if snr_val.size > 0 else 0
        test_count = np.sum(snr_test == snr_value)
        total_count = train_count + val_count + test_count
        
        if total_count > 0:
            results['snr_distribution'][snr_value] = {
                'train_count': train_count,
                'val_count': val_count,
                'test_count': test_count,
                'total_count': total_count,
                'train_ratio': train_count / total_count,
                'val_ratio': val_count / total_count,
                'test_ratio': test_count / total_count
            }
    
    # Count samples for each (modulation, SNR) combination
    for mod_idx, mod_name in enumerate(mods):
        for snr_value in unique_snrs:
            train_mask = (snr_train == snr_value) & (y_train_idx == mod_idx)
            val_mask = (snr_val == snr_value) & (y_val_idx == mod_idx) if snr_val.size > 0 and y_val_idx.size > 0 else np.array([False])
            test_mask = (snr_test == snr_value) & (y_test_idx == mod_idx)
            
            train_count = np.sum(train_mask)
            val_count = np.sum(val_mask) if val_mask.size > 0 else 0
            test_count = np.sum(test_mask)
            total_count = train_count + val_count + test_count
            
            if total_count > 0:
                results['modulation_snr_distribution'][mod_name][snr_value] = {
                    'train_count': train_count,
                    'val_count': val_count,
                    'test_count': test_count,
                    'total_count': total_count,
                    'train_ratio': train_count / total_count,
                    'val_ratio': val_count / total_count,
                    'test_ratio': test_count / total_count
                }
    
    # Calculate summary statistics
    train_ratios = [results['snr_distribution'][snr]['train_ratio'] for snr in unique_snrs]
    val_ratios = [results['snr_distribution'][snr]['val_ratio'] for snr in unique_snrs]
    test_ratios = [results['snr_distribution'][snr]['test_ratio'] for snr in unique_snrs]
    
    results['summary_stats'] = {
        'train_ratio_mean': np.mean(train_ratios),
        'train_ratio_std': np.std(train_ratios),
        'val_ratio_mean': np.mean(val_ratios),
        'val_ratio_std': np.std(val_ratios),
        'test_ratio_mean': np.mean(test_ratios),
        'test_ratio_std': np.std(test_ratios),
        'total_samples': {
            'train': len(snr_train),
            'val': len(snr_val) if snr_val.size > 0 else 0,
            'test': len(snr_test)
        }
    }
    
    return results


def print_snr_analysis(results):
    """Print detailed SNR distribution analysis."""
    print("=" * 80)
    print("SNR DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Summary statistics
    stats = results['summary_stats']
    print(f"\nOVERALL SPLIT RATIOS:")
    print(f"Total Samples - Train: {stats['total_samples']['train']}, Val: {stats['total_samples']['val']}, Test: {stats['total_samples']['test']}")
    total_all = sum(stats['total_samples'].values())
    if total_all > 0:
        print(f"Overall Ratios - Train: {stats['total_samples']['train']/total_all:.3f}, "
              f"Val: {stats['total_samples']['val']/total_all:.3f}, "
              f"Test: {stats['total_samples']['test']/total_all:.3f}")
    
    print(f"\nSNR-WISE SPLIT RATIO STATISTICS:")
    print(f"Train Ratio - Mean: {stats['train_ratio_mean']:.3f} ± {stats['train_ratio_std']:.3f}")
    print(f"Val Ratio   - Mean: {stats['val_ratio_mean']:.3f} ± {stats['val_ratio_std']:.3f}")
    print(f"Test Ratio  - Mean: {stats['test_ratio_mean']:.3f} ± {stats['test_ratio_std']:.3f}")
    
    # Per-SNR breakdown
    print(f"\nPER-SNR DISTRIBUTION:")
    print(f"{'SNR':<6} {'Train':<8} {'Val':<6} {'Test':<6} {'Total':<8} {'Train%':<8} {'Val%':<6} {'Test%':<6}")
    print("-" * 70)
    
    for snr_val in results['unique_snrs']:
        dist = results['snr_distribution'][snr_val]
        print(f"{snr_val:<6} {dist['train_count']:<8} {dist['val_count']:<6} {dist['test_count']:<6} "
              f"{dist['total_count']:<8} {dist['train_ratio']:<8.3f} {dist['val_ratio']:<6.3f} {dist['test_ratio']:<6.3f}")


def create_snr_distribution_plots(results, output_dir='../../output/test_results'):
    """Create visualization plots for SNR distribution."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: SNR distribution across splits
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    snrs = results['unique_snrs']
    train_counts = [results['snr_distribution'][snr]['train_count'] for snr in snrs]
    val_counts = [results['snr_distribution'][snr]['val_count'] for snr in snrs]
    test_counts = [results['snr_distribution'][snr]['test_count'] for snr in snrs]
    
    # Absolute counts
    x = np.arange(len(snrs))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Train', alpha=0.8)
    ax1.bar(x, val_counts, width, label='Validation', alpha=0.8)
    ax1.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('Sample Count Distribution Across SNRs')
    ax1.set_xticks(x)
    ax1.set_xticklabels(snrs, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Relative ratios
    train_ratios = [results['snr_distribution'][snr]['train_ratio'] for snr in snrs]
    val_ratios = [results['snr_distribution'][snr]['val_ratio'] for snr in snrs]
    test_ratios = [results['snr_distribution'][snr]['test_ratio'] for snr in snrs]
    
    ax2.bar(x - width, train_ratios, width, label='Train', alpha=0.8)
    ax2.bar(x, val_ratios, width, label='Validation', alpha=0.8)
    ax2.bar(x + width, test_ratios, width, label='Test', alpha=0.8)
    
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Split Ratio')
    ax2.set_title('Split Ratio Distribution Across SNRs')
    ax2.set_xticks(x)
    ax2.set_xticklabels(snrs, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/snr_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SNR distribution plot saved to: {output_dir}/snr_distribution_analysis.png")


def save_detailed_results(results, output_dir='../../output/test_results'):
    """Save detailed analysis results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # SNR distribution summary
    snr_data = []
    for snr_val in results['unique_snrs']:
        dist = results['snr_distribution'][snr_val]
        snr_data.append({
            'SNR': snr_val,
            'Train_Count': dist['train_count'],
            'Val_Count': dist['val_count'],
            'Test_Count': dist['test_count'],
            'Total_Count': dist['total_count'],
            'Train_Ratio': dist['train_ratio'],
            'Val_Ratio': dist['val_ratio'],
            'Test_Ratio': dist['test_ratio']
        })
    
    snr_df = pd.DataFrame(snr_data)
    snr_csv_path = f'{output_dir}/snr_distribution_summary.csv'
    snr_df.to_csv(snr_csv_path, index=False)
    print(f"SNR distribution summary saved to: {snr_csv_path}")
    
    # Modulation-SNR combination details
    mod_snr_data = []
    for mod_name, snr_dict in results['modulation_snr_distribution'].items():
        for snr_val, dist in snr_dict.items():
            mod_snr_data.append({
                'Modulation': mod_name,
                'SNR': snr_val,
                'Train_Count': dist['train_count'],
                'Val_Count': dist['val_count'],
                'Test_Count': dist['test_count'],
                'Total_Count': dist['total_count'],
                'Train_Ratio': dist['train_ratio'],
                'Val_Ratio': dist['val_ratio'],
                'Test_Ratio': dist['test_ratio']
            })
    
    mod_snr_df = pd.DataFrame(mod_snr_data)
    mod_snr_csv_path = f'{output_dir}/modulation_snr_distribution_details.csv'
    mod_snr_df.to_csv(mod_snr_csv_path, index=False)
    print(f"Modulation-SNR distribution details saved to: {mod_snr_csv_path}")


def main():
    """Main analysis function."""
    print("Starting SNR Split Analysis...")
    
    # Load dataset
    data_path = "../../data/RML2016.10a_dict.pkl"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure the dataset file exists at the specified path.")
        return
    
    try:
        dataset = load_data(data_path)
        print(f"Dataset loaded successfully from {data_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Prepare data using current splitting strategy
    print("Preparing data with current splitting strategy...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_data_by_snr(
            dataset, 
            test_size=0.2, 
            validation_split=0.1, 
            specific_snrs=None,
            augment_data=False,
            denoising_method='none'  # Skip denoising for faster analysis
        )
        print("Data preparation completed.")
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return
    
    # Analyze SNR distribution
    print("Analyzing SNR distribution...")
    results = analyze_snr_distribution(snr_train, snr_val, snr_test, y_train, y_val, y_test, mods)
    
    # Print analysis results
    print_snr_analysis(results)
    
    # Create visualizations
    print("\nCreating visualization plots...")
    create_snr_distribution_plots(results)
    
    # Save detailed results
    print("\nSaving detailed results...")
    save_detailed_results(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Key findings:")
    
    stats = results['summary_stats']
    if stats['train_ratio_std'] > 0.05 or stats['test_ratio_std'] > 0.05:
        print("⚠️  HIGH VARIANCE in split ratios across SNRs detected!")
        print("   Current splitting strategy may not maintain balanced SNR distribution.")
        print("   Consider implementing SNR-aware stratified splitting.")
    else:
        print("✅ Split ratios across SNRs are relatively consistent.")
    
    print(f"\nStandard deviations:")
    print(f"  Train ratio std: {stats['train_ratio_std']:.4f}")
    print(f"  Test ratio std:  {stats['test_ratio_std']:.4f}")


if __name__ == "__main__":
    main()