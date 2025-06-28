#!/usr/bin/env python3
"""
Standalone baseline model performance comparison plot
Baseline model comparison functionality extracted from ablation experiment visualization script
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os

# Set font and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def create_output_dir():
    """Create output directory"""
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'figure', 'baseline_comparison_en')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_baseline_model_comparison():
    """Plot baseline model performance comparison"""
    print("Starting to plot baseline model performance comparison...")
    
    # Baseline model data (from experimental results)
    models = ['FCNN', 'CNN2D', 'Transformer', 'CNN1D', 'ResNet', 'ComplexCNN', 'Hybrid\n(Baseline)', 'Hybrid+GPR+Aug\n(Best)']
    accuracies = [42.65, 47.31, 47.86, 54.94, 55.37, 57.11, 56.94, 65.38]
    
    # Color configuration - distinguish traditional models, single advanced models and hybrid models
    colors = ['#FF6B6B', '#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#45B7D1', '#2E86AB', '#C73E1D']
    edge_colors = ['darkred', 'darkred', 'darkred', 'darkgreen', 'darkgreen', 'darkblue', 'navy', 'darkred']
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Baseline model accuracy comparison
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, 
                    edgecolor=edge_colors, linewidth=1.5)
    
    ax1.set_title('Baseline Model Performance Comparison', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Model Architecture', fontsize=14, fontweight='bold')
    ax1.set_ylim(40, 70)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=15)
    
    # Add 64.59% SOTA baseline
    ax1.axhline(y=64.59, color='red', linestyle='--', alpha=0.7, linewidth=2, label='64.59% previous SOTA')
    
    # Add legend explaining different colors
    legend_elements = [
        # mpatches.Patch(color='#FF6B6B', label='Traditional Models'),
        # mpatches.Patch(color='#4ECDC4', label='Convolutional Neural Networks'),
        # mpatches.Patch(color='#45B7D1', label='Complex Neural Networks'),
        # mpatches.Patch(color='#2E86AB', label='Hybrid Models'),
        # mpatches.Patch(color='#C73E1D', label='Complete Hybrid Model'),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7, linewidth=2, label='64.59% previous SOTA')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    # Save figures
    output_dir = create_output_dir()
    for fmt in ['png', 'pdf']:
        filepath = os.path.join(output_dir, f'baseline_model_comparison.{fmt}')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {filepath}")
    
    plt.show()
    print("✅ Baseline model performance comparison plot generated")

def print_model_statistics():
    """Print model statistics"""
    models = ['FCNN', 'CNN2D', 'Transformer', 'CNN1D', 'ResNet', 'ComplexCNN', 'Hybrid(Baseline)', 'Hybrid+GPR+Aug(Best)']
    accuracies = [42.65, 47.31, 47.86, 54.94, 55.37, 57.11, 56.94, 65.38]
    
    print("\n" + "="*60)
    print("Baseline Model Performance Statistics")
    print("="*60)
    
    baseline_performance = 42.65  # FCNN as baseline
    
    for model, acc in zip(models, accuracies):
        improvement = acc - baseline_performance
        print(f"{model:<20} | Accuracy: {acc:>6.2f}% | Improvement: {improvement:>+6.2f}%")
    
    print("="*60)
    print(f"Best Performance: {max(accuracies):.2f}% ({models[accuracies.index(max(accuracies))]})")
    print(f"Maximum Improvement: {max(accuracies) - baseline_performance:.2f} percentage points")
    print("="*60)

def main():
    """Main function"""
    print("Baseline Model Performance Comparison Plot Generator")
    print("="*50)
    
    # Print statistics
    print_model_statistics()
    
    # Plot charts
    plot_baseline_model_comparison()
    
    print("\nProgram execution completed!")

if __name__ == "__main__":
    main()
