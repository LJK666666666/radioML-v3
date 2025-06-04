"""
Neural Network Architecture Visualization Script

This script creates detailed architecture diagrams for the lightweight hybrid model
combining ResNet and ComplexCNN components for radio signal classification.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np
import os

# Set up matplotlib for better figure quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

def create_figure_directory():
    """Create figure directory if it doesn't exist"""
    figure_dir = os.path.join(os.path.dirname(__file__), 'figure')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    return figure_dir



def draw_complex_residual_block():
    """
    Draw detailed structure of a complex residual block
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    colors = {
        'input': '#E8F4FD',
        'conv': '#FFE6CC',
        'bn': '#D4EDDA',
        'activation': '#FFF3CD',
        'add': '#F8D7DA',
        'shortcut': '#E2E3E5'
    }
    
    # Main path layers
    main_layers = [
        {'name': 'Input\n(batch, time, 2*filters)', 'type': 'input', 'pos': (2, 9), 'size': (2.5, 0.8)},
        {'name': 'ComplexConv1D\nfilters, kernel_size', 'type': 'conv', 'pos': (2, 7.5), 'size': (2.5, 0.8)},
        {'name': 'ComplexBN', 'type': 'bn', 'pos': (2, 6), 'size': (2, 0.8)},
        {'name': 'ComplexActivation', 'type': 'activation', 'pos': (2, 4.5), 'size': (2, 0.8)},
        {'name': 'ComplexConv1D\nfilters, kernel_size', 'type': 'conv', 'pos': (2, 3), 'size': (2.5, 0.8)},
        {'name': 'ComplexBN', 'type': 'bn', 'pos': (2, 1.5), 'size': (2, 0.8)},
    ]
    
    # Shortcut path layers
    shortcut_layers = [
        {'name': 'Shortcut\nConnection', 'type': 'shortcut', 'pos': (6, 6), 'size': (2, 0.8)},
        {'name': 'ComplexConv1D\n1x1 (if needed)', 'type': 'conv', 'pos': (6, 4.5), 'size': (2.5, 0.8)},
        {'name': 'ComplexBN\n(if needed)', 'type': 'bn', 'pos': (6, 3), 'size': (2, 0.8)},
    ]
    
    # Addition and output
    final_layers = [
        {'name': 'Complex\nAddition', 'type': 'add', 'pos': (9, 1.5), 'size': (2, 0.8)},
        {'name': 'ComplexActivation\n(Final)', 'type': 'activation', 'pos': (9, 0), 'size': (2, 0.8)},
    ]
    
    # Draw all layers
    all_layers = main_layers + shortcut_layers + final_layers
    for layer in all_layers:
        x, y = layer['pos']
        w, h = layer['size']
        color = colors[layer['type']]
        
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.2
        )
        ax.add_patch(box)
        
        ax.text(x, y, layer['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Draw main path connections
    main_connections = [
        ((2, 8.6), (2, 7.9)),
        ((2, 7.1), (2, 6.4)),
        ((2, 5.6), (2, 4.9)),
        ((2, 4.1), (2, 3.4)),
        ((2, 2.6), (2, 1.9)),
        ((2, 1.1), (9, 1.9))
    ]
    
    for start, end in main_connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Draw shortcut path
    shortcut_connections = [
        ((2, 9), (6, 6.4)),  # Input to shortcut
        ((6, 5.6), (6, 4.9)),
        ((6, 4.1), (6, 3.4)),
        ((6, 2.6), (9, 1.9))  # Shortcut to addition
    ]
    
    for start, end in shortcut_connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Final connection
    ax.annotate('', xy=(9, -0.4), xytext=(9, 1.1),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # Add mathematical annotations
    ax.text(4.5, 8, 'Main Path:\nF(x)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    ax.text(8, 5, 'Shortcut Path:\nx or W_s(x)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    ax.text(11, 1.5, 'y = F(x) + x\n(Complex Addition)', ha='left', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input'),
        patches.Patch(color=colors['conv'], label='Complex Convolution'),
        patches.Patch(color=colors['bn'], label='Complex BatchNorm'),
        patches.Patch(color=colors['activation'], label='Complex Activation'),
        patches.Patch(color=colors['shortcut'], label='Shortcut Path'),
        patches.Patch(color=colors['add'], label='Complex Addition')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('Complex Residual Block Architecture\n'
                'Combining ResNet Residual Learning with Complex Processing', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def draw_data_flow_pipeline():
    """
    Draw the complete data processing pipeline
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#FFE6CC',
        'network': '#D4EDDA',
        'output': '#FADBD8'
    }
    
    # Pipeline stages
    stages = [
        {'name': 'Raw I/Q Signal\n(2, 128)', 'type': 'input', 'pos': (1, 4), 'size': (1.8, 1)},
        {'name': 'GPR Denoising\n(Optional)', 'type': 'preprocessing', 'pos': (3.5, 4), 'size': (1.8, 1)},
        {'name': 'Rotation\nAugmentation\n(Optional)', 'type': 'preprocessing', 'pos': (6, 4), 'size': (1.8, 1)},
        {'name': 'Input\nReshaping\n(128, 2)', 'type': 'preprocessing', 'pos': (8.5, 4), 'size': (1.8, 1)},
        {'name': 'Complex Feature\nExtraction', 'type': 'network', 'pos': (11, 4), 'size': (1.8, 1)},
        {'name': 'Complex Residual\nProcessing', 'type': 'network', 'pos': (13.5, 4), 'size': (1.8, 1)},
        {'name': 'Complex→Real\nConversion', 'type': 'network', 'pos': (16, 4), 'size': (1.8, 1)},
        {'name': 'Classification\nOutput\n(11 classes)', 'type': 'output', 'pos': (18.5, 4), 'size': (1.8, 1)}
    ]
    
    # Draw stages
    for stage in stages:
        x, y = stage['pos']
        w, h = stage['size']
        color = colors[stage['type']]
        
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        ax.text(x, y, stage['name'], ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Draw connections
    for i in range(len(stages) - 1):
        start_x = stages[i]['pos'][0] + stages[i]['size'][0]/2
        end_x = stages[i+1]['pos'][0] - stages[i+1]['size'][0]/2
        y = stages[i]['pos'][1]
        
        ax.annotate('', xy=(end_x, y), xytext=(start_x, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add phase labels
    phase_labels = [
        {'text': 'Data Preprocessing', 'pos': (4.75, 2), 'color': 'lightblue'},
        {'text': 'Neural Network Processing', 'pos': (13.5, 2), 'color': 'lightgreen'},
        {'text': 'Output', 'pos': (18.5, 2), 'color': 'lightcoral'}
    ]
    
    for label in phase_labels:
        ax.text(label['pos'][0], label['pos'][1], label['text'], 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=label['color'], alpha=0.7))
    
    # Add technical details
    tech_details = [
        {'text': 'SNR-adaptive\nNoise Estimation', 'pos': (3.5, 6), 'size': (1.6, 0.8)},
        {'text': '90°, 180°, 270°\nRotations', 'pos': (6, 6), 'size': (1.6, 0.8)},
        {'text': 'Complex Arithmetic\nPreservation', 'pos': (12.25, 6), 'size': (1.6, 0.8)},
        {'text': 'Magnitude\nExtraction', 'pos': (16, 6), 'size': (1.6, 0.8)}
    ]
    
    for detail in tech_details:
        x, y = detail['pos']
        w, h = detail['size']
        
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor='lightyellow',
            edgecolor='gray',
            linewidth=1,
            linestyle='--'
        )
        ax.add_patch(box)
        
        ax.text(x, y, detail['text'], ha='center', va='center', 
                fontsize=8, style='italic')
        
        # Draw connection to main pipeline
        ax.plot([x, x], [y - h/2, 4.5], 'gray', linestyle=':', alpha=0.7)
    
    ax.set_title('Complete Data Processing Pipeline\n'
                'From Raw I/Q Signals to Modulation Classification', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 20)
    ax.set_ylim(1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """
    Generate all architecture diagrams
    """
    figure_dir = create_figure_directory()
    
    print("Generating neural network architecture diagrams...")
    
    # 1. Complex Residual Block Detail
    print("1. Creating complex residual block detail...")
    fig2 = draw_complex_residual_block()
    fig2.savefig(os.path.join(figure_dir, 'complex_residual_block.png'), 
                bbox_inches='tight', dpi=300)
    plt.close(fig2)
      # 2. Data Processing Pipeline
    print("2. Creating data processing pipeline...")
    fig3 = draw_data_flow_pipeline()
    fig3.savefig(os.path.join(figure_dir, 'data_processing_pipeline.png'), 
                bbox_inches='tight', dpi=300)
    plt.close(fig3)
    
    print(f"\nAll diagrams saved to: {figure_dir}")
    print("Generated files:")
    print("- complex_residual_block.png") 
    print("- data_processing_pipeline.png")

if __name__ == "__main__":
    main()
