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

def draw_lightweight_hybrid_architecture():
    """
    Draw the complete lightweight hybrid model architecture
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define colors for different layer types
    colors = {
        'input': '#E8F4FD',
        'complex_conv': '#FFE6CC',
        'complex_bn': '#D4EDDA',
        'complex_activation': '#FFF3CD',
        'complex_pooling': '#F8D7DA',
        'residual_block': '#E2E3E5',
        'complex_dense': '#D1ECF1',
        'magnitude': '#F5C6CB',
        'real_dense': '#C3E6CB',
        'output': '#FADBD8'
    }
    
    # Layer information
    layers = [
        {'name': 'Input\n(2, 128)', 'type': 'input', 'pos': (1, 10), 'size': (1.5, 0.8)},
        {'name': 'Permute\n(128, 2)', 'type': 'input', 'pos': (3, 10), 'size': (1.5, 0.8)},
        
        # Initial Complex Processing
        {'name': 'ComplexConv1D\nfilters=32, kernel=5', 'type': 'complex_conv', 'pos': (5, 10), 'size': (2.2, 0.8)},
        {'name': 'ComplexBN', 'type': 'complex_bn', 'pos': (7.5, 10), 'size': (1.5, 0.8)},
        {'name': 'ComplexActivation\n(LeakyReLU)', 'type': 'complex_activation', 'pos': (9.5, 10), 'size': (2, 0.8)},
        {'name': 'ComplexPooling1D\npool_size=2', 'type': 'complex_pooling', 'pos': (12, 10), 'size': (2, 0.8)},
        
        # Complex Residual Blocks
        {'name': 'ComplexResidualBlock\nfilters=64', 'type': 'residual_block', 'pos': (5, 8), 'size': (2.5, 1.2)},
        {'name': 'ComplexResidualBlock\nfilters=128, stride=2', 'type': 'residual_block', 'pos': (8, 8), 'size': (2.5, 1.2)},
        {'name': 'ComplexResidualBlock\nAdvanced\nfilters=256, stride=2', 'type': 'residual_block', 'pos': (11, 8), 'size': (2.5, 1.2)},
        
        # Global Features
        {'name': 'ComplexGlobal\nAveragePooling1D', 'type': 'complex_pooling', 'pos': (6, 6), 'size': (2.5, 0.8)},
        {'name': 'ComplexDense\n512 units', 'type': 'complex_dense', 'pos': (9, 6), 'size': (2, 0.8)},
        {'name': 'ComplexActivation\n(LeakyReLU)', 'type': 'complex_activation', 'pos': (11.5, 6), 'size': (2, 0.8)},
        {'name': 'Dropout\n(0.5)', 'type': 'complex_dense', 'pos': (14, 6), 'size': (1.5, 0.8)},
        
        # Conversion to Real
        {'name': 'ComplexMagnitude\n(Complex→Real)', 'type': 'magnitude', 'pos': (7, 4), 'size': (2.5, 0.8)},
        
        # Final Classification
        {'name': 'Dense\n256 units, ReLU', 'type': 'real_dense', 'pos': (10, 4), 'size': (2, 0.8)},
        {'name': 'Dropout\n(0.3)', 'type': 'real_dense', 'pos': (12.5, 4), 'size': (1.5, 0.8)},
        {'name': 'Dense\n11 classes, Softmax', 'type': 'output', 'pos': (8.5, 2), 'size': (2.5, 0.8)},
    ]
    
    # Draw layers
    for layer in layers:
        x, y = layer['pos']
        w, h = layer['size']
        color = colors[layer['type']]
        
        # Create rounded rectangle
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, layer['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
      # Draw connections with proper positioning to avoid text overlap
    connections = [
        # Horizontal connections at input level
        ((1.75, 10), (2.25, 10)),    # Input -> Permute
        ((3.75, 10), (4.25, 10)),    # Permute -> ComplexConv1D
        ((6.1, 10), (6.9, 10)),      # ComplexConv1D -> ComplexBN
        ((8.25, 10), (8.75, 10)),    # ComplexBN -> ComplexActivation
        ((10.5, 10), (11, 10)),      # ComplexActivation -> ComplexPooling
        
        # Vertical connections to residual blocks
        ((12, 9.6), (5, 8.6)),       # ComplexPooling -> ResBlock1 (curved)
        ((6.25, 7.4), (6.75, 8.6)),  # ResBlock1 -> ResBlock2
        ((9.25, 7.4), (9.75, 8.6)),  # ResBlock2 -> ResBlock3
        
        # From residual blocks to global processing
        ((11, 7.4), (6, 6.4)),       # ResBlock3 -> GlobalPooling
        ((7.25, 6), (7.75, 6)),      # GlobalPooling -> ComplexDense
        ((10, 6), (10.5, 6)),        # ComplexDense -> ComplexActivation
        ((12.5, 6), (13.25, 6)),     # ComplexActivation -> Dropout
        
        # To magnitude conversion
        ((14, 5.6), (7, 4.4)),       # Dropout -> ComplexMagnitude (curved)
        
        # Final classification
        ((8.25, 3.6), (9, 4)),       # ComplexMagnitude -> Dense
        ((11, 4), (11.75, 4)),       # Dense -> Dropout
        ((12.5, 3.6), (8.5, 2.4)),   # Final Dropout -> Output (curved)
    ]
    
    for i, (start, end) in enumerate(connections):
        # Use different arrow styles for different types of connections
        if i in [4, 6, 11, 15]:  # Curved connections
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue',
                                     connectionstyle="arc3,rad=0.3"))
        else:  # Straight connections
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=1.8, color='black'))
      # Add skip connections for residual blocks with proper positioning
    skip_connections = [
        ((5, 8.6), (8, 7.4)),   # First residual skip (above the blocks)
        ((8, 8.6), (11, 7.4)),  # Second residual skip (above the blocks)
    ]
    
    for start, end in skip_connections:
        # Draw curved skip connection with red color and higher curve
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='red',
                                 connectionstyle="arc3,rad=0.5"))
        
        # Add skip connection label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.5
        ax.text(mid_x, mid_y, 'Skip', ha='center', va='center',
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor='red', alpha=0.8))
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input/Reshape'),
        patches.Patch(color=colors['complex_conv'], label='Complex Convolution'),
        patches.Patch(color=colors['complex_bn'], label='Complex BatchNorm'),
        patches.Patch(color=colors['complex_activation'], label='Complex Activation'),
        patches.Patch(color=colors['complex_pooling'], label='Complex Pooling'),
        patches.Patch(color=colors['residual_block'], label='Complex Residual Block'),
        patches.Patch(color=colors['complex_dense'], label='Complex Dense'),
        patches.Patch(color=colors['magnitude'], label='Complex→Real'),
        patches.Patch(color=colors['real_dense'], label='Real Dense'),
        patches.Patch(color=colors['output'], label='Output')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add title and annotations
    ax.set_title('Lightweight Hybrid ResNet-ComplexCNN Architecture\n'
                'for Radio Signal Modulation Classification', 
                fontsize=14, fontweight='bold', pad=20)
      # Add phase annotations with better positioning to avoid arrow overlap
    ax.text(8, 11.5, 'Phase 1: Complex Feature Extraction', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    ax.text(8, 9.2, 'Phase 2: Complex Residual Learning', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    ax.text(10, 7.2, 'Phase 3: Complex Global Features', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    ax.text(10, 5.2, 'Phase 4: Real Classification', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(-1, 16)
    ax.set_ylim(1, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

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

def draw_complex_convolution_operation():
    """
    Draw detailed visualization of complex convolution operation
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Mathematical representation
    ax.text(7, 9, 'Complex Convolution Operation', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.text(7, 8.2, r'$z = x + jy$, $W = W_r + jW_i$', 
            ha='center', va='center', fontsize=14)
    
    # Draw input representation
    ax.text(2, 7, 'Complex Input', ha='center', fontweight='bold', fontsize=12)
    
    # Real part box
    real_box = Rectangle((1, 5.5), 2, 1, facecolor='lightblue', edgecolor='black')
    ax.add_patch(real_box)
    ax.text(2, 6, 'Real Part\n(x)', ha='center', va='center', fontweight='bold')
    
    # Imaginary part box
    imag_box = Rectangle((1, 4), 2, 1, facecolor='lightcoral', edgecolor='black')
    ax.add_patch(imag_box)
    ax.text(2, 4.5, 'Imaginary Part\n(y)', ha='center', va='center', fontweight='bold')
    
    # Draw weight representation
    ax.text(7, 7, 'Complex Weights', ha='center', fontweight='bold', fontsize=12)
    
    # Real weight box
    wr_box = Rectangle((6, 5.5), 2, 1, facecolor='lightgreen', edgecolor='black')
    ax.add_patch(wr_box)
    ax.text(7, 6, 'Real Weights\n(W_r)', ha='center', va='center', fontweight='bold')
    
    # Imaginary weight box
    wi_box = Rectangle((6, 4), 2, 1, facecolor='lightyellow', edgecolor='black')
    ax.add_patch(wi_box)
    ax.text(7, 4.5, 'Imaginary Weights\n(W_i)', ha='center', va='center', fontweight='bold')
    
    # Draw complex multiplication
    ax.text(12, 7, 'Complex Output', ha='center', fontweight='bold', fontsize=12)
    
    # Output real part
    out_real_box = Rectangle((11, 5.5), 2, 1, facecolor='mediumpurple', edgecolor='black')
    ax.add_patch(out_real_box)
    ax.text(12, 6, 'Re(z*W)\nx*W_r - y*W_i', ha='center', va='center', fontweight='bold')
    
    # Output imaginary part
    out_imag_box = Rectangle((11, 4), 2, 1, facecolor='orange', edgecolor='black')
    ax.add_patch(out_imag_box)
    ax.text(12, 4.5, 'Im(z*W)\nx*W_i + y*W_r', ha='center', va='center', fontweight='bold')
    
    # Draw arrows showing the computation
    # Real part computation
    ax.annotate('', xy=(11, 6), xytext=(3, 6),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(11, 6), xytext=(8, 6),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Cross terms for real part
    ax.annotate('', xy=(11, 6), xytext=(3, 4.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='red', linestyle='--'))
    ax.annotate('', xy=(11, 6), xytext=(8, 4.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='red', linestyle='--'))
    
    # Imaginary part computation
    ax.annotate('', xy=(11, 4.5), xytext=(3, 6),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='green', linestyle='--'))
    ax.annotate('', xy=(11, 4.5), xytext=(8, 4.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='green', linestyle='--'))
    
    ax.annotate('', xy=(11, 4.5), xytext=(3, 4.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(11, 4.5), xytext=(8, 6),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # Mathematical formulas
    ax.text(7, 3, 'Complex Multiplication Formula:', ha='center', fontsize=12, fontweight='bold')
    ax.text(7, 2.4, r'$(x + jy) \times (W_r + jW_i) = (xW_r - yW_i) + j(xW_i + yW_r)$', 
            ha='center', fontsize=11)
    
    ax.text(7, 1.5, 'Convolution Operation:', ha='center', fontsize=12, fontweight='bold')
    ax.text(7, 0.9, r'$\text{Re}(z * W) = x * W_r - y * W_i$', ha='center', fontsize=10)
    ax.text(7, 0.5, r'$\text{Im}(z * W) = x * W_i + y * W_r$', ha='center', fontsize=10)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
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
    
    # 1. Lightweight Hybrid Architecture Overview
    print("1. Creating lightweight hybrid architecture overview...")
    fig1 = draw_lightweight_hybrid_architecture()
    fig1.savefig(os.path.join(figure_dir, 'lightweight_hybrid_architecture.png'), 
                bbox_inches='tight', dpi=300)
    plt.close(fig1)
    
    # 2. Complex Residual Block Detail
    print("2. Creating complex residual block detail...")
    fig2 = draw_complex_residual_block()
    fig2.savefig(os.path.join(figure_dir, 'complex_residual_block.png'), 
                bbox_inches='tight', dpi=300)
    plt.close(fig2)
    
    # 3. Data Processing Pipeline
    print("3. Creating data processing pipeline...")
    fig3 = draw_data_flow_pipeline()
    fig3.savefig(os.path.join(figure_dir, 'data_processing_pipeline.png'), 
                bbox_inches='tight', dpi=300)
    plt.close(fig3)
    
    # 4. Complex Convolution Operation
    print("4. Creating complex convolution operation diagram...")
    fig4 = draw_complex_convolution_operation()
    fig4.savefig(os.path.join(figure_dir, 'complex_convolution_operation.png'), 
                bbox_inches='tight', dpi=300)
    plt.close(fig4)
    
    print(f"\nAll diagrams saved to: {figure_dir}")
    print("Generated files:")
    print("- lightweight_hybrid_architecture.png")
    print("- complex_residual_block.png") 
    print("- data_processing_pipeline.png")
    print("- complex_convolution_operation.png")

if __name__ == "__main__":
    main()
