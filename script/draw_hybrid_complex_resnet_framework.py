"""
Hybrid Complex ResNet Model Framework Visualization

This script creates a comprehensive visualization of the hybrid complex ResNet model architecture,
highlighting:
1. Residual connections (skip connections)
2. Complex number processing features
3. Multi-stage architecture
4. Different types of residual blocks
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np
import os

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_hybrid_complex_resnet_framework():
    """
    Create the main framework diagram showing the overall architecture
    with residual connections and complex processing highlights
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define colors for different components
    colors = {
        'input': '#E8F4FD',
        'complex_conv': '#4A90E2',
        'residual_basic': '#7ED321',
        'residual_advanced': '#F5A623',
        'complex_dense': '#BD10E0',
        'real_output': '#B8E986',
        'skip_connection': '#D0021B',
        'complex_feature': '#50E3C2'
    }
    
    # Stage positions and dimensions
    stage_width = 2.0
    stage_height = 1.2
    x_start = 1
    y_positions = [9, 7.5, 6, 4.5, 3, 1.5]
      # Stage 1: Input and Initial Complex Processing
    stage1_box = FancyBboxPatch(
        (x_start, y_positions[0]), stage_width*2, stage_height,
        boxstyle="round,pad=0.1", 
        facecolor=colors['input'],
        edgecolor='black', linewidth=2
    )
    ax.add_patch(stage1_box)
    ax.text(x_start + 1, y_positions[0] + 0.6, 'Stage 1: 输入层\n(2, 128) → (128, 2)\n维度转置', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Complex Conv1D block
    conv_box = FancyBboxPatch(
        (x_start + 3.5, y_positions[0]), stage_width, stage_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['complex_conv'],
        edgecolor='black', linewidth=1
    )
    ax.add_patch(conv_box)
    ax.text(x_start + 4.5, y_positions[0] + 0.6, 'ComplexConv1D\n64 filters, k=7\n(128, 2)→(64, 128)', 
            ha='center', va='center', fontsize=9, color='white', weight='bold')
    
    # Stage 2: Basic Residual Blocks
    stage2_box = FancyBboxPatch(
        (x_start, y_positions[1]), stage_width*4, stage_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['residual_basic'],
        edgecolor='black', linewidth=2
    )
    ax.add_patch(stage2_box)
    ax.text(x_start + 2, y_positions[1] + 0.6, 'Stage 2: 基础复数残差块\n2 × ComplexResidualBlock (64 filters)\n(64, 128)保持不变', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Stage 3: Deeper Residual Processing
    stage3_box = FancyBboxPatch(
        (x_start, y_positions[2]), stage_width*5, stage_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['residual_basic'],
        edgecolor='black', linewidth=2
    )
    ax.add_patch(stage3_box)
    ax.text(x_start + 2.5, y_positions[2] + 0.6, 'Stage 3: 深层复数残差处理\n2 × ComplexResidualBlock (128 filters)\n下采样: (32, 256)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Stage 4: Advanced Residual Blocks
    stage4_box = FancyBboxPatch(
        (x_start, y_positions[3]), stage_width*5, stage_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['residual_advanced'],
        edgecolor='black', linewidth=2
    )
    ax.add_patch(stage4_box)
    ax.text(x_start + 2.5, y_positions[3] + 0.6, 'Stage 4: 高级复数残差块\n2 × ComplexResidualBlockAdvanced (256 filters)\n注意力机制 + 深层残差', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Stage 5: High-level Feature Processing
    stage5_box = FancyBboxPatch(
        (x_start, y_positions[4]), stage_width*5, stage_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['residual_advanced'],
        edgecolor='black', linewidth=2
    )
    ax.add_patch(stage5_box)
    ax.text(x_start + 2.5, y_positions[4] + 0.6, 'Stage 5: 高层特征处理\n3 × ComplexResidualBlockAdvanced (512 filters)\n多层注意力 + 复数残差', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Stage 6: Output Processing
    stage6_box = FancyBboxPatch(
        (x_start, y_positions[5]), stage_width*2, stage_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['complex_dense'],
        edgecolor='black', linewidth=2
    )
    ax.add_patch(stage6_box)
    ax.text(x_start + 1, y_positions[5] + 0.6, 'Stage 6: 输出处理\n复数全连接\n复数→实数转换', 
            ha='center', va='center', fontsize=10, color='white', weight='bold')
    
    # Final classification
    final_box = FancyBboxPatch(
        (x_start + 3.5, y_positions[5]), stage_width, stage_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['real_output'],
        edgecolor='black', linewidth=2
    )
    ax.add_patch(final_box)
    ax.text(x_start + 4.5, y_positions[5] + 0.6, '分类层\nSoftmax\n类别输出', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw skip connections (residual connections)
    skip_connections = [
        # Stage 2 internal skip connections
        ((x_start + 0.5, y_positions[1] + 0.6), (x_start + 3.5, y_positions[1] + 0.6)),
        # Stage 3 internal skip connections  
        ((x_start + 0.5, y_positions[2] + 0.6), (x_start + 4.5, y_positions[2] + 0.6)),
        # Stage 4 internal skip connections
        ((x_start + 0.5, y_positions[3] + 0.6), (x_start + 4.5, y_positions[3] + 0.6)),
        # Stage 5 internal skip connections
        ((x_start + 0.5, y_positions[4] + 0.6), (x_start + 4.5, y_positions[4] + 0.6)),
    ]
    
    for start, end in skip_connections:
        # Draw curved skip connection arrows
        connection = ConnectionPatch(start, end, "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5,
                                   mutation_scale=20, fc=colors['skip_connection'],
                                   ec=colors['skip_connection'], linewidth=3,
                                   connectionstyle="arc3,rad=0.3")
        ax.add_patch(connection)
    
    # Draw main data flow arrows
    main_flow_connections = [
        ((x_start + 1, y_positions[0]), (x_start + 1, y_positions[1] + stage_height)),
        ((x_start + 2, y_positions[1]), (x_start + 2, y_positions[2] + stage_height)),
        ((x_start + 2.5, y_positions[2]), (x_start + 2.5, y_positions[3] + stage_height)),
        ((x_start + 2.5, y_positions[3]), (x_start + 2.5, y_positions[4] + stage_height)),
        ((x_start + 2.5, y_positions[4]), (x_start + 1, y_positions[5] + stage_height)),
        ((x_start + 2, y_positions[5] + 0.6), (x_start + 3.5, y_positions[5] + 0.6)),
    ]
    
    for start, end in main_flow_connections:
        connection = ConnectionPatch(start, end, "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5,
                                   mutation_scale=20, fc='black', ec='black', linewidth=2)
        ax.add_patch(connection)
    
    # Add complex processing indicators
    complex_indicators = [
        (x_start + 7, y_positions[0] + 0.6),
        (x_start + 7, y_positions[1] + 0.6),
        (x_start + 7, y_positions[2] + 0.6),
        (x_start + 7, y_positions[3] + 0.6),
        (x_start + 7, y_positions[4] + 0.6),
    ]
    
    for x, y in complex_indicators:
        circle = Circle((x, y), 0.2, facecolor=colors['complex_feature'], 
                       edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(x, y, 'C', ha='center', va='center', fontsize=8, weight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['complex_conv'], label='复数卷积层'),
        patches.Patch(color=colors['residual_basic'], label='基础残差块'),
        patches.Patch(color=colors['residual_advanced'], label='高级残差块'),
        patches.Patch(color=colors['complex_dense'], label='复数全连接'),
        patches.Patch(color=colors['skip_connection'], label='残差连接'),
        patches.Patch(color=colors['complex_feature'], label='复数处理标识')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Add title and annotations
    ax.set_title('混合复数ResNet模型架构图\nHybrid Complex ResNet Architecture', 
                fontsize=16, weight='bold', pad=20)
    
    # Add complex processing annotation
    ax.text(x_start + 8, y_positions[2], '复数域处理特点:\n• 实部+虚部分离处理\n• 复数卷积运算\n• 复数激活函数\n• 复数批归一化', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    # Add residual connection annotation
    ax.text(x_start + 8, y_positions[4], '残差连接特点:\n• 跳跃连接缓解梯度消失\n• 复数域残差学习\n• 多层级特征融合\n• 深层网络训练稳定', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_residual_block_detail():
    """
    Create detailed diagram of residual block internal structure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors
    colors = {
        'conv': '#4A90E2',
        'bn': '#7ED321',
        'activation': '#F5A623',
        'skip': '#D0021B',
        'attention': '#BD10E0'
    }
    
    # Basic Residual Block (left)
    ax1.set_title('基础复数残差块\nComplexResidualBlock', fontsize=14, weight='bold')
    
    # Input
    input_box = FancyBboxPatch((1, 8), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='lightgray', edgecolor='black')
    ax1.add_patch(input_box)
    ax1.text(2, 8.4, '输入特征\n(B, T, 2F)', ha='center', va='center', fontsize=10)
    
    # Main path
    # Conv1D + BN + Activation
    conv1_box = FancyBboxPatch((1, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                              facecolor=colors['conv'], edgecolor='black')
    ax1.add_patch(conv1_box)
    ax1.text(2, 6.9, 'ComplexConv1D\nk=3, s=1', ha='center', va='center', 
             fontsize=9, color='white', weight='bold')
    
    bn1_box = FancyBboxPatch((1, 5.2), 2, 0.8, boxstyle="round,pad=0.1",
                            facecolor=colors['bn'], edgecolor='black')
    ax1.add_patch(bn1_box)
    ax1.text(2, 5.6, 'ComplexBN', ha='center', va='center', fontsize=10, weight='bold')
    
    act1_box = FancyBboxPatch((1, 3.9), 2, 0.8, boxstyle="round,pad=0.1",
                             facecolor=colors['activation'], edgecolor='black')
    ax1.add_patch(act1_box)
    ax1.text(2, 4.3, 'ComplexActivation', ha='center', va='center', fontsize=10, weight='bold')
    
    # Conv2D + BN
    conv2_box = FancyBboxPatch((1, 2.6), 2, 0.8, boxstyle="round,pad=0.1",
                              facecolor=colors['conv'], edgecolor='black')
    ax1.add_patch(conv2_box)
    ax1.text(2, 3.0, 'ComplexConv1D\nk=3, s=1', ha='center', va='center', 
             fontsize=9, color='white', weight='bold')
    
    bn2_box = FancyBboxPatch((1, 1.3), 2, 0.8, boxstyle="round,pad=0.1",
                            facecolor=colors['bn'], edgecolor='black')
    ax1.add_patch(bn2_box)
    ax1.text(2, 1.7, 'ComplexBN', ha='center', va='center', fontsize=10, weight='bold')
    
    # Addition and final activation
    add_circle = Circle((2, 0.5), 0.3, facecolor='yellow', edgecolor='black', linewidth=2)
    ax1.add_patch(add_circle)
    ax1.text(2, 0.5, '+', ha='center', va='center', fontsize=16, weight='bold')
    
    # Skip connection
    skip_arrow = ConnectionPatch((3.2, 8.4), (2.8, 0.5), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc=colors['skip'], ec=colors['skip'], 
                               linewidth=3, connectionstyle="arc3,rad=0.5")
    ax1.add_patch(skip_arrow)
    ax1.text(4, 4.5, '跳跃连接\nSkip Connection', ha='center', va='center', 
             fontsize=10, color=colors['skip'], weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Main flow arrows
    main_arrows = [
        ((2, 7.7), (2, 7.3)),
        ((2, 6.3), (2, 6.0)),
        ((2, 5.0), (2, 4.7)),
        ((2, 3.7), (2, 3.4)),
        ((2, 2.4), (2, 2.1)),
        ((2, 1.1), (2, 0.8))
    ]
    
    for start, end in main_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=2, shrinkB=2,
                              mutation_scale=15, fc='black', ec='black', linewidth=2)
        ax1.add_patch(arrow)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 9)
    ax1.axis('off')
    
    # Advanced Residual Block (right)
    ax2.set_title('高级复数残差块\nComplexResidualBlockAdvanced', fontsize=14, weight='bold')
    
    # Input
    input_box2 = FancyBboxPatch((1, 9), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='lightgray', edgecolor='black')
    ax2.add_patch(input_box2)
    ax2.text(2, 9.4, '输入特征\n(B, T, 2F)', ha='center', va='center', fontsize=10)
    
    # Three conv layers instead of two
    conv1_box2 = FancyBboxPatch((1, 7.8), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['conv'], edgecolor='black')
    ax2.add_patch(conv1_box2)
    ax2.text(2, 8.2, 'ComplexConv1D\nk=3, s=2', ha='center', va='center', 
             fontsize=9, color='white', weight='bold')
    
    conv2_box2 = FancyBboxPatch((1, 6.6), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['conv'], edgecolor='black')
    ax2.add_patch(conv2_box2)
    ax2.text(2, 7.0, 'ComplexConv1D\nk=3, s=1', ha='center', va='center', 
             fontsize=9, color='white', weight='bold')
    
    conv3_box2 = FancyBboxPatch((1, 5.4), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['conv'], edgecolor='black')
    ax2.add_patch(conv3_box2)
    ax2.text(2, 5.8, 'ComplexConv1D\nk=1, s=1', ha='center', va='center', 
             fontsize=9, color='white', weight='bold')
    
    # Attention mechanism
    att_box = FancyBboxPatch((1, 4.2), 2, 0.8, boxstyle="round,pad=0.1",
                            facecolor=colors['attention'], edgecolor='black')
    ax2.add_patch(att_box)
    ax2.text(2, 4.6, 'Complex\nAttention', ha='center', va='center', 
             fontsize=10, color='white', weight='bold')
    
    # Addition
    add_circle2 = Circle((2, 3.0), 0.3, facecolor='yellow', edgecolor='black', linewidth=2)
    ax2.add_patch(add_circle2)
    ax2.text(2, 3.0, '+', ha='center', va='center', fontsize=16, weight='bold')
    
    # Skip connection with 1x1 conv
    skip_conv_box = FancyBboxPatch((4, 7), 1.5, 0.8, boxstyle="round,pad=0.1",
                                  facecolor=colors['conv'], edgecolor='black')
    ax2.add_patch(skip_conv_box)
    ax2.text(4.75, 7.4, 'Conv1D\n1×1', ha='center', va='center', 
             fontsize=9, color='white', weight='bold')
    
    # Skip connection arrows
    skip_arrow1 = ConnectionPatch((3, 9.4), (4, 7.4), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5,
                                mutation_scale=15, fc=colors['skip'], ec=colors['skip'], 
                                linewidth=2)
    ax2.add_patch(skip_arrow1)
    
    skip_arrow2 = ConnectionPatch((4.75, 6.8), (2.8, 3.0), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5,
                                mutation_scale=15, fc=colors['skip'], ec=colors['skip'], 
                                linewidth=2, connectionstyle="arc3,rad=-0.3")
    ax2.add_patch(skip_arrow2)
    
    # Main flow arrows
    main_arrows2 = [
        ((2, 8.8), (2, 8.6)),
        ((2, 7.6), (2, 7.4)),
        ((2, 6.4), (2, 6.2)),
        ((2, 5.2), (2, 5.0)),
        ((2, 4.0), (2, 3.3))
    ]
    
    for start, end in main_arrows2:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=2, shrinkB=2,
                              mutation_scale=15, fc='black', ec='black', linewidth=2)
        ax2.add_patch(arrow)
    
    # Add annotations
    ax2.text(5.5, 5, '注意力权重:\n复数域自适应\n特征重加权', ha='center', va='center', 
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightpink', alpha=0.7))
    
    ax2.set_xlim(0, 6.5)
    ax2.set_ylim(2, 10)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def create_complex_processing_illustration():
    """
    Create illustration showing complex number processing concepts
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    ax.set_title('复数域处理原理图解\nComplex Domain Processing Illustration', 
                fontsize=16, weight='bold', pad=20)
    
    # Complex number representation
    ax.text(2, 9, '复数表示 (Complex Representation)', fontsize=14, weight='bold')
    
    # Real and imaginary parts
    real_box = FancyBboxPatch((1, 7.5), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(real_box)
    ax.text(1.75, 8, '实部\nReal\nI', ha='center', va='center', fontsize=11, weight='bold')
    
    imag_box = FancyBboxPatch((3, 7.5), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(imag_box)
    ax.text(3.75, 8, '虚部\nImaginary\nQ', ha='center', va='center', fontsize=11, weight='bold')
      # Complex convolution illustration
    ax.text(7, 9, '复数卷积与维度变化 (Complex Convolution)', fontsize=14, weight='bold')
    
    # Formula and dimension explanation
    dimension_text = """输入: (batch, time, 2) → I/Q交替存储
卷积核: filters个复数滤波器 
输出: (batch, time, 2×filters)

第3维度解释:
• 每个复数输出需要2个实数表示
• filters个复数 = 2×filters个实数
• 实部和虚部交替存储"""
    
    ax.text(7, 7.8, dimension_text, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # Complex activation
    ax.text(2, 6, '复数激活函数 (Complex Activation)', fontsize=14, weight='bold')
    
    activation_box = FancyBboxPatch((1, 4.5), 4, 1.2, boxstyle="round,pad=0.1",
                                   facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(activation_box)
    ax.text(3, 5.1, 'f(z) = f(a + bi)\n分别处理实部和虚部\nor 复数域统一处理', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Residual connection in complex domain
    ax.text(7, 6, '复数域残差连接 (Complex Residual)', fontsize=14, weight='bold')
    
    # Residual formula
    ax.text(7, 5.1, 'H(z) = F(z) + z\n其中 z, F(z) ∈ ℂ\n复数域加法运算', 
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8))
    
    # Complex magnitude conversion
    ax.text(2, 3, '复数幅度转换 (Magnitude Conversion)', fontsize=14, weight='bold')
    
    mag_box = FancyBboxPatch((1, 1.5), 4, 1.2, boxstyle="round,pad=0.1",
                            facecolor='lightpink', edgecolor='purple', linewidth=2)
    ax.add_patch(mag_box)
    ax.text(3, 2.1, '|z| = √(a² + b²)\n复数转实数用于分类', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Benefits
    ax.text(7, 3, '复数处理优势 (Advantages)', fontsize=14, weight='bold')
    
    benefits_text = """• 保持I/Q信号相位关系
• 更好的特征表示能力  
• 减少参数数量
• 提高收敛速度
• 适合无线信号处理"""
    
    ax.text(7, 2.1, benefits_text, ha='left', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightsteelblue', alpha=0.8))
    
    # Add complex plane visualization
    ax.text(11, 8, '复平面表示', fontsize=12, weight='bold')
    
    # Draw coordinate system
    ax.arrow(10.5, 6.5, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(11.5, 5.5, 0, 2, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(12.7, 6.4, 'Real', fontsize=10)
    ax.text(11.3, 7.7, 'Imag', fontsize=10)
    
    # Plot a complex number
    ax.plot([11.5, 12.2], [6.5, 7.1], 'ro-', linewidth=2, markersize=6)
    ax.text(12.3, 7.2, 'z = a + bi', fontsize=10)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """
    Generate all visualization diagrams
    """
    # Create output directory if it doesn't exist
    output_dir = r'd:\1python programs\radioml\radioML-v3\script\figure'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成混合复数ResNet模型框架图...")
    
    # Generate main architecture diagram
    print("1. 生成主体架构图...")
    fig1 = create_hybrid_complex_resnet_framework()
    fig1.savefig(os.path.join(output_dir, 'hybrid_complex_resnet_framework.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Generate residual block details
    print("2. 生成残差块详细结构图...")
    fig2 = create_residual_block_detail()
    fig2.savefig(os.path.join(output_dir, 'residual_blocks_detail.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Generate complex processing illustration
    print("3. 生成复数处理原理图...")
    fig3 = create_complex_processing_illustration()
    fig3.savefig(os.path.join(output_dir, 'complex_processing_illustration.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("所有图表已生成完成!")
    print(f"图片保存位置: {output_dir}")
    print("生成的文件:")
    print("- hybrid_complex_resnet_framework.png: 主体架构图")
    print("- residual_blocks_detail.png: 残差块详细结构")
    print("- complex_processing_illustration.png: 复数处理原理")

if __name__ == "__main__":
    main()
