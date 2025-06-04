import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow, Polygon
from matplotlib.text import TextPath
import numpy as np
import matplotlib.font_manager as fm

# 设置全局样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

def draw_neural_network():
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_aspect('equal')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    plt.title('Lightweight Hybrid Neural Network Architecture', 
              fontsize=16, pad=20, fontweight='bold')
    
    # 定义颜色方案
    colors = {
        'input': '#5DADE2',
        'conv': '#3498DB',
        'complex': '#2E86C1',
        'residual': '#E74C3C',
        'dense': '#27AE60',
        'output': '#E67E22',
        'arrow': '#7F8C8D',
        'text': '#2C3E50'
    }
    
    # 绘制输入层
    draw_input_section(ax, 1, 8, colors)
    
    # 绘制复数处理部分
    draw_complex_processing(ax, 4, 8, colors)
    
    # 绘制残差块部分
    draw_residual_blocks(ax, 7, 8, colors)
    
    # 绘制输出部分
    draw_output_section(ax, 11, 8, colors)
    
    # 添加连接箭头
    draw_connections(ax, colors)
    
    # 添加图例
    draw_legend(ax, colors)
    
    # 添加注释
    plt.figtext(0.5, 0.02, 
                "Fig 1: Lightweight Hybrid Architecture with Complex Processing and Residual Connections",
                ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('lightweight_hybrid_architecture.png', bbox_inches='tight', dpi=300)
    plt.show()

def draw_input_section(ax, x, y, colors):
    """绘制输入部分"""
    # 输入框
    input_box = Rectangle((x-0.8, y-1.5), 1.6, 3, 
                          facecolor=colors['input'], edgecolor='k', alpha=0.8)
    ax.add_patch(input_box)
    
    # 输入文本
    ax.text(x, y+1.2, "Input", ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(x, y, "I/Q Data\n(2×128)", ha='center', va='center', 
            fontsize=9, color='white')
    
    # 箭头
    ax.annotate('', xy=(x+1, y), xytext=(x+0.8, y),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['arrow']))
    
    # Permute操作
    permute_box = Rectangle((x+1.2, y-0.7), 1.2, 1.4, 
                            facecolor=colors['complex'], edgecolor='k', alpha=0.7)
    ax.add_patch(permute_box)
    ax.text(x+1.8, y, "Permute\n(128×2)", ha='center', va='center', 
            fontsize=8, color='white')
    
    return x+2.4

def draw_complex_processing(ax, x, y, colors):
    """绘制复数处理部分"""
    # 主标题
    ax.text(x+1, y+2.2, "Complex Processing", 
            ha='center', va='center', fontsize=12, fontweight='bold', color=colors['complex'])
    
    # 复数卷积块
    conv_box = Rectangle((x, y-0.8), 2, 1.6, facecolor=colors['complex'], edgecolor='k', alpha=0.8)
    ax.add_patch(conv_box)
    ax.text(x+1, y, "ComplexConv1D\n32@5×5", ha='center', va='center', 
            fontsize=9, color='white')
    
    # 批归一化
    bn_box = Rectangle((x+2.2, y-0.6), 1.5, 1.2, facecolor=colors['complex'], edgecolor='k', alpha=0.7)
    ax.add_patch(bn_box)
    ax.text(x+2.95, y, "ComplexBN", ha='center', va='center', 
            fontsize=8, color='white')
    
    # 激活函数
    act_box = Rectangle((x+3.9, y-0.6), 1.5, 1.2, facecolor=colors['complex'], edgecolor='k', alpha=0.7)
    ax.add_patch(act_box)
    ax.text(x+4.65, y, "C-LeakyReLU", ha='center', va='center', 
            fontsize=8, color='white')
    
    # 池化
    pool_box = Rectangle((x+5.7, y-0.6), 1.5, 1.2, facecolor=colors['complex'], edgecolor='k', alpha=0.7)
    ax.add_patch(pool_box)
    ax.text(x+6.45, y, "ComplexPool", ha='center', va='center', 
            fontsize=8, color='white')
    
    # 复数处理细节
    ax.text(x+3.4, y-1.8, "Complex Number Processing:", 
            ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text'])
    
    # 复数运算图示
    ax.plot([x+1, x+5], [y-2.5, y-2.5], 'k-', lw=0.5)
    ax.text(x+1, y-2.2, "Input (Real)", ha='center', va='center', fontsize=7)
    ax.text(x+3, y-2.2, "Input (Imag)", ha='center', va='center', fontsize=7)
    
    # 复数卷积公式
    ax.text(x+3, y-3.0, r"$\begin{pmatrix} a \\ b \end{pmatrix} * \begin{pmatrix} c & -d \\ d & c \end{pmatrix} = \begin{pmatrix} ac - bd \\ ad + bc \end{pmatrix}$", 
            ha='center', va='center', fontsize=10)
    
    return x+7.2

def draw_residual_blocks(ax, x, y, colors):
    """绘制残差块部分"""
    # 主标题
    ax.text(x+1.5, y+2.2, "Residual Blocks", 
            ha='center', va='center', fontsize=12, fontweight='bold', color=colors['residual'])
    
    # 绘制三个残差块
    for i in range(3):
        draw_residual_block(ax, x+i*1.5, y, colors, i)
    
    # 残差连接细节
    ax.text(x+1.5, y-1.8, "Residual Connection Mechanism:", 
            ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text'])
    
    # 残差公式
    ax.text(x+1.5, y-2.6, r"$\mathcal{F}(x) + x$", 
            ha='center', va='center', fontsize=14, color=colors['residual'])
    
    # 残差图示
    ax.plot([x-0.5, x+3.5], [y-3.2, y-3.2], 'k-', lw=0.5)
    ax.annotate('', xy=(x+0.5, y-3.5), xytext=(x+0.5, y-3.0),
                arrowprops=dict(arrowstyle='->', lw=1, color=colors['residual']))
    ax.text(x+0.5, y-3.7, "Residual", ha='center', va='center', fontsize=8)
    
    ax.annotate('', xy=(x+2.5, y-3.5), xytext=(x+2.5, y-3.0),
                arrowprops=dict(arrowstyle='->', lw=1, color=colors['residual']))
    ax.text(x+2.5, y-3.7, "Identity", ha='center', va='center', fontsize=8)
    
    return x+4.5

def draw_residual_block(ax, x, y, colors, idx):
    """绘制单个残差块"""
    # 块容器
    block = Rectangle((x-0.7, y-1.2), 1.4, 2.4, facecolor='white', 
                      edgecolor=colors['residual'], lw=1.5, alpha=0.9)
    ax.add_patch(block)
    
    # 块标题
    ax.text(x, y+1.0, f"ResBlock {idx+1}", ha='center', va='center', 
            fontsize=9, color=colors['residual'], fontweight='bold')
    
    # 内部层
    layers = [
        ("ComplexConv", colors['conv']),
        ("C-BN", colors['complex']),
        ("C-LeakyReLU", colors['complex']),
        ("ComplexConv", colors['conv']),
        ("C-BN", colors['complex'])
    ]
    
    for i, (name, color) in enumerate(layers):
        layer = Rectangle((x-0.5, y-0.8+i*0.4), 1.0, 0.3, 
                         facecolor=color, edgecolor='k', alpha=0.7)
        ax.add_patch(layer)
        ax.text(x, y-0.65+i*0.4, name, ha='center', va='center', 
                fontsize=7, color='white')
    
    # 输入输出
    ax.plot([x-1.0, x-0.7], [y, y], 'k-', lw=1.5)
    ax.plot([x+0.7, x+1.0], [y, y], 'k-', lw=1.5)
    
    # 残差连接
    ax.plot([x-0.7, x+0.7], [y+0.8, y+0.8], color=colors['residual'], 
            linestyle='--', lw=1.5)
    ax.plot([x-0.7, x-0.7], [y, y+0.8], color=colors['residual'], 
            linestyle='--', lw=1.5)
    ax.plot([x+0.7, x+0.7], [y, y+0.8], color=colors['residual'], 
            linestyle='--', lw=1.5)
    
    # 加法操作
    add_circle = Circle((x, y-1.0), 0.2, facecolor=colors['residual'], 
                        edgecolor='k', alpha=0.8)
    ax.add_patch(add_circle)
    ax.text(x, y-1.0, "+", ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')

def draw_output_section(ax, x, y, colors):
    """绘制输出部分"""
    # 全局池化
    pool_box = Rectangle((x-0.7, y-0.6), 1.4, 1.2, 
                         facecolor=colors['complex'], edgecolor='k', alpha=0.8)
    ax.add_patch(pool_box)
    ax.text(x, y, "Global\nPooling", ha='center', va='center', 
            fontsize=9, color='white')
    
    # 复数全连接
    dense_box = Rectangle((x+1.0, y-0.8), 1.6, 1.6, 
                          facecolor=colors['dense'], edgecolor='k', alpha=0.8)
    ax.add_patch(dense_box)
    ax.text(x+1.8, y, "ComplexDense\n512", ha='center', va='center', 
            fontsize=9, color='white')
    
    # Dropout
    drop_box = Rectangle((x+2.8, y-0.6), 1.2, 1.2, 
                         facecolor=colors['dense'], edgecolor='k', alpha=0.7)
    ax.add_patch(drop_box)
    ax.text(x+3.4, y, "Dropout\n0.5", ha='center', va='center', 
            fontsize=8, color='white')
    
    # 复数转实数
    convert_box = Rectangle((x+4.2, y-0.7), 1.4, 1.4, 
                            facecolor=colors['output'], edgecolor='k', alpha=0.8)
    ax.add_patch(convert_box)
    ax.text(x+4.9, y, "Complex to\nReal", ha='center', va='center', 
            fontsize=9, color='white')
    
    # 实数全连接
    real_dense = Rectangle((x+5.8, y-0.8), 1.6, 1.6, 
                           facecolor=colors['dense'], edgecolor='k', alpha=0.8)
    ax.add_patch(real_dense)
    ax.text(x+6.6, y, "Dense\n256", ha='center', va='center', 
            fontsize=9, color='white')
    
    # 输出层
    output_box = Rectangle((x+7.8, y-0.7), 1.4, 1.4, 
                           facecolor=colors['output'], edgecolor='k', alpha=0.9)
    ax.add_patch(output_box)
    ax.text(x+8.5, y, "Output\nSoftmax", ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')
    
    # 转换细节
    ax.text(x+4.9, y-1.8, "Complex to Real Conversion:", 
            ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text'])
    
    # 幅度计算
    ax.text(x+4.9, y-2.6, r"$|z| = \sqrt{\text{Re}(z)^2 + \text{Im}(z)^2}$", 
            ha='center', va='center', fontsize=12, color=colors['output'])

def draw_connections(ax, colors):
    """绘制连接箭头"""
    # 主要数据流
    ax.annotate('', xy=(3.4, 8), xytext=(1.8, 8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['arrow']))
    
    ax.annotate('', xy=(7.2, 8), xytext=(4.4, 8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['arrow']))
    
    ax.annotate('', xy=(11.0, 8), xytext=(8.4, 8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['arrow']))
    
    # 复数处理部分的连接
    for i in range(3):
        ax.annotate('', xy=(4.0+i*1.1, 8), xytext=(3.4+i*1.1, 8),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color=colors['complex']))
    
    # 残差块间的连接
    for i in range(2):
        ax.annotate('', xy=(7.5+i*1.5, 8), xytext=(7.0+i*1.5, 8),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color=colors['residual']))
    
    # 输出部分的连接
    connections = [
        (11.7, 12.8), (13.0, 14.2), (15.2, 16.6), (17.4, 18.5)
    ]
    
    for i, (start, end) in enumerate(connections):
        ax.annotate('', xy=(end, 8), xytext=(start, 8),
                    arrowprops=dict(arrowstyle='->', lw=1.2, 
                    color=colors['dense'] if i < 3 else colors['output']))

def draw_legend(ax, colors):
    """绘制图例"""
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Input Layer',
                  markerfacecolor=colors['input'], markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Complex Processing',
                  markerfacecolor=colors['complex'], markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Residual Block',
                  markerfacecolor='white', markeredgecolor=colors['residual'], markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Dense Layer',
                  markerfacecolor=colors['dense'], markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Output Layer',
                  markerfacecolor=colors['output'], markersize=12),
        plt.Line2D([0], [0], color=colors['residual'], lw=2, label='Residual Connection',
                  linestyle='--'),
        plt.Line2D([0], [0], color=colors['arrow'], lw=2, label='Data Flow', 
                  marker='>', markersize=10)
    ]
    
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=9, frameon=True)

# 生成架构图
draw_neural_network()