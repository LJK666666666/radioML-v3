"""
残差连接可视化脚本

这个脚本创建清晰的图表来解释残差连接的工作原理和重要性
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_residual_connection_explanation():
    """
    创建残差连接的详细解释图
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # === 子图1: 传统网络 vs 残差网络 ===
    ax1.set_title('传统网络 vs 残差网络对比', fontsize=14, fontweight='bold')
    
    # 传统网络
    ax1.text(0.25, 0.8, '传统网络', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    # 画传统网络流程
    boxes_traditional = [
        {'pos': (0.1, 0.6), 'text': '输入\\nx', 'color': '#E3F2FD'},
        {'pos': (0.25, 0.4), 'text': 'F(x)\\n(多层网络)', 'color': '#FFCDD2'},
        {'pos': (0.4, 0.6), 'text': '输出\\ny = F(x)', 'color': '#E8F5E8'}
    ]
    
    for box in boxes_traditional:
        rect = FancyBboxPatch(
            (box['pos'][0]-0.05, box['pos'][1]-0.08), 0.1, 0.16,
            boxstyle="round,pad=0.02", facecolor=box['color'],
            edgecolor='black', linewidth=1
        )
        ax1.add_patch(rect)
        ax1.text(box['pos'][0], box['pos'][1], box['text'], 
                ha='center', va='center', fontsize=9)
    
    # 传统网络箭头
    ax1.annotate('', xy=(0.2, 0.45), xytext=(0.15, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax1.annotate('', xy=(0.35, 0.55), xytext=(0.3, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # 残差网络
    ax1.text(0.75, 0.8, '残差网络', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # 画残差网络流程
    boxes_residual = [
        {'pos': (0.6, 0.6), 'text': '输入\\nx', 'color': '#E3F2FD'},
        {'pos': (0.75, 0.4), 'text': 'F(x)\\n(多层网络)', 'color': '#FFCDD2'},
        {'pos': (0.9, 0.6), 'text': '输出\\ny = F(x) + x', 'color': '#E8F5E8'}
    ]
    
    for box in boxes_residual:
        rect = FancyBboxPatch(
            (box['pos'][0]-0.05, box['pos'][1]-0.08), 0.1, 0.16,
            boxstyle="round,pad=0.02", facecolor=box['color'],
            edgecolor='black', linewidth=1
        )
        ax1.add_patch(rect)
        ax1.text(box['pos'][0], box['pos'][1], box['text'], 
                ha='center', va='center', fontsize=9)
    
    # 残差网络箭头 - 主路径
    ax1.annotate('', xy=(0.7, 0.45), xytext=(0.65, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax1.annotate('', xy=(0.85, 0.55), xytext=(0.8, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # 残差网络箭头 - 跳跃连接
    ax1.annotate('', xy=(0.85, 0.6), xytext=(0.65, 0.6),
                arrowprops=dict(arrowstyle='->', lw=3, color='green',
                              connectionstyle="arc3,rad=0.3"))
    ax1.text(0.75, 0.7, '跳跃连接', ha='center', fontsize=10, color='green',
            fontweight='bold')
    
    # 添加数学公式
    ax1.text(0.25, 0.2, 'y = F(x)', ha='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    ax1.text(0.75, 0.2, 'y = F(x) + x', ha='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # === 子图2: 梯度流动对比 ===
    ax2.set_title('梯度流动：为什么残差连接有效', fontsize=14, fontweight='bold')
    
    # 传统深度网络的梯度消失
    ax2.text(0.5, 0.9, '传统深度网络：梯度消失问题', ha='center', fontsize=12, 
            fontweight='bold', color='red')
    
    layers_traditional = np.linspace(0.1, 0.9, 6)
    gradient_values = np.array([1.0, 0.8, 0.5, 0.2, 0.05, 0.01])
    
    for i, (x, grad) in enumerate(zip(layers_traditional, gradient_values)):
        color_intensity = grad
        ax2.add_patch(plt.Rectangle((x-0.03, 0.6), 0.06, 0.2, 
                                   facecolor=plt.cm.Reds(color_intensity), 
                                   edgecolor='black'))
        ax2.text(x, 0.7, f'Layer{i+1}', ha='center', va='center', fontsize=8)
        ax2.text(x, 0.55, f'{grad:.2f}', ha='center', va='center', fontsize=8)
        
        if i < len(layers_traditional) - 1:
            ax2.annotate('', xy=(layers_traditional[i+1]-0.03, 0.7), 
                        xytext=(x+0.03, 0.7),
                        arrowprops=dict(arrowstyle='->', lw=1, color='red'))
    
    # 残差网络的梯度流动
    ax2.text(0.5, 0.4, '残差网络：梯度畅通流动', ha='center', fontsize=12, 
            fontweight='bold', color='green')
    
    for i, x in enumerate(layers_traditional):
        ax2.add_patch(plt.Rectangle((x-0.03, 0.1), 0.06, 0.2, 
                                   facecolor=plt.cm.Greens(0.8), 
                                   edgecolor='black'))
        ax2.text(x, 0.2, f'Layer{i+1}', ha='center', va='center', fontsize=8)
        ax2.text(x, 0.05, '1.0+', ha='center', va='center', fontsize=8)
        
        if i < len(layers_traditional) - 1:
            ax2.annotate('', xy=(layers_traditional[i+1]-0.03, 0.2), 
                        xytext=(x+0.03, 0.2),
                        arrowprops=dict(arrowstyle='->', lw=1, color='blue'))
    
    # 跳跃连接
    ax2.annotate('', xy=(0.85, 0.2), xytext=(0.15, 0.2),
                arrowprops=dict(arrowstyle='->', lw=3, color='green',
                              connectionstyle="arc3,rad=-0.3"))
    ax2.text(0.5, 0.32, '跳跃连接保证梯度≥1', ha='center', fontsize=10, 
            color='green', fontweight='bold')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # === 子图3: 详细的残差块结构 ===
    ax3.set_title('复数残差块详细结构', fontsize=14, fontweight='bold')
    
    # 绘制详细的残差块
    components = [
        {'name': '输入\\n(batch, time, 2*filters)', 'pos': (0.2, 0.8), 'color': '#E3F2FD'},
        {'name': 'ComplexConv1D\\n+ BatchNorm\\n+ Activation', 'pos': (0.2, 0.6), 'color': '#FFE0B2'},
        {'name': 'ComplexConv1D\\n+ BatchNorm', 'pos': (0.2, 0.4), 'color': '#FFE0B2'},
        {'name': '跳跃连接\\n(1x1 Conv如需要)', 'pos': (0.7, 0.6), 'color': '#C8E6C9'},
        {'name': '相加\\n(Add)', 'pos': (0.2, 0.2), 'color': '#F8BBD9'},
        {'name': '最终激活\\n输出', 'pos': (0.2, 0.05), 'color': '#D1C4E9'}
    ]
    
    for comp in components:
        x, y = comp['pos']
        if '跳跃连接' in comp['name']:
            w, h = 0.25, 0.15
        else:
            w, h = 0.2, 0.12
        
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=1
        )
        ax3.add_patch(box)
        ax3.text(x, y, comp['name'], ha='center', va='center', fontsize=9)
    
    # 主路径箭头
    main_path_connections = [
        ((0.2, 0.74), (0.2, 0.66)),
        ((0.2, 0.54), (0.2, 0.46)),
        ((0.2, 0.34), (0.2, 0.26))
    ]
    
    for start, end in main_path_connections:
        ax3.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # 跳跃连接箭头
    ax3.annotate('', xy=(0.7, 0.53), xytext=(0.2, 0.73),
                arrowprops=dict(arrowstyle='->', lw=2, color='green',
                              connectionstyle="arc3,rad=0.3"))
    ax3.annotate('', xy=(0.32, 0.2), xytext=(0.58, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='green',
                              connectionstyle="arc3,rad=0.3"))
    
    # 最终激活箭头
    ax3.annotate('', xy=(0.2, 0.11), xytext=(0.2, 0.14),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    
    # 添加说明
    ax3.text(0.8, 0.3, '关键特点:\\n• 主路径学习残差\\n• 跳跃连接保持原信息\\n• 相加融合两路信息\\n• 解决梯度消失', 
            fontsize=10, va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # === 子图4: 性能对比 ===
    ax4.set_title('残差连接带来的性能提升', fontsize=14, fontweight='bold')
    
    # 模拟数据：深度vs准确率
    depths = np.array([6, 12, 18, 24, 30, 36])
    accuracy_traditional = np.array([85, 87, 88, 85, 80, 75])  # 传统网络：深度增加性能下降
    accuracy_residual = np.array([85, 89, 92, 94, 95, 96])     # 残差网络：深度增加性能提升
    
    ax4.plot(depths, accuracy_traditional, 'ro-', linewidth=2, markersize=8, 
            label='传统网络', color='red')
    ax4.plot(depths, accuracy_residual, 'go-', linewidth=2, markersize=8, 
            label='残差网络', color='green')
    
    ax4.fill_between(depths, accuracy_traditional, accuracy_residual, 
                    alpha=0.3, color='yellow', label='性能提升区域')
    
    ax4.set_xlabel('网络深度(层数)', fontsize=12)
    ax4.set_ylabel('准确率(%)', fontsize=12)
    ax4.set_ylim(70, 100)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # 添加注释
    ax4.annotate('传统网络：\\n深度增加导致\\n性能下降', 
                xy=(30, 80), xytext=(25, 72),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    ax4.annotate('残差网络：\\n深度增加带来\\n性能提升', 
                xy=(36, 96), xytext=(33, 85),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_complex_residual_detailed():
    """
    创建复数残差连接的详细技术图
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    ax.set_title('复数残差块技术实现细节', fontsize=16, fontweight='bold')
    
    # 层的详细规格
    layers = [
        # 输入
        {'name': '复数输入', 'detail': 'shape: (batch, time, 2*filters)\\nReal + Imag 交替排列', 
         'pos': (2, 9), 'size': (2.5, 1), 'color': '#E3F2FD'},
        
        # 主路径第一层
        {'name': 'ComplexConv1D', 'detail': 'filters: 64\\nkernel: 3\\nstrides: 1', 
         'pos': (2, 7.5), 'size': (2.5, 1), 'color': '#FFE0B2'},
        {'name': 'ComplexBN', 'detail': '复数批标准化\\n分别处理实部虚部', 
         'pos': (2, 6), 'size': (2.5, 1), 'color': '#C8E6C9'},
        {'name': 'ComplexActivation', 'detail': 'LeakyReLU\\n应用于实部和虚部', 
         'pos': (2, 4.5), 'size': (2.5, 1), 'color': '#FFCCBC'},
        
        # 主路径第二层
        {'name': 'ComplexConv1D', 'detail': 'filters: 64\\nkernel: 3\\nstrides: 1', 
         'pos': (2, 3), 'size': (2.5, 1), 'color': '#FFE0B2'},
        {'name': 'ComplexBN', 'detail': '复数批标准化\\n(无激活)', 
         'pos': (2, 1.5), 'size': (2.5, 1), 'color': '#C8E6C9'},
        
        # 跳跃连接
        {'name': '跳跃连接判断', 'detail': 'if 维度不匹配:\\n  1x1 ComplexConv\\nelse:\\n  直接连接', 
         'pos': (7, 6), 'size': (3, 2), 'color': '#F8BBD9'},
        
        # 相加和输出
        {'name': '复数相加', 'detail': 'output = main + skip\\n逐元素相加', 
         'pos': (2, 0), 'size': (2.5, 1), 'color': '#D1C4E9'},
        {'name': '最终激活', 'detail': 'ComplexActivation\\n输出结果', 
         'pos': (2, -1.5), 'size': (2.5, 1), 'color': '#DCEDC8'}
    ]
    
    # 绘制所有层
    for layer in layers:
        x, y = layer['pos']
        w, h = layer['size']
        
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # 层名称
        ax.text(x, y + h/4, layer['name'], ha='center', va='center', 
                fontsize=11, fontweight='bold')
        # 详细信息
        ax.text(x, y - h/4, layer['detail'], ha='center', va='center', 
                fontsize=9, style='italic')
    
    # 主路径连接
    main_connections = [
        ((2, 8.5), (2, 8)),      # 输入到Conv1
        ((2, 7), (2, 6.5)),      # Conv1到BN1
        ((2, 5.5), (2, 5)),      # BN1到Act1
        ((2, 4), (2, 3.5)),      # Act1到Conv2
        ((2, 2.5), (2, 2)),      # Conv2到BN2
        ((2, 1), (2, 0.5)),      # BN2到Add
        ((2, -0.5), (2, -1))     # Add到Final
    ]
    
    for start, end in main_connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # 跳跃连接
    ax.annotate('', xy=(5.5, 6), xytext=(3.25, 9),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='green',
                             connectionstyle="arc3,rad=0.3"))
    ax.annotate('', xy=(3.25, 0), xytext=(5.5, 6),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='green',
                             connectionstyle="arc3,rad=0.3"))
    
    # 添加数学公式说明
    math_text = """
复数残差连接数学表示：

设输入为 z = x + iy (复数形式)
主路径输出: F(z) = F(x + iy)
跳跃连接: z 或 W_s(z) (如需维度调整)

最终输出: output = F(z) + skip_connection
其中复数加法为: (a+bi) + (c+di) = (a+c) + (b+d)i

梯度流: ∂L/∂z = ∂L/∂F(z) · ∂F(z)/∂z + ∂L/∂z
最后一项确保梯度至少为1，避免消失
"""
    
    ax.text(10.5, 3, math_text, fontsize=10, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # 添加优势说明
    advantages_text = """
复数残差连接的优势：

✓ 保持I/Q信号的相位信息
✓ 复数域的梯度流动更自然
✓ 结合了ResNet和ComplexNN优点
✓ 适合射频信号处理任务
✓ 减少梯度消失问题
"""
    
    ax.text(10.5, 0, advantages_text, fontsize=10, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(-1, 14)
    ax.set_ylim(-3, 10)
    ax.axis('off')
    
    return fig

def save_residual_connection_visualizations():
    """
    保存残差连接的可视化图表
    """
    # 确保目录存在
    figure_dir = os.path.join('script', 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    
    # 创建并保存第一个图表
    fig1 = create_residual_connection_explanation()
    path1 = os.path.join(figure_dir, 'residual_connection_explanation.png')
    fig1.savefig(path1, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 创建并保存第二个图表
    fig2 = create_complex_residual_detailed()
    path2 = os.path.join(figure_dir, 'complex_residual_detailed.png')
    fig2.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 也保存PDF版本
    fig1.savefig(os.path.join(figure_dir, 'residual_connection_explanation.pdf'), 
                bbox_inches='tight', facecolor='white')
    fig2.savefig(os.path.join(figure_dir, 'complex_residual_detailed.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    print(f"残差连接可视化已保存:")
    print(f"1. 残差连接原理解释: {path1}")
    print(f"2. 复数残差块详细技术: {path2}")
    
    return path1, path2

if __name__ == "__main__":
    print("创建残差连接可视化图表...")
    
    # 保存可视化图表
    path1, path2 = save_residual_connection_visualizations()
    
    print("\\n🎯 残差连接核心概念总结:")
    print("• 主路径: 学习残差函数 F(x)")
    print("• 跳跃连接: 保持原始信息 x")  
    print("• 相加操作: output = F(x) + x")
    print("• 关键作用: 解决梯度消失，使深度网络训练成为可能")
    print("• 在复数域: 保持I/Q信号的完整性")
    
    print("\\n📊 生成的可视化文件:")
    print(f"  {path1}")
    print(f"  {path2}")
    print("\\n这些图表清楚地展示了残差连接的工作原理和重要性！")
