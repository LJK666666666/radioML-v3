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
    保存复数残差连接的详细技术图
    """
    # 确保目录存在
    figure_dir = os.path.join('script', 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    
    # 创建并保存复数残差详细图
    fig2 = create_complex_residual_detailed()
    path2 = os.path.join(figure_dir, 'complex_residual_detailed.png')
    fig2.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 也保存PDF版本
    fig2.savefig(os.path.join(figure_dir, 'complex_residual_detailed.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    print(f"复数残差连接可视化已保存:")
    print(f"复数残差块详细技术: {path2}")
    
    return path2

if __name__ == "__main__":
    print("创建复数残差连接可视化图表...")
    
    # 保存可视化图表
    path2 = save_residual_connection_visualizations()
    
    print("\n🎯 复数残差连接核心概念总结:")
    print("• 主路径: 学习残差函数 F(z)")
    print("• 跳跃连接: 保持原始复数信息 z")  
    print("• 复数相加: output = F(z) + z")
    print("• 关键作用: 解决梯度消失，保持I/Q信号完整性")
    print("• 在复数域: 适合射频信号处理任务")
    
    print("\n📊 生成的可视化文件:")
    print(f"  {path2}")
    print("\n这个图表详细展示了复数残差连接的技术实现！")
