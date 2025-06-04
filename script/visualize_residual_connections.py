"""
残差连接可视化 - 直观展示工作原理
让你一眼就能看懂残差连接是怎么工作的！
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns

# 设置中文字体和样式
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")

def draw_traditional_vs_residual():
    """对比传统网络和残差网络"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ============ 传统网络 ============
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    ax1.set_title('🔴 传统深度网络 - 梯度消失问题', fontsize=16, fontweight='bold')
    
    # 绘制层
    layers = ['输入', 'Conv1', 'Conv2', 'Conv3', 'Conv4', '输出']
    positions = [1, 2.5, 4, 5.5, 7, 8.5]
    
    for i, (layer, pos) in enumerate(zip(layers, positions)):
        if i == 0:
            color = '#4CAF50'  # 绿色输入
        elif i == len(layers) - 1:
            color = '#F44336'  # 红色输出
        else:
            color = '#2196F3'  # 蓝色隐藏层
            
        box = FancyBboxPatch((pos-0.3, 1), 0.6, 1, boxstyle="round,pad=0.1", 
                           facecolor=color, alpha=0.7, edgecolor='black')
        ax1.add_patch(box)
        ax1.text(pos, 1.5, layer, ha='center', va='center', fontweight='bold', color='white')
    
    # 绘制前向箭头
    for i in range(len(positions) - 1):
        arrow = ConnectionPatch((positions[i]+0.3, 1.5), (positions[i+1]-0.3, 1.5), 
                              "data", "data", arrowstyle="-|>", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="black", alpha=0.8)
        ax1.add_patch(arrow)
    
    # 绘制梯度消失
    gradient_values = [1.0, 0.7, 0.4, 0.2, 0.1, 0.05]
    for i, (pos, grad) in enumerate(zip(positions[::-1], gradient_values)):
        if i < len(positions) - 1:
            arrow = ConnectionPatch((pos-0.3, 0.5), (positions[::-1][i+1]+0.3, 0.5), 
                                  "data", "data", arrowstyle="-|>", shrinkA=0, shrinkB=0,
                                  mutation_scale=10*grad, fc="red", alpha=grad)
            ax1.add_patch(arrow)
            ax1.text(pos, 0.2, f'{grad:.1f}', ha='center', va='center', 
                    color='red', fontweight='bold')
    
    ax1.text(5, 0.5, '梯度反向传播 ←', ha='center', va='center', 
            color='red', fontsize=12, fontweight='bold')
    ax1.text(5, 2.7, '⚠️ 问题：梯度越来越小，前面的层学不到东西！', 
            ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # ============ 残差网络 ============
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 4)
    ax2.set_title('✅ 残差网络 - 梯度流畅传播', fontsize=16, fontweight='bold')
    
    # 绘制主要层
    for i, (layer, pos) in enumerate(zip(layers, positions)):
        if i == 0:
            color = '#4CAF50'
        elif i == len(layers) - 1:
            color = '#F44336'
        else:
            color = '#2196F3'
            
        box = FancyBboxPatch((pos-0.3, 2), 0.6, 1, boxstyle="round,pad=0.1", 
                           facecolor=color, alpha=0.7, edgecolor='black')
        ax2.add_patch(box)
        ax2.text(pos, 2.5, layer, ha='center', va='center', fontweight='bold', color='white')
    
    # 绘制主路径箭头
    for i in range(len(positions) - 1):
        arrow = ConnectionPatch((positions[i]+0.3, 2.5), (positions[i+1]-0.3, 2.5), 
                              "data", "data", arrowstyle="-|>", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="black", alpha=0.8)
        ax2.add_patch(arrow)
    
    # 绘制残差连接（跳跃连接）
    # 从输入到Conv3
    skip1 = ConnectionPatch((1.3, 2.2), (5.2, 2.2), "data", "data", 
                           arrowstyle="-|>", shrinkA=0, shrinkB=0,
                           mutation_scale=20, fc="orange", alpha=0.8,
                           connectionstyle="arc3,rad=0.3", linewidth=3)
    ax2.add_patch(skip1)
    ax2.text(3.2, 3.2, '跳跃连接 1', ha='center', va='center', 
            color='orange', fontsize=10, fontweight='bold')
    
    # 从Conv2到输出
    skip2 = ConnectionPatch((4.3, 2.2), (8.2, 2.2), "data", "data", 
                           arrowstyle="-|>", shrinkA=0, shrinkB=0,
                           mutation_scale=20, fc="orange", alpha=0.8,
                           connectionstyle="arc3,rad=-0.3", linewidth=3)
    ax2.add_patch(skip2)
    ax2.text(6.2, 1.2, '跳跃连接 2', ha='center', va='center', 
            color='orange', fontsize=10, fontweight='bold')
    
    # 绘制梯度流
    gradient_values_res = [1.0, 0.9, 0.8, 0.7, 0.8, 0.9]  # 残差网络梯度保持较大
    for i, (pos, grad) in enumerate(zip(positions[::-1], gradient_values_res)):
        if i < len(positions) - 1:
            arrow = ConnectionPatch((pos-0.3, 1.3), (positions[::-1][i+1]+0.3, 1.3), 
                                  "data", "data", arrowstyle="-|>", shrinkA=0, shrinkB=0,
                                  mutation_scale=15*grad, fc="green", alpha=0.8)
            ax2.add_patch(arrow)
            ax2.text(pos, 1.0, f'{grad:.1f}', ha='center', va='center', 
                    color='green', fontweight='bold')
    
    ax2.text(5, 1.3, '梯度反向传播 ←', ha='center', va='center', 
            color='green', fontsize=12, fontweight='bold')
    ax2.text(5, 3.7, '✅ 解决：梯度通过跳跃连接保持强度！', 
            ha='center', va='center', color='green', fontsize=12, fontweight='bold')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('d:/1python programs/radioml/radioML-v3/script/residual_vs_traditional.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def draw_residual_block_detail():
    """详细展示单个残差块的工作原理"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_title('🔍 残差块详细解析 - 看清楚每一步！', fontsize=16, fontweight='bold')
    
    # 输入
    input_box = FancyBboxPatch((0.5, 3.5), 1, 1, boxstyle="round,pad=0.1", 
                              facecolor='#4CAF50', alpha=0.8, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1, 4, '输入\nX', ha='center', va='center', fontweight='bold', 
           color='white', fontsize=12)
    
    # 主路径
    # Conv1
    conv1_box = FancyBboxPatch((2.5, 5), 1.2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='#2196F3', alpha=0.8, edgecolor='black')
    ax.add_patch(conv1_box)
    ax.text(3.1, 5.4, 'Conv1D\n+ BN + ReLU', ha='center', va='center', 
           fontweight='bold', color='white', fontsize=10)
    
    # Conv2
    conv2_box = FancyBboxPatch((4.5, 5), 1.2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='#2196F3', alpha=0.8, edgecolor='black')
    ax.add_patch(conv2_box)
    ax.text(5.1, 5.4, 'Conv1D\n+ BN', ha='center', va='center', 
           fontweight='bold', color='white', fontsize=10)
    
    # 相加操作
    add_circle = plt.Circle((7, 4), 0.4, facecolor='#FF9800', alpha=0.8, 
                           edgecolor='black', linewidth=2)
    ax.add_patch(add_circle)
    ax.text(7, 4, '+', ha='center', va='center', fontweight='bold', 
           color='white', fontsize=20)
    
    # 最终激活
    final_box = FancyBboxPatch((8.2, 3.5), 1, 1, boxstyle="round,pad=0.1", 
                              facecolor='#9C27B0', alpha=0.8, edgecolor='black')
    ax.add_patch(final_box)
    ax.text(8.7, 4, 'ReLU\n激活', ha='center', va='center', fontweight='bold', 
           color='white', fontsize=10)
    
    # 主路径箭头
    # 输入到Conv1
    arrow1 = ConnectionPatch((1.5, 4.2), (2.5, 5.2), "data", "data", 
                           arrowstyle="-|>", shrinkA=0, shrinkB=0,
                           mutation_scale=20, fc="black", alpha=0.8)
    ax.add_patch(arrow1)
    
    # Conv1到Conv2
    arrow2 = ConnectionPatch((3.7, 5.4), (4.5, 5.4), "data", "data", 
                           arrowstyle="-|>", shrinkA=0, shrinkB=0,
                           mutation_scale=20, fc="black", alpha=0.8)
    ax.add_patch(arrow2)
    
    # Conv2到加法
    arrow3 = ConnectionPatch((5.7, 5.2), (6.7, 4.3), "data", "data", 
                           arrowstyle="-|>", shrinkA=0, shrinkB=0,
                           mutation_scale=20, fc="black", alpha=0.8)
    ax.add_patch(arrow3)
    
    # 跳跃连接
    skip_arrow = ConnectionPatch((1.5, 3.8), (6.6, 3.8), "data", "data", 
                               arrowstyle="-|>", shrinkA=0, shrinkB=0,
                               mutation_scale=25, fc="red", alpha=0.9,
                               connectionstyle="arc3,rad=-0.3", linewidth=4)
    ax.add_patch(skip_arrow)
    
    # 加法到输出
    arrow4 = ConnectionPatch((7.4, 4), (8.2, 4), "data", "data", 
                           arrowstyle="-|>", shrinkA=0, shrinkB=0,
                           mutation_scale=20, fc="black", alpha=0.8)
    ax.add_patch(arrow4)
    
    # 标签
    ax.text(3.5, 6.2, '主路径', ha='center', va='center', 
           color='blue', fontsize=12, fontweight='bold')
    ax.text(4, 2.8, '跳跃连接（残差连接）', ha='center', va='center', 
           color='red', fontsize=12, fontweight='bold')
    ax.text(1, 2.5, '相同的输入！', ha='center', va='center', 
           color='green', fontsize=10, fontweight='bold')
    
    # 数学公式
    ax.text(5, 1.5, '数学表达式：', ha='center', va='center', 
           fontsize=14, fontweight='bold')
    ax.text(5, 1, 'output = ReLU(F(X) + X)', ha='center', va='center', 
           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    ax.text(5, 0.5, '其中 F(X) 是主路径的处理结果', ha='center', va='center', 
           fontsize=10, style='italic')
    
    # 解释
    explanation = """
    关键理解：
    1. X 同时进入主路径和跳跃连接
    2. 主路径学习特征变换 F(X)
    3. 跳跃连接保持原始信息 X
    4. 最终输出 = F(X) + X
    5. 如果 F(X) = 0，输出就是输入（恒等映射）
    """
    ax.text(1, 7, explanation, ha='left', va='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('d:/1python programs/radioml/radioML-v3/script/residual_block_detail.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def draw_performance_comparison():
    """展示残差网络的性能优势"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 训练损失对比
    epochs = np.arange(1, 101)
    traditional_loss = 2.5 * np.exp(-epochs/50) + 0.5 + 0.3 * np.random.normal(0, 0.1, 100)
    residual_loss = 2.5 * np.exp(-epochs/30) + 0.2 + 0.2 * np.random.normal(0, 0.1, 100)
    
    ax1.plot(epochs, traditional_loss, label='传统网络', linewidth=2, color='red', alpha=0.8)
    ax1.plot(epochs, residual_loss, label='残差网络', linewidth=2, color='blue', alpha=0.8)
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('训练损失')
    ax1.set_title('训练损失对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率对比
    traditional_acc = 1 - traditional_loss / 3
    residual_acc = 1 - residual_loss / 3
    
    ax2.plot(epochs, traditional_acc, label='传统网络', linewidth=2, color='red', alpha=0.8)
    ax2.plot(epochs, residual_acc, label='残差网络', linewidth=2, color='blue', alpha=0.8)
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('准确率')
    ax2.set_title('准确率对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 网络深度 vs 性能
    depths = [10, 20, 30, 40, 50, 60, 70, 80]
    traditional_perf = [85, 87, 88, 87, 85, 82, 78, 75]  # 传统网络性能下降
    residual_perf = [85, 89, 92, 94, 95, 96, 96.5, 97]  # 残差网络性能提升
    
    ax3.plot(depths, traditional_perf, 'o-', label='传统网络', linewidth=2, 
             color='red', alpha=0.8, markersize=8)
    ax3.plot(depths, residual_perf, 's-', label='残差网络', linewidth=2, 
             color='blue', alpha=0.8, markersize=8)
    ax3.set_xlabel('网络深度（层数）')
    ax3.set_ylabel('测试准确率 (%)')
    ax3.set_title('网络深度 vs 性能')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 梯度大小对比
    layers = ['Layer1', 'Layer10', 'Layer20', 'Layer30', 'Layer40', 'Layer50']
    traditional_grad = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    residual_grad = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4]
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax4.bar(x - width/2, traditional_grad, width, label='传统网络', 
           color='red', alpha=0.7)
    ax4.bar(x + width/2, residual_grad, width, label='残差网络', 
           color='blue', alpha=0.7)
    ax4.set_xlabel('网络层')
    ax4.set_ylabel('梯度大小（对数尺度）')
    ax4.set_title('梯度消失对比')
    ax4.set_yscale('log')
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:/1python programs/radioml/radioML-v3/script/residual_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def create_complete_visualization():
    """创建完整的残差连接可视化"""
    
    print("\n" + "🎨 " * 20)
    print("创建残差连接完整可视化")
    print("🎨 " * 20)
    
    print("\n1. 📊 绘制传统网络 vs 残差网络对比...")
    draw_traditional_vs_residual()
    
    print("\n2. 🔍 绘制残差块详细解析...")
    draw_residual_block_detail()
    
    print("\n3. 📈 绘制性能对比图...")
    draw_performance_comparison()
    
    print("\n✅ 所有可视化完成！")
    print("\n📝 生成的文件：")
    print("   - residual_vs_traditional.png: 传统 vs 残差网络对比")
    print("   - residual_block_detail.png: 残差块详细解析")
    print("   - residual_performance_comparison.png: 性能对比")


if __name__ == "__main__":
    create_complete_visualization()
