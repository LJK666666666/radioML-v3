#!/usr/bin/env python3
"""
独立绘制基线模型性能比较图
从消融实验可视化脚本中提取的基线模型比较功能
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def create_output_dir():
    """创建输出目录"""
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'figure', 'baseline_comparison')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_baseline_model_comparison():
    """绘制基线模型性能比较"""
    print("开始绘制基线模型性能比较图...")
    
    # 基线模型数据（来自实验结果）
    models = ['FCNN', 'CNN2D', 'Transformer', 'CNN1D', 'ResNet', 'ComplexCNN', '混合\n(基线)', '混合+GPR+增强\n(最佳)']
    accuracies = [42.65, 47.31, 47.86, 54.94, 55.37, 57.11, 56.94, 65.38]
    
    # 颜色配置 - 区分传统模型、单一先进模型和混合模型
    colors = ['#FF6B6B', '#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#45B7D1', '#2E86AB', '#C73E1D']
    edge_colors = ['darkred', 'darkred', 'darkred', 'darkgreen', 'darkgreen', 'darkblue', 'navy', 'darkred']
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    # 基线模型准确率对比
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, 
                    edgecolor=edge_colors, linewidth=1.5)
    
    ax1.set_title('基线模型性能比较\nBaseline Model Performance Comparison', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('分类准确率 (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('模型架构', fontsize=14, fontweight='bold')
    ax1.set_ylim(40, 70)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 旋转x轴标签
    ax1.tick_params(axis='x', rotation=15)
    
    # 添加64.59% SOTA基准线
    ax1.axhline(y=64.59, color='red', linestyle='--', alpha=0.7, linewidth=2, label='64.59% previous SOTA')
    
    # 添加图例说明不同颜色的含义
    legend_elements = [
        # mpatches.Patch(color='#FF6B6B', label='传统模型'),
        # mpatches.Patch(color='#4ECDC4', label='卷积神经网络'),
        # mpatches.Patch(color='#45B7D1', label='复数神经网络'),
        # mpatches.Patch(color='#2E86AB', label='混合模型'),
        # mpatches.Patch(color='#C73E1D', label='完整混合模型'),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7, linewidth=2, label='64.59% previous SOTA')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = create_output_dir()
    for fmt in ['png', 'pdf']:
        filepath = os.path.join(output_dir, f'baseline_model_comparison.{fmt}')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图片已保存到: {filepath}")
    
    plt.show()
    print("✅ 基线模型性能比较图已生成")

def print_model_statistics():
    """打印模型统计信息"""
    models = ['FCNN', 'CNN2D', 'Transformer', 'CNN1D', 'ResNet', 'ComplexCNN', '混合(基线)', '混合+GPR+增强(最佳)']
    accuracies = [42.65, 47.31, 47.86, 54.94, 55.37, 57.11, 56.94, 65.38]
    
    print("\n" + "="*60)
    print("基线模型性能统计")
    print("="*60)
    
    baseline_performance = 42.65  # FCNN作为基线
    
    for model, acc in zip(models, accuracies):
        improvement = acc - baseline_performance
        print(f"{model:<20} | 准确率: {acc:>6.2f}% | 提升: {improvement:>+6.2f}%")
    
    print("="*60)
    print(f"最佳性能: {max(accuracies):.2f}% ({models[accuracies.index(max(accuracies))]})")
    print(f"最大提升: {max(accuracies) - baseline_performance:.2f}个百分点")
    print("="*60)

def main():
    """主函数"""
    print("基线模型性能比较图生成器")
    print("="*50)
    
    # 打印统计信息
    print_model_statistics()
    
    # 绘制图表
    plot_baseline_model_comparison()
    
    print("\n程序执行完成！")

if __name__ == "__main__":
    main()
