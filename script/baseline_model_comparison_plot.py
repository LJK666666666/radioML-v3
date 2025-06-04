#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线模型性能比较图绘制脚本
Baseline Model Performance Comparison Plot Script

该脚本独立绘制基线模型的性能对比图，包括：
1. 各模型的分类准确率对比
2. 相对于基线的性能提升对比
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_output_dir():
    """创建输出目录"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'baseline_comparison')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_baseline_model_comparison():
    """绘制基线模型性能比较"""
    # 基线模型数据（来自实验结果）
    models = ['FCNN', 'CNN2D', 'Transformer', 'CNN1D', 'ResNet', 'ComplexCNN', '轻量级混合\n(基线)', '混合+GPR+增强\n(最佳)']
    accuracies = [42.65, 47.31, 47.86, 54.94, 55.37, 57.11, 56.94, 65.38]
    
    # 颜色配置 - 区分传统模型、单一先进模型和混合模型
    colors = ['#FF6B6B', '#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#45B7D1', '#2E86AB', '#C73E1D']
    edge_colors = ['darkred', 'darkred', 'darkred', 'darkgreen', 'darkgreen', 'darkblue', 'navy', 'darkred']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 子图1: 基线模型准确率对比
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, 
                    edgecolor=edge_colors, linewidth=1.5)
    
    ax1.set_title('基线模型性能比较\nBaseline Model Performance Comparison', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('分类准确率 (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('模型架构', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 70)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 旋转x轴标签
    ax1.tick_params(axis='x', rotation=15)
    
    # 添加参考线
    ax1.axhline(y=60, color='red', linestyle='--', alpha=0.7, linewidth=2, label='60%基准线')
    ax1.legend(loc='upper left')
    
    # 子图2: 模型性能提升对比
    baseline_performance = 42.65  # FCNN作为基线
    improvements = [acc - baseline_performance for acc in accuracies]
    
    bars2 = ax2.bar(models, improvements, color=colors, alpha=0.8,
                    edgecolor=edge_colors, linewidth=1.5)
    
    ax2.set_title('相对基线性能提升\nPerformance Improvement vs Baseline', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('性能提升 (百分点)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('模型架构', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'+{imp:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 旋转x轴标签
    ax2.tick_params(axis='x', rotation=15)
    
    # 添加零线
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # 添加图例说明不同颜色的含义
    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label='传统模型'),
        mpatches.Patch(color='#4ECDC4', label='卷积神经网络'),
        mpatches.Patch(color='#45B7D1', label='复数神经网络'),
        mpatches.Patch(color='#2E86AB', label='轻量级混合模型'),
        mpatches.Patch(color='#C73E1D', label='完整混合模型')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.95))
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = create_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for fmt in ['png', 'pdf']:
        filename = f'baseline_model_comparison_{timestamp}.{fmt}'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 图片已保存: {filepath}")
    
    plt.show()
    print("✅ 基线模型性能比较图已生成")

def plot_model_performance_summary():
    """绘制模型性能汇总表格"""
    models = ['FCNN', 'CNN2D', 'Transformer', 'CNN1D', 'ResNet', 'ComplexCNN', '轻量级混合', '混合+GPR+增强']
    accuracies = [42.65, 47.31, 47.86, 54.94, 55.37, 57.11, 56.94, 65.38]
    improvements = [acc - 42.65 for acc in accuracies]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格数据
    table_data = []
    for i, (model, acc, imp) in enumerate(zip(models, accuracies, improvements)):
        rank = i + 1
        table_data.append([rank, model, f'{acc:.2f}%', f'+{imp:.2f}'])
    
    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=['排名', '模型架构', '准确率', '相对提升'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # 设置标题
    ax.set_title('基线模型性能汇总表\nBaseline Model Performance Summary', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 保存表格
    output_dir = create_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for fmt in ['png', 'pdf']:
        filename = f'model_performance_summary_{timestamp}.{fmt}'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 表格已保存: {filepath}")
    
    plt.show()
    print("✅ 模型性能汇总表已生成")

def main():
    """主函数"""
    print("🚀 开始生成基线模型性能比较图...")
    print("=" * 50)
    
    # 绘制性能比较图
    plot_baseline_model_comparison()
    
    print("\n" + "=" * 50)
    print("🚀 开始生成模型性能汇总表...")
    
    # 绘制性能汇总表
    plot_model_performance_summary()
    
    print("\n" + "=" * 50)
    print("🎉 所有图表生成完成！")
    
    # 显示输出目录
    output_dir = create_output_dir()
    print(f"📁 输出目录: {output_dir}")

if __name__ == "__main__":
    main()
