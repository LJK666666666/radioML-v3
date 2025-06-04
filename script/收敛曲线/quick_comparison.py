"""
快速绘制模型验证准确率对比图
Quick comparison plot for model validation accuracy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def quick_plot():
    """快速绘制所有模型的验证准确率对比"""
    
    # 模型配置
    models = [
        ('cnn2d', 'CNN2D', '#1f77b4'),
        ('cnn1d', 'CNN1D', '#ff7f0e'), 
        ('transformer', 'Transformer', '#2ca02c'),
        ('resnet', 'ResNet', '#d62728'),
        ('complex_nn', 'Complex NN', '#9467bd'),
        ('lightweight_hybrid', 'Lightweight Hybrid', '#8c564b'),
        ('lightweight_hybrid_model_gpr_augment', 'Lightweight Hybrid (GPR)', '#e377c2')
    ]
    
    # 日志目录
    log_dir = "../../output/models/logs"
    plt.figure(figsize=(12, 8))
    
    for model_key, model_name, color in models:
        # 检查model_key是否已经包含'_model'，避免重复添加
        if model_key.endswith('_model_gpr_augment'):
            log_file = os.path.join(log_dir, f"{model_key}_detailed_log.csv")
        else:
            log_file = os.path.join(log_dir, f"{model_key}_model_detailed_log.csv")
        
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                epochs = df['epoch'].values
                val_acc = df['val_accuracy'].values
                
                plt.plot(epochs, val_acc, 
                        label=f'{model_name} (最高: {np.max(val_acc):.3f})',
                        linewidth=2, color=color, marker='o', markersize=3)
                
                print(f"{model_name}: {len(epochs)} epochs, 最高准确率: {np.max(val_acc):.4f}")
                
            except Exception as e:
                print(f"加载 {model_name} 失败: {e}")
        else:
            print(f"文件不存在: {log_file}")
    plt.xlabel('训练轮数 (Epochs)', fontsize=12)
    plt.ylabel('验证准确率 (Validation Accuracy)', fontsize=12)
    plt.title('模型验证准确率收敛曲线对比', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 110)
    plt.ylim(0, 0.7)
    
    plt.tight_layout()
      # 保存图片
    output_dir = "./figure"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'quick_validation_comparison.png'), 
                dpi=300, bbox_inches='tight')
    
    plt.show()
    print(f"图片已保存到: {output_dir}/quick_validation_comparison.png")

if __name__ == "__main__":
    quick_plot()
