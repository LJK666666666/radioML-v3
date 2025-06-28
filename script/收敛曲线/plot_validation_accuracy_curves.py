"""
绘制各个模型的验证集准确率收敛曲线
Plot validation accuracy convergence curves for different models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置绘图样式
# plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_data(log_dir, model_name):
    """
    加载模型训练日志数据
    
    Args:
        log_dir (str): 日志文件目录
        model_name (str): 模型名称
    
    Returns:
        pd.DataFrame: 训练日志数据，如果文件不存在返回None
    """    # 检查model_name是否已经包含'_model'，避免重复添加
    if model_name.endswith('_model_gpr_augment') or model_name.endswith('_gpr_augment'):
        log_file = os.path.join(log_dir, f"{model_name}_detailed_log.csv")
    else:
        log_file = os.path.join(log_dir, f"{model_name}_model_detailed_log.csv")
    
    if not os.path.exists(log_file):
        print(f"警告: 文件 {log_file} 不存在")
        return None
    
    try:
        df = pd.read_csv(log_file)
        print(f"成功加载 {model_name} 数据: {len(df)} epochs")
        return df
    except Exception as e:
        print(f"加载 {model_name} 数据时出错: {e}")
        return None

def plot_validation_accuracy_curves():
    """
    绘制所有模型的验证集准确率曲线
    """    # 定义模型列表和对应的显示名称
    models = {
        'cnn2d': 'CNN2D',
        'cnn1d': 'CNN1D', 
        'transformer': 'Transformer',
        'resnet': 'ResNet',
        'complex_nn': 'Complex NN',
        'lightweight_hybrid': 'Hybrid',
        'lightweight_hybrid_model_gpr_augment': 'Hybrid (GPR Augment)'
    }
    
    # 日志文件目录
    log_dir = "../../output/models/logs"
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 存储所有模型的数据用于统计
    all_data = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, (model_key, model_name) in enumerate(models.items()):
        # 加载数据
        df = load_model_data(log_dir, model_key)
        
        if df is not None and 'val_accuracy' in df.columns:
            epochs = df['epoch'].values
            val_accuracy = df['val_accuracy'].values
            
            # 绘制曲线
            plt.plot(epochs, val_accuracy, 
                    label=model_name, 
                    linewidth=2.5, 
                    color=colors[i],
                    marker='o', 
                    markersize=4,
                    alpha=0.8)
            
            # 存储数据
            all_data[model_name] = {
                'epochs': epochs,
                'val_accuracy': val_accuracy,
                'max_accuracy': np.max(val_accuracy),
                'final_accuracy': val_accuracy[-1] if len(val_accuracy) > 0 else 0,
                'total_epochs': len(epochs)
            }
            
            print(f"{model_name}: 最高验证准确率 = {np.max(val_accuracy):.4f}, "
                  f"最终验证准确率 = {val_accuracy[-1]:.4f}, "
                  f"训练轮数 = {len(epochs)}")
    
    # 设置图形属性
    plt.xlabel('训练轮数 (Epoch)', fontsize=14, fontweight='bold')
    plt.ylabel('验证集准确率 (Validation Accuracy)', fontsize=14, fontweight='bold')
    plt.title('各模型验证集准确率收敛曲线对比\nValidation Accuracy Convergence Curves Comparison', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置图例
    plt.legend(loc='lower right', fontsize=12, frameon=True, 
               fancybox=True, shadow=True, ncol=2)
      # 设置坐标轴
    plt.xlim(0, 110)
    plt.ylim(0, 0.7)
    
    # 添加精确度格式
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    # 调整布局
    plt.tight_layout()
      # 保存图片
    output_dir = "./figure"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, 'validation_accuracy_curves.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'validation_accuracy_curves.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    print(f"\n图形已保存到: {output_dir}")
    
    # 显示图形
    plt.show()
    
    # 打印统计信息
    print("\n=== 模型性能统计 ===")
    print(f"{'模型名称':<25} {'最高准确率':<12} {'最终准确率':<12} {'训练轮数':<8}")
    print("-" * 65)
    
    # 按最高准确率排序
    sorted_models = sorted(all_data.items(), 
                          key=lambda x: x[1]['max_accuracy'], 
                          reverse=True)
    
    for model_name, stats in sorted_models:
        print(f"{model_name:<25} {stats['max_accuracy']:<12.4f} "
              f"{stats['final_accuracy']:<12.4f} {stats['total_epochs']:<8d}")

def plot_individual_curves():
    """
    绘制每个模型的单独详细曲线
    """
    models = {
        'cnn2d': 'CNN2D',
        'cnn1d': 'CNN1D', 
        'transformer': 'Transformer',
        'resnet': 'ResNet',        'complex_nn': 'Complex NN',
        'lightweight_hybrid': 'Hybrid',
        'lightweight_hybrid_model_gpr_augment': 'Hybrid (GPR Augment)'
        }
    
    log_dir = "../../output/models/logs"
    output_dir = "./figure/individual_curves"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_key, model_name in models.items():
        df = load_model_data(log_dir, model_key)
        
        if df is not None and 'val_accuracy' in df.columns:
            plt.figure(figsize=(10, 6))
            
            epochs = df['epoch'].values
            train_accuracy = df['train_accuracy'].values if 'train_accuracy' in df.columns else None
            val_accuracy = df['val_accuracy'].values
            
            # 绘制训练和验证准确率
            plt.plot(epochs, val_accuracy, 'b-', linewidth=2, 
                    label='验证准确率 (Validation)', marker='o', markersize=3)
            
            if train_accuracy is not None:
                plt.plot(epochs, train_accuracy, 'r--', linewidth=2, 
                        label='训练准确率 (Training)', marker='s', markersize=3, alpha=0.7)
            plt.xlabel('训练轮数 (Epoch)', fontsize=12)
            plt.ylabel('准确率 (Accuracy)', fontsize=12)
            plt.title(f'{model_name} 模型训练过程', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(0, 110)
            plt.ylim(0, 0.7)
            
            # 添加最高点标注
            max_val_idx = np.argmax(val_accuracy)
            max_val_acc = val_accuracy[max_val_idx]
            max_val_epoch = epochs[max_val_idx]
            
            plt.annotate(f'最高: {max_val_acc:.4f}\nEpoch: {max_val_epoch}',
                        xy=(max_val_epoch, max_val_acc),
                        xytext=(max_val_epoch + len(epochs)*0.1, max_val_acc + 0.05),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            plt.tight_layout()
            
            # 保存个别模型图片
            plt.savefig(os.path.join(output_dir, f'{model_key}_accuracy_curve.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"个别模型曲线已保存到: {output_dir}")

if __name__ == "__main__":
    print("开始绘制模型验证集准确率收敛曲线...")
    
    # 检查日志目录是否存在
    log_dir = "../../output/models/logs"
    if not os.path.exists(log_dir):
        print(f"错误: 日志目录 {log_dir} 不存在")
        exit(1)
    
    try:
        # 绘制总览对比图
        plot_validation_accuracy_curves()
        
        # 绘制单独详细图
        plot_individual_curves()
        
        print("\n所有图形绘制完成！")
        
    except Exception as e:
        print(f"绘制过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
