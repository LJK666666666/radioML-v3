"""
生成高质量学术论文用的验证准确率收敛曲线图
Generate high-quality validation accuracy convergence curves for academic papers
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams
import seaborn as sns

# 设置学术论文图表样式
def setup_academic_style():
    """设置学术论文风格的绘图参数"""
    # 设置字体
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['legend.fontsize'] = 11
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11
    
    # 设置线条和标记
    rcParams['lines.linewidth'] = 2
    rcParams['lines.markersize'] = 6
    
    # 设置图表质量
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.format'] = 'pdf'
    
    # 设置颜色和样式
    plt.style.use('seaborn-v0_8-whitegrid')

def load_and_process_data(log_dir, models):
    """加载并处理所有模型数据"""
    model_data = {}
    
    for model_key, model_info in models.items():
        # 检查model_key是否已经包含'_model'，避免重复添加
        if model_key.endswith('_model_gpr_augment'):
            log_file = os.path.join(log_dir, f"{model_key}_detailed_log.csv")
        else:
            log_file = os.path.join(log_dir, f"{model_key}_model_detailed_log.csv")
        
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                if 'val_accuracy' in df.columns:
                    model_data[model_key] = {
                        'epochs': df['epoch'].values,
                        'val_accuracy': df['val_accuracy'].values,
                        'train_accuracy': df['train_accuracy'].values if 'train_accuracy' in df.columns else None,
                        'name': model_info['name'],
                        'color': model_info['color'],
                        'linestyle': model_info.get('linestyle', '-'),                        'marker': model_info.get('marker', 'o')
                    }
                    print(f"[OK] 成功加载 {model_info['name']}: {len(df)} epochs")
                else:
                    print(f"[ERROR] {model_info['name']}: 缺少 val_accuracy 列")
            except Exception as e:
                print(f"[ERROR] 加载 {model_info['name']} 失败: {e}")
        else:
            print(f"[ERROR] 文件不存在: {log_file}")
    
    return model_data

def create_academic_plot(model_data, save_path):
    """创建学术论文质量的图表"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 绘制每个模型的曲线
    for model_key, data in model_data.items():
        epochs = data['epochs']
        val_acc = data['val_accuracy']
        
        # 平滑处理（可选）
        # from scipy.signal import savgol_filter
        # val_acc_smooth = savgol_filter(val_acc, min(11, len(val_acc)//4*2+1), 2)
        
        ax.plot(epochs, val_acc,
               label=data['name'],
               color=data['color'],
               linestyle=data['linestyle'],
               marker=data['marker'],
               markevery=max(1, len(epochs)//20),  # 适当间隔显示标记点
               linewidth=2,
               markersize=4,
               alpha=0.9)
        
        # 计算并显示关键指标
        max_acc = np.max(val_acc)
        final_acc = val_acc[-1]
        max_epoch = epochs[np.argmax(val_acc)]
        
        print(f"{data['name']}: 最高={max_acc:.4f} (Epoch {max_epoch}), 最终={final_acc:.4f}")
    
    # 设置坐标轴
    ax.set_xlabel('Training Epochs', fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontweight='bold')
    ax.set_title('Validation Accuracy Convergence Curves', fontweight='bold', pad=15)
      # 设置坐标范围
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 0.7)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 设置图例
    ax.legend(loc='lower right', frameon=True, fancybox=True, 
             shadow=False, framealpha=0.9, edgecolor='black')
    
    # 设置刻度格式
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存为多种格式
    base_path = save_path.rsplit('.', 1)[0]
    plt.savefig(f"{base_path}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{base_path}.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{base_path}.eps", bbox_inches='tight', dpi=300)
    
    plt.show()
    return fig

def create_statistics_table(model_data, save_path):
    """创建统计表格"""
    
    stats_data = []
    for model_key, data in model_data.items():
        val_acc = data['val_accuracy']
        epochs = data['epochs']
        
        stats = {
            'Model': data['name'],
            'Max Accuracy': f"{np.max(val_acc):.4f}",
            'Final Accuracy': f"{val_acc[-1]:.4f}",
            'Best Epoch': int(epochs[np.argmax(val_acc)]),
            'Total Epochs': len(epochs),
            'Convergence Speed': f"{np.argmax(val_acc)/len(epochs):.2f}"
        }
        stats_data.append(stats)
    
    # 按最高准确率排序
    stats_data.sort(key=lambda x: float(x['Max Accuracy']), reverse=True)
    
    # 创建DataFrame并保存
    df_stats = pd.DataFrame(stats_data)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_stats.to_csv(save_path, index=False)
    df_stats.to_latex(save_path.replace('.csv', '.tex'), index=False, escape=False)
    
    print("\n=== 模型性能统计表 ===")
    print(df_stats.to_string(index=False))
    print(f"\n统计表已保存到: {save_path}")

def main():
    """主函数"""
    
    # 设置学术样式
    setup_academic_style()
    
    # 定义模型配置
    models = {
        'cnn2d': {
            'name': 'CNN2D',
            'color': '#1f77b4',
            'linestyle': '-',
            'marker': 'o'
        },
        'cnn1d': {
            'name': 'CNN1D',
            'color': '#ff7f0e',
            'linestyle': '-',
            'marker': 's'
        },
        'transformer': {
            'name': 'Transformer',
            'color': '#2ca02c',
            'linestyle': '-',
            'marker': '^'
        },
        'resnet': {
            'name': 'ResNet',
            'color': '#d62728',
            'linestyle': '-',
            'marker': 'v'
        },
        'complex_nn': {
            'name': 'Complex NN',
            'color': '#9467bd',
            'linestyle': '-',
            'marker': 'D'        },
        'lightweight_hybrid': {
            'name': 'Hybrid',
            'color': '#8c564b',
            'linestyle': '-',
            'marker': 'p'
        },
        'lightweight_hybrid_model_gpr_augment': {
            'name': 'GRCR-Net (Proposed)',
            'color': '#e377c2',
            'linestyle': '-',
            'marker': '*'
        }
    }
      # 路径配置
    log_dir = "../../output/models/logs"
    output_dir = "./figure/academic_figures"
    
    print("正在加载模型数据...")
    model_data = load_and_process_data(log_dir, models)
    
    if not model_data:
        print("错误: 没有成功加载任何模型数据!")
        return
    
    print(f"\n成功加载 {len(model_data)} 个模型的数据")
      # 创建学术图表
    print("\n正在生成学术论文图表...")
    fig = create_academic_plot(model_data, 
                              os.path.join(output_dir, "validation_accuracy_convergence.pdf"))
      # 创建统计表格
    print("\n正在生成统计表格...")
    create_statistics_table(model_data, 
                           os.path.join(output_dir, "model_performance_statistics.csv"))
    
    print(f"\n[OK] 所有文件已保存到: {output_dir}")
    print("  - validation_accuracy_convergence.pdf/png/eps")
    print("  - model_performance_statistics.csv/tex")

if __name__ == "__main__":
    main()
