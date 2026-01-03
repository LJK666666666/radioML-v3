import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置结果目录
results_dir = 'output/results'

# 存储所有模型的SNR-Accuracy数据
models_data = {}

# 遍历所有文件夹
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)

    # 检查是否为目录
    if os.path.isdir(folder_path):
        summary_file = os.path.join(folder_path, 'evaluation_summary.csv')

        # 检查evaluation_summary.csv是否存在
        if os.path.exists(summary_file):
            # 读取CSV文件
            df = pd.read_csv(summary_file)

            # 提取模型名称（从文件夹名称中提取）
            model_name = folder.replace('_evaluation_results', '')

            # 存储数据
            models_data[model_name] = df

# 模型名称映射函数
def format_model_name(base_model):
    if base_model == 'lightweight':
        return 'Complex-ResNet'
    elif base_model == 'micro':
        return 'Complex-ResNet-mini'
    elif base_model == 'complex':
        return 'ComplexCNN'
    elif base_model == 'resnet':
        return 'ResNet'
    else:
        return base_model.upper()

# 按基础模型分组
model_groups = {}
for model_name, df in models_data.items():
    # 提取基础模型名称
    base_model = model_name.split('_')[0]

    if base_model not in model_groups:
        model_groups[base_model] = {}

    # 格式化模型名称
    formatted_name = format_model_name(base_model)

    # 确定变体类型
    if '_efficient_gpr_per_sample_augment_' in model_name:
        variant = f'{formatted_name}+Aug+GPR'
    elif '_augment_' in model_name:
        variant = f'{formatted_name}+Aug'
    elif '_efficient_gpr_per_sample_' in model_name:
        variant = f'{formatted_name}+GPR'
    else:
        variant = formatted_name

    model_groups[base_model][variant] = df

# 输出目录
output_dir = 'script/snr_accuracy/figure'
os.makedirs(output_dir, exist_ok=True)

# 为每个基础模型创建单独的图形
for base_model, variants in model_groups.items():
    fig, ax = plt.subplots(figsize=(10, 6))

    # 定义绘制顺序和对应的标记形状
    formatted_name = format_model_name(base_model)
    order = [
        formatted_name,
        f'{formatted_name}+Aug',
        f'{formatted_name}+GPR',
        f'{formatted_name}+Aug+GPR'
    ]
    markers = ['o', '^', 's', '*']  # 圆点、三角形、方形、星形

    # 按顺序绘制折线
    for idx, variant_name in enumerate(order):
        if variant_name in variants:
            df = variants[variant_name]
            ax.plot(df['SNR'], df['Accuracy'], marker=markers[idx], label=variant_name,
                   linewidth=2, markersize=6 if markers[idx] != '*' else 10)

    # 设置图形属性
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 设置主图x轴刻度，每隔2标注
    import numpy as np
    ax.set_xticks(np.arange(-20, 20, 2))

    # 在右下角添加放大图（-12dB到-2dB）
    # 创建子图位置 [left, bottom, width, height]
    ax_inset = fig.add_axes([0.6, 0.15, 0.2, 0.6])

    # 在放大图中绘制-12dB到-2dB的数据
    for idx, variant_name in enumerate(order):
        if variant_name in variants:
            df = variants[variant_name]
            # 筛选-12dB到-2dB的数据
            df_zoom = df[(df['SNR'] >= -14) & (df['SNR'] <= 0)]
            ax_inset.plot(df_zoom['SNR'], df_zoom['Accuracy'], marker=markers[idx],
                         linewidth=1.5, markersize=4 if markers[idx] != '*' else 8)

    # ax_inset.set_xlabel('SNR (dB)', fontsize=8)
    # ax_inset.set_ylabel('Accuracy', fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_xlim(-14, 0)
    # 设置子图x轴刻度，每隔2标注
    ax_inset.set_xticks(np.arange(-14, 2, 2))

    plt.tight_layout()

    # 保存图形
    output_file = os.path.join(output_dir, f'{base_model}_snr_accuracy.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图形已保存到: {output_file}")

    plt.close()

print(f"\n所有图形已保存到: {output_dir}")
