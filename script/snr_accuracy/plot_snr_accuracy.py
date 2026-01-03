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

# 创建图形
plt.figure(figsize=(12, 8))

# 为每个模型绘制折线
for model_name, df in models_data.items():
    plt.plot(df['SNR'], df['Accuracy'], marker='o', label=model_name, linewidth=2, markersize=4)

# 设置图形属性
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('SNR vs Accuracy for All Models', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图形
output_dir = 'script/snr_accuracy/figure'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'snr_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
print(f"图形已保存到: {os.path.join(output_dir, 'snr_accuracy_comparison.png')}")

# 显示图形
plt.show()
