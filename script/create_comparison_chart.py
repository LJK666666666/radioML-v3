import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取baseline.xlsx文件
baseline_path = r'd:\1python programs\radioml\radioML-v3\arcticle\baseline.xlsx'
baseline_df = pd.read_excel(baseline_path)

print("Baseline data from Excel:")
print(baseline_df)
print("\nColumns:", baseline_df.columns.tolist())

# 从RESULT.md提取的数据
our_results = {
    'resnet': 0.5537,
    'complexnn': 0.5711,  # complexnn + relu
    'lightweight_hybrid': 0.5694,
    'resnet + gpr + augment': 0.6437,
    'complexnn + gpr + augment': 0.6341,
    'lightweight_hybrid + gpr + augment': 0.6538
}

# 提取其他论文的baseline数据
baseline_methods = []
baseline_accuracies = []

for idx, row in baseline_df.iterrows():
    if pd.notna(row['Method']) and pd.notna(row['Accuracy']) and row['Title'] != 'Our Method':
        method_name = str(row['Method'])
        accuracy = float(row['Accuracy'])
        baseline_methods.append(method_name)
        baseline_accuracies.append(accuracy)

print("\nExtracted baseline methods:")
for method, acc in zip(baseline_methods, baseline_accuracies):
    print(f"{method}: {acc:.4f}")

# 准备所有方法的数据
all_data = []

# 我们的方法（原始 + 改进）
models = ['ResNet', 'ComplexNN', 'Hybrid']
original_acc = [our_results['resnet'], our_results['complexnn'], our_results['lightweight_hybrid']]
improved_acc = [
    our_results['resnet + gpr + augment'], 
    our_results['complexnn + gpr + augment'], 
    our_results['lightweight_hybrid + gpr + augment']
]

for i, model in enumerate(models):
    all_data.append((model, original_acc[i], improved_acc[i], 'ours'))

# 基线方法（只有一个数值）
for method, acc in zip(baseline_methods, baseline_accuracies):
    all_data.append((method, acc, acc, 'baseline'))

# 按改进后的准确率（第3列）从小到大排序
all_data.sort(key=lambda x: x[2])

# 分离数据
sorted_methods = [item[0] for item in all_data]
sorted_original = [item[1] for item in all_data]
sorted_improved = [item[2] for item in all_data]
method_types = [item[3] for item in all_data]

# 计算改进部分（对于我们的方法）
sorted_improvement = []
for i, method_type in enumerate(method_types):
    if method_type == 'ours':
        sorted_improvement.append(sorted_improved[i] - sorted_original[i])
    else:
        sorted_improvement.append(0)  # 基线方法没有改进部分

# 创建主要的对比图
fig, ax = plt.subplots(1, 1, figsize=(16, 10))

# 动态计算y轴范围，确保所有数据和标签都在可见区域内
y_min = min(min(sorted_original), min(sorted_improved)) - 0.02
y_max = max(sorted_improved) + 0.03
print(f"设置y轴范围: {y_min:.3f} - {y_max:.3f}")

# 为我们的方法创建堆叠柱状图，为基线方法创建普通柱状图
bars_original = []
bars_improvement = []
bars_baseline = []

# 计算统一的基线位置（所有柱子从相同位置开始）
baseline_y = y_min

for i, method_type in enumerate(method_types):
    if method_type == 'ours':
        # 我们的方法：堆叠柱状图（从基线开始）
        # 第一部分：从基线到原始性能
        bar_baseline_to_orig = ax.bar(i, sorted_original[i] - baseline_y, 
                                    bottom=baseline_y, color='#87CEEB', alpha=0.8, 
                                    edgecolor='black', linewidth=0.5)
        # 第二部分：改进部分
        bar_imp = ax.bar(i, sorted_improvement[i], bottom=sorted_original[i], 
                        color='#2E86C1', alpha=0.9, edgecolor='black', linewidth=0.5)
        bars_original.append(bar_baseline_to_orig)
        bars_improvement.append(bar_imp)
    else:
        # 基线方法：从基线到最终性能的普通柱状图
        bar_base = ax.bar(i, sorted_improved[i] - baseline_y, bottom=baseline_y,
                         color='#F39C12', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars_baseline.append(bar_base)

# 设置图表属性
ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison: Our Methods (Stacked) vs Baseline Methods\n(Sorted by Final Performance)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(range(len(sorted_methods)))

# 修改横坐标标签，添加AbFTNet的SOTA标识
modified_labels = []
for method in sorted_methods:
    if 'AbFTNet' in method:
        modified_labels.append(f'{method}\n(previous SOTA)')
    else:
        modified_labels.append(method)

ax.set_xticklabels(modified_labels, rotation=0, ha='center')

# 动态计算y轴范围，确保所有数据和标签都在可见区域内
y_min = min(min(sorted_original), min(sorted_improved)) - 0.02
y_max = max(sorted_improved) + 0.03
ax.set_ylim(y_min, y_max)
print(f"设置y轴范围: {y_min:.3f} - {y_max:.3f}")

ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, method_type in enumerate(method_types):
    if method_type == 'ours':
        # 我们的方法：原始性能标签（在原始部分中心）
        mid_y = baseline_y + (sorted_original[i] - baseline_y) / 2
        ax.text(i, mid_y, f'{sorted_original[i]:.3f}', 
                ha='center', va='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 改进后总性能标签（在柱子顶部）
        ax.text(i, sorted_improved[i] + 0.005, f'{sorted_improved[i]:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 提升幅度标签（在改进部分中间）
        if sorted_improvement[i] > 0:
            improvement_pct = (sorted_improvement[i] / sorted_original[i]) * 100
            improvement_y_pos = sorted_original[i] + sorted_improvement[i]/2
            ax.text(i, improvement_y_pos,
                    f'+{sorted_improvement[i]:.3f}\n(+{improvement_pct:.1f}%)', 
                    ha='center', va='center', fontweight='bold', fontsize=9, color='white')
    else:
        # 基线方法：在柱子顶部显示数值
        ax.text(i, sorted_improved[i] + 0.005, f'{sorted_improved[i]:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#87CEEB', label='Our Original Performance'),
    Patch(facecolor='#2E86C1', label='Our Improvement (+GPR+Aug)'),
    Patch(facecolor='#F39C12', label='Baseline Methods')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()

# 确保输出目录存在
output_dir = r'd:\1python programs\radioml\radioML-v3\script\figure'
os.makedirs(output_dir, exist_ok=True)

# 保存对比图
output_path = os.path.join(output_dir, 'sorted_stacked_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n按最终准确率排序的堆叠对比图已保存到: {output_path}")

plt.show()

# 创建堆叠式对比图（显示改进效果）
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

models = ['ResNet', 'ComplexNN', 'Lightweight Hybrid']
original_acc = [our_results['resnet'], our_results['complexnn'], our_results['lightweight_hybrid']]
improved_acc = [
    our_results['resnet + gpr + augment'], 
    our_results['complexnn + gpr + augment'], 
    our_results['lightweight_hybrid + gpr + augment']
]
improvement = [imp - orig for imp, orig in zip(improved_acc, original_acc)]

# 创建堆叠柱状图
bars_original = ax2.bar(range(3), original_acc, 
                       color=['#87CEEB', '#FFB6C1', '#98FB98'], alpha=0.8, 
                       label='Original Performance')
bars_improvement = ax2.bar(range(3), improvement, bottom=original_acc,
                          color=['#2E86C1', '#E74C3C', '#28B463'], alpha=0.9,
                          label='Improvement (+GPR+Aug)')

# 设置图表属性
ax2.set_xlabel('Our Methods', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Our Methods: Original vs Improved Performance', 
              fontsize=14, fontweight='bold')
ax2.set_xticks(range(3))
ax2.set_xticklabels(models)
ax2.set_ylim(0.0, 0.7)  # 从0开始，充满整个画布
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
# 原始性能标签
for i, (bar, acc) in enumerate(zip(bars_original, original_acc)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
             f'{acc:.3f}', ha='center', va='center', fontweight='bold', fontsize=10)

# 总性能标签
for i, acc in enumerate(improved_acc):
    ax2.text(i, acc + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 提升幅度标签
for i in range(3):
    orig = original_acc[i]
    imp = improvement[i]
    improvement_pct = (imp / orig) * 100
    ax2.text(i, orig + imp/2,
             f'+{imp:.3f}\n(+{improvement_pct:.1f}%)', 
             ha='center', va='center', fontweight='bold', fontsize=9, color='white')

# 添加图例
ax2.legend(loc='upper left', fontsize=11)

plt.tight_layout()

# 保存堆叠对比图
stacked_path = os.path.join(output_dir, 'stacked_improvement.png')
plt.savefig(stacked_path, dpi=300, bbox_inches='tight')
print(f"堆叠改进对比图已保存到: {stacked_path}")

plt.show()
