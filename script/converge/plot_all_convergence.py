import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置日志目录
logs_dir = 'output/models/logs'

# ========== 模型显示控制 ==========
# 设置为1显示该模型，设置为0不显示
# model_display_config = {
#     'CGDNN': 0,
#     'ComplexCNN': 1,
#     'Complex-ResNet': 1,
#     'Complex-ResNet-mini': 0,
#     'MCLDNN': 1,
#     'MCNET': 0,
#     'PET': 1,
#     'ResNet': 1,
#     'ULCNN': 0
# }
model_display_config = {
    'CGDNN': 0,
    'ComplexCNN': 1,
    'Complex-ResNet': 1,
    'Complex-ResNet-mini': 0,
    'MCLDNN': 1,
    'MCNET': 0,
    'PET': 1,
    'ResNet': 1,
    'ULCNN': 0
}

# 设置epoch显示范围（None表示显示全部）
max_epoch = 50
# =================================

# 存储所有模型的训练日志数据
models_data = {}

# 遍历所有CSV文件
for file in os.listdir(logs_dir):
    if file.endswith('_detailed_log.csv'):
        file_path = os.path.join(logs_dir, file)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查是否有数据
        if len(df) > 0:
            # 提取模型名称（移除'_model_'后缀和'_detailed_log.csv'）
            model_name = file.replace('_model_', '_').replace('_detailed_log.csv', '')

            # 存储数据
            models_data[model_name] = df

# 模型名称映射函数
def format_model_name(base_model):
    if base_model == 'lightweight_hybrid':
        return 'Complex-ResNet'
    elif base_model == 'micro_lightweight_hybrid':
        return 'Complex-ResNet-mini'
    elif base_model == 'complex_nn':
        return 'ComplexCNN'
    elif base_model == 'resnet':
        return 'ResNet'
    else:
        return base_model.upper()

# 格式化所有模型名称
formatted_models_data = {}
for model_name, df in models_data.items():
    # 提取基础模型名称
    parts = model_name.split('_')

    # 处理带hybrid的模型名称
    if 'hybrid' in model_name:
        if model_name.startswith('micro'):
            base_model = 'micro_lightweight_hybrid'
        else:
            base_model = 'lightweight_hybrid'
    elif model_name.startswith('complex_nn'):
        base_model = 'complex_nn'
    else:
        base_model = parts[0]

    # 格式化模型名称
    formatted_name = format_model_name(base_model)

    # 确定变体类型
    if 'efficient_gpr_per_sample_augment' in model_name:
        variant = f'{formatted_name}+Aug+GPR'
    elif 'augment' in model_name:
        variant = f'{formatted_name}+Aug'
    elif 'efficient_gpr_per_sample' in model_name:
        variant = f'{formatted_name}+GPR'
    else:
        variant = formatted_name

    formatted_models_data[variant] = df

# 创建图形
plt.figure(figsize=(14, 8))

# 定义所有基础模型和绘制顺序（根据配置过滤）
base_models = ['CGDNN', 'ComplexCNN', 'Complex-ResNet', 'Complex-ResNet-mini', 'MCLDNN', 'MCNET', 'PET', 'ResNet', 'ULCNN']
all_order = []
for base in base_models:
    # 只添加配置中设置为1的模型
    if model_display_config.get(base, 0) == 1:
        all_order.extend([
            # base,
            f'{base}+Aug',
            # f'{base}+GPR',
            f'{base}+Aug+GPR'
        ])

# 定义每个基础模型的marker样式
markers = {
    'CGDNN': 'o',          # 圆形
    'ComplexCNN': 's',     # 方形
    'Complex-ResNet': '^', # 上三角
    'Complex-ResNet-mini': 'v', # 下三角
    'MCLDNN': 'D',         # 菱形
    'MCNET': 'p',          # 五边形
    'PET': '*',            # 星形
    'ResNet': 'X',         # X形
    'ULCNN': 'h'           # 六边形
}

# 按顺序绘制折线
for model_name in all_order:
    if model_name in formatted_models_data:
        df = formatted_models_data[model_name]

        # 过滤epoch数据，只显示到max_epoch
        if max_epoch is not None:
            df = df[df['epoch'] <= max_epoch]

        # 确定基础模型名称
        base_name = model_name.replace('+GPR', '').replace('+Aug', '')
        marker = markers.get(base_name, 'o')

        # 判断是否为基础模型（使用虚线）或+GPR模型（使用实线）
        if '+GPR' in model_name:
            linestyle = '-'  # 实线
        else:
            linestyle = '--'  # 虚线

        plt.plot(df['epoch'], df['val_accuracy'],
                label=model_name,
                linewidth=2,
                alpha=0.7,
                marker=marker,
                markersize=6,
                markevery=5,  # 每5个点显示一个marker
                linestyle=linestyle)

# 设置图形属性
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 输出目录
output_dir = 'script/converge/figure'
os.makedirs(output_dir, exist_ok=True)

# 保存图形
output_file = os.path.join(output_dir, 'all_models_convergence.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"图形已保存到: {output_file}")

plt.close()
