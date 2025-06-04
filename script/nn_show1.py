# 可视化代码
import tensorflow as tf
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
from hybrid_complex_resnet_model import build_lightweight_hybrid_model

def visualize_lightweight_hybrid_model(input_shape=(2, 128), num_classes=10):
    """
    构建并可视化 lightweight_hybrid_model 神经网络架构
    """
    # 构建模型
    model = build_lightweight_hybrid_model(input_shape, num_classes)
    
    # 设置可视化参数
    visual_config = {
        'to_file': 'lightweight_hybrid_model.png',
        'show_shapes': True,
        'show_layer_names': True,
        'rankdir': 'TB',  # 垂直布局 (TB=Top to Bottom)
        'dpi': 150,
        'expand_nested': True,  # 展开嵌套层
        'show_dtype': False,
        'show_layer_activations': True
    }
    
    # 生成模型图
    plot_model(model, **visual_config)
    
    # 添加标题并显示
    img = Image.open('lightweight_hybrid_model.png')
    plt.figure(figsize=(18, 24))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Lightweight Hybrid Model Architecture', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('enhanced_lightweight_hybrid_model.png', bbox_inches='tight', dpi=200)
    plt.show()
    
    return model

# 执行可视化
if __name__ == "__main__":
    # 设置输入参数
    input_shape = (2, 128)  # I/Q 数据形状
    num_classes = 10         # 分类类别数
    
    # 生成可视化
    model = visualize_lightweight_hybrid_model(input_shape, num_classes)
    
    # 打印模型摘要
    print("\n" + "="*80)
    print("模型摘要:")
    print("="*80)
    model.summary()