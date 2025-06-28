"""
快速查看特定模型参数数量的示例脚本
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# 导入自定义层和激活函数
try:
    from src.model.complex_nn_model import (
        ComplexConv1D, ComplexBatchNormalization, ComplexDense, ComplexMagnitude, 
        ComplexActivation, ComplexPooling1D,
        complex_relu, mod_relu, zrelu, crelu, cardioid, complex_tanh, phase_amplitude_activation,
        complex_elu, complex_leaky_relu, complex_swish, real_imag_mixed_relu
    )
    from src.model.hybrid_complex_resnet_model import (
        ComplexResidualBlock, ComplexResidualBlockAdvanced, ComplexGlobalAveragePooling1D
    )
    from src.model.hybrid_transition_resnet_model import (
        ComplexResidualBlock as TransitionComplexResidualBlock,
        HybridTransitionBlock
    )
    print("✅ 成功导入自定义层")
except ImportError as e:
    print(f"⚠️  警告: 导入自定义层失败: {e}")
    print("模型加载可能会失败")

def quick_count_parameters(model_path):
    """
    快速统计指定模型的参数数量
    
    Args:
        model_path (str): 模型文件路径
    """
    print(f"正在分析模型: {os.path.basename(model_path)}")
    print("-" * 50)
    
    try:
        # 加载模型
        model = keras.models.load_model(model_path)
        
        # 统计参数
        total_params = model.count_params()
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        non_trainable_params = total_params - trainable_params
        
        print(f"✅ 模型加载成功")
        print(f"总参数数量:     {total_params:,}")
        print(f"可训练参数:     {trainable_params:,}")
        print(f"不可训练参数:   {non_trainable_params:,}")
        print(f"模型层数:       {len(model.layers)}")
        
        # 估计模型大小 (假设float32，每个参数4字节)
        size_mb = (total_params * 4) / (1024 * 1024)
        print(f"估计大小:       {size_mb:.2f} MB")
        
        return total_params
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 0

# 示例用法
if __name__ == "__main__":
    # 项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # 目标模型路径
    target_model = "lightweight_hybrid_model_gpr_augment.keras"
    model_path = os.path.join(project_root, "model_weight_saved", target_model)
    
    print(f"🔍 模型参数统计")
    print(f"目标模型: {target_model}")
    print(f"完整路径: {model_path}")
    print("=" * 60)
    
    if os.path.exists(model_path):
        quick_count_parameters(model_path)
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        print("\n可用的模型文件:")
        model_dir = os.path.join(project_root, "model_weight_saved")
        if os.path.exists(model_dir):
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.keras'):
                    print(f"  - {f}")
