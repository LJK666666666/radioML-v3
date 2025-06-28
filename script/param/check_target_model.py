"""
专门针对lightweight_hybrid_model_gpr_augment.keras模型的参数统计脚本
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

def load_custom_objects():
    """加载自定义对象字典"""
    try:
        # 导入所有自定义层
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
        
        # 构建自定义对象字典
        custom_objects = {
            # 复数层
            'ComplexConv1D': ComplexConv1D,
            'ComplexBatchNormalization': ComplexBatchNormalization,
            'ComplexDense': ComplexDense,
            'ComplexMagnitude': ComplexMagnitude,
            'ComplexActivation': ComplexActivation,
            'ComplexPooling1D': ComplexPooling1D,
            'ComplexGlobalAveragePooling1D': ComplexGlobalAveragePooling1D,
            
            # 残差块
            'ComplexResidualBlock': ComplexResidualBlock,
            'ComplexResidualBlockAdvanced': ComplexResidualBlockAdvanced,
            'HybridTransitionBlock': HybridTransitionBlock,
            
            # 激活函数
            'complex_relu': complex_relu,
            'mod_relu': mod_relu,
            'zrelu': zrelu,
            'crelu': crelu,
            'cardioid': cardioid,
            'complex_tanh': complex_tanh,
            'phase_amplitude_activation': phase_amplitude_activation,
            'complex_elu': complex_elu,
            'complex_leaky_relu': complex_leaky_relu,
            'complex_swish': complex_swish,
            'real_imag_mixed_relu': real_imag_mixed_relu,
        }
        
        print("✅ 成功构建自定义对象字典")
        return custom_objects
        
    except ImportError as e:
        print(f"❌ 导入自定义层失败: {e}")
        return {}

def count_model_parameters_safe(model_path):
    """
    安全地加载模型并统计参数
    """
    print(f"正在尝试加载模型: {os.path.basename(model_path)}")
    print("-" * 60)
    
    try:
        # 首先尝试直接加载
        model = keras.models.load_model(model_path)
        print("✅ 直接加载成功")
        
    except Exception as e:
        print(f"直接加载失败: {e}")
        print("尝试使用自定义对象加载...")
        
        try:
            # 使用自定义对象加载
            custom_objects = load_custom_objects()
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            print("✅ 使用自定义对象加载成功")
            
        except Exception as e2:
            print(f"❌ 使用自定义对象加载也失败: {e2}")
            return None
    
    # 统计参数
    try:
        total_params = model.count_params()
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n📊 模型参数统计:")
        print(f"模型名称:       {os.path.basename(model_path)}")
        print(f"总参数数量:     {total_params:,}")
        print(f"可训练参数:     {trainable_params:,}")
        print(f"不可训练参数:   {non_trainable_params:,}")
        print(f"模型层数:       {len(model.layers)}")
        
        # 估计模型大小 (假设float32，每个参数4字节)
        size_mb = (total_params * 4) / (1024 * 1024)
        print(f"估计大小:       {size_mb:.2f} MB")
        
        # 显示模型结构摘要
        print(f"\n📋 模型结构摘要:")
        model.summary()
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'layers': len(model.layers),
            'size_mb': size_mb
        }
        
    except Exception as e:
        print(f"❌ 参数统计失败: {e}")
        return None

def main():
    """主函数"""
    print("🔍 Lightweight Hybrid Model 参数统计工具")
    print("=" * 70)
    
    # 目标模型路径
    target_model = "lightweight_hybrid_model_gpr_augment.keras"
    model_path = os.path.join(project_root, "model_weight_saved", target_model)
    
    print(f"目标模型: {target_model}")
    print(f"完整路径: {model_path}")
    
    if os.path.exists(model_path):
        result = count_model_parameters_safe(model_path)
        if result:
            print(f"\n✅ 参数统计完成!")
        else:
            print(f"\n❌ 参数统计失败!")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        print("\n可用的模型文件:")
        model_dir = os.path.join(project_root, "model_weight_saved")
        if os.path.exists(model_dir):
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.keras'):
                    print(f"  - {f}")

if __name__ == "__main__":
    main()
