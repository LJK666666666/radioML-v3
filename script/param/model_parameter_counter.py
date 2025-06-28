"""
模型参数数量统计工具
用于读取保存的Keras模型文件并输出参数数量信息
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

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

def count_model_parameters(model_path):
    """
    读取Keras模型文件并统计参数数量
    
    Args:
        model_path (str): 模型文件路径
        
    Returns:
        dict: 包含参数统计信息的字典
    """
    try:
        # 加载模型
        model = keras.models.load_model(model_path)
        
        # 统计总参数数量
        total_params = model.count_params()
        
        # 统计可训练参数数量
        trainable_params = sum([np.prod(var.shape) for var in model.trainable_variables])
        
        # 统计不可训练参数数量
        non_trainable_params = total_params - trainable_params
        
        # 获取模型结构信息
        model_layers = len(model.layers)
        
        return {
            'model_path': model_path,
            'model_name': os.path.basename(model_path),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'total_layers': model_layers,
            'model_loaded': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'model_path': model_path,
            'model_name': os.path.basename(model_path),
            'total_parameters': 0,
            'trainable_parameters': 0,
            'non_trainable_parameters': 0,
            'total_layers': 0,
            'model_loaded': False,
            'error': str(e)
        }

def format_number(num):
    """格式化数字，添加千位分隔符"""
    return f"{num:,}"

def print_model_info(model_info):
    """
    打印模型参数信息
    
    Args:
        model_info (dict): 模型信息字典
    """
    print(f"\n{'='*60}")
    print(f"模型名称: {model_info['model_name']}")
    print(f"模型路径: {model_info['model_path']}")
    print(f"{'='*60}")
    
    if model_info['model_loaded']:
        print(f"总参数数量:     {format_number(model_info['total_parameters'])}")
        print(f"可训练参数:     {format_number(model_info['trainable_parameters'])}")
        print(f"不可训练参数:   {format_number(model_info['non_trainable_parameters'])}")
        print(f"网络层数:       {model_info['total_layers']}")
        
        # 计算参数大小（假设float32，每个参数4字节）
        param_size_mb = (model_info['total_parameters'] * 4) / (1024 * 1024)
        print(f"估计模型大小:   {param_size_mb:.2f} MB")
    else:
        print(f"❌ 模型加载失败: {model_info['error']}")

def analyze_all_models():
    """分析所有保存的模型"""
    model_dir = os.path.join(project_root, 'model_weight_saved')
    
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return
    
    # 获取所有.keras文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    
    if not model_files:
        print(f"❌ 在 {model_dir} 中未找到.keras模型文件")
        return
    
    print(f"发现 {len(model_files)} 个模型文件")
    
    all_model_info = []
    
    for model_file in sorted(model_files):
        model_path = os.path.join(model_dir, model_file)
        model_info = count_model_parameters(model_path)
        all_model_info.append(model_info)
        print_model_info(model_info)
    
    # 打印汇总信息
    print_summary(all_model_info)

def print_summary(all_model_info):
    """打印所有模型的汇总信息"""
    print(f"\n{'='*80}")
    print("模型参数汇总表")
    print(f"{'='*80}")
    print(f"{'模型名称':<40} {'总参数':<15} {'可训练参数':<15} {'状态':<10}")
    print(f"{'-'*80}")
    
    for info in all_model_info:
        status = "✅ 成功" if info['model_loaded'] else "❌ 失败"
        model_name = info['model_name'][:37] + "..." if len(info['model_name']) > 40 else info['model_name']
        
        print(f"{model_name:<40} {format_number(info['total_parameters']):<15} "
              f"{format_number(info['trainable_parameters']):<15} {status:<10}")

def analyze_specific_model(model_name):
    """分析特定模型"""
    model_path = os.path.join(project_root, 'model_weight_saved', model_name)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    model_info = count_model_parameters(model_path)
    print_model_info(model_info)

def main():
    """主函数"""
    print("🔍 Keras模型参数统计工具")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，分析特定模型
        model_name = sys.argv[1]
        if not model_name.endswith('.keras'):
            model_name += '.keras'
        analyze_specific_model(model_name)
    else:
        # 否则分析所有模型
        analyze_all_models()

if __name__ == "__main__":
    main()
