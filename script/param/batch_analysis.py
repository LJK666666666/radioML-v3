"""
批量分析所有保存模型的参数数量
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import pandas as pd

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
            'ComplexConv1D': ComplexConv1D,
            'ComplexBatchNormalization': ComplexBatchNormalization,
            'ComplexDense': ComplexDense,
            'ComplexMagnitude': ComplexMagnitude,
            'ComplexActivation': ComplexActivation,
            'ComplexPooling1D': ComplexPooling1D,
            'ComplexGlobalAveragePooling1D': ComplexGlobalAveragePooling1D,
            'ComplexResidualBlock': ComplexResidualBlock,
            'ComplexResidualBlockAdvanced': ComplexResidualBlockAdvanced,
            'HybridTransitionBlock': HybridTransitionBlock,
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
        
        return custom_objects
        
    except ImportError as e:
        print(f"❌ 导入自定义层失败: {e}")
        return {}

def analyze_single_model(model_path, custom_objects):
    """分析单个模型"""
    model_name = os.path.basename(model_path)
    
    try:
        # 尝试加载模型
        try:
            model = keras.models.load_model(model_path)
        except:
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
        
        # 统计参数
        total_params = model.count_params()
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        non_trainable_params = total_params - trainable_params
        layers = len(model.layers)
        size_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'layers': layers,
            'size_mb': round(size_mb, 2),
            'status': '✅ 成功'
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'total_params': 0,
            'trainable_params': 0,
            'non_trainable_params': 0,
            'layers': 0,
            'size_mb': 0,
            'status': f'❌ 失败: {str(e)[:50]}...'
        }

def main():
    """主函数"""
    print("🔍 批量模型参数统计工具")
    print("=" * 80)
    
    model_dir = os.path.join(project_root, "model_weight_saved")
    
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return
    
    # 获取所有.keras文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    
    if not model_files:
        print(f"❌ 在 {model_dir} 中未找到.keras模型文件")
        return
    
    print(f"发现 {len(model_files)} 个模型文件")
    print("正在加载自定义对象...")
    
    custom_objects = load_custom_objects()
    
    results = []
    
    for i, model_file in enumerate(sorted(model_files), 1):
        print(f"\n[{i}/{len(model_files)}] 分析模型: {model_file}")
        model_path = os.path.join(model_dir, model_file)
        result = analyze_single_model(model_path, custom_objects)
        results.append(result)
        
        # 显示简要信息
        if '成功' in result['status']:
            print(f"  ✅ 参数: {result['total_params']:,} | 大小: {result['size_mb']} MB")
        else:
            print(f"  ❌ 加载失败")
    
    # 生成汇总报告
    print("\n" + "=" * 80)
    print("📊 模型参数汇总报告")
    print("=" * 80)
    
    # 创建DataFrame便于显示
    df = pd.DataFrame(results)
    
    # 按参数数量排序
    df_sorted = df.sort_values('total_params', ascending=False)
    
    print(f"{'模型名称':<45} {'总参数':<12} {'可训练参数':<12} {'大小(MB)':<8} {'状态'}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        model_name = row['model_name'][:42] + "..." if len(row['model_name']) > 45 else row['model_name']
        status_icon = "✅" if "成功" in row['status'] else "❌"
        
        print(f"{model_name:<45} {row['total_params']:<12,} {row['trainable_params']:<12,} "
              f"{row['size_mb']:<8} {status_icon}")
    
    # 统计信息
    successful_models = df[df['status'].str.contains('成功')]
    if len(successful_models) > 0:
        print("\n📈 统计信息:")
        print(f"成功加载模型数: {len(successful_models)}/{len(results)}")
        print(f"最大参数模型: {successful_models.loc[successful_models['total_params'].idxmax(), 'model_name']}")
        print(f"最小参数模型: {successful_models.loc[successful_models['total_params'].idxmin(), 'model_name']}")
        print(f"平均参数数量: {successful_models['total_params'].mean():,.0f}")
        print(f"总计模型大小: {successful_models['size_mb'].sum():.2f} MB")
    
    # 保存结果到CSV
    output_file = os.path.join(project_root, "script", "param", "model_parameters_analysis.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
