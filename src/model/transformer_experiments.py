"""
Experiment script to test different embed_dim values for transformer model performance.
"""

import numpy as np
from transformer_model import build_transformer_model

def compare_embed_dimensions():
    """
    Compare transformer models with different embedding dimensions
    """
    
    # 实验设置
    input_shape = (2, 128)  # (channels, sequence_length)
    num_classes = 11  # 假设11个调制类型
    
    # 不同的embed_dim值进行实验
    embed_dims = [16, 32, 64, 128, 256]
    
    models_info = []
    
    for embed_dim in embed_dims:
        print(f"\n=== Testing embed_dim = {embed_dim} ===")
        
        # 构建模型
        model = build_transformer_model(
            input_shape=input_shape,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=4,
            ff_dim=embed_dim,  # 通常ff_dim与embed_dim成比例
            num_transformer_blocks=1,
            dropout_rate=0.1
        )
        
        # 统计模型参数
        total_params = model.count_params()
        trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
        
        # 计算模型复杂度
        model_info = {
            'embed_dim': embed_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        }
        
        models_info.append(model_info)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_info['model_size_mb']:.2f} MB")
        
        # 显示模型架构摘要
        print("\nModel summary:")
        model.summary()
        
        # 清理内存
        del model
    
    # 分析结果
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'embed_dim':<12} {'Params':<15} {'Size(MB)':<10} {'Ratio':<8}")
    print("-" * 50)
    
    base_params = models_info[0]['total_params']
    for info in models_info:
        ratio = info['total_params'] / base_params
        print(f"{info['embed_dim']:<12} {info['total_params']:<15,} {info['model_size_mb']:<10.2f} {ratio:<8.1f}x")
    
    return models_info

def experiment_recommendations():
    """
    基于理论分析给出实验建议
    """
    print("\n" + "="*60)
    print("EXPERIMENTAL RECOMMENDATIONS")
    print("="*60)
    
    recommendations = {
        "baseline": {
            "embed_dim": 32,
            "reason": "当前设置，作为基准"
        },
        "smaller": {
            "embed_dim": 16,
            "reason": "测试是否过度参数化"
        },
        "moderate": {
            "embed_dim": 64,
            "reason": "适度增加，平衡性能和复杂度"
        },
        "larger": {
            "embed_dim": 128,
            "reason": "测试更强表示能力的效果"
        },
        "too_large": {
            "embed_dim": 256,
            "reason": "可能过拟合，但值得验证"
        }
    }
    
    for name, config in recommendations.items():
        print(f"\n{name.upper()}:")
        print(f"  embed_dim: {config['embed_dim']}")
        print(f"  理由: {config['reason']}")
    
    print("\n建议实验步骤:")
    print("1. 用相同的数据集训练所有模型")
    print("2. 比较验证集准确率")
    print("3. 观察训练曲线（过拟合迹象）")
    print("4. 测量训练时间和内存使用")
    print("5. 在测试集上评估最终性能")

if __name__ == "__main__":
    # 运行参数比较
    models_info = compare_embed_dimensions()
    
    # 显示建议
    experiment_recommendations()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("• 模型参数随embed_dim平方增长")
    print("• 对于无线电信号分类，embed_dim=32-64可能是最优的")
    print("• 过大的embed_dim可能导致过拟合")
    print("• 需要实际训练来验证最佳设置")
