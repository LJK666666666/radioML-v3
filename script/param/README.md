# 模型参数统计工具

本目录包含用于分析Keras模型参数数量的工具脚本。

## 文件说明

### 1. `model_parameter_counter.py`
完整的模型参数统计工具，支持分析单个模型或所有模型。

**功能特性：**
- 统计总参数数量、可训练参数、不可训练参数
- 显示模型层数和估计文件大小
- 支持批量分析所有模型
- 错误处理和详细的输出格式
- 自动导入项目自定义层

**使用方法：**
```bash
# 分析所有模型
python model_parameter_counter.py

# 分析特定模型
python model_parameter_counter.py lightweight_hybrid_model_gpr_augment.keras
```

### 2. `quick_param_check.py`
快速检查特定模型参数的简化脚本。

**使用方法：**
```bash
python quick_param_check.py
```

### 3. `check_target_model.py` ⭐ 推荐
专门针对`lightweight_hybrid_model_gpr_augment.keras`模型的参数统计脚本。

**功能特性：**
- 自动处理自定义层加载
- 详细的模型结构展示
- 安全的错误处理机制

**使用方法：**
```bash
python check_target_model.py
```

### 4. `batch_analysis.py` ⭐ 推荐
批量分析所有保存模型的参数统计工具。

**功能特性：**
- 批量分析所有.keras模型文件
- 生成详细的汇总报告
- 导出CSV格式的分析结果
- 统计信息和排序功能

**使用方法：**
```bash
python batch_analysis.py
```

### 5. `model_parameters_analysis.csv`
批量分析的结果文件，包含所有模型的参数统计信息。

## 分析结果示例

### 目标模型 (`lightweight_hybrid_model_gpr_augment.keras`) 统计结果：

```
📊 模型参数统计:
模型名称:       lightweight_hybrid_model_gpr_augment.keras
总参数数量:     1,400,299
可训练参数:     1,400,299
不可训练参数:   0
模型层数:       17
估计大小:       5.34 MB
```

### 所有模型排序（按参数数量）：

| 模型名称 | 总参数 | 可训练参数 | 大小(MB) |
|---------|--------|------------|----------|
| lightweight_hybrid_model_*.keras | 1,400,299 | 1,400,299 | 5.34 |
| cnn1d_model.keras | 1,177,163 | 1,176,267 | 4.49 |
| complex_nn_model_*.keras | 810,955 | 810,955 | 3.09 |
| resnet_model_*.keras | 577,483 | 575,563 | 2.20 |
| lightweight_transition_model_*.keras | 548,203 | 547,051 | 2.09 |
| fcnn_model.keras | 300,811 | 299,019 | 1.15 |
| transformer_model.keras | 12,203 | 12,203 | 0.05 |

## 依赖要求

- TensorFlow/Keras
- NumPy
- Pandas (用于batch_analysis.py)
- Python 3.7+

## 注意事项

1. **自定义层支持**：脚本已自动处理项目中的自定义复数层导入
2. **模型路径**：确保模型文件在`model_weight_saved`目录中
3. **文件格式**：仅支持`.keras`格式的模型文件
4. **内存要求**：需要足够的内存来加载大型模型
5. **错误处理**：如果某个模型加载失败，会跳过并继续分析其他模型

## 项目集成

这些工具已完全集成到radioML-v3项目中，能够：
- 自动识别和导入项目的自定义层（ComplexConv1D、ComplexResidualBlock等）
- 处理复数神经网络模型的特殊结构
- 支持混合架构模型的参数统计

## 快速开始

1. 分析目标模型：
```bash
python check_target_model.py
```

2. 批量分析所有模型：
```bash
python batch_analysis.py
```

结果将显示在终端并保存到CSV文件中。
