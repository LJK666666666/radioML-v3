# Transformer with Rotary Positional Encoding (RoPE) Models

本文档介绍了新添加的两种基于旋转位置编码（Rotary Positional Encoding, RoPE）的Transformer模型。

## 新增模型

### 1. transformer_rope_sequential
- **描述**: 使用标准序列位置编码的Transformer模型
- **位置编码方式**: 按照数据的前后顺序（0, 1, 2, ..., 127）作为位置
- **适用场景**: 适合时序性较强的信号分析

### 2. transformer_rope_phase
- **描述**: 使用I/Q复数相位角作为位置编码的Transformer模型
- **位置编码方式**: 使用I/Q构成复数的幅角（atan2(Q, I)）作为位置信息
- **适用场景**: 适合需要考虑信号相位特征的调制识别任务

## 技术实现

### 旋转位置编码 (RoPE)
RoPE通过旋转变换将位置信息直接编码到注意力机制中，相比传统的加性位置编码具有以下优势：
- 更好的位置感知能力
- 对序列长度的泛化性更强
- 在注意力计算中保持相对位置关系

### 相位位置编码
对于`transformer_rope_phase`模型，位置信息的计算方式：
1. 提取I/Q分量：`I = data[..., 0]`, `Q = data[..., 1]`
2. 计算相位角：`phase = atan2(Q, I)`
3. 归一化到[0,1]：`normalized_phase = (phase + π) / (2π)`
4. 生成位置编码：使用归一化相位角生成正弦/余弦位置编码

## 使用方法

### 训练单个模型
```bash
# 训练序列位置编码模型
python main.py --models transformer_rope_sequential --mode train --epochs 100

# 训练相位位置编码模型
python main.py --models transformer_rope_phase --mode train --epochs 100
```

### 训练多个模型
```bash
# 同时训练两种RoPE模型
python main.py --models transformer_rope_sequential transformer_rope_phase --mode train --epochs 100

# 与其他模型一起训练
python main.py --models transformer transformer_rope_sequential transformer_rope_phase --mode train
```

### 评估模型
```bash
# 评估RoPE模型
python main.py --models transformer_rope_sequential transformer_rope_phase --mode evaluate

# 完整流程（训练+评估）
python main.py --models transformer_rope_sequential transformer_rope_phase --mode all --epochs 100
```

### 高级选项
```bash
# 使用数据增强和去噪
python main.py --models transformer_rope_phase --mode train --augment_data --denoising_method gpr

# 使用分层分割
python main.py --models transformer_rope_sequential --mode train --stratified_split

# 自定义参数
python main.py --models transformer_rope_phase --mode train --epochs 200 --batch_size 64
```

## 模型参数

两种模型都支持以下参数（在模型构建函数中可调整）：
- `num_heads`: 注意力头数量（默认：4）
- `ff_dim`: 前馈网络隐藏层大小（默认：64）
- `num_transformer_blocks`: Transformer块数量（默认：3）
- `embed_dim`: 嵌入维度（默认：64，必须为偶数以支持RoPE）
- `dropout_rate`: Dropout率（默认：0.1）

## 输出文件

训练完成后，模型文件将保存为：
- `transformer_rope_sequential_model.keras` - 序列位置编码模型
- `transformer_rope_phase_model.keras` - 相位位置编码模型

评估结果将保存在results目录中：
- `transformer_rope_sequential_evaluation_results.png` - 序列模型评估结果
- `transformer_rope_phase_evaluation_results.png` - 相位模型评估结果

## 注意事项

1. **嵌入维度限制**: RoPE要求嵌入维度必须为偶数
2. **内存使用**: 相位位置编码模型可能需要稍多的内存来计算相位角
3. **收敛性**: 不同的位置编码方式可能导致不同的收敛速度和最终性能
4. **数据依赖**: 相位位置编码的效果很大程度上依赖于I/Q数据的质量

## 性能比较建议

建议同时训练以下模型进行性能比较：
```bash
python main.py --models transformer transformer_rope_sequential transformer_rope_phase --mode all --epochs 100
```

这将帮助您比较：
- 传统Transformer（无位置编码）
- 序列RoPE Transformer
- 相位RoPE Transformer

的性能差异，从而选择最适合您数据集的模型。
