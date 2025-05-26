# Complex Activation Functions Guide

## 概述

复数神经网络中的激活函数设计需要考虑复数的特殊性质。本文档详细介绍了实现的各种复数激活函数及其应用场景。

## 激活函数详解

### 1. `complex_relu` / `crelu`
**描述**: 分别对实部和虚部应用ReLU
```python
crelu(a + bi) = ReLU(a) + i·ReLU(b)
```

**特点**:
- ✅ 简单易实现
- ✅ 计算效率高
- ❌ 破坏复数的几何特性
- ❌ 可能导致相位信息丢失

**适用场景**: 基础测试、快速原型

### 2. `mod_relu` ⭐ **推荐**
**描述**: 对幅度应用ReLU，保持相位
```python
mod_relu(z) = ReLU(|z| + bias) · (z / |z|)
```

**特点**:
- ✅ 保持相位信息
- ✅ 控制信号强度
- ✅ 可学习的偏置参数
- ✅ 适合信号处理任务

**适用场景**: **无线电信号分类、I/Q数据处理**

### 3. `zrelu`
**描述**: 仅允许第一象限的复数通过
```python
zrelu(z) = z if Re(z)≥0 and Im(z)≥0, else 0
```

**特点**:
- ✅ 几何直观
- ✅ 稀疏性好
- ❌ 限制性较强
- ❌ 可能过度稀疏

**适用场景**: 需要稀疏表示的应用

### 4. `cardioid` ⭐ **推荐**
**描述**: 心形激活函数，具有方向选择性
```python
cardioid(z) = 0.5·(1 + cos(arg(z)))·z
```

**特点**:
- ✅ 保持幅度和相位信息
- ✅ 方向选择性强
- ✅ 生物学启发
- ✅ 平滑可微

**适用场景**: **需要方向敏感性的信号处理**

### 5. `complex_tanh`
**描述**: 对实部和虚部分别应用tanh
```python
complex_tanh(a + bi) = tanh(a) + i·tanh(b)
```

**特点**:
- ✅ 有界输出 [-1, 1]
- ✅ 平滑可微
- ✅ 训练稳定
- ❌ 可能出现饱和

**适用场景**: **需要稳定训练的深层网络**

### 6. `phase_amplitude_activation`
**描述**: 分别对相位和幅度应用不同激活
```python
phase_amplitude(z) = f_amp(|z|) · exp(i·f_phase(arg(z)))
```

**特点**:
- ✅ 高度可定制
- ✅ 理论基础扎实
- ❌ 计算复杂度高
- ❌ 参数较多

**适用场景**: 研究性应用、特殊需求

## 性能比较

| 激活函数 | 相位保持 | 计算效率 | 训练稳定性 | 信号处理适用性 |
|---------|---------|---------|-----------|-------------|
| complex_relu | ❌ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| mod_relu | ✅ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| zrelu | ✅ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| cardioid | ✅ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| complex_tanh | ❌ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| phase_amplitude | ✅ | ⭐ | ⭐⭐ | ⭐⭐⭐ |

## 使用建议

### 对于无线电信号分类：
1. **首选**: `mod_relu` - 最适合I/Q数据
2. **次选**: `cardioid` - 方向敏感性好
3. **备选**: `complex_tanh` - 训练稳定

### 模型构建示例：

```python
# 使用mod_relu (推荐)
model = build_complex_nn_model(
    input_shape=(2, 128),
    num_classes=11,
    activation_type='mod_relu'
)

# 使用cardioid
model = build_complex_nn_model(
    input_shape=(2, 128), 
    num_classes=11,
    activation_type='cardioid'
)
```

### 参数调优建议：

1. **学习率**: mod_relu和cardioid可以使用稍高的学习率 (0.001-0.01)
2. **正则化**: 复数激活函数天然具有一定的正则化效果
3. **批次大小**: 建议使用较大的批次大小以稳定复数统计量

## 理论背景

### 复数激活函数的设计原则：

1. **保持复数性质**: 输出应该是输入的复数函数
2. **相位保持**: 尽可能保持输入的相位信息
3. **可微性**: 确保梯度可以正确传播
4. **稳定性**: 避免梯度消失或爆炸

### 数学基础：

复数 z = a + bi 可以表示为：
- 直角坐标: (a, b)
- 极坐标: (r, θ) 其中 r = |z|, θ = arg(z)

不同的激活函数在这两种表示下有不同的效果。

## 实验建议

1. **基准测试**: 先用 `complex_relu` 建立基准
2. **改进测试**: 使用 `mod_relu` 或 `cardioid` 
3. **性能对比**: 比较收敛速度和最终准确率
4. **可视化**: 使用提供的测试脚本查看激活函数行为

运行测试脚本：
```bash
cd /home/test/2/2.4/radioML-v2
python test_complex_activations.py
```
