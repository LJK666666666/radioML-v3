# RadioML v3 - 先进的无线电信号分类系统

基于RadioML 2016.10a数据集的综合深度学习框架，用于自动调制分类。本项目实现了多种最先进的神经网络架构，包括复值神经网络、ResNet、Transformer和混合模型，专门用于无线电信号分类。

## 🚀 主要特性

- **多种模型架构**: CNN1D、CNN2D、ResNet、复值神经网络、Transformer和混合模型
- **复值神经网络**: 自定义实现，具有专门的复数激活函数，用于I/Q信号处理
- **高级预处理**: 使用GPR、小波变换和降噪自编码器(DDAE)进行信号降噪
- **数据增强**: 基于旋转的增强技术，提高模型鲁棒性
- **全面评估**: 基于SNR的分析和详细的性能指标
- **模块化设计**: 易于扩展和实验新架构

## 📊 数据集

本项目使用 **RadioML 2016.10a** 数据集，包含：
- **11种调制类型**: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
- **SNR范围**: -20 dB 到 +18 dB（2 dB步长）
- **信号格式**: I/Q采样，每个信号128个时间步
- **总样本数**: 220,000个样本（每个调制-SNR对1,000个样本）

## 🏗️ 模型架构

### 1. 复值神经网络 (ComplexNN)
- **复值层**: ComplexConv1D、ComplexDense、ComplexBatchNormalization
- **高级激活函数**: mod_relu、cardioid、zrelu、crelu、complex_tanh
- **最佳准确率**: 63.4%（使用GPR降噪 + 数据增强）

### 2. ResNet模型
- **残差连接**: 用于深度网络训练
- **多种块类型**: 具有跳跃连接
- **最佳准确率**: 64.4%（使用GPR降噪 + 数据增强）

### 3. 混合模型
- **复数到实数转换**: 结合ComplexNN和ResNet的优势
- **轻量级复数混合**: 项目最佳性能模型
- **最佳准确率**: 65.4%（轻量级复数混合模型）

### 4. 传统模型
- **CNN1D**: 时间序列数据的1D卷积（55.0%准确率）
- **CNN2D**: 将I/Q视为图像的2D卷积（47.3%准确率）
- **Transformer**: 基于注意力机制的架构（48.6%准确率）

## 🛠️ 安装配置

### 系统要求
- Python 3.12.9
- 支持CUDA的GPU（推荐）

### 依赖包安装
```bash

```

### 数据集设置
1. 下载RadioML 2016.10a数据集
2. 将`RML2016.10a_dict.pkl`文件放置在项目根目录

## 🚀 快速开始

### 基本使用
```bash
cd src

# 训练和评估SOTA模型
python main.py --mode evaluate --model_type lightweight_hybrid

# 训练和评估所有模型
python main.py --mode all --model_type all

# 训练特定模型
python main.py --mode train --model_type resnet --epochs 100

# 进行SNR分析评估
python main.py --mode evaluate --model_type complex_nn
```

### 高级选项
```bash
# 使用数据增强和GPR降噪
python main.py --model_type resnet --augment_data --denoising_method gpr

# 自定义配置
python main.py --model_type complex_nn --epochs 500 --batch_size 256 --denoising_method gpr --augment_data
```

## 📁 项目结构

```
radioML-v3/
├── src/
│   ├── main.py                 # 主执行脚本
│   ├── models.py              # 模型导入
│   ├── train.py               # 训练逻辑
│   ├── evaluate.py            # 评估和指标
│   ├── preprocess.py          # 数据预处理和降噪
│   ├── explore_dataset.py     # 数据集探索
│   ├── train_autoencoder.py   # 降噪自编码器训练
│   └── model/
│       ├── cnn1d_model.py
│       ├── cnn2d_model.py
│       ├── resnet_model.py
│       ├── complex_nn_model.py
│       ├── transformer_model.py
│       ├── hybrid_complex_resnet_model.py
│       ├── hybrid_transition_resnet_model.py
│       └── callbacks.py
├── output/                    # 生成的输出
│   ├── models/               # 保存的模型权重
│   ├── results/              # 评估结果
│   ├── training_plots/       # 训练曲线
│   └── exploration/          # 数据集可视化
├── model_weight_saved/       # 预训练模型
├── denoised_datasets/        # 缓存的降噪数据
├── RML2016.10a_dict.pkl     # 数据集文件
├── README.md
├── result.md                 # 详细实验结果
└── COMPLEX_ACTIVATIONS_GUIDE.md  # 复数激活函数指南
```

## ⚙️ 配置选项

### 命令行参数
- `--mode`: 操作模式 (explore, train, evaluate, all)
- `--model_type`: 模型架构 (cnn1d, cnn2d, resnet, complex_nn, transformer, hybrid_*, all)
- `--epochs`: 训练轮数 (默认: 500)
- `--batch_size`: 训练批次大小 (默认: 128)
- `--augment_data`: 启用数据增强
- `--denoising_method`: 预处理方法 (gpr, wavelet, ddae, none)
- `--random_seed`: 随机种子，确保可重现性 (默认: 42)

### 降噪方法
- **GPR**: 高斯过程回归（最佳性能）
- **Wavelet**: 基于小波的降噪
- **DDAE**: 深度降噪自编码器
- **None**: 无预处理

## 📈 性能结果

| 模型 | 基础准确率 | + 数据增强 | + GPR + 增强 | 最佳配置 |
|-------|-----------|-----------|------------|----------|
| **轻量级复数混合** | - | - | **65.4%** | **GPR + 增强** |
| ResNet | 55.0% | 59.9% | 64.4% | GPR + 增强 |
| ComplexNN (mod_relu) | 54.1% | - | 63.4% | GPR + 增强 |
| 轻量级转换模型 | - | - | 62.9% | GPR + 增强 |
| ComplexNN (leaky_relu) | 56.2% | - | 61.4% | GPR + 增强 |
| CNN1D | 55.0% | - | - | 基础 |
| CNN2D | 47.3% | - | - | 基础 |
| Transformer | 48.6% | - | - | 基础 |

### SNR性能分析
模型在高SNR水平下表现更好：
- **低SNR (-20 到 -10 dB)**: 9-34% 准确率
- **中等SNR (-8 到 0 dB)**: 45-86% 准确率
- **高SNR (2 到 18 dB)**: 85-93% 准确率

## 🔬 高级特性

### 复数激活函数
本项目实现了新颖的复值激活函数：
- **mod_relu**: 对幅度应用ReLU，同时保持相位
- **cardioid**: 方向敏感激活函数
- **zrelu**: 实部选择激活
- **complex_tanh**: 复值双曲正切

### 数据预处理流水线
1. **信号加载**: 从RadioML数据集加载I/Q采样
2. **降噪**: 应用GPR/小波/DDAE降噪
3. **标准化**: 标准化信号幅度
4. **增强**: 旋转I/Q信号提高鲁棒性
5. **训练/验证/测试分割**: 60/20/20分层分割

### 评估指标
- **总体准确率**: 所有SNR水平的分类准确率
- **基于SNR的分析**: 按信噪比的性能分解
- **混淆矩阵**: 每种调制类型的分析
- **分类报告**: 每类的精确率、召回率、F1分数

## 📖 使用示例

### 训练复值神经网络
```python
from models import build_complex_nn_model
from preprocess import prepare_data_by_snr

# 加载和预处理数据
dataset = load_radioml_data('RML2016.10a_dict.pkl')
X_train, X_val, X_test, y_train, y_val, y_test, _, _, _, mods = prepare_data_by_snr(
    dataset, augment_data=True, denoising_method='gpr'
)

# 使用mod_relu激活构建模型
model = build_complex_nn_model(
    input_shape=(2, 128), 
    num_classes=11,
    activation_type='mod_relu'
)

# 训练模型
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
```

### 自定义评估
```python
from evaluate import evaluate_by_snr

# 使用SNR分析评估模型
accuracy = evaluate_by_snr(
    model, X_test, y_test, snr_test, mods, output_dir='results/'
)
```

## 🤝 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/new-model`)
3. 提交更改 (`git commit -am 'Add new model architecture'`)
4. 推送到分支 (`git push origin feature/new-model`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详情请查看 [LICENSE](LICENSE) 文件。

## 📚 参考文献

1. T. O'Shea and J. Hoydis, "An introduction to deep learning for the physical layer," IEEE Transactions on Cognitive Communications and Networking, 2017.
2. RadioML 2016.10a 数据集: https://www.deepsig.ai/datasets
3. 复值神经网络: 信号处理中的理论与应用

## 🔗 相关工作

- [DeepSig RadioML 数据集](https://www.deepsig.ai/datasets)
- [GNU Radio](https://www.gnuradio.org/)
- [复值神经网络](https://github.com/wavefrontshaping/complexPyTorch)

## 📞 联系方式

如有问题和支持需求，请在GitHub上提出issue或联系维护者。

---

**注意**: 本项目仅用于研究和教育目的。在处理射频数据时，请确保遵守当地法规。