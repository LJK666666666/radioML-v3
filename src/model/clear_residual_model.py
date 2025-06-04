"""
清晰易懂的残差连接实现
用于RadioML信号分类的复数值残差网络

============ 什么是残差连接？ ============

简单来说，残差连接就是"抄近路"：
- 传统网络：输入 -> 层1 -> 层2 -> 层3 -> 输出
- 残差网络：输入 -> 层1 -> 层2 -> 层3 -> 输出
                 ↘________________↗ (抄近路)

核心思想：
- 让网络学习"变化"而不是"绝对值"
- 如果最优函数就是不变，网络只需要学会"什么都不做"
- 解决深度网络难以训练的问题

============ 为什么有效？ ============

1. 梯度流畅：梯度可以直接通过"抄近路"传到前面的层
2. 易于训练：即使其他层学不好，至少还有原始输入
3. 性能提升：允许网络更深，性能更好

============ 代码实现 ============
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
import numpy as np

# 简化的复数层实现（用于演示）
class SimpleComplexConv1D(tf.keras.layers.Layer):
    """简化的复数卷积层，便于理解"""
    
    def __init__(self, filters, kernel_size, strides=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
    def build(self, input_shape):
        # 输入是 [batch, time, 2*channels] (I和Q分量)
        input_dim = input_shape[-1] // 2
        
        # 为实部和虚部分别创建权重
        self.W_real = self.add_weight(
            shape=(self.kernel_size, input_dim, self.filters),
            initializer='glorot_uniform',
            name='W_real'
        )
        self.W_imag = self.add_weight(
            shape=(self.kernel_size, input_dim, self.filters),
            initializer='glorot_uniform',
            name='W_imag'
        )
        
    def call(self, inputs):
        # 分离I和Q分量
        input_real = inputs[..., :inputs.shape[-1]//2]  # I分量
        input_imag = inputs[..., inputs.shape[-1]//2:]  # Q分量
        
        # 复数卷积：(a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        output_real = tf.nn.conv1d(input_real, self.W_real, stride=self.strides, padding=self.padding.upper()) - \
                     tf.nn.conv1d(input_imag, self.W_imag, stride=self.strides, padding=self.padding.upper())
        
        output_imag = tf.nn.conv1d(input_real, self.W_imag, stride=self.strides, padding=self.padding.upper()) + \
                     tf.nn.conv1d(input_imag, self.W_real, stride=self.strides, padding=self.padding.upper())
        
        # 重新组合I和Q
        return tf.concat([output_real, output_imag], axis=-1)


class SimpleComplexBN(tf.keras.layers.Layer):
    """简化的复数批归一化"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        channels = input_shape[-1] // 2
        self.bn_real = BatchNormalization()
        self.bn_imag = BatchNormalization()
        
    def call(self, inputs, training=None):
        input_real = inputs[..., :inputs.shape[-1]//2]
        input_imag = inputs[..., inputs.shape[-1]//2:]
        
        output_real = self.bn_real(input_real, training=training)
        output_imag = self.bn_imag(input_imag, training=training)
        
        return tf.concat([output_real, output_imag], axis=-1)


class VerySimpleResidualBlock(tf.keras.layers.Layer):
    """
    最简单的残差块 - 一目了然的实现
    
    这个残差块做的事情：
    1. 输入进来
    2. 经过两层卷积处理（主路径）
    3. 把输入直接加到处理结果上（残差连接）
    4. 输出
    
    就这么简单！
    """
    
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        print(f"\n=== 构建残差块 ===")
        print(f"输入形状: {input_shape}")
        print(f"目标通道数: {self.filters}")
        
        # 主路径：两层卷积
        self.conv1 = SimpleComplexConv1D(self.filters, 3, padding='same')
        self.bn1 = SimpleComplexBN()
        
        self.conv2 = SimpleComplexConv1D(self.filters, 3, padding='same')
        self.bn2 = SimpleComplexBN()
        
        # 检查是否需要调整输入维度
        input_channels = input_shape[-1]
        output_channels = self.filters * 2  # 复数有2倍通道
        
        if input_channels != output_channels:
            print(f"⚠️ 维度不匹配: {input_channels} != {output_channels}")
            print("需要用1x1卷积调整维度")
            self.shortcut_conv = SimpleComplexConv1D(self.filters, 1, padding='same')
            self.shortcut_bn = SimpleComplexBN()
        else:
            print("✅ 维度匹配，可以直接相加")
            self.shortcut_conv = None
            
    def call(self, inputs, training=None):
        print(f"\n--- 残差块前向传播 ---")
        print(f"输入: {inputs.shape}")
        
        # 🚀 主路径处理
        print("🚀 主路径开始处理...")
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)  # 激活函数
        print(f"  第一层输出: {x.shape}")
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        # 注意：第二层后不加激活函数！
        print(f"  第二层输出: {x.shape}")
        
        # 🔗 跳跃连接处理
        print("🔗 跳跃连接处理...")
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
            print(f"  跳跃连接调整后: {shortcut.shape}")
        else:
            shortcut = inputs
            print(f"  跳跃连接直接使用: {shortcut.shape}")
        
        # ✨ 残差连接的魔法时刻！
        print("✨ 残差连接：主路径 + 跳跃连接")
        print(f"  主路径: {x.shape}")
        print(f"  跳跃连接: {shortcut.shape}")
        
        output = tf.add(x, shortcut)  # 这就是残差连接！
        print(f"  相加结果: {output.shape}")
        
        # 最后加激活函数
        output = tf.nn.relu(output)
        print(f"  最终输出: {output.shape}")
        
        return output


class ClearResidualModel:
    """
    清晰易懂的残差网络模型
    专为RadioML信号分类设计
    """
    
    def __init__(self, input_shape=(1024, 2), num_classes=11):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build_model(self):
        """构建模型 - 每一步都有清晰的解释"""
        
        print("\n" + "="*50)
        print("🏗️  开始构建清晰的残差网络模型")
        print("="*50)
        
        # 输入层
        inputs = Input(shape=self.input_shape, name='signal_input')
        print(f"📥 输入层: {self.input_shape}")
        
        # 第一层：简单的特征提取
        print("\n🎯 第一阶段：基础特征提取")
        x = SimpleComplexConv1D(32, 7, padding='same', name='initial_conv')(inputs)
        x = SimpleComplexBN(name='initial_bn')(x)
        x = tf.nn.relu(x)
        x = MaxPooling1D(2, name='initial_pool')(x)
        print(f"基础特征提取后: {x.shape}")
        
        # 残差块堆叠
        print("\n🏗️ 第二阶段：残差块堆叠")
        
        # 第一个残差块
        print("\n📦 第一个残差块 (32 filters)")
        x = VerySimpleResidualBlock(32, name='residual_1')(x)
        
        # 第二个残差块
        print("\n📦 第二个残差块 (64 filters)")
        x = VerySimpleResidualBlock(64, name='residual_2')(x)
        
        # 第三个残差块
        print("\n📦 第三个残差块 (128 filters)")
        x = VerySimpleResidualBlock(128, name='residual_3')(x)
        
        # 全局平均池化
        print("\n🌊 第三阶段：特征聚合")
        x = GlobalAveragePooling1D(name='global_pool')(x)
        print(f"全局池化后: {x.shape}")
        
        # 分类器
        print("\n🎯 第四阶段：分类")
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout')(x)
        outputs = Dense(self.num_classes, activation='softmax', name='classification')(x)
        print(f"最终输出: {outputs.shape}")
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs, name='ClearResidualModel')
        
        print("\n" + "="*50)
        print("✅ 模型构建完成！")
        print("="*50)
        
        return model
    
    def explain_residual_connection(self):
        """详细解释残差连接的工作原理"""
        
        explanation = """
        
    🔍 残差连接详细解释
    
    ==================== 传统网络的问题 ====================
    
    传统深度网络：
    输入 → Conv1 → Conv2 → Conv3 → ... → ConvN → 输出
    
    问题：
    1. 梯度消失：梯度从输出传到输入时会越来越小
    2. 难以训练：层数增加时，训练变得困难
    3. 性能下降：有时更深的网络性能反而更差
    
    ==================== 残差连接的解决方案 ====================
    
    残差网络：
    输入 → Conv1 → Conv2 → (+) → 输出
     ↓                        ↑
     └────── 跳跃连接 ─────────┘
    
    核心思想：
    - 输出 = F(输入) + 输入
    - 网络学习的是"变化量"F(输入)，而不是"绝对值"
    - 如果最优解是恒等映射，网络只需要让F(输入)=0即可
    
    ==================== 为什么有效？ ====================
    
    1. 💪 梯度流畅：
       - 梯度可以通过跳跃连接直接传播
       - 避免了梯度消失问题
    
    2. 🎯 易于优化：
       - 即使F(输入)学习失败，至少还有原始输入
       - 网络不会比没有残差连接时更差
    
    3. 🚀 性能提升：
       - 可以训练更深的网络
       - 在各种任务上都有性能提升
    
    ==================== 在复数信号中的应用 ====================
    
    RadioML信号是复数(I+jQ)，残差连接的好处：
    1. 保持I/Q相位关系
    2. 避免深度网络破坏信号特征
    3. 更好地学习调制特征
    
        """
        
        print(explanation)


def create_simple_demo():
    """创建一个简单的演示"""
    
    print("\n" + "🎭 " * 20)
    print("残差连接演示")
    print("🎭 " * 20)
    
    # 创建模型
    model_builder = ClearResidualModel()
    
    # 解释残差连接
    model_builder.explain_residual_connection()
    
    # 构建模型
    model = model_builder.build_model()
    
    # 打印模型摘要
    print("\n📊 模型摘要:")
    model.summary()
    
    # 创建一些假数据进行测试
    print("\n🧪 测试前向传播:")
    dummy_input = tf.random.normal((2, 1024, 2))  # 批大小=2
    output = model(dummy_input, training=False)
    print(f"测试输出形状: {output.shape}")
    
    return model


if __name__ == "__main__":
    model = create_simple_demo()
