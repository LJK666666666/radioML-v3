#!/usr/bin/env python3
"""
G-PET模型: 基于信号-噪声分解的创新架构

核心创新ultrathink思路：
1. 信号分解：原始信号 = 去噪信号 + 噪声信号
2. 功率约束：功率变化 ≈ 噪声功率（基于物理原理）
3. 双路径处理：干净信号和噪声信号分别提取特征
4. 四通道输入：[clean_I, clean_Q, noise_I, noise_Q]
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, Flatten, Reshape, Lambda,
    Conv1D, Conv2D, BatchNormalization, Activation,
    GlobalAveragePooling1D, Add, Multiply, Concatenate
)
from keras.saving import register_keras_serializable
import numpy as np


@register_keras_serializable(package="GPET")
class TransposeLayer(tf.keras.layers.Layer):
    """转置层：(batch, 2, seq_len) -> (batch, seq_len, 2)"""
    def call(self, inputs):
        return tf.transpose(inputs, perm=[0, 2, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[1])


@register_keras_serializable(package="GPET")
class ExtractChannelLayer(tf.keras.layers.Layer):
    """提取指定通道"""
    def __init__(self, channel_idx=0, **kwargs):
        super(ExtractChannelLayer, self).__init__(**kwargs)
        self.channel_idx = channel_idx

    def get_config(self):
        config = super().get_config()
        config.update({"channel_idx": self.channel_idx})
        return config

    def call(self, inputs):
        return inputs[:, :, self.channel_idx:self.channel_idx+1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)


@register_keras_serializable(package="GPET")
class TrigonometricLayer(tf.keras.layers.Layer):
    """三角函数层"""
    def __init__(self, func_type='cos', **kwargs):
        super(TrigonometricLayer, self).__init__(**kwargs)
        self.func_type = func_type

    def get_config(self):
        config = super().get_config()
        config.update({"func_type": self.func_type})
        return config

    def call(self, inputs):
        if self.func_type == 'cos':
            return tf.cos(inputs)
        elif self.func_type == 'sin':
            return tf.sin(inputs)
        else:
            raise ValueError(f"不支持的三角函数类型: {self.func_type}")

    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable(package="GPET")
class TileLayer(tf.keras.layers.Layer):
    """瓦片复制层，避免tf.tile的符号张量问题"""
    def __init__(self, multiples, **kwargs):
        super(TileLayer, self).__init__(**kwargs)
        self.multiples = multiples

    def get_config(self):
        config = super().get_config()
        config.update({"multiples": self.multiples})
        return config

    def call(self, inputs):
        return tf.tile(inputs, self.multiples)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        for i, mult in enumerate(self.multiples):
            if i < len(output_shape) and output_shape[i] is not None:
                output_shape[i] *= mult
        return tuple(output_shape)


@register_keras_serializable(package="GPET")
class ExpandDimsLayer(tf.keras.layers.Layer):
    """扩展维度层，避免tf.expand_dims的符号张量问题"""
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        axis = self.axis if self.axis >= 0 else len(output_shape) + self.axis + 1
        output_shape.insert(axis, 1)
        return tuple(output_shape)


@register_keras_serializable(package="GPET")
class UNetDenoising(tf.keras.layers.Layer):
    """
    U-Net架构的去噪网络

    特点：
    1. 编码器-解码器结构，保留多尺度信息
    2. 跳跃连接保持细节信息
    3. 专门用于信号去噪任务
    """

    def __init__(self, seq_len=128, **kwargs):
        super(UNetDenoising, self).__init__(**kwargs)
        self.seq_len = seq_len

    def build(self, input_shape):
        # 编码器路径（下采样）
        self.encoder_conv1 = Conv1D(16, 7, padding='same', activation='relu', name='enc_conv1')
        self.encoder_pool1 = tf.keras.layers.MaxPooling1D(2, name='enc_pool1')

        self.encoder_conv2 = Conv1D(32, 5, padding='same', activation='relu', name='enc_conv2')
        self.encoder_pool2 = tf.keras.layers.MaxPooling1D(2, name='enc_pool2')

        self.encoder_conv3 = Conv1D(64, 3, padding='same', activation='relu', name='enc_conv3')

        # 瓶颈层
        self.bottleneck = Conv1D(128, 3, padding='same', activation='relu', name='bottleneck')

        # 解码器路径（上采样）
        self.decoder_up1 = tf.keras.layers.UpSampling1D(2, name='dec_up1')
        self.decoder_conv1 = Conv1D(64, 3, padding='same', activation='relu', name='dec_conv1')

        self.decoder_up2 = tf.keras.layers.UpSampling1D(2, name='dec_up2')
        self.decoder_conv2 = Conv1D(32, 5, padding='same', activation='relu', name='dec_conv2')

        # 输出层
        self.output_conv = Conv1D(2, 7, padding='same', activation='linear', name='output_conv')

        # 手动构建所有子层以确保序列化正常
        seq_len = input_shape[1]

        # 构建编码器层
        self.encoder_conv1.build((None, seq_len, 2))
        self.encoder_conv2.build((None, seq_len//2, 16))
        self.encoder_conv3.build((None, seq_len//4, 32))

        # 构建瓶颈层
        self.bottleneck.build((None, seq_len//4, 64))

        # 构建解码器层
        self.decoder_conv1.build((None, seq_len//2, 128+32))  # 跳跃连接后的通道数
        self.decoder_conv2.build((None, seq_len, 64+16))      # 跳跃连接后的通道数

        # 构建输出层
        self.output_conv.build((None, seq_len, 32))

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len})
        return config

    def call(self, inputs):
        """
        U-Net前向传播

        Args:
            inputs: (batch, seq_len, 2)
        Returns:
            (batch, seq_len, 2)
        """
        # 编码器路径
        enc1 = self.encoder_conv1(inputs)       # (batch, seq_len, 16)
        pool1 = self.encoder_pool1(enc1)        # (batch, seq_len/2, 16)

        enc2 = self.encoder_conv2(pool1)        # (batch, seq_len/2, 32)
        pool2 = self.encoder_pool2(enc2)        # (batch, seq_len/4, 32)

        enc3 = self.encoder_conv3(pool2)        # (batch, seq_len/4, 64)

        # 瓶颈层
        bottleneck = self.bottleneck(enc3)      # (batch, seq_len/4, 128)

        # 解码器路径（跳跃连接）
        up1 = self.decoder_up1(bottleneck)     # (batch, seq_len/2, 128)
        # 跳跃连接：concatenate with enc2
        if up1.shape[1] == enc2.shape[1]:  # 确保维度匹配
            concat1 = tf.concat([up1, enc2], axis=-1)  # (batch, seq_len/2, 128+32)
        else:
            concat1 = up1
        dec1 = self.decoder_conv1(concat1)     # (batch, seq_len/2, 64)

        up2 = self.decoder_up2(dec1)           # (batch, seq_len, 64)
        # 跳跃连接：concatenate with enc1
        if up2.shape[1] == enc1.shape[1]:  # 确保维度匹配
            concat2 = tf.concat([up2, enc1], axis=-1)  # (batch, seq_len, 64+16)
        else:
            concat2 = up2
        dec2 = self.decoder_conv2(concat2)     # (batch, seq_len, 32)

        # 输出层
        output = self.output_conv(dec2)        # (batch, seq_len, 2)

        return output


@register_keras_serializable(package="GPET")
class SignalNoiseDecompositionUNet(tf.keras.layers.Layer):
    """
    基于U-Net的信号-噪声分解层

    核心创新：
    1. U-Net架构的去噪网络
    2. 功率约束：功率变化 ≈ 噪声功率
    3. 噪声高斯分布KL散度损失
    4. 输出4通道：[clean_I, clean_Q, noise_I, noise_Q]
    """

    def __init__(self, seq_len=128, power_constraint_weight=0.01, kl_loss_weight=0.001, **kwargs):
        super(SignalNoiseDecompositionUNet, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.power_constraint_weight = power_constraint_weight
        self.kl_loss_weight = kl_loss_weight

    def build(self, input_shape):
        # input_shape是列表: [signal_shape, snr_shape]
        signal_shape = input_shape[0]  # (batch, 2, seq_len)

        # U-Net去噪网络
        self.unet_denoiser = UNetDenoising(seq_len=signal_shape[2])

        # 建立U-Net
        self.unet_denoiser.build((None, signal_shape[2], 2))

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "power_constraint_weight": self.power_constraint_weight,
            "kl_loss_weight": self.kl_loss_weight
        })
        return config

    def compute_output_shape(self, input_shape):
        signal_shape = input_shape[0]
        # 输出4通道：[clean_I, clean_Q, noise_I, noise_Q]
        return (signal_shape[0], 4, signal_shape[2])

    def calculate_power_tf(self, signal):
        """计算信号功率"""
        # signal: (batch, 2, seq_len)
        i_component = signal[:, 0, :]  # (batch, seq_len)
        q_component = signal[:, 1, :]  # (batch, seq_len)
        power = tf.reduce_mean(i_component ** 2 + q_component ** 2, axis=1)  # (batch,)
        return power

    def compute_expected_noise_power(self, signal_power, snr_db):
        """根据SNR计算期望的噪声功率"""
        snr_linear = 10 ** (snr_db / 10)
        expected_noise_power = signal_power / (snr_linear + 1)
        return expected_noise_power

    def gaussian_kl_divergence_loss(self, noise_signal):
        """
        计算噪声信号与标准高斯分布的KL散度损失

        理论：噪声应该服从高斯分布 N(0, σ²)
        KL(p||q) = 0.5 * (σ²/σ₀² + μ²/σ₀² - 1 - log(σ²/σ₀²))
        其中 σ₀² = 1 (标准高斯)
        """
        # noise_signal: (batch, 2, seq_len)

        # 计算噪声统计量
        noise_mean = tf.reduce_mean(noise_signal, axis=[1, 2])  # (batch,)
        noise_var = tf.math.reduce_variance(noise_signal, axis=[1, 2])  # (batch,)

        # KL散度：假设目标是标准高斯分布 N(0, 1)
        kl_loss = 0.5 * (noise_var + noise_mean**2 - 1 - tf.math.log(tf.maximum(noise_var, 1e-8)))

        return tf.reduce_mean(kl_loss)

    def call(self, inputs):
        """
        信号-噪声分解（U-Net版本）

        Args:
            inputs: [signal, snr]
                signal: (batch, 2, seq_len) 原始信号
                snr: (batch, 1) SNR值

        Returns:
            decomposed: (batch, 4, seq_len) [clean_I, clean_Q, noise_I, noise_Q]
        """
        signal, snr = inputs
        snr_scalar = tf.squeeze(snr, axis=1)  # (batch,)

        # 计算原始信号功率
        original_power = self.calculate_power_tf(signal)  # (batch,)

        # 转换为时间序列格式进行U-Net去噪
        signal_transposed = tf.transpose(signal, perm=[0, 2, 1])  # (batch, seq_len, 2)

        # 应用U-Net去噪
        clean_transposed = self.unet_denoiser(signal_transposed)  # (batch, seq_len, 2)

        # 转回原格式
        clean_signal = tf.transpose(clean_transposed, perm=[0, 2, 1])  # (batch, 2, seq_len)

        # 计算去噪后功率
        clean_power = self.calculate_power_tf(clean_signal)  # (batch,)

        # 噪声信号 = 原始信号 - 去噪信号
        noise_signal = signal - clean_signal  # (batch, 2, seq_len)

        # === 损失函数设计 ===

        # 1. 功率约束损失（利用SNR信息的物理约束）
        expected_noise_power = self.compute_expected_noise_power(original_power, snr_scalar)
        actual_power_change = original_power - clean_power
        power_constraint_loss = tf.reduce_mean(tf.square(actual_power_change - expected_noise_power))
        self.add_loss(self.power_constraint_weight * power_constraint_loss)

        # 2. 噪声高斯分布KL散度损失
        kl_divergence_loss = self.gaussian_kl_divergence_loss(noise_signal)
        self.add_loss(self.kl_loss_weight * kl_divergence_loss)

        # 拼接干净信号和噪声信号：[clean_I, clean_Q, noise_I, noise_Q]
        decomposed = tf.concat([clean_signal, noise_signal], axis=1)  # (batch, 4, seq_len)

        return decomposed


def build_g_pet_model(input_shape, num_classes=11):
    """
    构建G-PET模型：U-Net信号-噪声分解版本

    Args:
        input_shape: 信号输入形状 (2, seq_len)
        num_classes: 分类数量

    Returns:
        编译后的Keras模型
    """

    # 双输入设计
    signal_input = Input(shape=input_shape, name='signal_input')       # (batch, 2, seq_len)
    snr_input = Input(shape=(1,), name='snr_input')                   # (batch, 1) SNR值

    # U-Net信号-噪声分解层（核心创新）
    decomposed = SignalNoiseDecompositionUNet(
        seq_len=input_shape[1],
        power_constraint_weight=0.01,   # 功率约束权重
        kl_loss_weight=0.001           # KL散度损失权重
    )([signal_input, snr_input])
    # 输出：(batch, 4, seq_len) = [clean_I, clean_Q, noise_I, noise_Q]

    # 转换为时间序列格式进行处理
    x = TransposeLayer()(decomposed)  # (batch, seq_len, 4)

    # 分离干净信号和噪声信号通道
    clean_i = ExtractChannelLayer(channel_idx=0)(x)  # (batch, seq_len, 1)
    clean_q = ExtractChannelLayer(channel_idx=1)(x)  # (batch, seq_len, 1)
    noise_i = ExtractChannelLayer(channel_idx=2)(x)  # (batch, seq_len, 1)
    noise_q = ExtractChannelLayer(channel_idx=3)(x)  # (batch, seq_len, 1)

    # 对干净信号应用三角函数变换（主要特征）
    cos_clean_i = TrigonometricLayer(func_type='cos')(clean_i)
    sin_clean_i = TrigonometricLayer(func_type='sin')(clean_i)
    cos_clean_q = TrigonometricLayer(func_type='cos')(clean_q)
    sin_clean_q = TrigonometricLayer(func_type='sin')(clean_q)

    # 对噪声信号应用统计特征提取（辅助特征）
    noise_features = Concatenate(axis=-1)([noise_i, noise_q])  # (batch, seq_len, 2)

    # 噪声特征的统计描述
    noise_conv = Conv1D(4, 3, activation='relu', padding='same', name='noise_conv')(noise_features)

    # 组合所有特征
    all_features = Concatenate(axis=-1, name='combine_all_features')([
        cos_clean_i, sin_clean_i, cos_clean_q, sin_clean_q,  # 干净信号的三角特征 (4通道)
        noise_conv                                             # 噪声特征 (4通道)
    ])  # (batch, seq_len, 8)

    # 添加SNR信息
    snr_features = Dense(8, activation='relu', name='snr_features')(snr_input)  # (batch, 8)
    snr_features = ExpandDimsLayer(axis=1)(snr_features)  # (batch, 1, 8)
    snr_features = TileLayer([1, input_shape[1], 1])(snr_features)  # (batch, seq_len, 8)

    # 最终特征融合
    final_features = Concatenate(axis=-1, name='final_feature_fusion')([all_features, snr_features])  # (batch, seq_len, 16)

    # 卷积特征提取
    x = Conv1D(64, 3, activation='relu', padding='same', name='conv1')(final_features)
    x = BatchNormalization(name='bn1')(x)
    x = Conv1D(128, 3, activation='relu', padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)

    # 全局平均池化
    x = GlobalAveragePooling1D(name='gap')(x)

    # 添加SNR信息到最终分类特征
    snr_final = Dense(32, activation='relu', name='snr_final')(snr_input)  # (batch, 32)
    x = Concatenate(axis=-1, name='classifier_features')([x, snr_final])  # (batch, 128+32)

    # 分类头
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    # 创建双输入模型
    model = Model(inputs=[signal_input, snr_input], outputs=outputs, name='G_PET_UNet')

    # 编译模型（包含多个损失项）
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # 主要损失：分类损失
        metrics=['accuracy']
        # 附加损失（功率约束+KL散度）会自动通过add_loss添加
    )

    return model


def build_gpet_lightweight_model(input_shape, num_classes=11):
    """
    构建轻量级G-PET模型（单输入版本，用于对比）

    Args:
        input_shape: 输入形状 (2, seq_len)
        num_classes: 分类数量

    Returns:
        编译后的Keras模型
    """

    # 输入层
    inputs = Input(shape=input_shape, name='input_signals')

    # 转换为时间序列格式
    x = TransposeLayer()(inputs)  # (batch, seq_len, 2)

    # 简化的特征提取
    i_data = ExtractChannelLayer(channel_idx=0)(x)
    q_data = ExtractChannelLayer(channel_idx=1)(x)

    # 三角函数变换
    cos_i = TrigonometricLayer(func_type='cos')(i_data)
    sin_i = TrigonometricLayer(func_type='sin')(i_data)
    cos_q = TrigonometricLayer(func_type='cos')(q_data)
    sin_q = TrigonometricLayer(func_type='sin')(q_data)

    # 特征融合
    features = Concatenate(axis=-1)([cos_i, sin_i, cos_q, sin_q])

    # 轻量卷积
    x = Conv1D(64, 3, activation='relu', padding='same')(features)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    # 简化分类头
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # 创建模型
    model = Model(inputs=inputs, outputs=outputs, name='G_PET_Lightweight')

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 导出函数供集成使用
def build_gpet_model(input_shape, num_classes=11):
    """构建完整的G-PET模型"""
    return build_g_pet_model(input_shape, num_classes)


if __name__ == "__main__":
    print("测试G-PET模型构建...")

    # 测试信号-噪声分解模型
    print("\\n=== G-PET信号-噪声分解模型 ===")
    try:
        model_full = build_gpet_model((2, 128), num_classes=11)
        model_full.summary()
        print(f"总参数: {model_full.count_params():,}")
        print("✅ G-PET信号-噪声分解模型创建成功")

        # 测试前向传播
        import numpy as np
        test_signal = np.random.randn(4, 2, 128).astype(np.float32)
        test_snr = np.random.uniform(-10, 20, (4, 1)).astype(np.float32)
        output = model_full([test_signal, test_snr])
        print(f"测试输出形状: {output.shape}")

    except Exception as e:
        print(f"❌ G-PET信号-噪声分解模型创建失败: {e}")

    # 测试轻量级模型
    print("\\n=== 轻量级G-PET模型 ===")
    try:
        model_lite = build_gpet_lightweight_model((2, 128), num_classes=11)
        model_lite.summary()
        print(f"总参数: {model_lite.count_params():,}")
        print("✅ 轻量级G-PET模型创建成功")
    except Exception as e:
        print(f"❌ 轻量级G-PET模型创建失败: {e}")