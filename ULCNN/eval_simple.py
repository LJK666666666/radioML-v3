# -*- coding: utf-8 -*-
"""
ULCNN模型评估脚本 - 直接从原始5ULCNN.py修改而来
只进行评估，不训练
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
n_neuron = 16
n_mobileunit = 6
ks = 5

from keras import models
from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import scipy.io as scio
import numpy as np
from numpy import array
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import backend as K
from keras import models
from keras.layers import *
from complexnn.conv import ComplexConv1D
from complexnn.bn import ComplexBatchNormalization
from complexnn.dense import ComplexDense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import random
import time

def TestDataset(snr):
    x = np.load(f"test_RML/x_snr={snr}.npy")
    x = x.transpose((0, 2, 1))
    y = np.load(f"test_RML/y_snr={snr}.npy")
    y = to_categorical(y)
    return x, y

def channel_shuffle(x):
    in_channels, D = K.int_shape(x)[1:]
    channels_per_group = in_channels // 2
    pre_shape = [-1, 2, channels_per_group, D]
    dim = (0, 2, 1, 3)
    later_shape = [-1, in_channels, D]

    x = Lambda(lambda z: K.reshape(z, pre_shape))(x)
    x = Lambda(lambda z: K.permute_dimensions(z, dim))(x)  
    x = Lambda(lambda z: K.reshape(z, later_shape))(x)

    return x

def dwconv_mobile(x, neurons):
    x = SeparableConv1D(int(2*neurons), ks, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = channel_shuffle(x)
    return x

def channelattention(x):
    x_GAP  = GlobalAveragePooling1D()(x)
    x_GMP  = GlobalMaxPooling1D()(x)
    channel = K.int_shape(x_GAP)[1]

    share_Dense1 = Dense(int(channel/16), activation = 'relu')
    share_Dense2 = Dense(channel)

    x_GAP = Reshape((1, channel))(x_GAP)
    x_GAP = share_Dense1(x_GAP)
    x_GAP = share_Dense2(x_GAP)

    
    x_GMP = Reshape((1, channel))(x_GMP)
    x_GMP = share_Dense1(x_GMP)
    x_GMP = share_Dense2(x_GMP)

    x_Mask = Add()([x_GAP, x_GMP])
    x_Mask = Activation('sigmoid')(x_Mask)

    x = Multiply()([x, x_Mask])
    return x

# 构建模型
x_input = Input(shape=[128, 2])
x = ComplexConv1D(n_neuron, ks, padding='same')(x_input)
x = ComplexBatchNormalization()(x)
x = Activation('relu')(x)

for i in range(n_mobileunit):
    x = dwconv_mobile(x, n_neuron)
    x = channelattention(x)
    if i==3:
        f4 = GlobalAveragePooling1D()(x)
    if i==4:
        f5 = GlobalAveragePooling1D()(x)
    if i==5:
        f6 = GlobalAveragePooling1D()(x)

f = Add()([f4, f5])
f = Add()([f,f6])
f = Dense(11)(f)
c = Activation('softmax', name='modulation')(f)

model = Model(inputs = x_input, outputs=c)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])

print("模型构建完成")
model.summary()

filename = f'ULCNN_MN={n_mobileunit}_N={n_neuron}_KS={ks}'
model_path = f"model/{filename}.hdf5"

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在 {model_path}")
    exit(1)

print(f"\n加载模型权重: {model_path}")

# 尝试多种方法加载权重
success = False

# 方法1: 直接加载
try:
    model.load_weights(model_path)
    print("方法1成功: 直接加载")
    success = True
except Exception as e:
    print(f"方法1失败: {e}")

# 方法2: 重新创建模型并加载
if not success:
    try:
        print("尝试方法2: 重新创建模型...")
        # 清除会话
        K.clear_session()
        
        # 重新构建模型
        x_input = Input(shape=[128, 2])
        x = ComplexConv1D(n_neuron, ks, padding='same')(x_input)
        x = ComplexBatchNormalization()(x)
        x = Activation('relu')(x)

        for i in range(n_mobileunit):
            x = dwconv_mobile(x, n_neuron)
            x = channelattention(x)
            if i==3:
                f4 = GlobalAveragePooling1D()(x)
            if i==4:
                f5 = GlobalAveragePooling1D()(x)
            if i==5:
                f6 = GlobalAveragePooling1D()(x)

        f = Add()([f4, f5])
        f = Add()([f,f6])
        f = Dense(11)(f)
        c = Activation('softmax', name='modulation')(f)

        model = Model(inputs = x_input, outputs=c)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', 
                      metrics=['accuracy'])
        
        # 尝试加载权重
        model.load_weights(model_path)
        print("方法2成功: 重新创建模型后加载")
        success = True
    except Exception as e:
        print(f"方法2失败: {e}")

# 方法3: 使用 by_name=False
if not success:
    try:
        print("尝试方法3: 使用 by_name=False...")
        model.load_weights(model_path, by_name=False)
        print("方法3成功: by_name=False")
        success = True
    except Exception as e:
        print(f"方法3失败: {e}")

if not success:
    print("所有加载方法都失败了，无法继续评估")
    exit(1)

print("权重加载成功！开始评估...")

# 评估模型
snrs = range(-20, 20, 2)
results = []

print("\nSNR(dB)\tAccuracy")
print("-" * 20)

for snr in snrs:
    try:
        x, y = TestDataset(snr)
        [loss, acc] = model.evaluate(x, y, batch_size=1000, verbose=0)
        results.append([snr, acc])
        print(f"{snr:6d}\t{acc:.4f}")
    except Exception as e:
        print(f"SNR {snr} 评估失败: {e}")
        continue

# 保存结果
if results:
    results_array = np.array(results)
    df = pd.DataFrame(results_array, columns=['SNR(dB)', 'Accuracy'])
    output_file = f'evaluation_results_{filename}.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n评估结果已保存到: {output_file}")

# 测试推理时间
print("\n测试推理时间...")
try:
    x, y = TestDataset(0)
    
    # 预热
    model.evaluate(x, y, batch_size=1000, verbose=0)
    
    # 测试推理时间
    batch_sizes = [1000]
    for bs in batch_sizes:
        t1 = time.time()
        model.evaluate(x, y, batch_size=bs, verbose=0)
        t2 = time.time()
        
        inference_time = t2 - t1
        samples_per_second = len(x) / inference_time
        print(f"批处理大小: {bs}")
        print(f"推理时间: {inference_time:.4f}秒")
        print(f"处理速度: {samples_per_second:.2f} samples/second")
        
except Exception as e:
    print(f"推理时间测试失败: {e}")

print("\n评估完成！")