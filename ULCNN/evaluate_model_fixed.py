# -*- coding: utf-8 -*-
"""
ULCNN模型评估脚本（兼容性修复版本）
直接加载训练好的模型权重，在测试集上进行评估
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 在导入任何其他库之前修复 h5py 兼容性问题
import sys

# 模型参数（必须与训练时一致）
n_neuron = 16
n_mobileunit = 6
ks = 5

# 直接修复 Keras 的权重加载函数
def apply_keras_fix():
    """在导入 Keras 之前修复兼容性问题"""
    # 方法：monkey patch Keras 的 load_weights_from_hdf5_group 函数
    import keras.engine.topology as topology
    original_load_weights = topology.load_weights_from_hdf5_group
    
    def fixed_load_weights_from_hdf5_group(f, layers, reshape=False):
        """修复后的权重加载函数"""
        # 备份原始的 decode 方法检查
        import h5py
        
        # 临时替换有问题的代码段
        if hasattr(f, 'attrs') and 'keras_version' in f.attrs:
            keras_version = f.attrs['keras_version']
            # 如果 keras_version 已经是字符串，直接使用
            if isinstance(keras_version, str):
                # 创建一个临时的包装器，让字符串有 decode 方法
                class StrWithDecode(str):
                    def decode(self, encoding='utf8'):
                        return str(self)
                
                # 临时替换属性
                original_version = f.attrs['keras_version']
                try:
                    # 删除原属性
                    del f.attrs['keras_version']
                    # 用包装后的版本替换
                    f.attrs['keras_version'] = StrWithDecode(original_version).encode('utf8')
                except:
                    # 如果无法修改，使用其他方法
                    pass
        
        # 调用原始函数
        try:
            return original_load_weights(f, layers, reshape)
        except AttributeError as e:
            if "'str' object has no attribute 'decode'" in str(e):
                # 直接修改 Keras 源码中的问题行
                import keras.engine.topology
                
                # 保存原始代码
                original_code = keras.engine.topology.load_weights_from_hdf5_group.__code__
                
                # 创建新的函数，跳过版本检查
                def bypass_version_check(f, layers, reshape=False):
                    """绕过版本检查的权重加载"""
                    import h5py
                    from keras.utils.generic_utils import to_list
                    from keras import backend as K
                    
                    if 'layer_names' not in f.attrs and 'model_weights' in f:
                        f = f['model_weights']
                    
                    layer_names = [n.decode('utf8') if hasattr(n, 'decode') else str(n) 
                                 for n in f.attrs.get('layer_names', [])]
                    filtered_layers = []
                    for layer in layers:
                        weights = layer.weights
                        if weights:
                            filtered_layers.append(layer)
                    
                    for k, name in enumerate(layer_names):
                        g = f[name]
                        weight_names = [n.decode('utf8') if hasattr(n, 'decode') else str(n) 
                                      for n in g.attrs.get('weight_names', [])]
                        weight_values = [g[weight_name] for weight_name in weight_names]
                        
                        if k < len(filtered_layers):
                            layer = filtered_layers[k]
                            symbolic_weights = layer.weights
                            weight_values_list = []
                            for weight_value in weight_values:
                                if hasattr(weight_value, 'value'):
                                    weight_values_list.append(weight_value.value)
                                else:
                                    weight_values_list.append(weight_value[:])
                            
                            if len(weight_values_list) != len(symbolic_weights):
                                continue
                                
                            K.batch_set_value(zip(symbolic_weights, weight_values_list))
                
                return bypass_version_check(f, layers, reshape)
            else:
                raise e
    
    # 应用补丁
    topology.load_weights_from_hdf5_group = fixed_load_weights_from_hdf5_group

# 在导入 Keras 后立即应用修复
apply_keras_fix()

from keras import models
from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
from keras import backend as K
from complexnn.conv import ComplexConv1D
from complexnn.bn import ComplexBatchNormalization
from complexnn.dense import ComplexDense
import pandas as pd
import time

def TestDataset(snr):
    """加载测试数据集"""
    x = np.load(f"test_RML/x_snr={snr}.npy")
    x = x.transpose((0, 2, 1))
    y = np.load(f"test_RML/y_snr={snr}.npy")
    y = to_categorical(y)
    return x, y

def channel_shuffle(x):
    """通道重排函数"""
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
    """深度可分离卷积移动单元"""
    x = SeparableConv1D(int(2*neurons), ks, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = channel_shuffle(x)
    return x

def channelattention(x):
    """通道注意力机制"""
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

def build_model():
    """构建ULCNN模型结构"""
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
    f = Add()([f, f6])
    f = Dense(11)(f)
    c = Activation('softmax', name='modulation')(f)

    model = Model(inputs = x_input, outputs=c)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model

def evaluate_model(model_path, snr_range=None):
    """评估模型在不同SNR下的性能"""
    print("构建模型结构...")
    model = build_model()
    
    print(f"加载模型权重: {model_path}")
    try:
        model.load_weights(model_path)
        print("模型权重加载成功！")
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        return
    
    if snr_range is None:
        snr_range = range(-20, 20, 2)
    
    results = []
    print("\n开始评估模型性能...")
    print("SNR(dB)\tAccuracy")
    print("-" * 20)
    
    for snr in snr_range:
        try:
            x, y = TestDataset(snr)
            [loss, acc] = model.evaluate(x, y, batch_size=1000, verbose=0)
            results.append([snr, acc])
            print(f"{snr:6d}\t{acc:.4f}")
        except Exception as e:
            print(f"SNR {snr}评估失败: {e}")
            continue
    
    # 保存结果
    results_array = np.array(results)
    df = pd.DataFrame(results_array, columns=['SNR(dB)', 'Accuracy'])
    output_file = f'evaluation_results_MN={n_mobileunit}_N={n_neuron}_KS={ks}.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n评估结果已保存到: {output_file}")
    
    return results

def test_inference_time(model_path, snr=0, batch_sizes=[1000]):
    """测试推理时间"""
    print("\n测试推理时间...")
    model = build_model()
    model.load_weights(model_path)
    
    x, y = TestDataset(snr)
    
    for bs in batch_sizes:
        print(f"\n批处理大小: {bs}")
        # 预热
        model.evaluate(x, y, batch_size=bs, verbose=0)
        
        # 测试推理时间
        t1 = time.time()
        model.evaluate(x, y, batch_size=bs, verbose=0)
        t2 = time.time()
        
        inference_time = t2 - t1
        samples_per_second = len(x) / inference_time
        print(f"推理时间: {inference_time:.4f}秒")
        print(f"处理速度: {samples_per_second:.2f} samples/second")

if __name__ == "__main__":
    # 模型权重文件路径
    filename = f'ULCNN_MN={n_mobileunit}_N={n_neuron}_KS={ks}'
    model_path = f"model/{filename}.hdf5"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请确保模型文件路径正确")
        exit(1)
    
    print("=" * 50)
    print("ULCNN 模型评估脚本")
    print("=" * 50)
    print(f"模型参数: MobileUnits={n_mobileunit}, Neurons={n_neuron}, KernelSize={ks}")
    print(f"模型文件: {model_path}")
    
    # 评估模型性能
    results = evaluate_model(model_path)
    
    # 测试推理时间
    if results:  # 只有评估成功时才测试推理时间
        test_inference_time(model_path)
    
    print("\n评估完成！")