# -*- coding: utf-8 -*-
"""
ULCNN模型评估脚本 - 终极修复版本
直接修补 Keras 源码以解决 h5py 兼容性问题
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 在导入 Keras 前修复问题
def patch_keras_topology():
    """直接修补 Keras 的 topology.py 文件"""
    import keras.engine.topology as topology
    
    # 保存原始函数
    original_load_weights = topology.load_weights_from_hdf5_group
    
    def patched_load_weights_from_hdf5_group(f, layers, reshape=False):
        """修复版本的权重加载函数"""
        import numpy as np
        from keras import backend as K
        import warnings
        
        # 跳过版本检查部分，直接加载权重
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        
        # 处理层名称
        layer_names = []
        if 'layer_names' in f.attrs:
            for name in f.attrs['layer_names']:
                if hasattr(name, 'decode'):
                    layer_names.append(name.decode('utf8'))
                else:
                    layer_names.append(str(name))
        
        # 过滤有权重的层
        filtered_layers = []
        for layer in layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)
        
        layer_names = layer_names[:len(filtered_layers)]
        
        # 加载每一层的权重
        for k, name in enumerate(layer_names):
            if k >= len(filtered_layers):
                break
                
            g = f[name]
            
            # 处理权重名称
            weight_names = []
            if 'weight_names' in g.attrs:
                for weight_name in g.attrs['weight_names']:
                    if hasattr(weight_name, 'decode'):
                        weight_names.append(weight_name.decode('utf8'))
                    else:
                        weight_names.append(str(weight_name))
            
            # 加载权重值
            weight_values = [g[weight_name] for weight_name in weight_names]
            
            # 获取层的符号权重
            layer = filtered_layers[k]
            symbolic_weights = layer.weights
            
            # 转换权重值
            weight_value_tuples = []
            for i, (symbolic_weight, weight_value) in enumerate(zip(symbolic_weights, weight_values)):
                if hasattr(weight_value, 'value'):
                    array = weight_value.value
                else:
                    array = np.array(weight_value)
                    
                weight_value_tuples.append((symbolic_weight, array))
            
            # 批量设置权重
            if weight_value_tuples:
                K.batch_set_value(weight_value_tuples)
    
    # 替换原函数
    topology.load_weights_from_hdf5_group = patched_load_weights_from_hdf5_group
    print("Keras topology 兼容性补丁已应用")

# 应用补丁
patch_keras_topology()

n_neuron = 16
n_mobileunit = 6
ks = 5

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

def build_model():
    """构建ULCNN模型"""
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
    return model

if __name__ == "__main__":
    print("=" * 50)
    print("ULCNN 模型评估脚本")
    print("=" * 50)
    
    # 构建模型
    print("构建模型...")
    model = build_model()
    
    # 加载权重
    filename = f'ULCNN_MN={n_mobileunit}_N={n_neuron}_KS={ks}'
    model_path = f"model/{filename}.hdf5"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        exit(1)
    
    print(f"加载模型权重: {model_path}")
    try:
        model.load_weights(model_path)
        print("权重加载成功！")
    except Exception as e:
        print(f"权重加载失败: {e}")
        exit(1)
    
    # 评估模型
    print("\n开始评估模型...")
    snrs = range(-20, 20, 2)
    results = []
    
    print("SNR(dB)\tAccuracy")
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
        
        # 打印平均精度
        avg_acc = np.mean(results_array[:, 1])
        print(f"平均准确率: {avg_acc:.4f}")
    
    # 测试推理时间
    print("\n测试推理时间...")
    try:
        x, y = TestDataset(0)
        
        # 预热
        model.evaluate(x, y, batch_size=1000, verbose=0)
        
        # 测试推理时间
        t1 = time.time()
        model.evaluate(x, y, batch_size=1000, verbose=0)
        t2 = time.time()
        
        inference_time = t2 - t1
        samples_per_second = len(x) / inference_time
        print(f"推理时间: {inference_time:.4f}秒")
        print(f"处理速度: {samples_per_second:.2f} samples/second")
        
    except Exception as e:
        print(f"推理时间测试失败: {e}")
    
    print("\n评估完成！")