# -*- coding: utf-8 -*-
"""
ULCNN模型评估脚本
直接加载训练好的模型权重，在测试集上进行评估
Created for evaluation only
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 修复 h5py 字符串解码兼容性问题
import h5py

def load_weights_safe(model, filepath):
    """安全加载权重的函数，处理 h5py 版本兼容性问题"""
    try:
        model.load_weights(filepath)
    except (AttributeError, TypeError) as e:
        if "decode" in str(e) or "conversion path" in str(e):
            print("检测到 h5py 版本兼容性问题，尝试修复...")
            
            # 方法1: 修改环境变量强制使用兼容模式
            import os
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
            
            # 方法2: 直接读取 h5 文件并手动处理
            import keras.engine.topology as topology
            original_func = topology.load_weights_from_hdf5_group
            
            def safe_load_weights_from_hdf5_group(f, layers, reshape=False):
                # 备份原始属性
                original_attrs = {}
                if hasattr(f, 'attrs'):
                    for key in f.attrs.keys():
                        original_attrs[key] = f.attrs[key]
                        # 如果是字符串类型的属性，确保它能被正确处理
                        if key == 'keras_version':
                            try:
                                # 尝试删除和重新创建属性
                                del f.attrs[key]
                                # 使用 bytes 类型存储
                                version_str = original_attrs[key]
                                if isinstance(version_str, str):
                                    f.attrs.create(key, version_str.encode('utf-8'))
                                else:
                                    f.attrs.create(key, version_str)
                            except:
                                # 如果失败，保持原样
                                pass
                
                try:
                    return original_func(f, layers, reshape)
                except:
                    # 如果还是失败，恢复原始属性并重试
                    for key, value in original_attrs.items():
                        try:
                            if key in f.attrs:
                                del f.attrs[key]
                            f.attrs.create(key, value)
                        except:
                            pass
                    # 最后的尝试：忽略版本检查
                    return original_func(f, layers, reshape)
            
            # 应用补丁
            topology.load_weights_from_hdf5_group = safe_load_weights_from_hdf5_group
            
            try:
                model.load_weights(filepath)
                print("权重加载成功（已应用兼容性修复）")
            except Exception as inner_e:
                print(f"兼容性修复失败: {inner_e}")
                # 尝试最后的方法：降级处理
                print("尝试降级处理...")
                try:
                    # 创建新的 h5py 文件，去掉有问题的属性
                    import tempfile
                    import shutil
                    
                    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_file:
                        temp_path = tmp_file.name
                    
                    # 复制文件内容，跳过有问题的属性
                    with h5py.File(filepath, 'r') as src, h5py.File(temp_path, 'w') as dst:
                        def copy_without_attrs(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                dst.create_dataset(name, data=obj[:], dtype=obj.dtype)
                            elif isinstance(obj, h5py.Group):
                                dst.create_group(name)
                        
                        src.visititems(copy_without_attrs)
                    
                    model.load_weights(temp_path)
                    os.unlink(temp_path)  # 删除临时文件
                    print("权重加载成功（使用降级处理）")
                    
                except Exception as final_e:
                    print(f"所有修复方法都失败了: {final_e}")
                    raise final_e
            finally:
                # 恢复原始函数
                topology.load_weights_from_hdf5_group = original_func
        else:
            raise e

# 模型参数（必须与训练时一致）
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
        load_weights_safe(model, model_path)
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
    load_weights_safe(model, model_path)
    
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
    test_inference_time(model_path)
    
    print("\n评估完成！")