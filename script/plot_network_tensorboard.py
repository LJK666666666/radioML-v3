import tensorflow as tf
from tensorflow.keras import layers, Model
import datetime
import os
import numpy as np

def create_complex_residual_block(inputs, filters, stride=1, is_advanced=False):
    """创建复数残差块"""
    x = layers.Conv1D(filters, kernel_size=3, strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if is_advanced:
        x = layers.LeakyReLU()(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
    
    # 残差连接
    if stride != 1 or inputs.shape[-1] != filters:
        inputs = layers.Conv1D(filters, kernel_size=1, strides=stride)(inputs)
    
    x = layers.Add()([x, inputs])
    return x

def build_hybrid_model():
    """构建混合复数ResNet模型"""
    inputs = layers.Input(shape=(128, 2))
    
    # 初始复数特征提取
    x = layers.Conv1D(32, kernel_size=5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # 残差块
    x = create_complex_residual_block(x, 64)
    x = create_complex_residual_block(x, 128, stride=2)
    x = create_complex_residual_block(x, 256, stride=2, is_advanced=True)
    
    # 全局特征提取
    x = layers.GlobalAveragePooling1D()(x)
    
    # 复数全连接层
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    # 实数域分类
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(11, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # 创建模型
    model = build_hybrid_model()
    
    # 创建日志目录（使用绝对路径）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"日志目录已创建: {log_dir}")
    
    # 创建TensorBoard回调
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 创建一些虚拟训练数据
    x_train = np.random.random((100, 128, 2))  # 100个样本，每个样本128个时间步，2个特征
    y_train = np.random.randint(0, 11, (100,))  # 11个类别的随机标签
    y_train = tf.keras.utils.to_categorical(y_train, 11)  # 转换为one-hot编码
    
    # 训练模型一个epoch来生成事件文件
    history = model.fit(
        x_train,
        y_train,
        epochs=1,
        batch_size=32,
        callbacks=[tensorboard_callback],
        verbose=1
    )
    
    # 打印模型结构
    model.summary()
    
    # 验证日志文件是否创建
    log_files = os.listdir(log_dir)
    print(f"\n日志目录中的文件: {log_files}")
    
    print(f"\nTensorBoard日志保存在: {log_dir}")
    print("运行以下命令查看网络结构:")
    print(f"tensorboard --logdir={log_dir}")

if __name__ == "__main__":
    main() 