# TensorBoard 可视化使用说明

## 启动TensorBoard
```bash
tensorboard --logdir=script/figure/tensorboard
```

## 查看模型图
1. 在浏览器中打开 http://localhost:6006
2. 点击 "Graphs" 标签页
3. 选择 "lightweight_hybrid_model" 模型
4. 探索模型的交互式图结构

## 生成的文件
- `model_architecture.png`: Keras生成的模型结构图
- `lightweight_hybrid_model/`: TensorFlow SavedModel格式
  - `saved_model.pb`: 模型图定义
  - `variables/`: 模型权重（如果有的话）

## TensorBoard功能
- **Graphs**: 交互式模型架构图
- **Images**: 模型结构的静态图像
- **Scalars**: 训练指标（如果有训练日志）
- **Histograms**: 权重分布（如果有权重数据）

## 导出高质量图像
在TensorBoard的Graphs页面：
1. 右键点击图形
2. 选择 "Save image as..."
3. 保存为PNG或SVG格式

## 自定义可视化
可以通过修改模型定义来改变可视化效果：
- 添加更多层名称注释
- 使用更描述性的层名
- 添加自定义操作说明
