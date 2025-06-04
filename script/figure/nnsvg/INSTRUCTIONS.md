# NN-SVG 使用说明

## 在线使用
1. 访问 NN-SVG 网站: https://alexlenail.me/NN-SVG/
2. 点击 "LeNet Style" 或其他样式
3. 将 `lightweight_hybrid_complete.json` 的内容粘贴到输入框
4. 调整参数和样式
5. 导出 SVG 格式图像

## 配置文件说明
- `metadata`: 模型基本信息
- `architecture`: 架构概览
- `layers`: 详细的层配置
- `connections`: 层之间的连接
- `skip_connections`: 跳跃连接（残差连接）
- `annotations`: 区域标注

## 自定义选项
- 修改 `color` 字段来改变层的颜色
- 调整 `position` 来改变布局
- 修改 `description` 来改变提示信息

## 输出格式
- SVG: 矢量图格式，可无限缩放
- PNG: 光栅图格式，适合展示
- PDF: 适合学术论文使用
