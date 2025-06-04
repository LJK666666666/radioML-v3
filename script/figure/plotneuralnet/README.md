# PlotNeuralNet 使用说明

## 安装 PlotNeuralNet
```bash
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
cd PlotNeuralNet
```

## 编译生成图像
1. 将 `lightweight_hybrid_advanced.tex` 复制到 PlotNeuralNet 目录
2. 运行编译命令：
```bash
bash png_latex.sh lightweight_hybrid_advanced
```

## 输出文件
- PDF: `lightweight_hybrid_advanced.pdf`
- PNG: `lightweight_hybrid_advanced.png`

## 自定义说明
- 可以修改颜色定义部分来改变视觉效果
- 调整 height, width, depth 参数来改变层的大小
- 修改 shift 参数来调整层的位置

## 依赖要求
- LaTeX (推荐 TeX Live)
- Python 3.x
- ImageMagick (用于PNG转换)
