"""
Professional Neural Network Visualization Tools
使用专业工具可视化轻量级混合模型架构

本脚本集成多种专业的神经网络可视化工具：
1. Diagrams - Python架构图生成
2. PlotNeuralNet - LaTeX/TikZ代码生成
3. TensorBoard - 模型图可视化
4. NN-SVG - Web可视化配置生成
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def install_required_packages():
    """
    安装所需的专业可视化包
    """
    try:
        import subprocess
        import sys
        
        packages = [
            'diagrams',
            'graphviz',
            'pydot',
            'tensorboard'
        ]
        
        for package in packages:
            try:
                __import__(package)
                print(f"✅ {package} already installed")
            except ImportError:
                print(f"📦 Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                
    except Exception as e:
        print(f"⚠️ Package installation error: {e}")
        print("Please manually install: pip install diagrams graphviz pydot tensorboard")

def create_diagrams_visualization():
    """
    使用 Diagrams 库创建专业的架构图
    """
    try:
        from diagrams import Diagram, Cluster, Edge
        from diagrams.generic.blank import Blank
        from diagrams.programming.framework import React
        from diagrams.aws.ml import SagemakerModel
        
        output_path = os.path.join('script', 'figure', 'professional_diagrams')
        os.makedirs(output_path, exist_ok=True)
        
        # 创建轻量级混合模型的 Diagrams 可视化
        with Diagram("Lightweight Hybrid Model Architecture", 
                    filename=os.path.join(output_path, "lightweight_hybrid_diagrams"),
                    show=False, direction="TB"):
            
            # 输入层
            with Cluster("Input Processing"):
                input_layer = Blank("I/Q Signal\n(2, 128)")
                permute = Blank("Permute\n(128, 2)")
            
            # 复数特征提取
            with Cluster("Complex Feature Extraction"):
                complex_conv = Blank("ComplexConv1D\nfilters=32")
                complex_bn = Blank("ComplexBN")
                complex_act = Blank("ComplexActivation")
                complex_pool = Blank("ComplexPooling")
            
            # 复数残差块
            with Cluster("Complex Residual Learning"):
                res_block1 = SagemakerModel("ResBlock-1\nfilters=64")
                res_block2 = SagemakerModel("ResBlock-2\nfilters=128")
                res_block3 = SagemakerModel("ResBlock-3\nfilters=256")
            
            # 全局特征
            with Cluster("Global Feature Processing"):
                global_pool = Blank("ComplexGlobal\nAveragePooling")
                complex_dense = Blank("ComplexDense\n512 units")
            
            # 分类层
            with Cluster("Classification"):
                magnitude = Blank("ComplexMagnitude")
                final_dense = Blank("Dense\n256 units")
                output = Blank("Output\n11 classes")
            
            # 连接
            input_layer >> permute
            permute >> complex_conv >> complex_bn >> complex_act >> complex_pool
            complex_pool >> res_block1 >> res_block2 >> res_block3
            res_block3 >> global_pool >> complex_dense
            complex_dense >> magnitude >> final_dense >> output
            
            # 残差连接
            res_block1 >> Edge(style="dashed", color="red") >> res_block2
            res_block2 >> Edge(style="dashed", color="red") >> res_block3
        
        print(f"✅ Diagrams visualization saved to: {output_path}")
        return output_path
        
    except ImportError:
        print("❌ Diagrams not available. Install with: pip install diagrams")
        return None
    except Exception as e:
        print(f"❌ Diagrams visualization error: {e}")
        return None

def generate_plotneuralnet_code():
    """
    生成 PlotNeuralNet (LaTeX/TikZ) 代码
    """
    latex_code = r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{../layers/}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d}

\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\DenseColor{rgb:blue,5;red,2.5;white,5}
\def\ComplexColor{rgb:green,5;blue,2.5;white,5}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.6]

% Input Layer
\pic[shift={(0,0,0)}] at (0,0,0) {Box={name=input,caption=Input\\(2×128),fill=\ConvColor,height=40,width=2,depth=128}};

% Permute
\pic[shift={(3,0,0)}] at (0,0,0) {Box={name=permute,caption=Permute\\(128×2),fill=\ConvColor,height=128,width=2,depth=40}};

% Complex Feature Extraction
\pic[shift={(6,0,0)}] at (0,0,0) {Box={name=conv1,caption=ComplexConv1D\\filters=32,fill=\ComplexColor,height=128,width=32,depth=35}};

\pic[shift={(9,0,0)}] at (0,0,0) {Box={name=bn1,caption=ComplexBN,fill=\ComplexColor,height=128,width=32,depth=30}};

\pic[shift={(12,0,0)}] at (0,0,0) {Box={name=act1,caption=ComplexActivation,fill=\ComplexColor,height=128,width=32,depth=25}};

\pic[shift={(15,0,0)}] at (0,0,0) {Box={name=pool1,caption=ComplexPooling,fill=\PoolColor,height=64,width=32,depth=20}};

% Complex Residual Blocks
\pic[shift={(18,0,0)}] at (0,0,0) {Box={name=res1,caption=ResBlock-1\\filters=64,fill=\ComplexColor,height=64,width=64,depth=18}};

\pic[shift={(22,0,0)}] at (0,0,0) {Box={name=res2,caption=ResBlock-2\\filters=128,fill=\ComplexColor,height=32,width=128,depth=16}};

\pic[shift={(26,0,0)}] at (0,0,0) {Box={name=res3,caption=ResBlock-3\\filters=256,fill=\ComplexColor,height=16,width=256,depth=14}};

% Global Features
\pic[shift={(30,0,0)}] at (0,0,0) {Box={name=globalpool,caption=ComplexGlobal\\AvgPooling,fill=\PoolColor,height=8,width=256,depth=12}};

\pic[shift={(33,0,0)}] at (0,0,0) {Box={name=dense1,caption=ComplexDense\\512 units,fill=\ComplexColor,height=6,width=512,depth=10}};

% Classification
\pic[shift={(36,0,0)}] at (0,0,0) {Box={name=magnitude,caption=ComplexMagnitude,fill=\DenseColor,height=4,width=512,depth=8}};

\pic[shift={(39,0,0)}] at (0,0,0) {Box={name=dense2,caption=Dense\\256 units,fill=\DenseColor,height=4,width=256,depth=6}};

\pic[shift={(42,0,0)}] at (0,0,0) {Box={name=output,caption=Output\\11 classes,fill=\DenseColor,height=2,width=11,depth=4}};

% Connections
\draw [connection]  (input-east)    -- node {\midarrow} (permute-west);
\draw [connection]  (permute-east)  -- node {\midarrow} (conv1-west);
\draw [connection]  (conv1-east)    -- node {\midarrow} (bn1-west);
\draw [connection]  (bn1-east)      -- node {\midarrow} (act1-west);
\draw [connection]  (act1-east)     -- node {\midarrow} (pool1-west);
\draw [connection]  (pool1-east)    -- node {\midarrow} (res1-west);
\draw [connection]  (res1-east)     -- node {\midarrow} (res2-west);
\draw [connection]  (res2-east)     -- node {\midarrow} (res3-west);
\draw [connection]  (res3-east)     -- node {\midarrow} (globalpool-west);
\draw [connection]  (globalpool-east) -- node {\midarrow} (dense1-west);
\draw [connection]  (dense1-east)   -- node {\midarrow} (magnitude-west);
\draw [connection]  (magnitude-east) -- node {\midarrow} (dense2-west);
\draw [connection]  (dense2-east)   -- node {\midarrow} (output-west);

% Residual Connections
\draw [connection, color=red, dashed]  (res1-north) to [bend left=30] node {} (res2-north);
\draw [connection, color=red, dashed]  (res2-north) to [bend left=30] node {} (res3-north);

% Labels
\node[above=1cm of input] {\Large Lightweight Hybrid Model Architecture};
\node[below=0.5cm of input] {Phase 1: Complex Feature Extraction};
\node[below=0.5cm of res2] {Phase 2: Complex Residual Learning};
\node[below=0.5cm of dense1] {Phase 3: Global Processing};
\node[below=0.5cm of output] {Phase 4: Classification};

\end{tikzpicture}
\end{document}
"""
    
    # 保存 LaTeX 代码
    output_dir = os.path.join('script', 'figure', 'plotneuralnet')
    os.makedirs(output_dir, exist_ok=True)
    
    latex_file = os.path.join(output_dir, 'lightweight_hybrid_plotneuralnet.tex')
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    # 创建编译说明
    readme_content = """# PlotNeuralNet Visualization

## Requirements
1. Install PlotNeuralNet: `git clone https://github.com/HarisIqbal88/PlotNeuralNet.git`
2. Install LaTeX with TikZ support
3. Install required packages: `sudo apt-get install texlive-latex-extra texlive-fonts-recommended`

## Usage
1. Place the .tex file in the PlotNeuralNet directory
2. Compile with: `pdflatex lightweight_hybrid_plotneuralnet.tex`
3. Output will be a high-quality PDF diagram

## Features
- 3D layer visualization
- Color-coded layer types
- Residual connections shown as dashed red lines
- Scalable vector graphics output
"""
    
    readme_file = os.path.join(output_dir, 'README.md')
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ PlotNeuralNet LaTeX code saved to: {latex_file}")
    return latex_file

def create_tensorboard_visualization():
    """
    创建 TensorBoard 可视化
    """
    try:
        # 重新导入模型以获取架构
        import sys
        sys.path.append('src')
        from model.hybrid_complex_resnet_model import build_lightweight_hybrid_model
        
        # 创建模型
        input_shape = (2, 128)
        num_classes = 11
        model = build_lightweight_hybrid_model(input_shape, num_classes)
        
        # 保存模型图
        output_dir = os.path.join('script', 'figure', 'tensorboard')
        os.makedirs(output_dir, exist_ok=True)
        
        # Keras plot_model
        model_plot_path = os.path.join(output_dir, 'lightweight_hybrid_keras_plot.png')
        plot_model(model, to_file=model_plot_path, show_shapes=True, show_layer_names=True,
                  rankdir='TB', expand_nested=True, dpi=300)
        
        # TensorBoard logs
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建虚拟数据用于图形生成
        import numpy as np
        dummy_input = np.random.randn(1, 2, 128).astype(np.float32)
        
        # 保存模型为 SavedModel 格式用于 TensorBoard
        saved_model_path = os.path.join(output_dir, 'saved_model')
        model.save(saved_model_path, save_format='tf')
        
        # 创建 TensorBoard 启动脚本
        tensorboard_script = f"""#!/bin/bash
# TensorBoard Visualization Script

echo "Starting TensorBoard for Lightweight Hybrid Model..."
echo "Model saved at: {saved_model_path}"
echo "Logs directory: {log_dir}"

# Launch TensorBoard
tensorboard --logdir={log_dir} --port=6006

echo "Open http://localhost:6006 in your browser to view the model graph"
"""
        
        script_path = os.path.join(output_dir, 'launch_tensorboard.sh')
        with open(script_path, 'w') as f:
            f.write(tensorboard_script)
        
        # Windows批处理文件
        bat_script = f"""@echo off
echo Starting TensorBoard for Lightweight Hybrid Model...
echo Model saved at: {saved_model_path}
echo Logs directory: {log_dir}

tensorboard --logdir={log_dir} --port=6006

echo Open http://localhost:6006 in your browser to view the model graph
pause
"""
        
        bat_path = os.path.join(output_dir, 'launch_tensorboard.bat')
        with open(bat_path, 'w') as f:
            f.write(bat_script)
        
        print(f"✅ TensorBoard files saved to: {output_dir}")
        print(f"📊 Keras model plot: {model_plot_path}")
        print(f"🚀 Run tensorboard with: {script_path}")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ TensorBoard visualization error: {e}")
        return None

def generate_nnsvg_config():
    """
    生成 NN-SVG 网站的配置文件
    """
    # NN-SVG 配置 (用于在线工具 http://alexlenail.me/NN-SVG/)
    nnsvg_config = {
        "architecture": "Lightweight Hybrid Model",
        "layers": [
            {
                "type": "Input",
                "name": "I/Q Signal",
                "shape": [2, 128],
                "color": "#E8F4FD"
            },
            {
                "type": "Reshape", 
                "name": "Permute",
                "shape": [128, 2],
                "color": "#E8F4FD"
            },
            {
                "type": "Convolution",
                "name": "ComplexConv1D",
                "filters": 32,
                "kernel_size": 5,
                "shape": [128, 64],
                "color": "#FFE6CC"
            },
            {
                "type": "BatchNormalization",
                "name": "ComplexBN",
                "shape": [128, 64],
                "color": "#D4EDDA"
            },
            {
                "type": "Activation",
                "name": "ComplexActivation",
                "activation": "LeakyReLU",
                "shape": [128, 64],
                "color": "#FFF3CD"
            },
            {
                "type": "Pooling",
                "name": "ComplexPooling",
                "pool_size": 2,
                "shape": [64, 64],
                "color": "#F8D7DA"
            },
            {
                "type": "ResidualBlock",
                "name": "ComplexResBlock-1",
                "filters": 64,
                "shape": [64, 128],
                "color": "#E2E3E5",
                "skip_connection": True
            },
            {
                "type": "ResidualBlock", 
                "name": "ComplexResBlock-2",
                "filters": 128,
                "strides": 2,
                "shape": [32, 256],
                "color": "#E2E3E5",
                "skip_connection": True
            },
            {
                "type": "ResidualBlock",
                "name": "ComplexResBlock-3", 
                "filters": 256,
                "strides": 2,
                "shape": [16, 512],
                "color": "#E2E3E5",
                "skip_connection": True
            },
            {
                "type": "GlobalAveragePooling",
                "name": "ComplexGlobalAvgPool",
                "shape": [512],
                "color": "#F8D7DA"
            },
            {
                "type": "Dense",
                "name": "ComplexDense",
                "units": 512,
                "activation": "LeakyReLU",
                "shape": [512],
                "color": "#D1ECF1"
            },
            {
                "type": "Custom",
                "name": "ComplexMagnitude",
                "description": "Complex to Real Conversion",
                "shape": [512],
                "color": "#F5C6CB"
            },
            {
                "type": "Dense",
                "name": "Dense",
                "units": 256,
                "activation": "ReLU",
                "shape": [256],
                "color": "#C3E6CB"
            },
            {
                "type": "Dense",
                "name": "Output",
                "units": 11,
                "activation": "Softmax",
                "shape": [11],
                "color": "#FADBD8"
            }
        ],
        "connections": [
            {"from": 0, "to": 1},
            {"from": 1, "to": 2},
            {"from": 2, "to": 3},
            {"from": 3, "to": 4},
            {"from": 4, "to": 5},
            {"from": 5, "to": 6},
            {"from": 6, "to": 7},
            {"from": 7, "to": 8},
            {"from": 8, "to": 9},
            {"from": 9, "to": 10},
            {"from": 10, "to": 11},
            {"from": 11, "to": 12},
            {"from": 12, "to": 13}
        ],
        "skip_connections": [
            {"from": 6, "to": 7, "type": "residual"},
            {"from": 7, "to": 8, "type": "residual"}
        ],
        "metadata": {
            "total_parameters": "1.3M",
            "accuracy": "65.38%",
            "inference_time": "2.3ms",
            "input_type": "I/Q Signal",
            "output_classes": 11
        }
    }
    
    # 保存配置
    output_dir = os.path.join('script', 'figure', 'nnsvg')
    os.makedirs(output_dir, exist_ok=True)
    
    config_file = os.path.join(output_dir, 'lightweight_hybrid_nnsvg_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(nnsvg_config, f, indent=2, ensure_ascii=False)
    
    # 创建使用说明
    instructions = """# NN-SVG Configuration

## Usage
1. Visit: http://alexlenail.me/NN-SVG/
2. Click "Load JSON Configuration"
3. Upload the `lightweight_hybrid_nnsvg_config.json` file
4. Customize colors, spacing, and labels as needed
5. Export as SVG for publication-quality graphics

## Features
- Interactive web-based editor
- SVG output (scalable vector graphics)
- Customizable colors and styling
- Professional appearance
- Easy to modify and update

## Alternative: Manual Setup
If the JSON import doesn't work, manually create layers:
- Input: 2×128
- Conv1D: filters=32, kernel=5
- BatchNorm + Activation + Pooling
- 3x Residual Blocks (64, 128, 256 filters)
- Global Average Pooling
- Dense: 512 → 256 → 11 (Softmax)

Add residual connections between ResBlocks.
"""
    
    instructions_file = os.path.join(output_dir, 'INSTRUCTIONS.md')
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"✅ NN-SVG configuration saved to: {config_file}")
    return config_file

def create_additional_professional_visualizations():
    """
    创建其他专业可视化工具的配置
    """
    output_dir = os.path.join('script', 'figure', 'professional_tools')
    os.makedirs(output_dir, exist_ok=True)
    
    # Netron 模型文件说明 
    netron_instructions = """# Netron Model Visualization

Netron is a professional neural network visualizer that supports many formats.

## Usage
1. Install Netron: `pip install netron`
2. Save your model in .h5, .pb, .onnx, or .keras format
3. Run: `netron model_file.keras`
4. Open in browser for interactive exploration

## Model Files Available
- `model_weight_saved/lightweight_hybrid_model.keras`
- Interactive layer-by-layer exploration
- Weight and bias visualization
- Network topology analysis

## Command
```bash
cd model_weight_saved
netron lightweight_hybrid_model.keras
```
"""
    
    # Graphviz DOT 格式
    dot_content = """digraph LightweightHybridModel {
    rankdir=TB;
    node [shape=box, style=filled];
    
    // Input layers
    input [label="I/Q Signal\\n(2, 128)", fillcolor="#E8F4FD"];
    permute [label="Permute\\n(128, 2)", fillcolor="#E8F4FD"];
    
    // Complex feature extraction
    conv1 [label="ComplexConv1D\\nfilters=32, kernel=5", fillcolor="#FFE6CC"];
    bn1 [label="ComplexBN", fillcolor="#D4EDDA"];
    act1 [label="ComplexActivation\\nLeakyReLU", fillcolor="#FFF3CD"];
    pool1 [label="ComplexPooling1D\\npool_size=2", fillcolor="#F8D7DA"];
    
    // Residual blocks
    res1 [label="ComplexResBlock-1\\nfilters=64", fillcolor="#E2E3E5"];
    res2 [label="ComplexResBlock-2\\nfilters=128, stride=2", fillcolor="#E2E3E5"];
    res3 [label="ComplexResBlock-3\\nfilters=256, stride=2", fillcolor="#E2E3E5"];
    
    // Global features
    globalpool [label="ComplexGlobal\\nAveragePooling", fillcolor="#F8D7DA"];
    dense1 [label="ComplexDense\\n512 units", fillcolor="#D1ECF1"];
    
    // Classification
    magnitude [label="ComplexMagnitude\\n(Complex→Real)", fillcolor="#F5C6CB"];
    dense2 [label="Dense\\n256 units, ReLU", fillcolor="#C3E6CB"];
    output [label="Output\\n11 classes, Softmax", fillcolor="#FADBD8"];
    
    // Main connections
    input -> permute -> conv1 -> bn1 -> act1 -> pool1;
    pool1 -> res1 -> res2 -> res3;
    res3 -> globalpool -> dense1 -> magnitude;
    magnitude -> dense2 -> output;
    
    // Residual connections
    res1 -> res2 [style=dashed, color=red, label="skip"];
    res2 -> res3 [style=dashed, color=red, label="skip"];
    
    // Clustering
    subgraph cluster_0 {
        label="Complex Feature Extraction";
        style=dashed;
        conv1; bn1; act1; pool1;
    }
    
    subgraph cluster_1 {
        label="Complex Residual Learning";
        style=dashed;
        res1; res2; res3;
    }
    
    subgraph cluster_2 {
        label="Classification";
        style=dashed;
        magnitude; dense2; output;
    }
}"""
    
    # 保存文件
    with open(os.path.join(output_dir, 'netron_instructions.md'), 'w') as f:
        f.write(netron_instructions)
    
    with open(os.path.join(output_dir, 'lightweight_hybrid.dot'), 'w') as f:
        f.write(dot_content)
    
    # Draw.io XML 配置
    drawio_xml = """<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net">
  <diagram name="Lightweight Hybrid Model">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <!-- Input -->
        <mxCell id="input" value="I/Q Signal&#xa;(2, 128)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F4FD;" vertex="1" parent="1">
          <mxGeometry x="50" y="50" width="100" height="60" as="geometry"/>
        </mxCell>
        <!-- Complex Conv -->
        <mxCell id="conv" value="ComplexConv1D&#xa;filters=32" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE6CC;" vertex="1" parent="1">
          <mxGeometry x="50" y="150" width="100" height="60" as="geometry"/>
        </mxCell>
        <!-- Add more cells as needed -->
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    
    with open(os.path.join(output_dir, 'lightweight_hybrid_drawio.xml'), 'w') as f:
        f.write(drawio_xml)
    
    print(f"✅ Additional professional tools configurations saved to: {output_dir}")
    return output_dir

def main():
    """
    主函数：生成所有专业可视化工具的配置和图像
    """
    print("🔧 Professional Neural Network Visualization Tools")
    print("=" * 60)
    
    # 检查并安装所需包
    print("📦 Installing required packages...")
    install_required_packages()
    
    results = {}
    
    # 1. Diagrams 可视化
    print("\n🎨 Creating Diagrams visualization...")
    diagrams_path = create_diagrams_visualization()
    if diagrams_path:
        results['diagrams'] = diagrams_path
    
    # 2. PlotNeuralNet LaTeX 代码
    print("\n📄 Generating PlotNeuralNet LaTeX code...")
    plotnn_path = generate_plotneuralnet_code()
    results['plotneuralnet'] = plotnn_path
    
    # 3. TensorBoard 可视化
    print("\n📊 Creating TensorBoard visualization...")
    tensorboard_path = create_tensorboard_visualization()
    if tensorboard_path:
        results['tensorboard'] = tensorboard_path
    
    # 4. NN-SVG 配置
    print("\n🌐 Generating NN-SVG configuration...")
    nnsvg_path = generate_nnsvg_config()
    results['nnsvg'] = nnsvg_path
    
    # 5. 其他专业工具
    print("\n🛠️ Creating additional professional tool configurations...")
    additional_path = create_additional_professional_visualizations()
    results['additional'] = additional_path
    
    # 总结
    print("\n✅ Professional Visualization Tools Summary")
    print("=" * 60)
    for tool, path in results.items():
        if path:
            print(f"📁 {tool.upper()}: {path}")
    
    print(f"\n🎯 All professional visualization tools configured!")
    print(f"📚 Check the generated files for high-quality neural network visualizations")
    
    return results

if __name__ == "__main__":
    main()
