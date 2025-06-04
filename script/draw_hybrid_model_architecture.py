#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合ComplexCNN-ResNet架构整体流程图绘制工具

基于论文grok.tex和enhanced_hybrid_model.tex中的架构描述，
绘制完整的模型架构流程图，包括：
1. 左侧：整体架构纵向流程图（层级合并简化）
2. 右侧：核心模块详细图解
   - 复数卷积模块简图
   - 复数批归一化简图
   - 残差连接简图

Author: RadioML-v3 Team
Date: 2025-06-04
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 设置数学字体
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'sans-serif'

# 设置颜色方案
COLORS = {
    'complex_domain': '#4A90E2',      # 复数域 - 蓝色
    'real_domain': '#F5A623',         # 实数域 - 橙色
    'input_output': '#7ED321',        # 输入输出 - 绿色
    'residual_basic': '#9013FE',      # 基础残差 - 紫色
    'residual_advanced': '#BD10E0',   # 高级残差 - 深紫
    'attention': '#F8E71C',           # 注意力 - 黄色
    'pooling': '#E74C3C',             # 池化 - 红色
    'transition': '#50E3C2',          # 转换 - 青色
    'text': '#2C3E50',                # 文本 - 深灰
    'arrow': '#34495E'                # 箭头 - 灰色
}

def create_architecture_diagram():
    """创建完整的架构图"""
    # 创建主图形
    fig = plt.figure(figsize=(24, 16))
    
    # 重新设计网格布局：左侧主架构，右侧三个等宽模块图
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 1, 1], height_ratios=[1, 1, 1])
    
    # 左侧主架构图
    ax_main = fig.add_subplot(gs[:, 0])
    draw_main_architecture(ax_main)
    
    # 右侧模块详细图 - 统一使用单列布局
    ax_conv = fig.add_subplot(gs[0, 1:])  # 第0行，第1-2列
    draw_complex_convolution_module(ax_conv)
    
    ax_bn = fig.add_subplot(gs[1, 1:])    # 第1行，第1-2列
    draw_complex_batch_norm_module(ax_bn)
    
    ax_res = fig.add_subplot(gs[2, 1:])   # 第2行，第1-2列
    draw_residual_connection_module(ax_res)
    
    # 设置总标题
    fig.suptitle('混合ComplexCNN-ResNet架构详细流程图\nLightweight Hybrid Complex-ResNet Architecture', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 调整布局 - 增加右侧模块间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.2, hspace=0.4)
    
    return fig

def draw_main_architecture(ax):
    """绘制主架构流程图（左侧纵向）"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 架构层级定义（纵向排列，从上到下）
    layers = [
        # (y_pos, width, height, color, text, description)
        (18.5, 8, 1, COLORS['input_output'], 'I/Q输入信号', 'Input (2, 128)'),
        (17.0, 8, 0.8, COLORS['complex_domain'], '预处理层', 'Permute (128, 2)'),
        
        # Stage 1: 初始特征提取
        (15.5, 8, 1.5, COLORS['complex_domain'], 'Stage 1: 初始特征提取', 
         'ComplexConv1D(32) + BN + Activation + Pool'),
        
        # Stage 2: 残差处理（合并显示）
        (13.5, 8, 1.2, COLORS['residual_basic'], '基础残差块', 'ComplexResidualBlock(64)'),
        (12.0, 8, 1.2, COLORS['residual_basic'], '下采样残差块', 'ComplexResidualBlock(128, s=2)'),
        (10.5, 8, 1.2, COLORS['residual_advanced'], '高级残差块', 'ComplexResidualBlockAdvanced(256, s=2)'),
        
        # Stage 3: 全局特征
        (8.5, 8, 1, COLORS['pooling'], '全局特征聚合', 'ComplexGlobalAveragePooling1D'),
        
        # Stage 4: 复数全连接
        (7.0, 8, 1, COLORS['complex_domain'], '复数全连接层', 'ComplexDense(512) + Dropout'),
        
        # 域转换
        (5.5, 8, 1, COLORS['transition'], '复数→实数转换', 'Complex Magnitude'),
        
        # 实数分类
        (4.0, 8, 1, COLORS['real_domain'], '实数全连接', 'Dense(256) + ReLU + Dropout'),
        (2.5, 8, 1, COLORS['real_domain'], '输出分类层', 'Dense(11) + Softmax'),
        
        # 输出
        (1.0, 8, 0.8, COLORS['input_output'], '分类结果', '11个调制类别'),
    ]
    
    # 绘制层级框
    boxes = []
    for y_pos, width, height, color, text, desc in layers:
        x_pos = 1
        
        # 创建圆角矩形
        box = FancyBboxPatch(
            (x_pos, y_pos - height/2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(box)
        boxes.append((x_pos + width/2, y_pos))
        
        # 添加主要文本
        ax.text(x_pos + width/2, y_pos + 0.1, text, 
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white' if color in [COLORS['residual_basic'], COLORS['residual_advanced']] else COLORS['text'])
        
        # 添加描述文本
        ax.text(x_pos + width/2, y_pos - 0.2, desc,
                ha='center', va='center', fontsize=9, style='italic',
                color='white' if color in [COLORS['residual_basic'], COLORS['residual_advanced']] else COLORS['text'])
    
    # 绘制连接箭头
    for i in range(len(boxes) - 1):
        x1, y1 = boxes[i]
        x2, y2 = boxes[i + 1]
        
        # 计算箭头起点和终点
        arrow_start_y = y1 - layers[i][2]/2 - 0.1
        arrow_end_y = y2 + layers[i+1][2]/2 + 0.1
        
        ax.annotate('', xy=(x2, arrow_end_y), xytext=(x1, arrow_start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['arrow']))
    
    # 添加域标识
    # 复数域背景
    complex_bg = patches.Rectangle((0.5, 4.5), 9, 14, linewidth=2, 
                                  edgecolor=COLORS['complex_domain'], facecolor='none', 
                                  linestyle='--', alpha=0.7)
    ax.add_patch(complex_bg)
    ax.text(0.3, 11.5, '复数域处理\nComplex Domain', rotation=90, ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['complex_domain'])
    
    # 实数域背景  
    real_bg = patches.Rectangle((0.5, 1.5), 9, 3.5, linewidth=2,
                               edgecolor=COLORS['real_domain'], facecolor='none',
                               linestyle='--', alpha=0.7)
    ax.add_patch(real_bg)
    ax.text(0.3, 3.5, '实数域分类\nReal Domain', rotation=90, ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['real_domain'])
    
    # 添加性能指标
    ax.text(5, 0.2, '模型性能: ~400K参数 | 65.4%准确率 | 2.3ms推理时间', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

def draw_complex_convolution_module(ax):
    """绘制复数卷积模块详细图（右上）"""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('复数卷积模块 (Complex Convolution)', fontsize=14, fontweight='bold', pad=20)
    
    # 输入复数信号
    input_box = FancyBboxPatch((1, 6.5), 3, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['complex_domain'], alpha=0.7)
    ax.add_patch(input_box)
    ax.text(2.5, 7, 'z = x + jy\n复数输入', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 分离实部虚部
    real_box = FancyBboxPatch((0.5, 5), 1.8, 0.8,
                             boxstyle="round,pad=0.05", facecolor='lightblue', alpha=0.7)
    ax.add_patch(real_box)
    ax.text(1.4, 5.4, '实部 x', ha='center', va='center', fontsize=10)
    
    imag_box = FancyBboxPatch((2.7, 5), 1.8, 0.8,
                             boxstyle="round,pad=0.05", facecolor='lightgreen', alpha=0.7)
    ax.add_patch(imag_box)
    ax.text(3.6, 5.4, '虚部 y', ha='center', va='center', fontsize=10)
    
    # 复数权重
    kernel_box = FancyBboxPatch((6, 5.5), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['attention'], alpha=0.7)
    ax.add_patch(kernel_box)
    ax.text(8, 6.25, 'W = Wr + jWi\n复数权重矩阵', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 卷积运算
    conv_operations = [
        (1.5, 3.5, 'x*Wr', 'lightblue'),
        (3, 3.5, 'x*Wi', 'lightblue'),
        (4.5, 3.5, 'y*Wr', 'lightgreen'),
        (6, 3.5, 'y*Wi', 'lightgreen')
    ]
    
    for x, y, text, color in conv_operations:
        box = FancyBboxPatch((x-0.5, y-0.3), 1, 0.6,
                            boxstyle="round,pad=0.05", facecolor=color, alpha=0.6)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
    
    # 复数乘法结果
    result_real = FancyBboxPatch((1, 2), 3.5, 0.8,
                                boxstyle="round,pad=0.1", facecolor='orange', alpha=0.7)
    ax.add_patch(result_real)
    ax.text(2.75, 2.4, 'Real: x*Wr - y*Wi', ha='center', va='center', fontsize=10, fontweight='bold')
    
    result_imag = FancyBboxPatch((5.5, 2), 3.5, 0.8,
                                boxstyle="round,pad=0.1", facecolor='purple', alpha=0.7)
    ax.add_patch(result_imag)
    ax.text(7.25, 2.4, 'Imag: x*Wi + y*Wr', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 最终输出
    output_box = FancyBboxPatch((3.5, 0.5), 4, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['complex_domain'], alpha=0.7)
    ax.add_patch(output_box)
    ax.text(5.5, 0.9, '复数卷积输出', ha='center', va='center', fontsize=11, fontweight='bold')
      # 绘制连接箭头
    arrows = [
        ((2.5, 6.5), (1.4, 5.8)),   # 输入到实部
        ((2.5, 6.5), (3.6, 5.8)),   # 输入到虚部
        ((1.4, 5), (1.5, 3.8)),     # 实部x到x*Wr运算
        ((1.4, 5), (3, 3.8)),       # 实部x到x*Wi运算
        ((3.6, 5), (4.5, 3.8)),     # 虚部y到y*Wr运算
        ((3.6, 5), (6, 3.8)),       # 虚部y到y*Wi运算
        ((1.5, 3.2), (2.75, 2.8)),  # x*Wr到实部结果
        ((3, 3.2), (2.75, 2.8)),    # x*Wi到实部结果
        ((4.5, 3.2), (7.25, 2.8)),  # y*Wr到虚部结果
        ((6, 3.2), (7.25, 2.8)),    # y*Wi到虚部结果
        ((2.75, 2), (4.5, 1.3)),    # 实部结果到输出
        ((7.25, 2), (6.5, 1.3))     # 虚部结果到输出
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['arrow']))

def draw_complex_batch_norm_module(ax):
    """绘制复数批归一化模块详细图（右中）"""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('复数批归一化模块 (Complex Batch Normalization)', fontsize=14, fontweight='bold', pad=20)
    
    # 输入
    input_box = FancyBboxPatch((1, 6.5), 3.5, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['complex_domain'], alpha=0.7)
    ax.add_patch(input_box)
    ax.text(2.75, 7, 'z = x + jy\n复数特征', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 统计计算
    stats_box = FancyBboxPatch((6, 6), 5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['attention'], alpha=0.7)
    ax.add_patch(stats_box)
    # 添加统计量文本
    ax.text(8.5, 6.75, '统计量计算\nμr, μi, σrr, σii, σri', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 协方差矩阵
    cov_box = FancyBboxPatch((1, 4.5), 4, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor='lightcyan', alpha=0.8)
    ax.add_patch(cov_box)
    ax.text(3, 5.1, '协方差矩阵\nC = [σrr  σri]\n    [σri  σii]', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 白化变换
    whiten_box = FancyBboxPatch((6.5, 4.5), 4, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='lightyellow', alpha=0.8)
    ax.add_patch(whiten_box)
    ax.text(8.5, 5.1, '白化变换\nW = C^(-1/2)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 归一化
    norm_box = FancyBboxPatch((2, 2.5), 3.5, 1,
                             boxstyle="round,pad=0.1",
                             facecolor='lightgreen', alpha=0.8)
    ax.add_patch(norm_box)
    ax.text(3.75, 3, '归一化\nz_hat = W(z - μ)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 可学习变换
    transform_box = FancyBboxPatch((6.5, 2.5), 3.5, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightpink', alpha=0.8)
    ax.add_patch(transform_box)
    ax.text(8.25, 3, '可学习变换\nz_out = γ*z_hat + β', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 输出
    output_box = FancyBboxPatch((4, 0.8), 4.5, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['complex_domain'], alpha=0.7)
    ax.add_patch(output_box)
    ax.text(6.25, 1.3, '归一化后的复数特征', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 绘制连接箭头
    arrows = [
        ((2.75, 6.5), (6, 6.9)),      # 输入到统计
        ((2.75, 6.5), (3, 5.7)),      # 输入到协方差
        ((8.5, 6), (8.5, 5.7)),       # 统计到白化
        ((3, 4.5), (3.75, 3.5)),      # 协方差到归一化
        ((8.5, 4.5), (8.25, 3.5)),    # 白化到变换
        ((3.75, 2.5), (5.5, 1.8)),    # 归一化到输出
        ((8.25, 2.5), (7, 1.8))       # 变换到输出
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['arrow']))

def draw_residual_connection_module(ax):
    """绘制残差连接模块详细图（右下）"""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('残差连接模块 (Residual Connection)', fontsize=14, fontweight='bold', pad=20)
    
    # 输入
    input_box = FancyBboxPatch((0.5, 6), 2.5, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['complex_domain'], alpha=0.7)
    ax.add_patch(input_box)
    ax.text(1.75, 6.5, '复数输入\nz', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 主路径（上路径）
    main_path_boxes = [
        (4.5, 6.5, 1.8, 0.8, 'Conv1'),
        (6.8, 6.5, 1.8, 0.8, 'BN1'),
        (9.1, 6.5, 1.8, 0.8, 'ReLU'),
        (9.1, 5, 1.8, 0.8, 'Conv2'),
        (6.8, 5, 1.8, 0.8, 'BN2')
    ]
    
    for x, y, w, h, text in main_path_boxes:
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.05",
                            facecolor=COLORS['residual_basic'], alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
      # 跳跃连接（下路径）
    skip_start = (1.75, 5.5)        # 从输入框下方开始
    skip_end = (5.5, 3.1)         # 到加法圆圈下方
    
    # 绘制跳跃连接曲线 - 从下方绕过
    skip_path = patches.FancyArrowPatch(skip_start, skip_end,
                                       connectionstyle="arc3,rad=.5",  # 正值表示从下方绕过
                                       arrowstyle='->', mutation_scale=20,
                                       color=COLORS['attention'], linewidth=3)
    ax.add_patch(skip_path)
    ax.text(3, 2.8, '跳跃连接\nSkip Connection', ha='center', va='center', 
            fontsize=10, fontweight='bold', color=COLORS['attention'])
    
    # 加法操作
    add_circle = Circle((5.5, 3.5), 0.4, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(add_circle)
    ax.text(5.5, 3.5, '+', ha='center', va='center', fontsize=16, fontweight='bold')
      # 输出激活
    output_act_box = FancyBboxPatch((7, 3), 1.8, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['residual_basic'], alpha=0.7)
    ax.add_patch(output_act_box)
    ax.text(7.9, 3.5, 'ReLU', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 最终输出
    final_output_box = FancyBboxPatch((9.5, 2.5), 2, 1.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor=COLORS['complex_domain'], alpha=0.7)
    ax.add_patch(final_output_box)
    ax.text(10.5, 3.25, '残差输出\nH(z) = F(z) + z', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制主路径连接箭头
    main_path_arrows = [
        ((3, 6.5), (3.6, 6.5)),      # 输入到Conv1
        ((5.4, 6.5), (5.9, 6.5)),    # Conv1到BN1
        ((7.7, 6.5), (8.2, 6.5)),    # BN1到ReLU
        ((9.1, 6.1), (9.1, 5.4)),    # ReLU到Conv2
        ((8.2, 5), (7.7, 5)),        # Conv2到BN2
        ((6.8, 4.6), (5.9, 3.7)),    # BN2到加法
        ((5.9, 3.5), (6.6, 3.5)),    # 加法到激活
        ((8.8, 3.5), (9.5, 3.5))     # 激活到输出
    ]
    
    for start, end in main_path_arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['arrow']))
    
    # 添加数学公式
    ax.text(6, 1.8, '数学表示: H(z) = F(z) + z\n其中 F(z) 是学习的复数残差函数', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    # 添加优势说明
    advantages = ['• 解决梯度消失问题', '• 保持复数相位信息', '• 增强特征表达能力']
    for i, adv in enumerate(advantages):
        ax.text(0.5, 2.5 - i*0.4, adv, ha='left', va='center', fontsize=9, 
                color=COLORS['text'], fontweight='bold')

def save_and_show_diagram():
    """保存并显示图表"""
    fig = create_architecture_diagram()
    
    # 保存高质量图片
    output_path = 'd:/1python programs/radioml/radioML-v3/script/hybrid_model_architecture_complete.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"架构图已保存到: {output_path}")
    
    # 显示图片
    plt.show()
    
    return fig

if __name__ == "__main__":
    # 创建并保存架构图
    print("开始绘制混合ComplexCNN-ResNet架构图...")
    fig = save_and_show_diagram()
    print("架构图绘制完成！")
    
    # 输出架构摘要信息
    print("\n" + "="*60)
    print("混合ComplexCNN-ResNet架构摘要")
    print("="*60)
    print("模型特点:")
    print("• 端到端复数域处理，保持I/Q信号相位信息")
    print("• 轻量级设计，约400K参数")
    print("• 残差连接解决梯度消失问题")
    print("• 复数批归一化提升训练稳定性")
    print("• 全局平均池化减少过拟合")
    print("• 在RadioML2016.10a数据集上达到65.4%准确率")
    print("• 推理时间仅2.3ms，适合实时应用")
    print("="*60)
