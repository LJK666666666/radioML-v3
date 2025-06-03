#!/usr/bin/env python3
"""
从RESULT.md中提取关键实验数据并生成LaTeX表格内容
用于更新 radio_modulation_classification.tex 中的表格
"""

import re
import os

def read_result_file(filepath):
    """读取RESULT.md文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def extract_accuracy_from_section(content, section_name):
    """从指定章节提取准确率信息"""
    # 匹配章节开始
    pattern = rf"# {re.escape(section_name)}\s*\n(.*?)(?=\n# |\n$)"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return None
    
    section_content = match.group(1)
    
    # 提取总体准确率
    overall_pattern = r"Overall accuracy: (0\.\d+)"
    overall_match = re.search(overall_pattern, section_content)
    overall_accuracy = float(overall_match.group(1)) if overall_match else None
    
    # 提取各SNR下的准确率
    snr_pattern = r"SNR ([+-]?\d+\.?\d*) dB: Accuracy = (0\.\d+)"
    snr_matches = re.findall(snr_pattern, section_content)
    snr_accuracies = {float(snr): float(acc) for snr, acc in snr_matches}
    
    return {
        'overall': overall_accuracy,
        'snr_accuracies': snr_accuracies
    }

def calculate_snr_ranges(snr_accuracies):
    """计算不同SNR范围的平均准确率"""
    low_snr = []  # -20dB到-2dB
    mid_snr = []  # 0dB到8dB
    high_snr = [] # 10dB到18dB
    
    for snr, acc in snr_accuracies.items():
        if -20 <= snr <= -2:
            low_snr.append(acc)
        elif 0 <= snr <= 8:
            mid_snr.append(acc)
        elif 10 <= snr <= 18:
            high_snr.append(acc)
    
    return {
        'low': sum(low_snr) / len(low_snr) if low_snr else 0,
        'mid': sum(mid_snr) / len(mid_snr) if mid_snr else 0,
        'high': sum(high_snr) / len(high_snr) if high_snr else 0
    }

def main():
    # 文件路径
    result_file = r"d:\1python programs\radioml\radioML-v3\guide\RESULT.md"
    
    if not os.path.exists(result_file):
        print(f"文件不存在: {result_file}")
        return
    
    # 读取文件内容
    content = read_result_file(result_file)
    
    # 要提取的模型配置
    models = {
        'lightweight_hybrid': 'lightweight_hybrid',
        'lightweight hybrid + augment': 'lightweight hybrid + augment',
        'lightweight hybrid + gpr': 'lightweight hybrid + gpr', 
        'lightweight hybrid + gpr + augment': 'lightweight hybrid + gpr + augment'
    }
    
    print("=== 轻量级混合模型实验结果 ===\n")
    
    results = {}
    
    # 提取各模型的数据
    for display_name, section_name in models.items():
        data = extract_accuracy_from_section(content, section_name)
        if data:
            results[display_name] = data
            snr_ranges = calculate_snr_ranges(data['snr_accuracies'])
            
            print(f"## {display_name}")
            print(f"总体准确率: {data['overall']:.4f} ({data['overall']*100:.2f}%)")
            print(f"低SNR (-20dB到-2dB): {snr_ranges['low']:.4f} ({snr_ranges['low']*100:.2f}%)")
            print(f"中SNR (0dB到8dB): {snr_ranges['mid']:.4f} ({snr_ranges['mid']*100:.2f}%)")
            print(f"高SNR (10dB到18dB): {snr_ranges['high']:.4f} ({snr_ranges['high']*100:.2f}%)")
            print()
    
    # 生成消融研究表格（基于混合架构的消融研究）
    print("=== 消融研究表格（LaTeX格式）===\n")
    
    base_name = 'lightweight_hybrid'
    aug_name = 'lightweight hybrid + augment'
    gpr_name = 'lightweight hybrid + gpr'
    full_name = 'lightweight hybrid + gpr + augment'
    
    if all(name in results for name in [base_name, aug_name, gpr_name, full_name]):
        base_acc = results[base_name]['overall']
        aug_acc = results[aug_name]['overall']
        gpr_acc = results[gpr_name]['overall']
        full_acc = results[full_name]['overall']
        
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{消融研究结果（以混合架构为基线）}")
        print("\\label{tab:ablation_study}")
        print("\\begin{tabular}{@{}lccc@{}}")
        print("\\toprule")
        print("组件组合 & 准确率(\\%) & 提升(\\%) & 累计提升(\\%) \\\\")
        print("\\midrule")
        print(f"混合架构（基线） & {base_acc*100:.2f} & - & - \\\\")
        print(f"混合架构 + 数据增强 & {aug_acc*100:.2f} & +{(aug_acc-base_acc)*100:.2f} & +{(aug_acc-base_acc)*100:.2f} \\\\")
        print(f"混合架构 + GPR去噪 & {gpr_acc*100:.2f} & +{(gpr_acc-base_acc)*100:.2f} & +{(gpr_acc-base_acc)*100:.2f} \\\\")
        print(f"混合架构 + GPR去噪 + 数据增强 & {full_acc*100:.2f} & +{(full_acc-base_acc)*100:.2f} & +{(full_acc-base_acc)*100:.2f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print()
    
    # 生成GPR去噪影响表格
    print("=== GPR去噪影响表格（LaTeX格式）===\n")
    
    if base_name in results and gpr_name in results:
        base_snr = calculate_snr_ranges(results[base_name]['snr_accuracies'])
        gpr_snr = calculate_snr_ranges(results[gpr_name]['snr_accuracies'])
        
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{GPR去噪对不同SNR范围的影响}")
        print("\\label{tab:gpr_impact}")
        print("\\begin{tabular}{@{}lccc@{}}")
        print("\\toprule")
        print("SNR范围 & 去噪前(\\%) & 去噪后(\\%) & 提升(\\%) \\\\")
        print("\\midrule")
        print(f"低SNR (-20dB到-2dB) & {base_snr['low']*100:.1f} & {gpr_snr['low']*100:.1f} & +{(gpr_snr['low']-base_snr['low'])*100:.1f} \\\\")
        print(f"中SNR (0dB到8dB) & {base_snr['mid']*100:.1f} & {gpr_snr['mid']*100:.1f} & +{(gpr_snr['mid']-base_snr['mid'])*100:.1f} \\\\")
        print(f"高SNR (10dB到18dB) & {base_snr['high']*100:.1f} & {gpr_snr['high']*100:.1f} & +{(gpr_snr['high']-base_snr['high'])*100:.1f} \\\\")
        print(f"总体 & {results[base_name]['overall']*100:.1f} & {results[gpr_name]['overall']*100:.1f} & +{(results[gpr_name]['overall']-results[base_name]['overall'])*100:.1f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print()
    
    # 生成数据增强影响的数据
    print("=== 数据增强影响分析 ===\n")
    if base_name in results and aug_name in results:
        base_acc = results[base_name]['overall']
        aug_acc = results[aug_name]['overall']
        improvement = (aug_acc - base_acc) * 100
        
        print(f"基线准确率: {base_acc*100:.2f}%")
        print(f"增强后准确率: {aug_acc*100:.2f}%")
        print(f"总体提升: +{improvement:.2f}个百分点")
        print()
    
    # 为文中表述生成具体数值
    print("=== 论文中的关键数值 ===\n")
    if full_name in results:
        final_acc = results[full_name]['overall']
        print(f"最终混合模型准确率: {final_acc*100:.2f}%")
        
        if base_name in results:
            base_acc = results[base_name]['overall']
            total_improvement = (final_acc - base_acc) * 100
            print(f"相比基线混合架构提升: +{total_improvement:.2f}个百分点")
            
        # 如果有ComplexCNN基线数据，也可以比较
        complexnn_data = extract_accuracy_from_section(content, "complexnn + relu")
        if complexnn_data and complexnn_data['overall']:
            complexnn_acc = complexnn_data['overall']
            vs_complexnn = (final_acc - complexnn_acc) * 100
            print(f"相比ComplexCNN基线提升: +{vs_complexnn:.2f}个百分点")

if __name__ == "__main__":
    main()
