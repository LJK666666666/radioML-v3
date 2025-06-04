#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNR Performance Calculator
为论文表格提供GPR去噪在各SNR水平下的详细影响计算和验证

Created on June 4, 2025
Author: RadioML-v3 Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import argparse

class SNRPerformanceCalculator:
    """计算和分析GPR去噪在不同SNR水平下的性能影响"""
    
    def __init__(self):
        # 论文中的实验数据
        self.snr_data = {
            'SNR(dB)': [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 
                       0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
            '基线(%)': [8.93, 8.68, 9.85, 11.08, 12.65, 20.15, 34.66, 54.86, 64.02, 75.66,
                      79.43, 82.96, 84.56, 83.93, 83.17, 84.73, 85.81, 85.31, 82.25, 83.87],
            '基线+GPR(%)': [9.96, 10.22, 12.69, 17.32, 24.18, 35.05, 47.36, 61.21, 70.84, 80.89,
                          83.17, 87.07, 89.00, 89.38, 89.10, 89.85, 90.31, 88.81, 88.15, 88.98]
        }
        
        # 计算提升百分点
        self.snr_data['提升(%)'] = [
            round(gpr - baseline, 2) 
            for baseline, gpr in zip(self.snr_data['基线(%)'], self.snr_data['基线+GPR(%)'])
        ]
    
    def generate_latex_table(self) -> str:
        """生成LaTeX表格代码"""
        latex_table = """\\begin{table}[h]
\\centering
\\caption{GPR去噪在各SNR水平下的详细影响}
\\label{tab:gpr_detailed_snr}
\\begin{tabular}{@{}lccc@{}}
\\toprule
SNR(dB) & 基线(\\%) & 基线+GPR(\\%) & 提升(\\%) \\\\
\\midrule
"""
        
        for i in range(len(self.snr_data['SNR(dB)'])):
            snr = self.snr_data['SNR(dB)'][i]
            baseline = self.snr_data['基线(%)'][i]
            gpr = self.snr_data['基线+GPR(%)'][i]
            improvement = self.snr_data['提升(%)'][i]
            
            latex_table += f"{snr} & {baseline} & {gpr} & +{improvement} \\\\\n"
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return latex_table
    
    def calculate_statistics(self) -> Dict[str, float]:
        """计算统计数据"""
        improvements = self.snr_data['提升(%)']
        baseline_acc = self.snr_data['基线(%)']
        gpr_acc = self.snr_data['基线+GPR(%)']
        
        stats = {
            '平均提升': np.mean(improvements),
            '最大提升': np.max(improvements),
            '最小提升': np.min(improvements),
            '提升标准差': np.std(improvements),
            '平均基线准确率': np.mean(baseline_acc),
            '平均GPR准确率': np.mean(gpr_acc),
            '相对改善率': np.mean(improvements) / np.mean(baseline_acc) * 100
        }
        
        return stats
    
    def analyze_snr_ranges(self) -> Dict[str, Dict[str, float]]:
        """分析不同SNR范围的性能"""
        snr_values = self.snr_data['SNR(dB)']
        baseline = self.snr_data['基线(%)']
        gpr = self.snr_data['基线+GPR(%)']
        improvements = self.snr_data['提升(%)']
        
        ranges = {
            '低SNR (-20dB到-8dB)': {'indices': [i for i, x in enumerate(snr_values) if x <= -8]},
            '中SNR (-6dB到4dB)': {'indices': [i for i, x in enumerate(snr_values) if -6 <= x <= 4]},
            '高SNR (6dB到18dB)': {'indices': [i for i, x in enumerate(snr_values) if x >= 6]}
        }
        
        results = {}
        for range_name, range_data in ranges.items():
            indices = range_data['indices']
            results[range_name] = {
                '平均基线准确率': np.mean([baseline[i] for i in indices]),
                '平均GPR准确率': np.mean([gpr[i] for i in indices]),
                '平均提升': np.mean([improvements[i] for i in indices]),
                '最大提升': np.max([improvements[i] for i in indices]),
                '最小提升': np.min([improvements[i] for i in indices])
            }
        
        return results
    
    def plot_performance_curves(self, save_path: str = None):
        """绘制性能曲线图"""
        plt.figure(figsize=(12, 8))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        snr_values = self.snr_data['SNR(dB)']
        baseline = self.snr_data['基线(%)']
        gpr = self.snr_data['基线+GPR(%)']
        improvements = self.snr_data['提升(%)']
        
        # 子图1：准确率对比
        plt.subplot(2, 2, 1)
        plt.plot(snr_values, baseline, 'o-', label='基线', linewidth=2, markersize=6)
        plt.plot(snr_values, gpr, 's-', label='基线+GPR', linewidth=2, markersize=6)
        plt.xlabel('SNR (dB)')
        plt.ylabel('准确率 (%)')
        plt.title('GPR去噪性能对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：提升幅度
        plt.subplot(2, 2, 2)
        colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' for x in improvements]
        plt.bar(snr_values, improvements, color=colors, alpha=0.7)
        plt.xlabel('SNR (dB)')
        plt.ylabel('提升幅度 (%)')
        plt.title('各SNR水平下的提升幅度')
        plt.grid(True, alpha=0.3)
        
        # 子图3：累积分布
        plt.subplot(2, 2, 3)
        sorted_improvements = sorted(improvements)
        cumulative = np.arange(1, len(sorted_improvements) + 1) / len(sorted_improvements)
        plt.plot(sorted_improvements, cumulative, 'o-', linewidth=2)
        plt.xlabel('提升幅度 (%)')
        plt.ylabel('累积概率')
        plt.title('提升幅度累积分布')
        plt.grid(True, alpha=0.3)
        
        # 子图4：SNR范围分析
        plt.subplot(2, 2, 4)
        range_analysis = self.analyze_snr_ranges()
        ranges = list(range_analysis.keys())
        avg_improvements = [range_analysis[r]['平均提升'] for r in ranges]
        
        plt.bar(ranges, avg_improvements, color=['lightcoral', 'lightblue', 'lightgreen'])
        plt.ylabel('平均提升 (%)')
        plt.title('不同SNR范围的平均提升')
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
    
    def export_to_csv(self, filename: str):
        """导出数据到CSV文件"""
        df = pd.DataFrame(self.snr_data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已导出至: {filename}")
    
    def validate_data(self) -> bool:
        """验证数据的一致性"""
        errors = []
        
        # 检查数据长度一致性
        lengths = [len(v) for v in self.snr_data.values()]
        if len(set(lengths)) != 1:
            errors.append("数据长度不一致")
        
        # 检查提升计算正确性
        for i in range(len(self.snr_data['SNR(dB)'])):
            calculated = round(self.snr_data['基线+GPR(%)'][i] - self.snr_data['基线(%)'][i], 2)
            recorded = self.snr_data['提升(%)'][i]
            if abs(calculated - recorded) > 0.01:
                errors.append(f"SNR {self.snr_data['SNR(dB)'][i]}dB 提升计算错误")
        
        # 检查准确率范围
        for key in ['基线(%)', '基线+GPR(%)']:
            values = self.snr_data[key]
            if any(v < 0 or v > 100 for v in values):
                errors.append(f"{key} 数值超出有效范围")
        
        if errors:
            print("数据验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("数据验证通过")
            return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SNR性能计算器')
    parser.add_argument('--export-csv', type=str, help='导出CSV文件路径')
    parser.add_argument('--save-plot', type=str, help='保存图表路径')
    parser.add_argument('--latex', action='store_true', help='生成LaTeX表格代码')
    parser.add_argument('--stats', action='store_true', help='显示统计信息')
    parser.add_argument('--validate', action='store_true', help='验证数据')
    
    args = parser.parse_args()
    
    calculator = SNRPerformanceCalculator()
    
    # 验证数据
    if args.validate or not any(vars(args).values()):
        calculator.validate_data()
    
    # 生成LaTeX表格
    if args.latex or not any(vars(args).values()):
        print("\nLaTeX表格代码:")
        print("=" * 50)
        print(calculator.generate_latex_table())
    
    # 显示统计信息
    if args.stats or not any(vars(args).values()):
        print("\n统计信息:")
        print("=" * 50)
        stats = calculator.calculate_statistics()
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        
        print("\nSNR范围分析:")
        print("-" * 30)
        range_analysis = calculator.analyze_snr_ranges()
        for range_name, data in range_analysis.items():
            print(f"\n{range_name}:")
            for metric, value in data.items():
                print(f"  {metric}: {value:.2f}")
    
    # 导出CSV
    if args.export_csv:
        calculator.export_to_csv(args.export_csv)
    
    # 绘制图表
    if args.save_plot or not any(vars(args).values()):
        calculator.plot_performance_curves(args.save_plot)

if __name__ == "__main__":
    main()
