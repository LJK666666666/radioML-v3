#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的SNR表格验证脚本
验证论文中GPR去噪表格数据的正确性

Created on June 4, 2025
"""

def validate_gpr_table():
    """验证GPR去噪表格数据"""
    
    # 论文表格数据
    snr_data = [
        (-20, 8.93, 9.96, 1.03),
        (-18, 8.68, 10.22, 1.54),
        (-16, 9.85, 12.69, 2.84),
        (-14, 11.08, 17.32, 6.24),
        (-12, 12.65, 24.18, 11.53),
        (-10, 20.15, 35.05, 14.90),
        (-8, 34.66, 47.36, 12.70),
        (-6, 54.86, 61.21, 6.35),
        (-4, 64.02, 70.84, 6.82),
        (-2, 75.66, 80.89, 5.23),
        (0, 79.43, 83.17, 3.74),
        (2, 82.96, 87.07, 4.11),
        (4, 84.56, 89.00, 4.44),
        (6, 83.93, 89.38, 5.45),
        (8, 83.17, 89.10, 5.93),
        (10, 84.73, 89.85, 5.12),
        (12, 85.81, 90.31, 4.50),
        (14, 85.31, 88.81, 3.50),
        (16, 82.25, 88.15, 5.90),
        (18, 83.87, 88.98, 5.11)
    ]
    
    print("GPR去噪表格数据验证")
    print("=" * 60)
    print("SNR(dB) | 基线(%) | 基线+GPR(%) | 提升(%) | 计算提升 | 误差")
    print("-" * 60)
    
    errors = []
    total_improvement = 0
    
    for snr, baseline, gpr, improvement in snr_data:
        calculated_improvement = round(gpr - baseline, 2)
        error = abs(calculated_improvement - improvement)
        
        status = "✓" if error < 0.01 else "✗"
        
        print(f"{snr:6d} | {baseline:6.2f} | {gpr:10.2f} | {improvement:6.2f} | "
              f"{calculated_improvement:8.2f} | {error:5.2f} {status}")
        
        if error >= 0.01:
            errors.append(f"SNR {snr}dB: 期望 {improvement}, 计算 {calculated_improvement}")
        
        total_improvement += improvement
    
    print("-" * 60)
    print(f"总提升: {total_improvement:.2f}%")
    print(f"平均提升: {total_improvement/len(snr_data):.2f}%")
    
    if errors:
        print(f"\n发现 {len(errors)} 个错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ 所有数据验证通过!")
    
    # 分析不同SNR范围
    print("\n" + "=" * 60)
    print("SNR范围分析")
    print("=" * 60)
    
    ranges = {
        "低SNR (-20dB到-8dB)": [d for d in snr_data if d[0] <= -8],
        "中SNR (-6dB到4dB)": [d for d in snr_data if -6 <= d[0] <= 4],
        "高SNR (6dB到18dB)": [d for d in snr_data if d[0] >= 6]
    }
    
    for range_name, range_data in ranges.items():
        avg_baseline = sum(d[1] for d in range_data) / len(range_data)
        avg_gpr = sum(d[2] for d in range_data) / len(range_data)
        avg_improvement = sum(d[3] for d in range_data) / len(range_data)
        max_improvement = max(d[3] for d in range_data)
        
        print(f"\n{range_name}:")
        print(f"  平均基线准确率: {avg_baseline:.2f}%")
        print(f"  平均GPR准确率: {avg_gpr:.2f}%")
        print(f"  平均提升: {avg_improvement:.2f}%")
        print(f"  最大提升: {max_improvement:.2f}%")
        print(f"  样本数: {len(range_data)}")

def generate_latex_table():
    """生成更新后的LaTeX表格"""
    
    print("\n" + "=" * 60)
    print("LaTeX表格代码")
    print("=" * 60)
    
    latex_code = """\\begin{table}[h]
\\centering
\\caption{GPR去噪在各SNR水平下的详细影响}
\\label{tab:gpr_detailed_snr}
\\begin{tabular}{@{}lccc@{}}
\\toprule
SNR(dB) & 基线(\\%) & 基线+GPR(\\%) & 提升(\\%) \\\\
\\midrule
-20 & 8.93 & 9.96 & +1.03 \\\\
-18 & 8.68 & 10.22 & +1.54 \\\\
-16 & 9.85 & 12.69 & +2.84 \\\\
-14 & 11.08 & 17.32 & +6.24 \\\\
-12 & 12.65 & 24.18 & +11.53 \\\\
-10 & 20.15 & 35.05 & +14.90 \\\\
-8 & 34.66 & 47.36 & +12.70 \\\\
-6 & 54.86 & 61.21 & +6.35 \\\\
-4 & 64.02 & 70.84 & +6.82 \\\\
-2 & 75.66 & 80.89 & +5.23 \\\\
0 & 79.43 & 83.17 & +3.74 \\\\
2 & 82.96 & 87.07 & +4.11 \\\\
4 & 84.56 & 89.00 & +4.44 \\\\
6 & 83.93 & 89.38 & +5.45 \\\\
8 & 83.17 & 89.10 & +5.93 \\\\
10 & 84.73 & 89.85 & +5.12 \\\\
12 & 85.81 & 90.31 & +4.50 \\\\
14 & 85.31 & 88.81 & +3.50 \\\\
16 & 82.25 & 88.15 & +5.90 \\\\
18 & 83.87 & 88.98 & +5.11 \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""
    
    print(latex_code)

if __name__ == "__main__":
    validate_gpr_table()
    generate_latex_table()
