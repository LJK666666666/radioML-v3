"""
主运行脚本 - 生成所有验证准确率收敛曲线图
Main script to generate all validation accuracy convergence curve plots
"""

import os
import sys
import subprocess

def run_script(script_name, description):
    """运行指定的Python脚本"""
    print(f"\n{'='*60}")
    print(f"正在运行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")
    
    try:
        # 使用subprocess运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("✓ 脚本运行成功!")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print("✗ 脚本运行失败!")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
            
    except Exception as e:
        print(f"✗ 运行脚本时出现异常: {e}")

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'pandas', 'matplotlib', 'numpy', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - 未安装")
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_files():
    """检查数据文件是否存在"""
    log_dir = "../../output/models/logs"
      # 模型配置，与其他脚本保持一致
    models = [
        'cnn2d',
        'cnn1d', 
        'transformer',
        'resnet',
        'complex_nn',
        'lightweight_hybrid',
        'lightweight_hybrid_model_gpr_augment'
    ]
    
    required_files = []
    for model_key in models:
        # 检查model_key是否已经包含'_model'，避免重复添加
        if model_key.endswith('_model_gpr_augment'):
            filename = f"{model_key}_detailed_log.csv"
        else:
            filename = f"{model_key}_model_detailed_log.csv"
        required_files.append(filename)
    
    print(f"\n检查数据文件 (目录: {log_dir}):")
    
    if not os.path.exists(log_dir):
        print(f"✗ 日志目录不存在: {log_dir}")
        return False
    
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(log_dir, file_name)
        if os.path.exists(file_path):
            print(f"✓ {file_name}")
        else:
            missing_files.append(file_name)
            print(f"✗ {file_name} - 文件不存在")
    
    if missing_files:
        print(f"\n警告: {len(missing_files)} 个文件缺失，将跳过对应模型")
    
    return len(missing_files) < len(required_files)  # 至少要有一些文件存在

def main():
    """主函数"""
    print("验证准确率收敛曲线绘制工具")
    print("Validation Accuracy Convergence Curve Plotting Tool")
    print("=" * 60)
    
    # 检查依赖
    print("\n1. 检查Python依赖包...")
    if not check_dependencies():
        print("请先安装缺失的依赖包后再运行此脚本")
        return
    
    # 检查数据文件
    print("\n2. 检查训练日志文件...")
    if not check_data_files():
        print("没有找到足够的训练日志文件，请确保模型训练已完成")
        return
      # 创建输出目录
    output_dirs = [
        "./figure",
        "./figure/individual_curves", 
        "./figure/academic_figures"
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 创建输出目录: {dir_path}")
    
    print("\n3. 开始生成图表...")
    
    # 运行各个脚本
    scripts = [
        ("quick_comparison.py", "快速对比图 (Quick Comparison)"),
        ("plot_validation_accuracy_curves.py", "详细分析图 (Detailed Analysis)"),
        ("academic_plot.py", "学术论文图 (Academic Quality)")
    ]
    
    success_count = 0
    
    for script_name, description in scripts:
        if os.path.exists(script_name):
            run_script(script_name, description)
            success_count += 1
        else:
            print(f"\n✗ 脚本文件不存在: {script_name}")
      # 总结
    print(f"\n{'='*60}")
    print(f"任务完成! 成功运行 {success_count}/{len(scripts)} 个脚本")
    print(f"{'='*60}")
    
    print("\n生成的文件:")
    print("📊 快速对比图: ./figure/quick_validation_comparison.png")
    print("📈 详细分析图: ./figure/validation_accuracy_curves.png/pdf")
    print("📋 个别模型图: ./figure/individual_curves/")
    print("🎓 学术论文图: ./figure/academic_figures/")
    
    print("\n🎉 所有验证准确率收敛曲线已生成完毕!")

if __name__ == "__main__":
    main()
