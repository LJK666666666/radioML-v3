@echo off
echo PDF to PowerPoint Converter
echo ========================

REM 检查Python是否已安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python。请先安装Python。
    pause
    exit /b 1
)

REM 安装依赖包
echo 正在安装所需的Python包...
pip install -r requirements.txt

REM 运行转换脚本
echo.
echo 开始转换PDF文件...
python pdf_to_ppt.py

echo.
echo 按任意键退出...
pause
