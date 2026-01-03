# PDF to PowerPoint Converter - 完整安装指南

## 📦 一键安装脚本

### Windows PowerShell 安装脚本

创建并运行以下PowerShell脚本来自动安装所有依赖：

```powershell
# install_dependencies.ps1
Write-Host "PDF to PowerPoint Converter - 依赖安装脚本" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

# 检查Python
Write-Host "`n1. 检查Python安装..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>$null
    Write-Host "   ✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python未安装或未添加到PATH" -ForegroundColor Red
    Write-Host "   请从 https://www.python.org/ 下载并安装Python" -ForegroundColor Red
    exit 1
}

# 检查pip
Write-Host "`n2. 检查pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>$null
    Write-Host "   ✅ pip已安装" -ForegroundColor Green
} catch {
    Write-Host "   ❌ pip未找到" -ForegroundColor Red
    exit 1
}

# 安装Python包
Write-Host "`n3. 安装Python依赖包..." -ForegroundColor Yellow
$packages = @("pdf2image", "python-pptx", "Pillow")
foreach ($package in $packages) {
    Write-Host "   安装 $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ $package 安装成功" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $package 安装失败" -ForegroundColor Red
    }
}

# 检查Poppler
Write-Host "`n4. 检查Poppler..." -ForegroundColor Yellow
try {
    $popplerTest = pdftoppm -h 2>$null
    Write-Host "   ✅ Poppler已安装" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Poppler未找到" -ForegroundColor Yellow
    Write-Host "   请按照以下步骤安装Poppler:" -ForegroundColor Yellow
    Write-Host "   1. 访问: https://github.com/oschwartz10612/poppler-windows/releases" -ForegroundColor Cyan
    Write-Host "   2. 下载最新版本的zip文件" -ForegroundColor Cyan
    Write-Host "   3. 解压到 C:\poppler" -ForegroundColor Cyan
    Write-Host "   4. 将 C:\poppler\Library\bin 添加到系统PATH" -ForegroundColor Cyan
}

Write-Host "`n5. 安装完成!" -ForegroundColor Green
Write-Host "   现在可以运行转换脚本了" -ForegroundColor Green

# 测试运行
Write-Host "`n6. 测试转换功能..." -ForegroundColor Yellow
if (Test-Path "presentation.pdf") {
    Write-Host "   发现presentation.pdf，正在测试转换..." -ForegroundColor Cyan
    python pdf_to_ppt.py
} else {
    Write-Host "   请将PDF文件重命名为 presentation.pdf 然后运行:" -ForegroundColor Cyan
    Write-Host "   python pdf_to_ppt.py" -ForegroundColor Cyan
}
```

## 🚀 快速开始

### 方法1: 使用简单版本

1. **安装依赖**:
```bash
pip install pdf2image python-pptx Pillow
```

2. **运行转换**:
```bash
# 转换当前目录的presentation.pdf
python pdf_to_ppt.py

# 转换指定文件
python pdf_to_ppt.py "your_file.pdf"
```

### 方法2: 使用高级版本

```bash
# 查看帮助
python advanced_pdf_to_ppt.py --help

# 转换单个文件
python advanced_pdf_to_ppt.py presentation.pdf

# 批量转换目录中的所有PDF
python advanced_pdf_to_ppt.py -b /path/to/pdf/directory

# 高质量转换
python advanced_pdf_to_ppt.py presentation.pdf -d 300 -o output.pptx
```

## 🔧 详细安装步骤

### 1. 安装Python
- 访问 https://www.python.org/downloads/
- 下载并安装Python 3.8或更高版本
- ⚠️ 安装时勾选"Add Python to PATH"

### 2. 安装Python包
```bash
pip install pdf2image python-pptx Pillow
```

### 3. 安装Poppler (Windows重要步骤)

#### 自动安装方法:
```bash
# 使用conda (推荐)
conda install -c conda-forge poppler

# 或使用choco
choco install poppler
```

#### 手动安装方法:
1. 下载: https://github.com/oschwartz10612/poppler-windows/releases
2. 解压到 `C:\poppler`
3. 添加 `C:\poppler\Library\bin` 到系统PATH

#### 验证安装:
```bash
pdftoppm -h
```

## 📝 使用示例

### 基本转换
```python
from pdf_to_ppt import pdf_to_ppt

# 基本转换
pdf_to_ppt("presentation.pdf")

# 指定输出文件和质量
pdf_to_ppt("input.pdf", "output.pptx", dpi=300)
```

### 高级功能
```python
from advanced_pdf_to_ppt import PDFToPPTConverter

# 创建转换器
converter = PDFToPPTConverter(dpi=250)

# 转换单个文件
converter.convert_single_pdf("file.pdf", "output.pptx")

# 批量转换
pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]
converter.convert_batch(pdf_files, output_dir="./output")
```

## 🐛 常见问题解决

### 1. "poppler not found" 错误
```bash
# 解决方案1: 指定poppler路径
python advanced_pdf_to_ppt.py presentation.pdf -p "C:\poppler\Library\bin"

# 解决方案2: 使用conda安装
conda install -c conda-forge poppler
```

### 2. 内存不足错误
```bash
# 降低分辨率
python advanced_pdf_to_ppt.py presentation.pdf -d 150
```

### 3. 权限错误
- 以管理员身份运行命令提示符
- 确保输出目录有写入权限

### 4. 中文路径问题
```python
# 在脚本开头添加
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

## 📊 性能优化建议

### DPI设置建议:
- **演示用**: 150-200 DPI (文件小，速度快)
- **打印用**: 250-300 DPI (质量好，文件大)
- **高质量**: 400+ DPI (最佳质量，文件很大)

### 批量处理:
```bash
# 处理整个目录
python advanced_pdf_to_ppt.py -b "./pdf_files" -o "./output" -d 200
```

## 🎯 高级配置

### 自定义Poppler路径
```python
converter = PDFToPPTConverter(
    dpi=200,
    poppler_path="C:/poppler/Library/bin"
)
```

### 进度监控
```python
def progress_callback(message):
    print(f"[进度] {message}")

converter.convert_single_pdf(
    "large_file.pdf",
    progress_callback=progress_callback
)
```

## 📁 项目结构
```
pdf to ppt/
├── pdf_to_ppt.py              # 简单版本转换器
├── advanced_pdf_to_ppt.py     # 高级版本转换器
├── requirements.txt           # Python依赖
├── run_converter.bat         # Windows批处理文件
├── README.md                 # 基本说明
├── INSTALL_GUIDE.md          # 本安装指南
└── presentation.pdf          # 示例PDF文件
```

现在你可以开始使用这个完整的PDF转PPT解决方案了！
