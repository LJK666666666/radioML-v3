Write-Host "PDF to PowerPoint Converter - 依赖安装脚本" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

# 检查Python
Write-Host "`n1. 检查Python安装..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>$null
    Write-Host "   ✅ 发现: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python未安装或未添加到PATH" -ForegroundColor Red
    Write-Host "   请从 https://www.python.org/ 下载并安装Python" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}

# 检查pip
Write-Host "`n2. 检查pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>$null
    Write-Host "   ✅ pip已可用" -ForegroundColor Green
} catch {
    Write-Host "   ❌ pip未找到" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}

# 安装Python包
Write-Host "`n3. 安装Python依赖包..." -ForegroundColor Yellow
Write-Host "   这可能需要几分钟时间..." -ForegroundColor Cyan

$packages = @("pdf2image>=3.1.0", "python-pptx>=0.6.21", "Pillow>=9.0.0")
$allSuccess = $true

foreach ($package in $packages) {
    Write-Host "   正在安装 $package..." -ForegroundColor Cyan
    $result = pip install $package 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ $package 安装成功" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $package 安装失败" -ForegroundColor Red
        Write-Host "   错误信息: $result" -ForegroundColor Red
        $allSuccess = $false
    }
}

# 检查Poppler
Write-Host "`n4. 检查Poppler..." -ForegroundColor Yellow
try {
    $null = Get-Command pdftoppm -ErrorAction Stop
    Write-Host "   ✅ Poppler已安装并可用" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Poppler未找到，这是Windows上必需的组件" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   请选择安装方法:" -ForegroundColor Cyan
    Write-Host "   1. 自动安装 (推荐，需要conda)" -ForegroundColor Cyan
    Write-Host "   2. 手动安装说明" -ForegroundColor Cyan
    Write-Host "   3. 跳过 (稍后手动安装)" -ForegroundColor Cyan
    
    $choice = Read-Host "   请输入选择 (1-3)"
    
    switch ($choice) {
        "1" {
            Write-Host "   尝试使用conda安装poppler..." -ForegroundColor Cyan
            try {
                conda install -c conda-forge poppler -y
                Write-Host "   ✅ Poppler安装成功" -ForegroundColor Green
            } catch {
                Write-Host "   ❌ conda未找到，请使用手动安装" -ForegroundColor Red
                $allSuccess = $false
            }
        }
        "2" {
            Write-Host ""
            Write-Host "   手动安装Poppler步骤:" -ForegroundColor Yellow
            Write-Host "   1. 访问: https://github.com/oschwartz10612/poppler-windows/releases" -ForegroundColor Cyan
            Write-Host "   2. 下载最新版本的zip文件" -ForegroundColor Cyan
            Write-Host "   3. 解压到 C:\poppler" -ForegroundColor Cyan
            Write-Host "   4. 将 C:\poppler\Library\bin 添加到系统PATH环境变量" -ForegroundColor Cyan
            Write-Host "   5. 重启命令提示符/PowerShell" -ForegroundColor Cyan
            $allSuccess = $false
        }
        "3" {
            Write-Host "   已跳过Poppler安装" -ForegroundColor Yellow
            $allSuccess = $false
        }
        default {
            Write-Host "   无效选择，已跳过" -ForegroundColor Yellow
            $allSuccess = $false
        }
    }
}

# 最终状态检查
Write-Host "`n5. 安装完成报告" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

if ($allSuccess) {
    Write-Host "   ✅ 所有依赖安装成功！" -ForegroundColor Green
    Write-Host "   🚀 现在可以使用转换器了" -ForegroundColor Green
    
    # 测试转换
    if (Test-Path "presentation.pdf") {
        Write-Host "`n6. 发现presentation.pdf文件" -ForegroundColor Yellow
        $runTest = Read-Host "   是否现在测试转换? (y/n)"
        if ($runTest -eq "y" -or $runTest -eq "Y") {
            Write-Host "   正在测试转换..." -ForegroundColor Cyan
            python pdf_to_ppt.py
        }
    } else {
        Write-Host "`n📝 使用说明:" -ForegroundColor Cyan
        Write-Host "   1. 将PDF文件重命名为 'presentation.pdf' 并放在此目录" -ForegroundColor White
        Write-Host "   2. 运行: python pdf_to_ppt.py" -ForegroundColor White
        Write-Host "   或运行: python advanced_pdf_to_ppt.py your_file.pdf" -ForegroundColor White
    }
} else {
    Write-Host "   ⚠️  安装过程中遇到一些问题" -ForegroundColor Yellow
    Write-Host "   请查看上面的错误信息并手动解决" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   常见解决方案:" -ForegroundColor Cyan
    Write-Host "   - 确保Python和pip已正确安装" -ForegroundColor White
    Write-Host "   - 以管理员身份运行此脚本" -ForegroundColor White
    Write-Host "   - 检查网络连接" -ForegroundColor White
    Write-Host "   - 手动安装Poppler (见上面的说明)" -ForegroundColor White
}

Write-Host ""
Read-Host "按Enter键退出"
