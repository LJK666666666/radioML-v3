# PDF to PowerPoint Converter

这是一个将PDF文件转换为PowerPoint演示文稿的Python脚本。脚本会将PDF的每一页转换为图片，然后插入到PPT的对应幻灯片中。

## 功能特点

- 🔄 将PDF的每一页转换为PPT中的一张幻灯片
- 📐 自动调整图片尺寸以适应幻灯片（保持宽高比）
- 🎯 图片自动居中显示
- 🔧 可自定义输出文件名和图片分辨率
- 💻 支持命令行参数和默认文件处理

## 安装要求

### 系统要求
- Python 3.6 或更高版本
- Windows 操作系统（需要安装poppler）

### Python依赖包
```
pdf2image>=3.1.0
python-pptx>=0.6.21
Pillow>=9.0.0
```

### 安装Poppler（重要！）

在Windows上，`pdf2image`需要poppler工具。请按以下步骤安装：

1. 下载Windows版本的poppler：
   - 访问：https://github.com/oschwartz10612/poppler-windows/releases
   - 下载最新版本的zip文件

2. 解压到一个目录（例如：`C:\poppler`）

3. 将poppler的bin目录添加到系统PATH环境变量中：
   - 例如：`C:\poppler\Library\bin`

4. 或者，在脚本中指定poppler路径（修改脚本中的convert_from_path调用）

## 使用方法

### 方法一：使用批处理文件（推荐）

1. 将PDF文件重命名为 `presentation.pdf` 并放在脚本目录下
2. 双击运行 `run_converter.bat`
3. 脚本会自动安装依赖并进行转换

### 方法二：命令行使用

```bash
# 安装依赖
pip install -r requirements.txt

# 转换默认文件（presentation.pdf）
python pdf_to_ppt.py

# 转换指定文件
python pdf_to_ppt.py "路径/到/你的文件.pdf"
```

### 方法三：在代码中使用

```python
from pdf_to_ppt import pdf_to_ppt

# 基本用法
pdf_to_ppt("presentation.pdf")

# 指定输出文件名和分辨率
pdf_to_ppt("input.pdf", "output.pptx", dpi=300)
```

## 参数说明

- `pdf_path`: PDF文件路径
- `output_path`: 输出PPT文件路径（可选，默认为PDF文件名+.pptx）
- `dpi`: 图片分辨率（可选，默认200，数值越高质量越好但文件越大）

## 输出结果

- 生成的PPT文件包含与PDF相同数量的幻灯片
- 每张幻灯片包含对应PDF页面的图片
- 图片自动调整大小以适应幻灯片，保持原始宽高比
- 图片在幻灯片中居中显示

## 注意事项

1. **Poppler安装**：这是最常见的问题，确保正确安装并配置poppler
2. **文件路径**：确保PDF文件路径正确，支持中文路径
3. **文件大小**：高DPI设置会产生更大的PPT文件
4. **内存使用**：处理大型PDF时可能需要较多内存

## 故障排除

### 常见错误及解决方法

1. **"poppler not found"错误**
   - 确保已正确安装poppler并添加到PATH

2. **"PDF文件不存在"错误**
   - 检查文件路径是否正确
   - 确保文件名拼写正确

3. **内存不足错误**
   - 降低DPI设置
   - 分批处理大型PDF文件

4. **权限错误**
   - 确保对输出目录有写入权限
   - 尝试以管理员身份运行

## 文件说明

- `pdf_to_ppt.py`: 主转换脚本
- `requirements.txt`: Python依赖包列表
- `run_converter.bat`: Windows批处理文件，一键运行转换
- `README.md`: 本说明文档

## 示例

假设你有一个名为`presentation.pdf`的文件：

```bash
# 使用默认设置转换
python pdf_to_ppt.py

# 输出：presentation.pptx
```

转换后的PPT将包含PDF的所有页面，每页作为一张幻灯片中的图片。

## 许可证

本脚本供学习和个人使用，请遵守相关软件的许可协议。
