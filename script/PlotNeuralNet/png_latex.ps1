# PowerShell script to convert LaTeX to PNG
# Usage: .\png_latex.ps1 filename
# Example: .\png_latex.ps1 lightweight_hybrid_advanced

param(
    [Parameter(Mandatory=$true)]
    [string]$FileName
)

# Check if the .tex file exists
if (-not (Test-Path "${FileName}.tex")) {
    Write-Error "Error: ${FileName}.tex not found!"
    exit 1
}

Write-Host "Converting ${FileName}.tex to PDF..." -ForegroundColor Green

# Compile LaTeX to PDF
try {
    pdflatex "${FileName}.tex"
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex failed"
    }
} catch {
    Write-Error "Error: Failed to compile LaTeX. Make sure pdflatex is installed and in your PATH."
    Write-Host "You can install MiKTeX from: https://miktex.org/download" -ForegroundColor Yellow
    exit 1
}

# Check if PDF was created successfully
if (-not (Test-Path "${FileName}.pdf")) {
    Write-Error "Error: Failed to generate PDF from LaTeX"
    exit 1
}

Write-Host "PDF generated successfully: ${FileName}.pdf" -ForegroundColor Green

# Convert PDF to PNG (high resolution)
Write-Host "Converting PDF to PNG..." -ForegroundColor Green

# Try different methods to convert PDF to PNG
$converted = $false

# Try Python pdf2image (most reliable)
if (Get-Command python -ErrorAction SilentlyContinue) {
    try {
        python pdf_to_png.py "${FileName}.pdf"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "PNG generated using Python pdf2image: ${FileName}.png" -ForegroundColor Green
            $converted = $true
        }
    } catch {
        Write-Warning "Python pdf2image failed, trying other methods..."
    }
}

# Try ImageMagick convert
if (-not $converted -and (Get-Command convert -ErrorAction SilentlyContinue)) {
    try {
        convert -density 300 "${FileName}.pdf" -quality 90 "${FileName}.png"
        Write-Host "PNG generated using ImageMagick: ${FileName}.png" -ForegroundColor Green
        $converted = $true
    } catch {
        Write-Warning "ImageMagick convert failed, trying other methods..."
    }
}

# Try ImageMagick magick (newer versions)
if (-not $converted -and (Get-Command magick -ErrorAction SilentlyContinue)) {
    try {
        magick convert -density 300 "${FileName}.pdf" -quality 90 "${FileName}.png"
        Write-Host "PNG generated using ImageMagick: ${FileName}.png" -ForegroundColor Green
        $converted = $true
    } catch {
        Write-Warning "ImageMagick magick failed, trying other methods..."
    }
}

# Try pdftoppm
if (-not $converted -and (Get-Command pdftoppm -ErrorAction SilentlyContinue)) {
    try {
        pdftoppm -png -r 300 "${FileName}.pdf" "${FileName}"
        # pdftoppm adds page numbers, rename if needed
        if (Test-Path "${FileName}-1.png") {
            Move-Item "${FileName}-1.png" "${FileName}.png"
        }
        Write-Host "PNG generated using pdftoppm: ${FileName}.png" -ForegroundColor Green
        $converted = $true
    } catch {
        Write-Warning "pdftoppm failed"
    }
}

if (-not $converted) {
    Write-Warning "No PDF to PNG converter found!"
    Write-Host "Please install Python and run: pip install pdf2image" -ForegroundColor Yellow
    Write-Host "Or install one of the following:" -ForegroundColor Yellow
    Write-Host "  - ImageMagick: https://imagemagick.org/script/download.php#windows" -ForegroundColor Yellow
    Write-Host "  - poppler-utils: https://blog.alivate.com.au/poppler-windows/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "For now, you can manually convert ${FileName}.pdf to PNG using online tools" -ForegroundColor Yellow
    Write-Host "or other software."
}

# Clean up auxiliary files
Write-Host "Cleaning up auxiliary files..." -ForegroundColor Green
Remove-Item -Path "*.aux", "*.log", "*.fdb_latexmk", "*.fls", "*.synctex.gz" -ErrorAction SilentlyContinue

Write-Host "Done!" -ForegroundColor Green
