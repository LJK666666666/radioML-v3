#!/bin/bash

# Script to convert LaTeX to PNG using pdflatex and convert
# Usage: bash png_latex.sh filename (without .tex extension)

if [ $# -eq 0 ]; then
    echo "Usage: bash png_latex.sh filename"
    echo "Example: bash png_latex.sh lightweight_hybrid_advanced"
    exit 1
fi

FILENAME=$1

# Check if the .tex file exists
if [ ! -f "${FILENAME}.tex" ]; then
    echo "Error: ${FILENAME}.tex not found!"
    exit 1
fi

echo "Converting ${FILENAME}.tex to PDF..."

# Compile LaTeX to PDF
pdflatex "${FILENAME}.tex"

# Check if PDF was created successfully
if [ ! -f "${FILENAME}.pdf" ]; then
    echo "Error: Failed to generate PDF from LaTeX"
    exit 1
fi

echo "PDF generated successfully: ${FILENAME}.pdf"

# Convert PDF to PNG (high resolution)
echo "Converting PDF to PNG..."

# Try different methods to convert PDF to PNG
if command -v convert >/dev/null 2>&1; then
    # ImageMagick convert
    convert -density 300 "${FILENAME}.pdf" -quality 90 "${FILENAME}.png"
    echo "PNG generated using ImageMagick: ${FILENAME}.png"
elif command -v magick >/dev/null 2>&1; then
    # ImageMagick magick (newer versions)
    magick convert -density 300 "${FILENAME}.pdf" -quality 90 "${FILENAME}.png"
    echo "PNG generated using ImageMagick: ${FILENAME}.png"
elif command -v pdftoppm >/dev/null 2>&1; then
    # poppler-utils pdftoppm
    pdftoppm -png -r 300 "${FILENAME}.pdf" "${FILENAME}"
    # pdftoppm adds page numbers, rename if needed
    if [ -f "${FILENAME}-1.png" ]; then
        mv "${FILENAME}-1.png" "${FILENAME}.png"
    fi
    echo "PNG generated using pdftoppm: ${FILENAME}.png"
else
    echo "Warning: No PDF to PNG converter found!"
    echo "Please install one of the following:"
    echo "  - ImageMagick: https://imagemagick.org/script/download.php#windows"
    echo "  - poppler-utils: https://blog.alivate.com.au/poppler-windows/"
    echo ""
    echo "For now, you can manually convert ${FILENAME}.pdf to PNG using online tools"
    echo "or other software."
fi

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.fdb_latexmk *.fls *.synctex.gz

echo "Done!"
