#!/usr/bin/env python3
"""
PDF to PNG converter using pdf2image library
Usage: python pdf_to_png.py input.pdf [output_name]
"""

import sys
import os

def install_and_import():
    """Install pdf2image if not available"""
    try:
        from pdf2image import convert_from_path
        return convert_from_path
    except ImportError:
        print("pdf2image not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdf2image"])
        from pdf2image import convert_from_path
        return convert_from_path

def convert_pdf_to_png(pdf_path, output_name=None):
    """Convert PDF to PNG with high resolution"""
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
        return False
    
    try:
        convert_from_path = install_and_import()
        
        # Convert PDF to images with high DPI
        print(f"Converting {pdf_path} to PNG...")
        images = convert_from_path(pdf_path, dpi=300)
        
        if not images:
            print("No pages found in PDF!")
            return False
        
        # Save the first page (assuming single page PDF)
        if output_name is None:
            output_name = os.path.splitext(pdf_path)[0]
        
        output_path = f"{output_name}.png"
        images[0].save(output_path, 'PNG')
        
        print(f"Successfully converted to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_png.py input.pdf [output_name]")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_pdf_to_png(pdf_file, output_name)
    sys.exit(0 if success else 1)
