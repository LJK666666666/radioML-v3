import os
import re
import csv
from pathlib import Path

def extract_snr_accuracy(txt_file):
    """Extract SNR and accuracy data from evaluation_summary.txt"""
    snr_data = []

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Find the "ACCURACY BY SNR:" section
    in_snr_section = False
    for line in lines:
        if "ACCURACY BY SNR:" in line:
            in_snr_section = True
            continue

        if in_snr_section and line.strip() and not line.startswith('-'):
            # Match lines like "SNR -20 dB: 0.0936" or "SNR  0 dB: 0.8500"
            match = re.match(r'SNR\s+(-?\d+)\s+dB:\s+(\d+\.\d+)', line)
            if match:
                snr = int(match.group(1))
                accuracy = float(match.group(2))
                snr_data.append({'SNR': snr, 'Accuracy': accuracy})
            elif line.strip() and not line.startswith('SNR'):
                # End of SNR section
                break

    return snr_data

def convert_txt_to_csv(txt_file):
    """Convert evaluation_summary.txt to evaluation_summary.csv"""
    snr_data = extract_snr_accuracy(txt_file)

    if not snr_data:
        print(f"No SNR data found in {txt_file}")
        return

    # Create CSV file in the same directory
    csv_file = txt_file.replace('evaluation_summary.txt', 'evaluation_summary.csv')

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SNR', 'Accuracy'])
        writer.writeheader()
        writer.writerows(snr_data)

    print(f"Created: {csv_file}")

def main():
    # Find all evaluation_summary.txt files in subdirectories
    current_dir = Path('.')
    txt_files = list(current_dir.glob('*/evaluation_summary.txt'))

    if not txt_files:
        print("No evaluation_summary.txt files found in subdirectories")
        return

    print(f"Found {len(txt_files)} evaluation_summary.txt files")
    print("-" * 50)

    for txt_file in txt_files:
        convert_txt_to_csv(str(txt_file))

    print("-" * 50)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
