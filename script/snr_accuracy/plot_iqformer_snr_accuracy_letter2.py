"""Generate IQFormer SNR accuracy comparison figure with larger fonts for IEEE CL."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths (relative to project root)
base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
results_dir = os.path.join(base_dir, 'output', 'results')

# Read data
baseline_csv = os.path.join(results_dir, 'iqformer_evaluation_results_stratified', 'accuracy_by_snr.csv')
gpr_csv = os.path.join(results_dir, 'iqformer_evaluation_results_efficient_gpr_per_sample_stratified', 'accuracy_by_snr.csv')

df_base = pd.read_csv(baseline_csv)
df_gpr = pd.read_csv(gpr_csv)

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(df_base['SNR'], df_base['Accuracy'], marker='o', label='IQFormer',
        linewidth=2.5, markersize=8, color='#1f77b4')
ax.plot(df_gpr['SNR'], df_gpr['Accuracy'], marker='s', label='IQFormer+GPR',
        linewidth=2.5, markersize=8, color='#ff7f0e')

ax.set_xlabel('SNR (dB)', fontsize=18)
ax.set_ylabel('Accuracy', fontsize=18)
ax.legend(fontsize=15, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xticks(np.arange(-20, 20, 4))
ax.tick_params(labelsize=14)

plt.tight_layout()

# Save to Letter2 figure directory
output_path = os.path.join(base_dir, 'paper', 'CL', 'Letter2', 'figure', 'snr_accuracy', 'iqformer_snr_accuracy.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()
