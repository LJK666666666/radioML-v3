"""Generate IQFormer SNR accuracy comparison figure (before/after GPR denoising)."""
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

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_base['SNR'], df_base['Accuracy'], marker='o', label='IQFormer',
        linewidth=2, markersize=6)
ax.plot(df_gpr['SNR'], df_gpr['Accuracy'], marker='s', label='IQFormer+GPR',
        linewidth=2, markersize=6)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(np.arange(-20, 20, 2))

# Inset for medium SNR range
ax_inset = fig.add_axes([0.6, 0.15, 0.2, 0.6])
for df, marker in [(df_base, 'o'), (df_gpr, 's')]:
    df_zoom = df[(df['SNR'] >= -14) & (df['SNR'] <= 0)]
    ax_inset.plot(df_zoom['SNR'], df_zoom['Accuracy'], marker=marker,
                  linewidth=1.5, markersize=4)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.3)
ax_inset.set_xlim(-14, 0)
ax_inset.set_xticks(np.arange(-14, 2, 2))

plt.tight_layout()

# Save to paper figure directory
output_path = os.path.join(base_dir, 'paper', 'CL', 'double3', 'figure', 'snr_accuracy', 'iqformer_snr_accuracy.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()
