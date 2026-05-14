"""Plot SNR robustness curve: IQFormer accuracy vs SNR estimation error.
Includes both original (exact-SNR trained) and finetuned (error-aware) results."""
import matplotlib.pyplot as plt
import numpy as np
import os

# Original weights (trained on exact-SNR GPR data)
sigma_orig = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10])
acc_orig = np.array([67.48, 66.65, 65.25, 63.94, 62.80, 61.98, 61.40, 59.95, 59.29])

# Finetuned weights (finetuned on exact+error merged data, sigma_err=3)
acc_finetuned = np.array([66.94, 66.56, 65.77, 65.06, 64.38, 63.81, 63.22, 62.41, 61.85])

baseline_no_gpr = 63.08  # IQFormer without GPR denoising

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sigma_orig, acc_orig, 'o-', color='#1f77b4', linewidth=2, markersize=7,
        label='GPR + IQFormer (original)')
ax.plot(sigma_orig, acc_finetuned, 's-', color='#2ca02c', linewidth=2, markersize=7,
        label='GPR + IQFormer (error-aware finetuned)')
ax.axhline(y=baseline_no_gpr, color='#d62728', linestyle='--', linewidth=1.5,
           label=f'IQFormer baseline (no GPR): {baseline_no_gpr}%')

# Fill the improvement region between original and finetuned
ax.fill_between(sigma_orig, acc_orig, acc_finetuned, alpha=0.15, color='#2ca02c')

ax.set_xlabel(r'SNR Estimation Error $\sigma_{\mathrm{err}}$ (dB)', fontsize=12)
ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
ax.set_xticks(sigma_orig)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(58, 69)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# Annotate crossover points
# Original crosses baseline at ~4 dB
ax.annotate(r'$\sigma_{\mathrm{err}} \approx 4$ dB',
            xy=(4, baseline_no_gpr), xytext=(5.5, 60.5),
            fontsize=9, color='#1f77b4',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2))

# Finetuned crosses baseline at ~6 dB
ax.annotate(r'$\sigma_{\mathrm{err}} \approx 6$ dB',
            xy=(6, baseline_no_gpr), xytext=(7.5, 60.0),
            fontsize=9, color='#2ca02c',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), '..', 'paper', 'CL', 'double3',
                        'figure', 'snr_robustness.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved to: {out_path}')
plt.close()
