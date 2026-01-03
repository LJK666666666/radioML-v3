#!/usr/bin/env python3
"""
Create a flowchart visualization for GPR denoising mathematical derivation process.
This script generates a clear flow diagram showing the step-by-step derivation
from I/Q power calculation to final noise estimation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_derivation_flowchart():
    """Create a flowchart showing the GPR denoising derivation process."""
    
    # Set up the figure with high DPI for presentation quality
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    step_color = '#E3F2FD'  # Light blue
    formula_color = '#FFF3E0'  # Light orange
    result_color = '#E8F5E8'  # Light green
    arrow_color = '#1976D2'  # Blue
    
    # Step 1: I/Q Power Calculation
    step1_box = FancyBboxPatch((2, 9.5), 6, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=step_color, 
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(step1_box)
    ax.text(5, 10.3, 'Step 1: Calculate Total Received Power from I/Q Channels', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 9.8, r'$P_r = \frac{1}{M}\sum_{k=0}^{M-1}(r_I[k]^2 + r_Q[k]^2)$', 
            ha='center', va='center', fontsize=10)
    
    # Arrow 1
    arrow1 = ConnectionPatch((5, 9.5), (5, 8.8), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=arrow_color, ec=arrow_color, lw=2)
    ax.add_patch(arrow1)
    
    # Step 2: SNR Relationship
    step2_box = FancyBboxPatch((2, 7.5), 6, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=step_color, 
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(step2_box)
    ax.text(5, 8.3, 'Step 2: Establish Signal-to-Noise Power Ratio', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 7.8, r'$\text{SNR}_{\text{linear}} = \frac{P_s}{P_w} = 10^{\text{SNR}_{\text{dB}}/10}$', 
            ha='center', va='center', fontsize=10)
    
    # Arrow 2
    arrow2 = ConnectionPatch((5, 7.5), (5, 6.8), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=arrow_color, ec=arrow_color, lw=2)
    ax.add_patch(arrow2)
    
    # Step 3: Power Combination
    step3_box = FancyBboxPatch((2, 5.5), 6, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=step_color, 
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(step3_box)
    ax.text(5, 6.3, 'Step 3: Combine Total Power and SNR to Find Noise Power', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 5.8, r'$P_w = \frac{P_r}{10^{\text{SNR}_{\text{dB}}/10} + 1}$', 
            ha='center', va='center', fontsize=10)
    
    # Arrow 3
    arrow3 = ConnectionPatch((5, 5.5), (5, 4.8), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=arrow_color, ec=arrow_color, lw=2)
    ax.add_patch(arrow3)
    
    # Step 4: Gaussian Distribution
    step4_box = FancyBboxPatch((2, 3.5), 6, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=step_color, 
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(step4_box)
    ax.text(5, 4.3, 'Step 4: Estimate Noise Standard Deviation from Gaussian Distribution', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 3.8, r'$\sigma_n^2 = \frac{P_w}{2}$ (for complex AWGN)', 
            ha='center', va='center', fontsize=10)
    
    # Arrow 4
    arrow4 = ConnectionPatch((5, 3.5), (5, 2.8), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=arrow_color, ec=arrow_color, lw=2)
    ax.add_patch(arrow4)
    
    # Final Result
    result_box = FancyBboxPatch((2, 1.7), 6, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=result_color, 
                                edgecolor='darkgreen', linewidth=2)
    ax.add_patch(result_box)
    ax.text(5, 2.3, 'Final Result: Theoretically Grounded Noise Estimation', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')
    ax.text(5, 1.9, r'$\sigma_n = \sqrt{\frac{P_r}{2(10^{\text{SNR}_{\text{dB}}/10} + 1)}}$', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Add side annotations
    ax.text(1.7, 10.1, '1', ha='center', va='center', fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', edgecolor='black'))
    ax.text(1.7, 8.1, '2', ha='center', va='center', fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', edgecolor='black'))
    ax.text(1.7, 6.1, '3', ha='center', va='center', fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', edgecolor='black'))
    ax.text(1.7, 4.1, '4', ha='center', va='center', fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', edgecolor='black'))
    
    # Title
    ax.text(5, 11.5, 'GPR Denoising: Mathematical Derivation Flow', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_power_visualization():
    """Create a visualization showing power relationships."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Power composition
    snr_db = -20
    snr_linear = 10**(snr_db/10)
    
    # Simulate power values
    P_total = 1.0  # Normalized
    P_noise = P_total / (snr_linear + 1)
    P_signal = P_total - P_noise
    
    # Pie chart
    sizes = [P_signal, P_noise]
    labels = [f'Signal Power\n{P_signal:.1%}', f'Noise Power\n{P_noise:.1%}']
    colors = ['#4CAF50', '#F44336']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Power Composition at SNR = {snr_db} dB', fontsize=12, fontweight='bold')
    
    # Right plot: SNR vs Noise Power
    snr_range = np.linspace(-20, 18, 100)
    snr_linear_range = 10**(snr_range/10)
    noise_power_ratio = 1 / (snr_linear_range + 1)
    
    ax2.plot(snr_range, noise_power_ratio, 'b-', linewidth=2, label='Noise Power Ratio')
    ax2.axvline(x=-20, color='r', linestyle='--', alpha=0.7, label='SNR = -20 dB')
    ax2.axhline(y=P_noise, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('SNR (dB)', fontsize=11)
    ax2.set_ylabel('Noise Power Ratio', fontsize=11)
    ax2.set_title('Noise Power vs SNR Relationship', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-20, 18)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create derivation flowchart
    fig1 = create_derivation_flowchart()
    fig1.savefig('../../presentation/figure/gpr_derivation_flow.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print("Created: gpr_derivation_flow.png")
    
    # Create power visualization
    fig2 = create_power_visualization()
    fig2.savefig('../../presentation/figure/power_analysis.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print("Created: power_analysis.png")
    
    plt.show()
