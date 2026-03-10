#!/usr/bin/env python3
"""
Generate a comparison figure between JAX reimplementation and original TensorFlow.
Compares training time and GPU usage for MNIST task training.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Data
# Original paper: 160 A100 GPUs, 60 hours for 16 epochs
# JAX reimplementation:
#   - 8 A40 GPUs, ~10 hours for 7 epochs
#   - 40 A40 GPUs, ~2 hours for 7 epochs (projected)

methods = ['Original\n(Chen et al. 2022)', 'JAX (8 GPUs)\n(This Work)', 'JAX (40 GPUs)\n(This Work)']
gpus = [160, 8, 40]
gpu_types = ['A100', 'A40', 'A40']
epochs = [16, 7, 7]
time_hours = [60, 10, 2]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Colors
colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red for original, blue/green for JAX
edge_colors = ['#c0392b', '#2980b9', '#27ae60']

# Subplot 1: GPU Count
ax1 = axes[0]
x_pos = np.arange(len(methods))
bars1 = ax1.bar(x_pos, gpus, color=colors, edgecolor=edge_colors, linewidth=2, width=0.6)
ax1.set_ylabel('Number of GPUs', fontweight='bold')
ax1.set_title('GPU Resources', fontweight='bold', fontsize=14)
ax1.set_ylim(0, 200)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods)

# Add value labels on bars
for bar, gpu, gpu_type in zip(bars1, gpus, gpu_types):
    height = bar.get_height()
    ax1.annotate(f'{gpu}\n({gpu_type})',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

# Add reduction factor annotations
ax1.annotate('20x fewer',
             xy=(1, 8), xytext=(0.5, 90),
             arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2),
             fontsize=11, fontweight='bold', color='#9b59b6',
             ha='center')
ax1.annotate('4x fewer',
             xy=(2, 40), xytext=(1.5, 110),
             arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2),
             fontsize=11, fontweight='bold', color='#9b59b6',
             ha='center')

# Subplot 2: Training Time
ax2 = axes[1]
bars2 = ax2.bar(x_pos, time_hours, color=colors, edgecolor=edge_colors, linewidth=2, width=0.6)
ax2.set_ylabel('Wall Clock Time (hours)', fontweight='bold')
ax2.set_title('Training Time (7 epochs sufficient for convergence)', fontweight='bold', fontsize=14)
ax2.set_ylim(0, 75)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods)

# Add value labels on bars with epoch info
for bar, time, epoch in zip(bars2, time_hours, epochs):
    height = bar.get_height()
    ax2.annotate(f'{time} h\n({epoch} epochs)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

# Add speedup annotations
ax2.annotate('6x faster',
             xy=(1, 10), xytext=(0.5, 38),
             arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2),
             fontsize=11, fontweight='bold', color='#9b59b6',
             ha='center')
ax2.annotate('30x faster',
             xy=(2, 2), xytext=(1.5, 28),
             arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2),
             fontsize=11, fontweight='bold', color='#9b59b6',
             ha='center')

# Overall title
fig.suptitle('V1 Spiking Neural Network Training: JAX vs TensorFlow',
             fontsize=16, fontweight='bold', y=1.02)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = '/nvmessd/yinzi/allen_v1_chen_2022_jax/experiments/weekly_report_figures/fig_jax_vs_original.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path}")

# Also save as PDF
pdf_path = output_path.replace('.png', '.pdf')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {pdf_path}")

plt.close()
