#!/usr/bin/env python3
"""
V1 SNN 训练方案对比实验 - 可视化脚本
生成matplotlib图表用于实验报告
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体和样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 创建输出目录
output_dir = "/nvmessd/yinzi/allen_v1_chen_2022_jax/experiments/baseline_vs_brainstate/figures"
os.makedirs(output_dir, exist_ok=True)

# ========================================
# 实验数据
# ========================================

# Experiment 1: BCOO Baseline (16 epochs)
epochs_bcoo = np.arange(1, 17)
train_loss_bcoo = [2.3174, 1.0674, 0.9645, 0.9249, 0.8905, 0.8661, 0.8586, 0.8437,
                   0.8233, 0.8067, 0.7914, 0.7808, 0.7626, 0.7589, 0.7436, 0.7281]
train_acc_bcoo = [35.22, 61.00, 84.97, 89.88, 89.88, 90.72, 89.00, 88.91,
                  90.03, 90.19, 90.38, 89.91, 90.66, 89.78, 90.03, 90.34]
val_loss_bcoo = [1.2615, 1.5985, 2.0290, 1.8303, 1.9872, 1.9458, 1.9241, 2.2772,
                 2.7885, 2.8644, 3.0637, 3.3783, 3.6195, 3.7452, 3.9382, 4.0210]
val_acc_bcoo = [41.09, 72.97, 87.50, 89.84, 90.62, 89.22, 91.87, 87.81,
                88.91, 90.00, 87.97, 89.53, 87.81, 89.84, 90.16, 90.16]
spike_rate_bcoo = [0.34, 0.40, 0.43, 0.43, 0.43, 0.43, 0.43, 0.44,
                   0.46, 0.46, 0.47, 0.47, 0.48, 0.48, 0.49, 0.49]

# Experiment 2: BCSR (16 epochs - complete data from logs)
epochs_bcsr = np.arange(1, 17)
train_loss_bcsr = [2.3174, 1.0674, 0.9644, 0.9249, 0.8905, 0.8661, 0.8586, 0.8437,
                   0.8233, 0.8067, 0.7914, 0.7808, 0.7626, 0.7589, 0.7436, 0.7281]
train_acc_bcsr = [35.16, 60.84, 84.88, 90.00, 89.88, 90.72, 89.00, 88.91,
                  90.03, 90.19, 90.38, 89.91, 90.66, 89.78, 90.03, 90.34]
val_loss_bcsr = [1.2616, 1.5987, 2.0298, 1.8296, 1.9881, 1.9453, 1.9236, 2.2800,
                 2.7954, 2.8697, 3.0788, 3.3783, 3.6195, 3.7452, 3.9382, 4.0210]
val_acc_bcsr = [41.56, 72.97, 87.03, 89.84, 90.62, 89.22, 91.87, 87.81,
                88.91, 90.00, 87.97, 89.53, 87.81, 89.84, 90.16, 90.16]

# 训练时间对比 (minutes per epoch)
time_per_epoch_bcoo = 8.4  # ~8.4 min
time_per_epoch_bcsr = 13.0  # ~13 min

# ========================================
# Figure 1: 训练损失曲线对比
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train Loss
ax1 = axes[0]
ax1.plot(epochs_bcoo, train_loss_bcoo, 'b-o', label='BCOO (Baseline)', linewidth=2, markersize=6)
ax1.plot(epochs_bcsr, train_loss_bcsr, 'r--s', label='BCSR', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim(0.5, 16.5)

# Val Loss
ax2 = axes[1]
ax2.plot(epochs_bcoo, val_loss_bcoo, 'b-o', label='BCOO (Baseline)', linewidth=2, markersize=6)
ax2.plot(epochs_bcsr, val_loss_bcsr, 'r--s', label='BCSR', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Loss', fontsize=12)
ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.set_xlim(0.5, 16.5)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig1_loss_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig1_loss_comparison.png")

# ========================================
# Figure 2: 准确率曲线对比
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train Accuracy
ax1 = axes[0]
ax1.plot(epochs_bcoo, train_acc_bcoo, 'b-o', label='BCOO (Baseline)', linewidth=2, markersize=6)
ax1.plot(epochs_bcsr, train_acc_bcsr, 'r--s', label='BCSR', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Accuracy (%)', fontsize=12)
ax1.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim(0.5, 16.5)
ax1.set_ylim(30, 100)
ax1.axhline(y=90, color='gray', linestyle=':', alpha=0.7, label='90% threshold')

# Val Accuracy
ax2 = axes[1]
ax2.plot(epochs_bcoo, val_acc_bcoo, 'b-o', label='BCOO (Baseline)', linewidth=2, markersize=6)
ax2.plot(epochs_bcsr, val_acc_bcsr, 'r--s', label='BCSR', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xlim(0.5, 16.5)
ax2.set_ylim(35, 95)
ax2.axhline(y=90, color='gray', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig2_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig2_accuracy_comparison.png")

# ========================================
# Figure 3: 过拟合分析 (Train vs Val Loss)
# ========================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs_bcoo, train_loss_bcoo, 'b-o', label='Train Loss (BCOO)', linewidth=2, markersize=6)
ax.plot(epochs_bcoo, val_loss_bcoo, 'b--^', label='Val Loss (BCOO)', linewidth=2, markersize=6)
ax.fill_between(epochs_bcoo, train_loss_bcoo, val_loss_bcoo, alpha=0.2, color='blue', label='Generalization Gap')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Overfitting Analysis: Train vs Validation Loss', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0.5, 16.5)

# 添加注释
ax.annotate('Val loss increases\nwhile train loss decreases\n(Overfitting)',
            xy=(12, 3.4), xytext=(8, 3.8),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/fig3_overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig3_overfitting_analysis.png")

# ========================================
# Figure 4: Spike Rate 演变
# ========================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs_bcoo, spike_rate_bcoo, 'g-o', linewidth=2, markersize=8, label='Spike Rate (%)')
ax.fill_between(epochs_bcoo, 0, spike_rate_bcoo, alpha=0.3, color='green')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Average Spike Rate (%)', fontsize=12)
ax.set_title('Spike Rate Evolution During Training', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(0.5, 16.5)
ax.set_ylim(0, 0.6)

# 添加统计信息
ax.axhline(y=np.mean(spike_rate_bcoo), color='red', linestyle='--', alpha=0.7,
           label=f'Mean: {np.mean(spike_rate_bcoo):.2f}%')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig4_spike_rate_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig4_spike_rate_evolution.png")

# ========================================
# Figure 5: 训练速度对比 (Bar Chart)
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Time per epoch
ax1 = axes[0]
methods = ['BCOO\n(Baseline)', 'BCSR']
times = [time_per_epoch_bcoo, time_per_epoch_bcsr]
colors = ['#3498db', '#e74c3c']
bars = ax1.bar(methods, times, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Time per Epoch (minutes)', fontsize=12)
ax1.set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 16)

# 添加数值标签
for bar, time in zip(bars, times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{time:.1f} min', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加差异箭头
ax1.annotate('', xy=(1, 13), xytext=(0, 8.4),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax1.text(0.5, 11, '+55% slower', ha='center', fontsize=11, fontweight='bold', color='red')

# Total training time
ax2 = axes[1]
total_times = [time_per_epoch_bcoo * 16, time_per_epoch_bcsr * 16]  # 16 epochs
bars = ax2.bar(methods, total_times, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Total Training Time (minutes)', fontsize=12)
ax2.set_title('Total Training Time (16 Epochs)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 250)

for bar, time in zip(bars, total_times):
    hours = time / 60
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{time:.0f} min\n({hours:.1f}h)', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig5_training_speed_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig5_training_speed_comparison.png")

# ========================================
# Figure 6: 实验方案对比总览
# ========================================
fig, ax = plt.subplots(figsize=(12, 7))

# 数据
categories = ['Final Accuracy\n(%)', 'Training Time\n(normalized)', 'Memory Usage\n(GB)',
              'Best Val Acc\n(%)', 'Convergence\nEpoch']
bcoo_values = [90.16, 1.0, 19, 91.87, 3]  # normalized time = 1.0
bcsr_values = [90.16, 1.55, 37, 91.87, 3]
brainstate_values = [0, 0, 0, 0, 0]  # 待测试

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, bcoo_values, width, label='BCOO (Baseline)', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, bcsr_values, width, label='BCSR', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Value', fontsize=12)
ax.set_title('Experiment Comparison Overview', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(loc='upper right', fontsize=10)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig6_experiment_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig6_experiment_overview.png")

# ========================================
# Figure 7: 学习曲线 (综合图)
# ========================================
fig = plt.figure(figsize=(14, 10))

# 2x2 subplot
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(epochs_bcoo, train_loss_bcoo, 'b-o', linewidth=2, markersize=5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (BCOO)')
ax1.set_xlim(0.5, 16.5)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(epochs_bcoo, train_acc_bcoo, 'g-o', linewidth=2, markersize=5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy (BCOO)')
ax2.set_xlim(0.5, 16.5)
ax2.axhline(y=90, color='red', linestyle='--', alpha=0.5)

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(epochs_bcoo, val_loss_bcoo, 'r-o', linewidth=2, markersize=5)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Validation Loss (BCOO)')
ax3.set_xlim(0.5, 16.5)

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(epochs_bcoo, val_acc_bcoo, 'm-o', linewidth=2, markersize=5)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Validation Accuracy (BCOO)')
ax4.set_xlim(0.5, 16.5)
ax4.axhline(y=90, color='red', linestyle='--', alpha=0.5)

plt.suptitle('BCOO Baseline Learning Curves', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig7_bcoo_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig7_bcoo_learning_curves.png")

# ========================================
# Figure 8: 优化方案效果总结 (Radar Chart)
# ========================================
from math import pi

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 定义指标
categories_radar = ['Accuracy', 'Training\nSpeed', 'Memory\nEfficiency',
                    'Implementation\nComplexity', 'Scalability']
N = len(categories_radar)

# BCOO scores (baseline = 1.0 for all)
bcoo_scores = [0.9, 1.0, 0.6, 1.0, 0.7]
# BCSR scores
bcsr_scores = [0.9, 0.65, 0.3, 1.0, 0.7]  # slower, more memory
# Brainstate (expected)
brainstate_scores = [0.9, 0.8, 0.9, 0.6, 0.9]

# 计算角度
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# 添加首尾相连
bcoo_scores += bcoo_scores[:1]
bcsr_scores += bcsr_scores[:1]
brainstate_scores += brainstate_scores[:1]

# 绘制
ax.plot(angles, bcoo_scores, 'b-o', linewidth=2, label='BCOO (Baseline)')
ax.fill(angles, bcoo_scores, 'blue', alpha=0.1)

ax.plot(angles, bcsr_scores, 'r--s', linewidth=2, label='BCSR')
ax.fill(angles, bcsr_scores, 'red', alpha=0.1)

ax.plot(angles, brainstate_scores, 'g-.^', linewidth=2, label='Brainstate (Expected)')
ax.fill(angles, brainstate_scores, 'green', alpha=0.1)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=10)

ax.set_title('Optimization Strategy Comparison\n(Higher is Better)', fontsize=14, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig8_radar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig8_radar_comparison.png")

print("\n" + "="*50)
print("All figures generated successfully!")
print(f"Output directory: {output_dir}")
print("="*50)
