#!/usr/bin/env python3
"""Generate figures for weekly report 2026-03-08."""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (10, 6)

# Create output directory path
OUTPUT_DIR = "/nvmessd/yinzi/allen_v1_chen_2022_jax/experiments/weekly_report_figures"


def fig1_training_time_comparison():
    """Figure 1: Training time comparison across implementations."""
    implementations = ['TensorFlow\n(Original)', 'JAX + BCOO']
    times = [12, 2]  # hours
    colors = ['#e74c3c', '#27ae60']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(implementations, times, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{time}h',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

    # Add speedup annotations
    ax.annotate('6x speedup', xy=(1, 2), xytext=(1.25, 6),
                fontsize=12, color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

    ax.set_ylabel('Training Time (hours)')
    ax.set_title('Garrett Binary Classification Training Time\n(8x A40 GPU, 52K neurons, 600 timesteps)')
    ax.set_ylim(0, 14)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_training_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig1_training_time.pdf', bbox_inches='tight')
    print("Saved: fig1_training_time.png/pdf")
    plt.close()


def fig2_memory_comparison():
    """Figure 2: Memory usage comparison BPTT vs IODim."""
    methods = ['BPTT\n(Standard Backprop)', 'IODim\n(Online Eligibility Trace)']
    memory_mb = [8000, 16]  # MB
    colors = ['#e74c3c', '#27ae60']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, memory_mb, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, mem in zip(bars, memory_mb):
        height = bar.get_height()
        label = f'{mem/1000:.0f} GB' if mem >= 1000 else f'{mem} MB'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

    # Add reduction annotation
    ax.annotate('500x\nmemory reduction', xy=(1, 16), xytext=(1.3, 4000),
                fontsize=12, color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

    ax.set_ylabel('Activation Storage (MB)')
    ax.set_title('BPTT vs IODim Memory Footprint\n(52K neurons, 600 timesteps, batch=64)')
    ax.set_yscale('log')
    ax.set_ylim(1, 20000)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_memory_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig2_memory_comparison.pdf', bbox_inches='tight')
    print("Saved: fig2_memory_comparison.png/pdf")
    plt.close()


def fig3_iodim_training_curve():
    """Figure 3: IODim training loss curve."""
    epochs = [1, 2, 3, 4, 5]
    loss = [0.125106, 0.120277, 0.117505, 0.114832, 0.111703]
    grad_norm = [0.117, 0.136, 0.141, 0.142, 0.140]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, loss, 'o-', color='#3498db', linewidth=2.5, markersize=10)
    ax1.fill_between(epochs, loss, alpha=0.3, color='#3498db')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('IODim Training Loss')
    ax1.set_xticks(epochs)

    # Add percentage reduction
    reduction = (loss[0] - loss[-1]) / loss[0] * 100
    ax1.annotate(f'{reduction:.1f}% reduction',
                 xy=(5, loss[-1]), xytext=(4, 0.123),
                 fontsize=12, color='#27ae60',
                 arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

    # Gradient norm curve
    ax2.plot(epochs, grad_norm, 's-', color='#9b59b6', linewidth=2.5, markersize=10)
    ax2.fill_between(epochs, grad_norm, alpha=0.3, color='#9b59b6')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm Evolution')
    ax2.set_xticks(epochs)

    plt.suptitle('IODim Training Validation (52K neurons, 5 epochs)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_iodim_training.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig3_iodim_training.pdf', bbox_inches='tight')
    print("Saved: fig3_iodim_training.png/pdf")
    plt.close()


def fig4_bug_fix_impact():
    """Figure 4: Bug fix impact visualization."""
    metrics = ['_current_factor', 'Recurrent\ncontribution (%)', 'grad_norm']
    before = [0.00025, 0.21, 0.0008]
    after = [0.0053, 4.45, 0.14]
    improvement = [21, 21, 175]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, before, width, label='Before fix', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, after, width, label='After fix', color='#27ae60', edgecolor='black')

    ax.set_ylabel('Value (log scale)')
    ax.set_title('_current_factor Bug Fix Impact')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_yscale('log')

    # Add improvement annotations
    for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvement)):
        ax.annotate(f'{imp}x',
                    xy=(i, after[i]),
                    xytext=(i, after[i] * 2),
                    ha='center', fontsize=11, color='#27ae60', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_bug_fix_impact.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig4_bug_fix_impact.pdf', bbox_inches='tight')
    print("Saved: fig4_bug_fix_impact.png/pdf")
    plt.close()


def fig5_optimization_overview():
    """Figure 5: Overview of optimization attempts."""
    optimizations = ['BCOO format\n(JAX sparse)', 'IODim\nonline gradient', 'BCSR format\n(cuSPARSE)', 'Gradient\nCheckpoint', 'ZeRO-2\ndistributed', 'Mixed precision\n(bfloat16)']
    effectiveness = [6, 5, -1, 0, 0, -1]  # Effectiveness score
    colors = ['#27ae60', '#27ae60', '#e74c3c', '#f39c12', '#f39c12', '#e74c3c']

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(optimizations, effectiveness, color=colors, edgecolor='black', linewidth=1.5)

    # Add labels
    labels = ['6x speedup', '500x mem reduction', '55% slower\n(not recommended)', 'Ineffective\n(wrong bottleneck)', 'Ineffective\n(too few params)', 'Failed\n(numerical instability)']
    for bar, label in zip(bars, labels):
        width = bar.get_width()
        x_pos = width + 0.2 if width >= 0 else width - 0.2
        ha = 'left' if width >= 0 else 'right'
        ax.annotate(label,
                    xy=(x_pos, bar.get_y() + bar.get_height()/2),
                    va='center', ha=ha, fontsize=11)

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Effectiveness Score')
    ax.set_title('Optimization Attempts Summary')
    ax.set_xlim(-2, 8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_optimization_overview.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig5_optimization_overview.pdf', bbox_inches='tight')
    print("Saved: fig5_optimization_overview.png/pdf")
    plt.close()


def fig6_gradient_chain():
    """Figure 6: Gradient chain analysis."""
    stages = ['Loss\n(MSE)', 'Output\n(surrogate)', 'Membrane V\n(_current_factor)', 'PSC', 'Weights']
    gradient_scale = [1.0, 0.3, 0.001, 0.5, 0.3]
    cumulative = [1.0]
    for g in gradient_scale[1:]:
        cumulative.append(cumulative[-1] * g)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Stage-wise gradient
    colors = ['#3498db', '#3498db', '#e74c3c', '#3498db', '#3498db']
    bars = ax1.bar(stages, gradient_scale, color=colors, edgecolor='black')
    ax1.set_ylabel('Gradient Scale Factor')
    ax1.set_title('Gradient Scale at Each Stage')
    ax1.set_yscale('log')

    # Highlight problematic stage
    ax1.annotate('Bottleneck!', xy=(2, 0.001), xytext=(2, 0.01),
                fontsize=12, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    # Cumulative gradient
    ax2.plot(range(len(stages)), cumulative, 'o-', color='#9b59b6', linewidth=2.5, markersize=10)
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages, rotation=15)
    ax2.set_ylabel('Cumulative Gradient')
    ax2.set_title('Cumulative Gradient Through Network')
    ax2.set_yscale('log')

    # Final gradient annotation
    ax2.annotate(f'Final: {cumulative[-1]:.1e}', xy=(4, cumulative[-1]), xytext=(3.5, 1e-3),
                fontsize=12, color='#9b59b6',
                arrowprops=dict(arrowstyle='->', color='#9b59b6'))

    plt.suptitle('Gradient Chain Analysis: Why loss_scale is Needed', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig6_gradient_chain.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig6_gradient_chain.pdf', bbox_inches='tight')
    print("Saved: fig6_gradient_chain.png/pdf")
    plt.close()


if __name__ == '__main__':
    print("Generating weekly report figures...")
    fig1_training_time_comparison()
    fig2_memory_comparison()
    fig3_iodim_training_curve()
    fig4_bug_fix_impact()
    fig5_optimization_overview()
    fig6_gradient_chain()
    print("\nAll figures generated successfully!")
