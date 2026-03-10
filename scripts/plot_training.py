import json
import matplotlib.pyplot as plt
from pathlib import Path

checkpoint_dir = Path("results/mnist/checkpoints")
epochs, losses, accuracies, rate_losses, voltage_losses = [], [], [], [], []

for i in range(1, 20):
    metrics_file = checkpoint_dir / f"metrics_{i}.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        epochs.append(i)
        losses.append(m['loss'])
        accuracies.append(m['accuracy'] * 100)
        rate_losses.append(m['rate_loss'])
        voltage_losses.append(m['voltage_loss'])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('V1 Model Training on MNIST (7 GPU, no LGN)', fontsize=14, fontweight='bold')

ax1 = axes[0, 0]
ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Total Loss'); ax1.set_title('Total Loss')
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(epochs, accuracies, 'g-o', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Classification Accuracy')
ax2.grid(True, alpha=0.3); ax2.set_ylim([50, 100])
ax2.axhline(y=92.11, color='r', linestyle='--', alpha=0.7, label='Paper (92.11%)')
ax2.legend()

ax3 = axes[1, 0]
ax3.plot(epochs, rate_losses, 'r-o', linewidth=2, markersize=6)
ax3.set_xlabel('Epoch'); ax3.set_ylabel('Rate Loss'); ax3.set_title('Rate Distribution Loss')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.plot(epochs, voltage_losses, 'm-o', linewidth=2, markersize=6)
ax4.set_xlabel('Epoch'); ax4.set_ylabel('Voltage Loss'); ax4.set_title('Voltage Regularization Loss')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/mnist/training_curve.png', dpi=150, bbox_inches='tight')
print(f"Best: {max(accuracies):.2f}% | Paper: 92.11% | Gap: {92.11 - max(accuracies):.2f}%")
