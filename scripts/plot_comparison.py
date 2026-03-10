import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paper results - V1 model (dark blue)
paper_epochs = np.arange(0, 17)
v1_full = [0.40, 0.82, 0.88, 0.90, 0.91, 0.92, 0.925, 0.93, 0.932, 0.935, 0.937, 0.94, 0.942, 0.943, 0.944, 0.945, 0.945]

# Paper results - V1 without LGN (cyan)
v1_no_lgn = [0.40, 0.50, 0.70, 0.82, 0.86, 0.88, 0.895, 0.90, 0.905, 0.91, 0.912, 0.915, 0.917, 0.918, 0.919, 0.92, 0.921]

# Our reproduction
checkpoint_dir = Path("results/mnist/checkpoints")
our_epochs, our_acc = [], []
for i in range(1, 20):
    metrics_file = checkpoint_dir / f"metrics_{i}.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        our_epochs.append(i)
        our_acc.append(m['accuracy'])

# Convert to numpy for easier manipulation
v1_full = np.array(v1_full)
v1_no_lgn = np.array(v1_no_lgn)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot paper results with shaded regions (simulating variance bands)
ax.plot(paper_epochs, v1_full, 'b-', linewidth=2.5, label='V1 model (Paper)')
ax.fill_between(paper_epochs, v1_full - 0.01, v1_full + 0.01, alpha=0.2, color='blue')

ax.plot(paper_epochs, v1_no_lgn, 'c-', linewidth=2.5, label='V1 without LGN (Paper)')
ax.fill_between(paper_epochs, v1_no_lgn - 0.01, v1_no_lgn + 0.01, alpha=0.2, color='cyan')

# Plot our results
ax.plot(our_epochs, our_acc, 'ro-', linewidth=2.5, markersize=8, 
        label='Our JAX reproduction (no LGN)', markeredgecolor='darkred', markerfacecolor='red')

# Styling
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Averaged accuracy', fontsize=12)
ax.set_title('V1 Model MNIST Classification: Paper vs Our Reproduction', fontsize=14, fontweight='bold')
ax.set_xlim([0, 16])
ax.set_ylim([0.35, 1.0])
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=11)

# Add text box with summary
paper_final = v1_no_lgn[-1]
our_best = max(our_acc)
textstr = f'Paper (no LGN) final: {paper_final*100:.1f}%\nOurs best: {our_best*100:.1f}%\nDifference: {(paper_final-our_best)*100:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('results/mnist/paper_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to results/mnist/paper_comparison.png")
print(f"\nPaper (no LGN) final: {paper_final*100:.1f}%")
print(f"Ours best: {our_best*100:.1f}%")
