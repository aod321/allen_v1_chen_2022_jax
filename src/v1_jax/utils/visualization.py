"""Visualization utilities for V1 model training and analysis.

Provides functions for:
- Training metrics visualization (loss curves, accuracy)
- Spike activity visualization (raster plots, firing rates)
- Network state analysis (voltages, ASC currents)
- Weight visualization (connectivity patterns)

Reference: Chen et al., Science Advances 2022
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array
import numpy as np

# Lazy import matplotlib to avoid issues on headless servers
_plt = None
_mpl = None


def _ensure_matplotlib():
    """Ensure matplotlib is imported."""
    global _plt, _mpl
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        _plt = plt
        _mpl = matplotlib
    return _plt, _mpl


# =============================================================================
# Training Metrics Visualization
# =============================================================================

@dataclass
class TrainingHistory:
    """Container for training history.

    Attributes:
        steps: List of step numbers
        losses: Dictionary of loss type -> list of values
        accuracies: Dictionary of accuracy type -> list of values
        learning_rates: List of learning rate values
        firing_rates: List of mean firing rates
    """
    steps: List[int]
    losses: Dict[str, List[float]]
    accuracies: Dict[str, List[float]]
    learning_rates: List[float]
    firing_rates: List[float]

    @classmethod
    def from_metrics_list(
        cls,
        metrics_list: List[Dict[str, float]],
        steps: Optional[List[int]] = None,
    ) -> 'TrainingHistory':
        """Create TrainingHistory from list of metric dictionaries.

        Args:
            metrics_list: List of metric dicts from training
            steps: Optional step numbers (defaults to 0, 1, 2, ...)

        Returns:
            TrainingHistory instance
        """
        if steps is None:
            steps = list(range(len(metrics_list)))

        losses = {
            'total': [],
            'classification': [],
            'rate': [],
            'voltage': [],
            'weight': [],
        }
        accuracies = {'train': []}
        learning_rates = []
        firing_rates = []

        for m in metrics_list:
            losses['total'].append(m.get('loss', 0.0))
            losses['classification'].append(m.get('classification_loss', 0.0))
            losses['rate'].append(m.get('rate_loss', 0.0))
            losses['voltage'].append(m.get('voltage_loss', 0.0))
            losses['weight'].append(m.get('weight_loss', 0.0))
            accuracies['train'].append(m.get('accuracy', 0.0))
            learning_rates.append(m.get('learning_rate', 0.0))
            firing_rates.append(m.get('mean_rate', 0.0))

        return cls(
            steps=steps,
            losses=losses,
            accuracies=accuracies,
            learning_rates=learning_rates,
            firing_rates=firing_rates,
        )


def plot_training_curves(
    history: TrainingHistory,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    show_components: bool = True,
) -> Any:
    """Plot training loss and accuracy curves.

    Args:
        history: TrainingHistory instance
        save_path: Path to save figure (None to return figure)
        figsize: Figure size
        show_components: Whether to show loss components

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, _ = _ensure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Total loss
    ax = axes[0, 0]
    ax.plot(history.steps, history.losses['total'], 'b-', linewidth=2, label='Total')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Loss components
    ax = axes[0, 1]
    if show_components:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        labels = ['Classification', 'Rate', 'Voltage', 'Weight']
        keys = ['classification', 'rate', 'voltage', 'weight']
        for key, color, label in zip(keys, colors, labels):
            values = history.losses[key]
            if any(v > 0 for v in values):
                ax.plot(history.steps, values, color=color, label=label, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Accuracy
    ax = axes[1, 0]
    ax.plot(history.steps, history.accuracies['train'], 'g-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Firing rate
    ax = axes[1, 1]
    ax.plot(history.steps, history.firing_rates, 'm-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Rate')
    ax.set_title('Mean Firing Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_loss_landscape(
    loss_surface: Array,
    param_ranges: Tuple[Array, Array],
    labels: Tuple[str, str] = ('param1', 'param2'),
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Any:
    """Plot 2D loss landscape.

    Args:
        loss_surface: 2D array of loss values
        param_ranges: Tuple of (range1, range2) parameter values
        labels: Parameter axis labels
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, _ = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    x, y = param_ranges
    loss_np = np.asarray(loss_surface)

    im = ax.contourf(x, y, loss_np, levels=50, cmap='viridis')
    ax.contour(x, y, loss_np, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loss')

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title('Loss Landscape')

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


# =============================================================================
# Spike Activity Visualization
# =============================================================================

def plot_raster(
    spikes: Array,
    neuron_indices: Optional[Array] = None,
    time_range: Optional[Tuple[int, int]] = None,
    neuron_types: Optional[Array] = None,
    dt_ms: float = 1.0,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Any:
    """Plot spike raster.

    Args:
        spikes: Spike array (time, neurons) or (time, batch, neurons)
        neuron_indices: Which neurons to plot (None for all)
        time_range: (start, end) time steps to plot
        neuron_types: Array indicating E (1) or I (-1) for coloring
        dt_ms: Time step in milliseconds
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, _ = _ensure_matplotlib()

    spikes_np = np.asarray(spikes)

    # Handle batch dimension
    if spikes_np.ndim == 3:
        spikes_np = spikes_np[:, 0, :]  # Take first batch

    n_time, n_neurons = spikes_np.shape

    # Select time range
    if time_range is not None:
        t_start, t_end = time_range
        spikes_np = spikes_np[t_start:t_end]
        t_offset = t_start
    else:
        t_offset = 0

    # Select neurons
    if neuron_indices is not None:
        neuron_indices = np.asarray(neuron_indices)
        spikes_np = spikes_np[:, neuron_indices]
        if neuron_types is not None:
            neuron_types = np.asarray(neuron_types)[neuron_indices]

    fig, ax = plt.subplots(figsize=figsize)

    # Find spike times and neuron indices
    spike_times, spike_neurons = np.where(spikes_np > 0.5)
    spike_times_ms = (spike_times + t_offset) * dt_ms

    # Color by neuron type if provided
    if neuron_types is not None:
        colors = np.where(neuron_types[spike_neurons] > 0, 'tab:blue', 'tab:red')
        ax.scatter(spike_times_ms, spike_neurons, c=colors, s=1, marker='|')
        # Legend
        ax.scatter([], [], c='tab:blue', s=20, marker='|', label='Excitatory')
        ax.scatter([], [], c='tab:red', s=20, marker='|', label='Inhibitory')
        ax.legend(loc='upper right')
    else:
        ax.scatter(spike_times_ms, spike_neurons, c='black', s=1, marker='|')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron Index')
    ax.set_title(f'Spike Raster ({spikes_np.shape[1]} neurons)')
    ax.set_xlim(t_offset * dt_ms, (t_offset + spikes_np.shape[0]) * dt_ms)
    ax.set_ylim(-0.5, spikes_np.shape[1] - 0.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_firing_rate_distribution(
    spikes: Array,
    bins: int = 50,
    dt_ms: float = 1.0,
    target_distribution: Optional[Array] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Any:
    """Plot distribution of neuron firing rates.

    Args:
        spikes: Spike array (time, neurons) or (time, batch, neurons)
        bins: Number of histogram bins
        dt_ms: Time step in milliseconds
        target_distribution: Target rate distribution for comparison
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, _ = _ensure_matplotlib()

    spikes_np = np.asarray(spikes)

    # Handle batch dimension
    if spikes_np.ndim == 3:
        # Average over batch
        spikes_np = np.mean(spikes_np, axis=1)

    # Compute firing rates (spikes per second)
    duration_s = spikes_np.shape[0] * dt_ms / 1000.0
    firing_rates = np.sum(spikes_np, axis=0) / duration_s

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    ax.hist(firing_rates, bins=bins, density=True, alpha=0.7,
            color='tab:blue', label='Model')

    # Plot target distribution if provided
    if target_distribution is not None:
        target_np = np.asarray(target_distribution)
        ax.hist(target_np, bins=bins, density=True, alpha=0.5,
                color='tab:orange', label='Target')
        ax.legend()

    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Density')
    ax.set_title('Firing Rate Distribution')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_rate = np.mean(firing_rates)
    std_rate = np.std(firing_rates)
    ax.axvline(mean_rate, color='red', linestyle='--',
               label=f'Mean: {mean_rate:.2f} Hz')
    ax.text(0.95, 0.95, f'Mean: {mean_rate:.2f} Hz\nStd: {std_rate:.2f} Hz',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_population_activity(
    spikes: Array,
    window_ms: float = 10.0,
    dt_ms: float = 1.0,
    neuron_types: Optional[Array] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> Any:
    """Plot population firing rate over time.

    Args:
        spikes: Spike array (time, neurons) or (time, batch, neurons)
        window_ms: Sliding window size in milliseconds
        dt_ms: Time step in milliseconds
        neuron_types: Array indicating E (1) or I (-1) for separate traces
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, _ = _ensure_matplotlib()

    spikes_np = np.asarray(spikes)

    # Handle batch dimension
    if spikes_np.ndim == 3:
        spikes_np = np.mean(spikes_np, axis=1)

    n_time = spikes_np.shape[0]
    time_ms = np.arange(n_time) * dt_ms

    # Sliding window averaging
    window_steps = int(window_ms / dt_ms)
    kernel = np.ones(window_steps) / window_steps

    fig, ax = plt.subplots(figsize=figsize)

    if neuron_types is not None:
        neuron_types_np = np.asarray(neuron_types)

        # Excitatory population
        exc_mask = neuron_types_np > 0
        exc_rate = np.sum(spikes_np[:, exc_mask], axis=1)
        exc_rate_smooth = np.convolve(exc_rate, kernel, mode='same')
        exc_rate_hz = exc_rate_smooth * (1000 / dt_ms) / np.sum(exc_mask)
        ax.plot(time_ms, exc_rate_hz, 'tab:blue', label=f'Excitatory (n={np.sum(exc_mask)})')

        # Inhibitory population
        inh_mask = neuron_types_np < 0
        if np.any(inh_mask):
            inh_rate = np.sum(spikes_np[:, inh_mask], axis=1)
            inh_rate_smooth = np.convolve(inh_rate, kernel, mode='same')
            inh_rate_hz = inh_rate_smooth * (1000 / dt_ms) / np.sum(inh_mask)
            ax.plot(time_ms, inh_rate_hz, 'tab:red', label=f'Inhibitory (n={np.sum(inh_mask)})')

        ax.legend()
    else:
        total_rate = np.sum(spikes_np, axis=1)
        total_rate_smooth = np.convolve(total_rate, kernel, mode='same')
        total_rate_hz = total_rate_smooth * (1000 / dt_ms) / spikes_np.shape[1]
        ax.plot(time_ms, total_rate_hz, 'black')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Population Rate (Hz)')
    ax.set_title('Population Activity')
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


# =============================================================================
# Network State Visualization
# =============================================================================

def plot_voltage_traces(
    voltages: Array,
    neuron_indices: Optional[List[int]] = None,
    time_range: Optional[Tuple[int, int]] = None,
    spikes: Optional[Array] = None,
    dt_ms: float = 1.0,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Any:
    """Plot membrane voltage traces.

    Args:
        voltages: Voltage array (time, neurons) or (time, batch, neurons)
        neuron_indices: Which neurons to plot (default: first 5)
        time_range: (start, end) time steps
        spikes: Optional spike array to overlay
        dt_ms: Time step in milliseconds
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, _ = _ensure_matplotlib()

    voltages_np = np.asarray(voltages)

    # Handle batch dimension
    if voltages_np.ndim == 3:
        voltages_np = voltages_np[:, 0, :]

    if neuron_indices is None:
        neuron_indices = list(range(min(5, voltages_np.shape[1])))

    # Select time range
    if time_range is not None:
        t_start, t_end = time_range
        voltages_np = voltages_np[t_start:t_end]
        t_offset = t_start
    else:
        t_offset = 0

    n_time = voltages_np.shape[0]
    time_ms = (np.arange(n_time) + t_offset) * dt_ms

    fig, axes = plt.subplots(len(neuron_indices), 1, figsize=figsize, sharex=True)
    if len(neuron_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, neuron_indices):
        v_trace = voltages_np[:, idx]
        ax.plot(time_ms, v_trace, 'b-', linewidth=1)

        # Overlay spikes if provided
        if spikes is not None:
            spikes_np = np.asarray(spikes)
            if spikes_np.ndim == 3:
                spikes_np = spikes_np[:, 0, :]
            if time_range is not None:
                spikes_np = spikes_np[t_start:t_end]

            spike_times = np.where(spikes_np[:, idx] > 0.5)[0]
            for st in spike_times:
                ax.axvline((st + t_offset) * dt_ms, color='red', alpha=0.5, linewidth=1)

        ax.set_ylabel(f'Neuron {idx}')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Membrane Voltage Traces')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_state_distribution(
    states: Dict[str, Array],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Any:
    """Plot distribution of GLIF3 state variables.

    Args:
        states: Dictionary with keys like 'v', 'asc_1', 'asc_2'
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, _ = _ensure_matplotlib()

    state_names = ['v', 'asc_1', 'asc_2', 'r']
    available_states = [k for k in state_names if k in states]

    n_states = len(available_states)
    fig, axes = plt.subplots(1, n_states, figsize=figsize)
    if n_states == 1:
        axes = [axes]

    titles = {
        'v': 'Membrane Voltage',
        'asc_1': 'ASC Current 1',
        'asc_2': 'ASC Current 2',
        'r': 'Refractory Counter',
    }

    for ax, name in zip(axes, available_states):
        data = np.asarray(states[name]).flatten()
        ax.hist(data, bins=50, density=True, alpha=0.7, color='tab:blue')
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(titles.get(name, name))
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.axvline(mean_val, color='red', linestyle='--')
        ax.text(0.95, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


# =============================================================================
# Weight Visualization
# =============================================================================

def plot_weight_matrix(
    weights: Array,
    neuron_types: Optional[Array] = None,
    sort_by_type: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    vmax: Optional[float] = None,
) -> Any:
    """Plot weight matrix as heatmap.

    Args:
        weights: Weight matrix (n_post, n_pre) or sparse indices + values
        neuron_types: Array indicating E (1) or I (-1)
        sort_by_type: Whether to sort neurons by type
        save_path: Path to save figure
        figsize: Figure size
        vmax: Max value for colormap

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, mpl = _ensure_matplotlib()

    weights_np = np.asarray(weights)

    # If sparse, convert to dense for visualization
    if weights_np.ndim == 1 or weights_np.shape[0] != weights_np.shape[1]:
        raise ValueError("Expected dense square weight matrix for visualization")

    n = weights_np.shape[0]

    # Sort by neuron type if requested
    if sort_by_type and neuron_types is not None:
        types_np = np.asarray(neuron_types)
        sort_idx = np.argsort(-types_np)  # Excitatory first
        weights_np = weights_np[sort_idx][:, sort_idx]
        types_np = types_np[sort_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Determine colormap range
    if vmax is None:
        vmax = np.percentile(np.abs(weights_np[weights_np != 0]), 99)

    # Use diverging colormap
    im = ax.imshow(weights_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight')

    ax.set_xlabel('Pre-synaptic Neuron')
    ax.set_ylabel('Post-synaptic Neuron')
    ax.set_title('Synaptic Weight Matrix')

    # Add E/I boundaries if sorted
    if sort_by_type and neuron_types is not None:
        n_exc = np.sum(types_np > 0)
        ax.axhline(n_exc - 0.5, color='black', linewidth=2)
        ax.axvline(n_exc - 0.5, color='black', linewidth=2)

        ax.text(n_exc / 2, -n * 0.02, 'E', ha='center', fontweight='bold')
        ax.text(n_exc + (n - n_exc) / 2, -n * 0.02, 'I', ha='center', fontweight='bold')

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_weight_distribution(
    weights: Array,
    neuron_types: Optional[Array] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Any:
    """Plot distribution of synaptic weights.

    Args:
        weights: Weight array
        neuron_types: Array indicating E (1) or I (-1) for pre-synaptic neurons
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure if save_path is None
    """
    plt, _ = _ensure_matplotlib()

    weights_np = np.asarray(weights).flatten()

    # Remove zeros for sparse weights
    nonzero_weights = weights_np[weights_np != 0]

    fig, ax = plt.subplots(figsize=figsize)

    if neuron_types is not None:
        # Separate E and I weights
        exc_weights = nonzero_weights[nonzero_weights > 0]
        inh_weights = nonzero_weights[nonzero_weights < 0]

        if len(exc_weights) > 0:
            ax.hist(exc_weights, bins=50, density=True, alpha=0.7,
                    color='tab:blue', label=f'Excitatory (n={len(exc_weights)})')
        if len(inh_weights) > 0:
            ax.hist(inh_weights, bins=50, density=True, alpha=0.7,
                    color='tab:red', label=f'Inhibitory (n={len(inh_weights)})')
        ax.legend()
    else:
        ax.hist(nonzero_weights, bins=50, density=True, alpha=0.7, color='tab:blue')

    ax.set_xlabel('Weight')
    ax.set_ylabel('Density')
    ax.set_title('Weight Distribution (Non-zero)')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Statistics
    mean_w = np.mean(nonzero_weights)
    std_w = np.std(nonzero_weights)
    ax.text(0.95, 0.95, f'Mean: {mean_w:.4f}\nStd: {std_w:.4f}\nN: {len(nonzero_weights)}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


# =============================================================================
# Summary Report Generation
# =============================================================================

def generate_training_report(
    history: TrainingHistory,
    output_dir: Union[str, Path],
    prefix: str = 'training',
) -> Dict[str, str]:
    """Generate comprehensive training report with all visualizations.

    Args:
        history: TrainingHistory instance
        output_dir: Directory to save figures
        prefix: Prefix for filenames

    Returns:
        Dictionary mapping figure names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Training curves
    path = output_dir / f'{prefix}_curves.png'
    plot_training_curves(history, save_path=path)
    paths['training_curves'] = str(path)

    return paths


def generate_activity_report(
    spikes: Array,
    voltages: Optional[Array] = None,
    neuron_types: Optional[Array] = None,
    dt_ms: float = 1.0,
    output_dir: Union[str, Path] = '.',
    prefix: str = 'activity',
) -> Dict[str, str]:
    """Generate comprehensive activity report with visualizations.

    Args:
        spikes: Spike array
        voltages: Optional voltage array
        neuron_types: Optional neuron type array
        dt_ms: Time step in milliseconds
        output_dir: Directory to save figures
        prefix: Prefix for filenames

    Returns:
        Dictionary mapping figure names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Raster plot
    path = output_dir / f'{prefix}_raster.png'
    plot_raster(spikes, neuron_types=neuron_types, dt_ms=dt_ms, save_path=path)
    paths['raster'] = str(path)

    # Firing rate distribution
    path = output_dir / f'{prefix}_rate_dist.png'
    plot_firing_rate_distribution(spikes, dt_ms=dt_ms, save_path=path)
    paths['rate_distribution'] = str(path)

    # Population activity
    path = output_dir / f'{prefix}_population.png'
    plot_population_activity(spikes, neuron_types=neuron_types, dt_ms=dt_ms, save_path=path)
    paths['population_activity'] = str(path)

    # Voltage traces if provided
    if voltages is not None:
        path = output_dir / f'{prefix}_voltages.png'
        plot_voltage_traces(voltages, spikes=spikes, dt_ms=dt_ms, save_path=path)
        paths['voltage_traces'] = str(path)

    return paths
