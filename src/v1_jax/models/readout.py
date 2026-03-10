"""Readout layers for V1 network outputs.

This module provides readout layers for mapping V1 population activity
to task-relevant outputs (e.g., classification logits).

The readout layers implement:
- Dense linear readout
- Sparse readout from selected neurons
- Pooled readout with spike rate computation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Callable, Union, List
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np


@dataclass
class ReadoutParams:
    """Parameters for readout layer.

    Attributes:
        weights: Readout weights (n_neurons, n_outputs) or (n_selected, n_outputs)
        bias: Optional bias (n_outputs,)
        neuron_indices: Optional indices of neurons to read from
    """
    weights: Array
    bias: Optional[Array] = None
    neuron_indices: Optional[Array] = None


# =============================================================================
# Dense Readout
# =============================================================================

def dense_readout(
    spikes: Array,
    params: ReadoutParams,
    temporal_pooling: str = 'mean',
    chunk_size: int = 50,
) -> Array:
    """Dense linear readout from spike trains.

    Computes: logits = pool(spikes) @ weights + bias

    Args:
        spikes: Spike trains (time, batch, n_neurons) or (batch, time, n_neurons)
        params: Readout parameters
        temporal_pooling: 'mean', 'sum', 'last', or 'chunks'
        chunk_size: Size of temporal chunks for 'chunks' pooling

    Returns:
        Logits of shape (batch, n_outputs) or (batch, n_chunks, n_outputs)
    """
    # Ensure time-first format
    if spikes.ndim == 3:
        # Assume (time, batch, n_neurons) format
        time_axis = 0
    else:
        raise ValueError(f"Expected 3D spikes, got shape {spikes.shape}")

    # Select neurons if indices provided
    if params.neuron_indices is not None:
        spikes = spikes[..., params.neuron_indices]

    # Temporal pooling
    if temporal_pooling == 'mean':
        pooled = jnp.mean(spikes, axis=time_axis)  # (batch, n_neurons)
    elif temporal_pooling == 'sum':
        pooled = jnp.sum(spikes, axis=time_axis)
    elif temporal_pooling == 'last':
        pooled = spikes[-1]  # Last timestep
    elif temporal_pooling == 'chunks':
        # Pool within chunks
        seq_len = spikes.shape[time_axis]
        n_chunks = seq_len // chunk_size

        # Reshape to (n_chunks, chunk_size, batch, n_neurons)
        spikes_chunked = spikes.reshape(n_chunks, chunk_size, *spikes.shape[1:])
        pooled = jnp.mean(spikes_chunked, axis=1)  # (n_chunks, batch, n_neurons)

        # Transpose to (batch, n_chunks, n_neurons)
        pooled = jnp.transpose(pooled, (1, 0, 2))
    else:
        raise ValueError(f"Unknown temporal_pooling: {temporal_pooling}")

    # Linear projection
    logits = pooled @ params.weights

    # Add bias
    if params.bias is not None:
        logits = logits + params.bias

    return logits


# =============================================================================
# Sparse Readout (from selected neurons)
# =============================================================================

def select_readout_neurons(
    n_neurons: int,
    n_select: int,
    neuron_types: Optional[Array] = None,
    excitatory_only: bool = False,
    key: Optional[Array] = None,
) -> Array:
    """Select neurons for sparse readout.

    Args:
        n_neurons: Total number of neurons
        n_select: Number of neurons to select
        neuron_types: Optional array indicating neuron types (0=exc, 1=inh)
        excitatory_only: If True, only select excitatory neurons
        key: Random key for random selection

    Returns:
        Indices of selected neurons (n_select,)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    if neuron_types is not None and excitatory_only:
        # Find excitatory neuron indices
        exc_mask = neuron_types == 0
        exc_indices = jnp.where(exc_mask)[0]

        # Randomly sample from excitatory neurons
        n_exc = len(exc_indices)
        if n_select > n_exc:
            raise ValueError(f"Requested {n_select} neurons but only {n_exc} excitatory")

        perm = jax.random.permutation(key, n_exc)
        selected = exc_indices[perm[:n_select]]
    else:
        # Random selection from all neurons
        perm = jax.random.permutation(key, n_neurons)
        selected = perm[:n_select]

    return jnp.sort(selected)


def sparse_readout(
    spikes: Array,
    params: ReadoutParams,
    temporal_pooling: str = 'mean',
) -> Array:
    """Sparse readout from selected neurons.

    Args:
        spikes: Spike trains (time, batch, n_neurons)
        params: Readout parameters with neuron_indices set
        temporal_pooling: How to pool over time

    Returns:
        Logits (batch, n_outputs)
    """
    if params.neuron_indices is None:
        raise ValueError("neuron_indices must be set for sparse_readout")

    return dense_readout(spikes, params, temporal_pooling)


# =============================================================================
# Chunk-wise Readout (for VCD, classification tasks)
# =============================================================================

def chunk_readout(
    spikes: Array,
    params: ReadoutParams,
    chunk_size: int = 50,
    apply_softmax: bool = False,
) -> Array:
    """Readout with chunk-wise temporal pooling.

    Computes logits for each time chunk independently.

    Args:
        spikes: Spike trains (time, batch, n_neurons)
        params: Readout parameters
        chunk_size: Duration of each chunk in timesteps
        apply_softmax: If True, apply softmax to each chunk (matches TF implementation)

    Returns:
        Logits or probabilities (batch, n_chunks, n_outputs)
    """
    logits = dense_readout(spikes, params, temporal_pooling='chunks', chunk_size=chunk_size)
    if apply_softmax:
        return jax.nn.softmax(logits, axis=-1)
    return logits


# =============================================================================
# Readout Layer Classes
# =============================================================================

class DenseReadout:
    """Dense readout layer.

    Maps V1 population activity to output logits via learned linear projection.

    Attributes:
        n_neurons: Number of input neurons
        n_outputs: Number of output classes/values
        temporal_pooling: Temporal pooling method
        chunk_size: Chunk size for chunk-wise pooling
    """

    def __init__(
        self,
        n_neurons: int,
        n_outputs: int,
        temporal_pooling: str = 'mean',
        chunk_size: int = 50,
        neuron_indices: Optional[Array] = None,
        key: Optional[Array] = None,
    ):
        """Initialize dense readout layer.

        Args:
            n_neurons: Number of input neurons (or selected neurons)
            n_outputs: Number of output classes
            temporal_pooling: 'mean', 'sum', 'last', or 'chunks'
            chunk_size: Size of temporal chunks
            neuron_indices: Optional indices of neurons to read from
            key: Random key for weight initialization
        """
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.temporal_pooling = temporal_pooling
        self.chunk_size = chunk_size
        self.neuron_indices = neuron_indices

        # Initialize weights
        if key is None:
            key = jax.random.PRNGKey(0)

        key1, key2 = jax.random.split(key)

        # Xavier initialization
        n_in = n_neurons if neuron_indices is None else len(neuron_indices)
        scale = jnp.sqrt(2.0 / (n_in + n_outputs))
        weights = jax.random.normal(key1, (n_in, n_outputs)) * scale
        bias = jnp.zeros(n_outputs)

        self.params = ReadoutParams(
            weights=weights,
            bias=bias,
            neuron_indices=neuron_indices,
        )

    def __call__(self, spikes: Array) -> Array:
        """Apply readout to spike trains.

        Args:
            spikes: Spike trains (time, batch, n_neurons)

        Returns:
            Logits
        """
        return dense_readout(
            spikes,
            self.params,
            temporal_pooling=self.temporal_pooling,
            chunk_size=self.chunk_size,
        )

    def get_params(self) -> ReadoutParams:
        """Get readout parameters."""
        return self.params

    def set_params(self, params: ReadoutParams) -> 'DenseReadout':
        """Set readout parameters.

        Returns new instance with updated parameters.
        """
        new_readout = DenseReadout(
            n_neurons=self.n_neurons,
            n_outputs=self.n_outputs,
            temporal_pooling=self.temporal_pooling,
            chunk_size=self.chunk_size,
            neuron_indices=self.neuron_indices,
        )
        new_readout.params = params
        return new_readout


class BinaryReadout:
    """Binary classification readout.

    Specialized readout for binary tasks like change detection.
    Uses sigmoid activation and BCE loss.
    """

    def __init__(
        self,
        n_neurons: int,
        temporal_pooling: str = 'chunks',
        chunk_size: int = 50,
        neuron_indices: Optional[Array] = None,
        key: Optional[Array] = None,
    ):
        """Initialize binary readout.

        Args:
            n_neurons: Number of input neurons
            temporal_pooling: Temporal pooling method
            chunk_size: Chunk size for chunk-wise pooling
            neuron_indices: Optional neuron indices for sparse readout
            key: Random key
        """
        self.dense_readout = DenseReadout(
            n_neurons=n_neurons,
            n_outputs=1,
            temporal_pooling=temporal_pooling,
            chunk_size=chunk_size,
            neuron_indices=neuron_indices,
            key=key,
        )

    def __call__(self, spikes: Array) -> Array:
        """Apply binary readout.

        Args:
            spikes: Spike trains

        Returns:
            Logits for binary classification (before sigmoid)
        """
        logits = self.dense_readout(spikes)
        return logits.squeeze(-1)  # Remove last dimension

    def probability(self, spikes: Array) -> Array:
        """Get probability of positive class.

        Args:
            spikes: Spike trains

        Returns:
            Probabilities
        """
        return jax.nn.sigmoid(self(spikes))


class MultiClassReadout:
    """Multi-class classification readout.

    Specialized readout for classification tasks.
    Uses softmax activation and cross-entropy loss.

    To match TF implementation, use temporal_pooling='chunks' with apply_softmax=True.
    This will output softmax probabilities for each chunk, compatible with from_logits=False loss.
    """

    def __init__(
        self,
        n_neurons: int,
        n_classes: int,
        temporal_pooling: str = 'chunks',
        chunk_size: int = 50,
        neuron_indices: Optional[Array] = None,
        key: Optional[Array] = None,
        apply_softmax: bool = False,
    ):
        """Initialize multi-class readout.

        Args:
            n_neurons: Number of input neurons
            n_classes: Number of classes
            temporal_pooling: Temporal pooling method ('mean', 'chunks', etc.)
            chunk_size: Chunk size for chunk-wise pooling
            neuron_indices: Optional neuron indices
            key: Random key
            apply_softmax: If True, apply softmax to output (matches TF implementation)
        """
        self.n_classes = n_classes
        self.apply_softmax = apply_softmax
        self.temporal_pooling = temporal_pooling
        self.chunk_size = chunk_size
        self.dense_readout = DenseReadout(
            n_neurons=n_neurons,
            n_outputs=n_classes,
            temporal_pooling=temporal_pooling,
            chunk_size=chunk_size,
            neuron_indices=neuron_indices,
            key=key,
        )

    def __call__(self, spikes: Array) -> Array:
        """Apply multi-class readout.

        Args:
            spikes: Spike trains

        Returns:
            Logits or probabilities for each class (depending on apply_softmax)
        """
        output = self.dense_readout(spikes)
        if self.apply_softmax:
            return jax.nn.softmax(output, axis=-1)
        return output

    def logits(self, spikes: Array) -> Array:
        """Get raw logits (before softmax).

        Args:
            spikes: Spike trains

        Returns:
            Raw logits
        """
        return self.dense_readout(spikes)

    def probability(self, spikes: Array) -> Array:
        """Get class probabilities.

        Args:
            spikes: Spike trains

        Returns:
            Softmax probabilities
        """
        return jax.nn.softmax(self.logits(spikes), axis=-1)

    def predict(self, spikes: Array) -> Array:
        """Get predicted class.

        Args:
            spikes: Spike trains

        Returns:
            Predicted class indices
        """
        return jnp.argmax(self.logits(spikes), axis=-1)


# =============================================================================
# L5 Pyramidal Cell Voting Readout (Biologically Realistic)
# =============================================================================

@dataclass
class L5VotingConfig:
    """Configuration for L5 voting readout.

    Reference: Chen et al., Science Advances 2022

    Attributes:
        n_classes: Number of output classes
        neurons_per_class: Number of L5 neurons per class pool
        temporal_pooling: 'chunks' for chunk-wise voting or 'mean' for overall
        chunk_size: Size of temporal chunks in timesteps (default 50 = 50ms at 1ms dt)
        response_window: Optional (start, end) tuple for restricting to specific time window
    """
    n_classes: int = 10
    neurons_per_class: int = 16
    temporal_pooling: str = 'chunks'
    chunk_size: int = 50
    response_window: Optional[Tuple[int, int]] = None


class L5VotingReadout:
    """L5 pyramidal cell competitive voting readout.

    Implements biologically realistic readout using L5 pyramidal cells as described
    in Chen et al., Science Advances 2022. This readout has NO trainable parameters -
    it simply counts spikes from predefined L5 neuron pools.

    Key features:
    - Uses ~30 L5 pyramidal neurons per class (localized spatially)
    - Competition via spike counting (which pool fires more)
    - Simulates L5 pyramidal cells projecting to subcortical regions for motor output

    The output represents the mean firing rate of each class's L5 pool, which can
    be used directly for classification (argmax of rates = predicted class).

    Example:
        >>> config = L5VotingConfig(n_classes=10, neurons_per_class=16)
        >>> neuron_indices = {i: np.array([...]) for i in range(10)}  # 10 pools
        >>> readout = L5VotingReadout(neuron_indices, config)
        >>> scores = readout(spikes)  # (batch, n_chunks, n_classes)
        >>> predictions = jnp.argmax(scores, axis=-1)
    """

    def __init__(
        self,
        neuron_indices: Dict[int, Array],
        config: Optional[L5VotingConfig] = None,
    ):
        """Initialize L5 voting readout.

        Args:
            neuron_indices: Dict mapping class_id (int) to neuron indices array
            config: L5VotingConfig (default uses standard settings)
        """
        self.pools = neuron_indices  # class_id -> neuron indices
        self.config = config or L5VotingConfig(n_classes=len(neuron_indices))
        self.n_classes = len(neuron_indices)

        # Validate
        if self.n_classes != self.config.n_classes:
            raise ValueError(
                f"Number of pools ({self.n_classes}) doesn't match "
                f"config.n_classes ({self.config.n_classes})"
            )

    def __call__(self, spikes: Array) -> Array:
        """Apply L5 voting readout to spike trains.

        Computes the mean firing rate for each L5 pool (class), optionally
        chunked over time for per-chunk predictions.

        Args:
            spikes: Spike trains (time, batch, n_neurons)

        Returns:
            Scores (batch, n_classes) or (batch, n_chunks, n_classes) if chunk pooling
        """
        # Extract response window if specified
        if self.config.response_window is not None:
            start, end = self.config.response_window
            spikes = spikes[start:end]

        # Get dimensions
        seq_len, batch_size, n_neurons = spikes.shape

        if self.config.temporal_pooling == 'chunks':
            # Chunk-wise voting
            n_chunks = seq_len // self.config.chunk_size

            # Reshape to (n_chunks, chunk_size, batch, n_neurons)
            truncated_len = n_chunks * self.config.chunk_size
            spikes_truncated = spikes[:truncated_len]
            spikes_chunked = spikes_truncated.reshape(
                n_chunks, self.config.chunk_size, batch_size, n_neurons
            )

            # Compute mean spike rate per chunk for each class pool
            scores = []
            for class_id in range(self.n_classes):
                pool_indices = self.pools[class_id]
                # Extract spikes for this pool: (n_chunks, chunk_size, batch, pool_size)
                pool_spikes = spikes_chunked[:, :, :, pool_indices]
                # Mean over time (chunk_size) and neurons (pool_size)
                # -> (n_chunks, batch)
                pool_rate = jnp.mean(pool_spikes, axis=(1, 3))
                scores.append(pool_rate)

            # Stack: (n_chunks, batch, n_classes)
            scores = jnp.stack(scores, axis=-1)
            # Transpose to (batch, n_chunks, n_classes) to match MultiClassReadout output
            scores = jnp.transpose(scores, (1, 0, 2))

        elif self.config.temporal_pooling == 'mean':
            # Overall mean rate
            scores = []
            for class_id in range(self.n_classes):
                pool_indices = self.pools[class_id]
                pool_spikes = spikes[:, :, pool_indices]  # (time, batch, pool_size)
                # Mean over time and neurons -> (batch,)
                pool_rate = jnp.mean(pool_spikes, axis=(0, 2))
                scores.append(pool_rate)

            # Stack: (batch, n_classes)
            scores = jnp.stack(scores, axis=-1)

        else:
            raise ValueError(f"Unknown temporal_pooling: {self.config.temporal_pooling}")

        return scores

    def logits(self, spikes: Array) -> Array:
        """Alias for __call__, returns raw scores (mean spike rates).

        The scores can be treated as "logits" for compatibility with
        cross-entropy loss, though they're actually firing rates.
        """
        return self(spikes)

    def predict(self, spikes: Array) -> Array:
        """Get predicted class from spike trains.

        Args:
            spikes: Spike trains (time, batch, n_neurons)

        Returns:
            Predicted class indices (batch,) or (batch, n_chunks)
        """
        scores = self(spikes)
        return jnp.argmax(scores, axis=-1)

    def get_params(self) -> Dict:
        """Get readout parameters (empty for voting readout)."""
        return {}

    def set_params(self, params: Dict) -> 'L5VotingReadout':
        """Set readout parameters (no-op for voting readout).

        Returns self since voting readout has no trainable parameters.
        """
        return self


def create_l5_voting_readout(
    network_data: Dict[str, Any],
    n_classes: int,
    config: Optional[L5VotingConfig] = None,
    class_offset: int = 0,
) -> L5VotingReadout:
    """Create L5 voting readout from network data.

    Factory function that extracts L5 neuron pools from network_data and
    creates an L5VotingReadout instance.

    Args:
        network_data: Network dict containing 'localized_readout_neuron_ids_{i}' keys
        n_classes: Number of output classes
        config: L5VotingConfig (optional)
        class_offset: Offset into readout neuron pools (default 0)
            - For 10-class MNIST: class_offset=5 uses pools 5-14

    Returns:
        L5VotingReadout instance

    Example:
        >>> network = load_billeh(data_dir, localized_readout=True)
        >>> readout = create_l5_voting_readout(network, n_classes=10, class_offset=5)
    """
    if config is None:
        config = L5VotingConfig(n_classes=n_classes)

    neuron_indices = {}
    for i in range(n_classes):
        key = f'localized_readout_neuron_ids_{i + class_offset}'
        if key not in network_data:
            raise KeyError(
                f"Network data missing '{key}'. "
                f"Make sure load_billeh was called with localized_readout=True"
            )
        neuron_indices[i] = jnp.array(network_data[key])

    return L5VotingReadout(neuron_indices, config)


# =============================================================================
# L5 Threshold Readout (Binary Tasks)
# =============================================================================

@dataclass
class L5ThresholdConfig:
    """Configuration for threshold-based binary readout (single pool).

    Reference: Chen et al. 2022 - used for orientation discrimination and change detection.
    Output format: [threshold, firing_rate] for softmax compatibility.

    Attributes:
        threshold: r₀ threshold value (default 0.01 as in paper)
        temporal_pooling: 'chunks' for chunk-wise or 'mean' for overall
        chunk_size: Size of temporal chunks in timesteps (default 50)
        response_window: Optional (start, end) tuple for restricting to specific time window
    """
    threshold: float = 0.01
    temporal_pooling: str = 'chunks'
    chunk_size: int = 50
    response_window: Optional[Tuple[int, int]] = None


class L5ThresholdReadout:
    """L5 pyramidal cell threshold-based readout for binary classification.

    Implements biologically realistic binary readout using a single L5 pool with
    threshold judgment as described in Chen et al., Science Advances 2022.

    Decision rule:
    - If mean firing rate r > r₀ (threshold) → class 1
    - Otherwise → class 0

    Output format: [threshold, firing_rate] for each time chunk.
    After softmax, this effectively becomes sigmoid(rate - threshold).

    Used for tasks:
    - Fine orientation discrimination (pool 0)
    - Grating change detection (pool 1)
    - Orientation differentiation (pool 2)

    Example:
        >>> config = L5ThresholdConfig(threshold=0.01)
        >>> neuron_indices = np.array([100, 101, 102, ...])  # L5 neuron indices
        >>> readout = L5ThresholdReadout(neuron_indices, config)
        >>> scores = readout(spikes)  # (batch, n_chunks, 2)
        >>> predictions = jnp.argmax(scores, axis=-1)  # 0 or 1
    """

    def __init__(
        self,
        neuron_indices: Array,
        config: Optional[L5ThresholdConfig] = None,
    ):
        """Initialize L5 threshold readout.

        Args:
            neuron_indices: Array of L5 neuron indices for the single readout pool
            config: L5ThresholdConfig (default uses r₀=0.01)
        """
        self.pool = neuron_indices
        self.config = config or L5ThresholdConfig()
        self.n_neurons = len(neuron_indices)

    def __call__(self, spikes: Array) -> Array:
        """Apply L5 threshold readout to spike trains.

        Computes mean firing rate of the L5 pool and stacks with threshold
        for softmax-based binary classification.

        Args:
            spikes: Spike trains (time, batch, n_neurons)

        Returns:
            Scores (batch, n_chunks, 2) or (batch, 2) where each entry is [threshold, rate]
        """
        # Extract response window if specified
        if self.config.response_window is not None:
            start, end = self.config.response_window
            spikes = spikes[start:end]

        # Get dimensions
        seq_len, batch_size, n_neurons = spikes.shape

        # Extract spikes from the L5 pool
        pool_spikes = spikes[:, :, self.pool]  # (time, batch, pool_size)

        if self.config.temporal_pooling == 'chunks':
            # Chunk-wise pooling
            n_chunks = seq_len // self.config.chunk_size

            # Reshape to (n_chunks, chunk_size, batch, pool_size)
            truncated_len = n_chunks * self.config.chunk_size
            pool_spikes_truncated = pool_spikes[:truncated_len]
            pool_spikes_chunked = pool_spikes_truncated.reshape(
                n_chunks, self.config.chunk_size, batch_size, self.n_neurons
            )

            # Mean firing rate per chunk: average over time (axis 1) and neurons (axis 3)
            # -> (n_chunks, batch)
            rates = jnp.mean(pool_spikes_chunked, axis=(1, 3))

            # Transpose to (batch, n_chunks)
            rates = jnp.transpose(rates, (1, 0))

            # Stack [threshold, rate] for each chunk -> (batch, n_chunks, 2)
            thresh = jnp.full_like(rates, self.config.threshold)
            scores = jnp.stack([thresh, rates], axis=-1)

        elif self.config.temporal_pooling == 'mean':
            # Overall mean rate: average over all time and neurons
            # -> (batch,)
            rates = jnp.mean(pool_spikes, axis=(0, 2))

            # Stack [threshold, rate] -> (batch, 2)
            thresh = jnp.full_like(rates, self.config.threshold)
            scores = jnp.stack([thresh, rates], axis=-1)

        else:
            raise ValueError(f"Unknown temporal_pooling: {self.config.temporal_pooling}")

        return scores

    def logits(self, spikes: Array) -> Array:
        """Alias for __call__, returns raw scores."""
        return self(spikes)

    def predict(self, spikes: Array) -> Array:
        """Get predicted class from spike trains.

        Args:
            spikes: Spike trains (time, batch, n_neurons)

        Returns:
            Predicted class indices (batch,) or (batch, n_chunks)
            0 = rate <= threshold, 1 = rate > threshold
        """
        scores = self(spikes)
        return jnp.argmax(scores, axis=-1)

    def get_params(self) -> Dict:
        """Get readout parameters (empty for threshold readout)."""
        return {}

    def set_params(self, params: Dict) -> 'L5ThresholdReadout':
        """Set readout parameters (no-op for threshold readout)."""
        return self


def create_l5_threshold_readout(
    network_data: Dict[str, Any],
    pool_index: int,
    config: Optional[L5ThresholdConfig] = None,
) -> L5ThresholdReadout:
    """Create L5 threshold readout from network data.

    Factory function that extracts a single L5 neuron pool from network_data
    and creates an L5ThresholdReadout instance for binary classification.

    Args:
        network_data: Network dict containing 'localized_readout_neuron_ids_{i}' keys
        pool_index: Index of the L5 pool to use (0-14)
        config: L5ThresholdConfig (optional)

    Returns:
        L5ThresholdReadout instance

    Example:
        >>> network = load_billeh(data_dir, localized_readout=True)
        >>> # For orientation discrimination (pool 0)
        >>> readout = create_l5_threshold_readout(network, pool_index=0)
        >>> # For grating change detection (pool 1)
        >>> readout = create_l5_threshold_readout(network, pool_index=1)
    """
    key = f'localized_readout_neuron_ids_{pool_index}'
    if key not in network_data:
        raise KeyError(
            f"Network data missing '{key}'. "
            f"Make sure load_billeh was called with localized_readout=True"
        )
    neuron_indices = jnp.array(network_data[key])
    return L5ThresholdReadout(neuron_indices, config)


# =============================================================================
# Pool Assignments and Strategies (from TF classification_tools.py)
# =============================================================================

# Pool assignments for each task
L5_POOL_ASSIGNMENTS = {
    'garrett': [0],           # Orientation discrimination (threshold)
    'vcd_grating': [1],       # Grating change detection (threshold)
    'ori_diff': [2],          # Orientation differentiation (threshold)
    'evidence': [3, 4],       # Evidence accumulation (competition)
    '10class': list(range(5, 15)),  # 10-class classification (competition)
    'mnist': list(range(5, 15)),    # MNIST (alias for 10class)
}

# Readout strategy for each task
L5_READOUT_STRATEGY = {
    'garrett': 'threshold',      # Single pool + threshold judgment
    'vcd_grating': 'threshold',  # Single pool + threshold judgment
    'ori_diff': 'threshold',     # Single pool + threshold judgment
    'evidence': 'competition',   # Two pools compete
    '10class': 'competition',    # 10 pools compete
    'mnist': 'competition',      # 10 pools compete
}


# =============================================================================
# Factory Functions
# =============================================================================

def create_readout(
    n_neurons: int,
    task: str,
    n_classes: int = 10,
    temporal_pooling: str = 'chunks',
    chunk_size: int = 50,
    neuron_indices: Optional[Array] = None,
    key: Optional[Array] = None,
) -> Union[DenseReadout, BinaryReadout, MultiClassReadout]:
    """Create appropriate readout layer for task.

    Args:
        n_neurons: Number of input neurons
        task: Task type ('binary', 'classification', 'regression')
        n_classes: Number of classes for classification
        temporal_pooling: Temporal pooling method
        chunk_size: Chunk size
        neuron_indices: Optional neuron indices
        key: Random key

    Returns:
        Appropriate readout layer
    """
    if task == 'binary':
        return BinaryReadout(
            n_neurons=n_neurons,
            temporal_pooling=temporal_pooling,
            chunk_size=chunk_size,
            neuron_indices=neuron_indices,
            key=key,
        )
    elif task == 'classification':
        return MultiClassReadout(
            n_neurons=n_neurons,
            n_classes=n_classes,
            temporal_pooling=temporal_pooling,
            chunk_size=chunk_size,
            neuron_indices=neuron_indices,
            key=key,
        )
    elif task == 'regression':
        return DenseReadout(
            n_neurons=n_neurons,
            n_outputs=1,
            temporal_pooling=temporal_pooling,
            chunk_size=chunk_size,
            neuron_indices=neuron_indices,
            key=key,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# =============================================================================
# JIT-compatible Pure Functions
# =============================================================================

@jax.jit
def apply_readout_jit(
    spikes: Array,
    weights: Array,
    bias: Array,
    neuron_indices: Optional[Array] = None,
) -> Array:
    """JIT-compiled readout application with mean pooling.

    Args:
        spikes: Spike trains (time, batch, n_neurons)
        weights: Readout weights
        bias: Readout bias
        neuron_indices: Optional neuron indices

    Returns:
        Logits (batch, n_outputs)
    """
    if neuron_indices is not None:
        spikes = spikes[..., neuron_indices]

    # Mean over time
    pooled = jnp.mean(spikes, axis=0)

    # Linear projection
    return pooled @ weights + bias


def make_readout_fn(
    params: ReadoutParams,
    temporal_pooling: str = 'mean',
    chunk_size: int = 50,
) -> Callable:
    """Create a JIT-compiled readout function.

    Args:
        params: Readout parameters
        temporal_pooling: Pooling method
        chunk_size: Chunk size for chunk pooling

    Returns:
        A function: spikes -> logits
    """
    @jax.jit
    def readout_fn(spikes: Array) -> Array:
        return dense_readout(
            spikes,
            params,
            temporal_pooling=temporal_pooling,
            chunk_size=chunk_size,
        )

    return readout_fn
