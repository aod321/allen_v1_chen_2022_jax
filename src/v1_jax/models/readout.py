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
) -> Array:
    """Readout with chunk-wise temporal pooling.

    Computes logits for each time chunk independently.

    Args:
        spikes: Spike trains (time, batch, n_neurons)
        params: Readout parameters
        chunk_size: Duration of each chunk in timesteps

    Returns:
        Logits (batch, n_chunks, n_outputs)
    """
    return dense_readout(spikes, params, temporal_pooling='chunks', chunk_size=chunk_size)


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
    """

    def __init__(
        self,
        n_neurons: int,
        n_classes: int,
        temporal_pooling: str = 'chunks',
        chunk_size: int = 50,
        neuron_indices: Optional[Array] = None,
        key: Optional[Array] = None,
    ):
        """Initialize multi-class readout.

        Args:
            n_neurons: Number of input neurons
            n_classes: Number of classes
            temporal_pooling: Temporal pooling method
            chunk_size: Chunk size for chunk-wise pooling
            neuron_indices: Optional neuron indices
            key: Random key
        """
        self.n_classes = n_classes
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
            Logits for each class
        """
        return self.dense_readout(spikes)

    def probability(self, spikes: Array) -> Array:
        """Get class probabilities.

        Args:
            spikes: Spike trains

        Returns:
            Softmax probabilities
        """
        return jax.nn.softmax(self(spikes), axis=-1)

    def predict(self, spikes: Array) -> Array:
        """Get predicted class.

        Args:
            spikes: Spike trains

        Returns:
            Predicted class indices
        """
        return jnp.argmax(self(spikes), axis=-1)


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
