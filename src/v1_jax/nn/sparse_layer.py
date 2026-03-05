"""Sparse layer implementations for JAX using BCOO format.

This module provides JAX-compatible sparse matrix operations for input
and recurrent connectivity in the V1 model.

Reference: models.py:50-99 (SparseLayer)
"""

from __future__ import annotations

from typing import Tuple, Optional, NamedTuple, Dict, Any
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse

import numpy as np


class SparseConnectivity(NamedTuple):
    """Sparse connectivity representation using BCOO format.

    Attributes:
        indices: COO indices, shape (nnz, 2) where each row is (target, source)
        weights: Connection weights, shape (nnz,)
        shape: Dense matrix shape (n_targets, n_sources)
    """
    indices: Array  # (nnz, 2)
    weights: Array  # (nnz,)
    shape: Tuple[int, int]

    def to_bcoo(self) -> sparse.BCOO:
        """Convert to JAX BCOO sparse matrix."""
        return sparse.BCOO(
            (self.weights, self.indices),
            shape=self.shape,
        )

    @staticmethod
    def from_arrays(
        indices: Array,
        weights: Array,
        shape: Tuple[int, int],
    ) -> 'SparseConnectivity':
        """Create SparseConnectivity from arrays.

        Args:
            indices: COO indices (nnz, 2)
            weights: Weights (nnz,)
            shape: Dense shape

        Returns:
            SparseConnectivity instance
        """
        return SparseConnectivity(
            indices=jnp.asarray(indices),
            weights=jnp.asarray(weights),
            shape=shape,
        )


def sparse_matmul_bcoo(
    connectivity: SparseConnectivity,
    x: Array,
    transpose_x: bool = True,
) -> Array:
    """Sparse matrix-vector multiplication using BCOO.

    Computes: y = W @ x (or y = W @ x.T if transpose_x=True)

    Args:
        connectivity: Sparse connectivity
        x: Input tensor (batch, n_sources) or (n_sources, batch)
        transpose_x: Whether to transpose x before multiplication

    Returns:
        Output tensor (n_targets, batch) or (batch, n_targets)
    """
    bcoo = connectivity.to_bcoo()

    if transpose_x:
        # x: (batch, n_sources) -> x.T: (n_sources, batch)
        # result: (n_targets, batch)
        result = bcoo @ x.T
        return result.T  # (batch, n_targets)
    else:
        return bcoo @ x


def sparse_input_layer(
    input_connectivity: SparseConnectivity,
    inputs: Array,
    bkg_weights: Optional[Array] = None,
    key: Optional[Array] = None,
    use_decoded_noise: bool = False,
    noise_data: Optional[Array] = None,
    noise_scale: Tuple[float, float] = (1.0, 1.0),
) -> Array:
    """Apply sparse input layer transformation.

    Note: This function is NOT JIT-compiled directly. Use InputLayer class
    for a JIT-compatible wrapper.

    This implements the SparseLayer from TensorFlow:
    1. Sparse matmul: W_in @ inputs
    2. Add background noise (rest-of-brain activity)

    Args:
        input_connectivity: Sparse input weights
        inputs: Input tensor (batch, seq_len, n_inputs)
        bkg_weights: Background weights (n_neurons * n_receptors,)
        key: Random key for noise generation
        use_decoded_noise: Whether to use decoded noise (from file)
        noise_data: Pre-loaded noise data for decoded noise mode
        noise_scale: (quick_scale, slow_scale) for decoded noise

    Returns:
        Input current (batch, seq_len, n_neurons * n_receptors)
    """
    batch_size = inputs.shape[0]
    seq_len = inputs.shape[1]
    n_inputs = inputs.shape[2]
    n_targets = input_connectivity.shape[0]

    # Reshape for sparse matmul: (batch * seq_len, n_inputs)
    flat_inputs = inputs.reshape(batch_size * seq_len, n_inputs)

    # Sparse matmul
    bcoo = input_connectivity.to_bcoo()
    input_current = (bcoo @ flat_inputs.T).T  # (batch * seq_len, n_targets)
    input_current = input_current.astype(jnp.float32)

    # Reshape back
    input_current = input_current.reshape(batch_size, seq_len, n_targets)

    # Add background noise
    if bkg_weights is not None:
        if use_decoded_noise and noise_data is not None and key is not None:
            # Use decoded noise from file
            key1, key2 = jax.random.split(key)

            # Quick noise: sample every step
            quick_idx = jax.random.randint(
                key1, (batch_size, seq_len, n_targets),
                minval=0, maxval=noise_data.shape[0]
            )
            quick_noise = noise_data[quick_idx]

            # Slow noise: sample per trial, tile over sequence
            slow_idx = jax.random.randint(
                key2, (batch_size, 1, n_targets),
                minval=0, maxval=noise_data.shape[0]
            )
            slow_idx = jnp.tile(slow_idx, (1, seq_len, 1))
            slow_noise = noise_data[slow_idx]

            noise_input = (
                noise_scale[0] * quick_noise +
                noise_scale[1] * slow_noise
            )
        else:
            # Simple Poisson-like background
            if key is None:
                key = jax.random.PRNGKey(0)

            rest_of_brain = jnp.sum(
                (jax.random.uniform(key, (batch_size, seq_len, 10)) < 0.1).astype(jnp.float32),
                axis=-1
            )
            noise_input = bkg_weights[None, None, :] * rest_of_brain[..., None] / 10.0

        input_current = input_current + noise_input

    return input_current


def create_recurrent_matmul_fn(
    connectivity: SparseConnectivity,
    n_neurons: int,
    max_delay: int,
) -> callable:
    """Create a JIT-compiled recurrent matmul function.

    The recurrent connectivity matrix has shape:
    (n_neurons * n_receptors, n_neurons * max_delay)

    This accounts for different delays by indexing into the spike buffer
    at different offsets.

    Args:
        connectivity: Sparse recurrent connectivity with delays encoded
        n_neurons: Number of neurons
        max_delay: Maximum synaptic delay

    Returns:
        Function: (z_buf) -> recurrent_current
        where z_buf has shape (batch, n_neurons * max_delay)
        and output has shape (batch, n_neurons * n_receptors)
    """
    bcoo = connectivity.to_bcoo()

    @jax.jit
    def recurrent_matmul(z_buf: Array) -> Array:
        """Compute recurrent current from spike buffer.

        Args:
            z_buf: Spike buffer (batch, n_neurons * max_delay)

        Returns:
            Recurrent current (batch, n_neurons * n_receptors)
        """
        # Sparse matmul: W_rec @ z_buf.T
        result = (bcoo @ z_buf.T).T
        return result.astype(jnp.float32)

    return recurrent_matmul


class InputLayer:
    """Input layer with sparse connectivity and background noise.

    This class wraps the sparse input connectivity and provides a
    callable interface for processing external inputs.

    Attributes:
        connectivity: Sparse input connectivity
        bkg_weights: Background noise weights
        n_neurons: Number of target neurons
        n_receptors: Number of receptor types
        n_inputs: Number of input sources
    """

    def __init__(
        self,
        indices: Array,
        weights: Array,
        dense_shape: Tuple[int, int],
        bkg_weights: Array,
        use_decoded_noise: bool = False,
        noise_data: Optional[Array] = None,
        noise_scale: Tuple[float, float] = (1.0, 1.0),
    ):
        """Initialize input layer.

        Args:
            indices: COO indices (nnz, 2) where each row is (target, source)
            weights: Connection weights (nnz,)
            dense_shape: (n_neurons * n_receptors, n_inputs)
            bkg_weights: Background weights (n_neurons * n_receptors,)
            use_decoded_noise: Whether to use decoded noise
            noise_data: Pre-loaded noise data
            noise_scale: Noise scaling factors
        """
        self.connectivity = SparseConnectivity(
            indices=jnp.asarray(indices),
            weights=jnp.asarray(weights, dtype=jnp.float32),
            shape=dense_shape,
        )
        self.bkg_weights = jnp.asarray(bkg_weights, dtype=jnp.float32)
        self.n_targets = dense_shape[0]
        self.n_inputs = dense_shape[1]
        self.use_decoded_noise = use_decoded_noise
        self.noise_data = noise_data
        self.noise_scale = noise_scale

    def __call__(
        self,
        inputs: Array,
        key: Optional[Array] = None,
    ) -> Array:
        """Process inputs through sparse layer.

        Args:
            inputs: Input tensor (batch, seq_len, n_inputs)
            key: Random key for noise

        Returns:
            Input current (batch, seq_len, n_targets)
        """
        return sparse_input_layer(
            self.connectivity,
            inputs,
            self.bkg_weights,
            key,
            self.use_decoded_noise,
            self.noise_data,
            self.noise_scale,
        )


class RecurrentLayer:
    """Recurrent layer with sparse connectivity and delays.

    Handles the recurrent connections with synaptic delays.
    Delays are encoded in the connectivity matrix by indexing
    into the spike buffer at different offsets.

    Attributes:
        connectivity: Sparse recurrent connectivity
        n_neurons: Number of neurons
        n_receptors: Number of receptor types
        max_delay: Maximum synaptic delay
    """

    def __init__(
        self,
        indices: Array,
        weights: Array,
        dense_shape: Tuple[int, int],
        n_neurons: int,
        n_receptors: int,
        max_delay: int,
    ):
        """Initialize recurrent layer.

        Args:
            indices: COO indices (nnz, 2) with delays encoded in source index
            weights: Connection weights (nnz,)
            dense_shape: (n_neurons * n_receptors, n_neurons * max_delay)
            n_neurons: Number of neurons
            n_receptors: Number of receptor types
            max_delay: Maximum synaptic delay
        """
        self.connectivity = SparseConnectivity(
            indices=jnp.asarray(indices),
            weights=jnp.asarray(weights, dtype=jnp.float32),
            shape=dense_shape,
        )
        self.n_neurons = n_neurons
        self.n_receptors = n_receptors
        self.max_delay = max_delay

        # Pre-compile the matmul function
        self._matmul_fn = create_recurrent_matmul_fn(
            self.connectivity, n_neurons, max_delay
        )

    def __call__(self, z_buf: Array) -> Array:
        """Compute recurrent current from spike buffer.

        Args:
            z_buf: Spike buffer (batch, n_neurons * max_delay)

        Returns:
            Recurrent current (batch, n_neurons * n_receptors)
        """
        return self._matmul_fn(z_buf)


def prepare_recurrent_connectivity(
    indices: np.ndarray,
    weights: np.ndarray,
    delays: np.ndarray,
    n_neurons: int,
    n_receptors: int,
    max_delay: int,
    dt: float = 1.0,
    voltage_scale: Optional[np.ndarray] = None,
    node_type_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Prepare recurrent connectivity with delay encoding.

    Converts raw connectivity to the format used by RecurrentLayer:
    - Target indices encode receptor type: target_neuron * n_receptors + receptor
    - Source indices encode delay: source_neuron + n_neurons * (delay - 1)

    Args:
        indices: Raw COO indices (nnz, 2) with (target, source)
        weights: Raw weights (nnz,)
        delays: Synaptic delays in ms (nnz,) or scalar
        n_neurons: Number of neurons
        n_receptors: Number of receptor types
        max_delay: Maximum delay in time steps
        dt: Time step in ms
        voltage_scale: Voltage normalization per neuron type
        node_type_ids: Neuron type indices for voltage scaling

    Returns:
        Tuple of (indices, weights, dense_shape) for RecurrentLayer
    """
    # Clip and convert delays to time steps
    delays_steps = np.clip(
        np.round(delays / dt).astype(np.int32),
        1, max_delay
    )

    # Encode delays in source index
    # Original source index is in indices[:, 1]
    # New source index = original_source + n_neurons * (delay - 1)
    new_indices = indices.copy()
    new_indices[:, 1] = indices[:, 1] + n_neurons * (delays_steps - 1)

    # Scale weights by voltage if provided
    new_weights = weights.copy().astype(np.float32)
    if voltage_scale is not None and node_type_ids is not None:
        # Target neuron index (without receptor encoding)
        target_neurons = indices[:, 0] // n_receptors
        target_types = node_type_ids[target_neurons]
        new_weights = new_weights / voltage_scale[target_types]

    dense_shape = (n_receptors * n_neurons, n_neurons * max_delay)

    return new_indices, new_weights, dense_shape


def prepare_input_connectivity(
    indices: np.ndarray,
    weights: np.ndarray,
    n_neurons: int,
    n_receptors: int,
    n_inputs: int,
    voltage_scale: Optional[np.ndarray] = None,
    node_type_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Prepare input connectivity.

    Args:
        indices: COO indices (nnz, 2) with (target, source)
        weights: Connection weights (nnz,)
        n_neurons: Number of neurons
        n_receptors: Number of receptor types
        n_inputs: Number of input sources
        voltage_scale: Voltage normalization per neuron type
        node_type_ids: Neuron type indices

    Returns:
        Tuple of (indices, weights, dense_shape) for InputLayer
    """
    new_weights = weights.copy().astype(np.float32)

    if voltage_scale is not None and node_type_ids is not None:
        target_neurons = indices[:, 0] // n_receptors
        target_types = node_type_ids[target_neurons]
        new_weights = new_weights / voltage_scale[target_types]

    dense_shape = (n_receptors * n_neurons, n_inputs)

    return indices.copy(), new_weights, dense_shape
