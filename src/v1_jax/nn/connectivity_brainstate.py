"""Connectivity and synaptic delay handling using brainstate + brainevent.

This module provides classes for managing sparse connectivity matrices
and synaptic delay buffers for GLIF3 networks.

Uses brainevent.CSR sparse format for event-driven spike propagation
(no autodiff needed since we use IODim for gradient computation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import jax.numpy as jnp
import brainstate

# Apply JAX compatibility patches BEFORE importing brainevent
from ..compat import apply_jax_compat_patches
apply_jax_compat_patches()

from brainevent import CSR

__all__ = ['Connection', 'SynapticDelayBuffer', 'build_connection_from_billeh']


@dataclass
class Connection:
    """Sparse connectivity representation using brainevent CSR format.

    This dataclass organizes connectivity matrices by synaptic properties
    to enable efficient event-driven spike propagation.

    Attributes
    ----------
    csr_matrices : Dict[Tuple[int, int], CSR]
        CSR matrices grouped by (delay_steps, receptor_type) combinations.
        Each matrix has shape (num_neurons, num_neurons).
    num_receptors : int
        Total number of receptor types in the network.
    max_delay : int
        Maximum synaptic delay in simulation steps.
    num_neurons : int
        Number of neurons in the network.
    input_csr_matrices : Dict[Tuple[int, int], CSR], optional
        CSR matrices for input connections (from LGN), grouped by
        (delay_steps, receptor_type). Shape: (num_neurons, num_inputs).
    """

    # Recurrent CSR matrices grouped by (delay, receptor_type)
    csr_matrices: Dict[Tuple[int, int], CSR]

    # Global properties
    num_receptors: int
    max_delay: int
    num_neurons: int

    # Input CSR matrices (optional, for LGN inputs)
    input_csr_matrices: Optional[Dict[Tuple[int, int], CSR]] = None
    num_inputs: int = 0


class SynapticDelayBuffer(brainstate.nn.Module):
    """Ring buffer for handling delayed synaptic inputs.

    This class implements a circular buffer to manage synaptic delays
    efficiently. The buffer maintains separate delay lines for each
    neuron and receptor type.

    Parameters
    ----------
    max_delay : int
        Maximum delay in simulation steps.
    num_neurons : int
        Number of target neurons.
    num_receptors : int
        Number of receptor types.
    dtype : jnp.dtype
        Data type for buffer (default: jnp.float32).

    Attributes
    ----------
    buffer : brainstate.ShortTermState
        Ring buffer with shape (buffer_size, num_receptors, num_neurons).
        buffer_size = max_delay + 1 to handle delays from 1 to max_delay.
    current_step : brainstate.ShortTermState
        Current time step counter for buffer indexing.
    """

    def __init__(
        self,
        max_delay: int,
        num_neurons: int,
        num_receptors: int,
        dtype=jnp.float32,
    ) -> None:
        """Initialize synaptic delay buffer."""
        super().__init__()
        self.max_delay = int(max_delay)
        self.num_neurons = int(num_neurons)
        self.num_receptors = int(num_receptors)
        # Buffer size needs to accommodate max_delay + 1
        self.buffer_size = self.max_delay + 1
        self.dtype = dtype

        self.buffer = brainstate.ShortTermState(
            jnp.zeros((self.buffer_size, self.num_receptors, self.num_neurons), dtype=dtype)
        )
        self.current_step = brainstate.ShortTermState(jnp.array(0, dtype=jnp.int32))

        self._zero_slice = jnp.zeros((self.num_receptors, self.num_neurons), dtype=dtype)

        # JIT compile internal methods for performance
        self._jit_advance_and_clear = brainstate.compile.jit(self._advance_and_clear_impl)
        self._jit_add_delayed = brainstate.compile.jit(self._add_delayed_impl)

    def get_current_synaptic_input(self) -> jnp.ndarray:
        """Get synaptic input for current time step.

        Returns
        -------
        jnp.ndarray
            Synaptic inputs, shape (num_receptors, num_neurons)
        """
        current_slot = self.current_step.value % self.buffer_size
        return self.buffer.value[current_slot]

    def advance_and_clear_current(self) -> None:
        """Advance time step and clear current slot for next use."""
        new_buffer, new_step = self._jit_advance_and_clear(
            self.buffer.value, self.current_step.value
        )
        self.buffer.value = new_buffer
        self.current_step.value = new_step

    def add_delayed_synaptic_input(
        self,
        delay_steps: int,
        receptor: int,
        synaptic_input: jnp.ndarray,
    ) -> None:
        """Add synaptic input to be delivered after a delay.

        Parameters
        ----------
        delay_steps : int
            Number of steps until delivery (1 to max_delay)
        receptor : int
            Receptor type index (0 to num_receptors-1)
        synaptic_input : jnp.ndarray
            Input values, shape (num_neurons,) or (batch, num_neurons)
        """
        if delay_steps <= 0 or delay_steps > self.max_delay:
            return

        new_buffer = self._jit_add_delayed(
            self.buffer.value, self.current_step.value,
            int(delay_steps), int(receptor), synaptic_input
        )
        self.buffer.value = new_buffer

    def reset(self) -> None:
        """Reset buffer to zero state."""
        self.buffer.value = jnp.zeros_like(self.buffer.value)
        self.current_step.value = jnp.array(0, dtype=jnp.int32)

    def _advance_and_clear_impl(
        self, buffer: jnp.ndarray, current_step: int
    ) -> Tuple[jnp.ndarray, int]:
        """Internal implementation for advance_and_clear_current."""
        current_slot = current_step % self.buffer_size
        new_buffer = buffer.at[current_slot].set(self._zero_slice)
        new_step = current_step + 1
        return new_buffer, new_step

    def _add_delayed_impl(
        self,
        buffer: jnp.ndarray,
        current_step: int,
        delay_steps: int,
        receptor: int,
        synaptic_input: jnp.ndarray,
    ) -> jnp.ndarray:
        """Internal implementation for add_delayed_synaptic_input."""
        target_slot = (current_step + delay_steps) % self.buffer_size
        new_buffer = buffer.at[target_slot, receptor, :].add(synaptic_input)
        return new_buffer


def nest_delay_round(delay_ms: float, resolution_ms: float) -> int:
    """Round delay to steps using NEST's 'round half up' method.

    Parameters
    ----------
    delay_ms : float or np.ndarray
        Delay in milliseconds
    resolution_ms : float
        Time step / resolution in milliseconds

    Returns
    -------
    int or np.ndarray
        Number of time steps (minimum 1)
    """
    steps_float = delay_ms / resolution_ms
    steps_float = np.round(steps_float, 10)  # Handle floating point precision
    steps = np.floor(steps_float + 0.5).astype(np.int32)
    steps = np.maximum(steps, 1)
    return steps


def build_connection_from_billeh(
    synapses: Dict,
    n_neurons: int,
    n_receptors: int,
    dt: float = 1.0,
    include_input: bool = False,
    input_data: Optional[Dict] = None,
    n_inputs: int = 0,
    dtype=jnp.float32,
) -> Connection:
    """Build Connection from Billeh network synapse data.

    Parameters
    ----------
    synapses : dict
        Synapse data with keys:
        - indices: (n_edges, 2) source-target pairs
        - weights: (n_edges,) synaptic weights
        - delays: (n_edges,) delays in ms
        - dense_shape: (n_post * n_receptors, n_pre)
    n_neurons : int
        Number of neurons
    n_receptors : int
        Number of receptor types
    dt : float
        Time step in ms
    include_input : bool
        Whether to include input (LGN) connections
    input_data : dict, optional
        Input connectivity data
    n_inputs : int
        Number of input neurons
    dtype : jnp.dtype
        Data type for matrices

    Returns
    -------
    Connection
        Connectivity structure with CSR matrices grouped by (delay, receptor)
    """
    indices = synapses['indices']
    weights = synapses['weights']
    delays = synapses['delays']

    # Convert delays to steps
    delay_steps = nest_delay_round(delays, dt)
    max_delay = int(np.max(delay_steps))

    # Extract receptor type from target index (target_idx = neuron_idx * n_receptors + receptor)
    target_indices = indices[:, 0]
    source_indices = indices[:, 1]

    target_neurons = target_indices // n_receptors
    receptor_types = target_indices % n_receptors

    # Group by (delay, receptor) and build CSR matrices
    csr_matrices = {}

    unique_delays = np.unique(delay_steps)
    unique_receptors = np.unique(receptor_types)

    for d in unique_delays:
        for r in unique_receptors:
            mask = (delay_steps == d) & (receptor_types == r)
            if not np.any(mask):
                continue

            row_indices = target_neurons[mask]
            col_indices = source_indices[mask]
            data = weights[mask].astype(np.float32 if dtype == jnp.float32 else np.float64)

            nnz = len(data)
            if nnz == 0:
                continue

            # Build CSR matrix using brainevent.CSR
            # CSR format: (data, indices, indptr)
            # Sort by row for CSR construction
            sort_idx = np.argsort(row_indices)
            sorted_rows = row_indices[sort_idx]
            sorted_cols = col_indices[sort_idx]
            sorted_data = data[sort_idx]

            # Compute indptr
            indptr = np.zeros(n_neurons + 1, dtype=np.int32)
            np.add.at(indptr[1:], sorted_rows, 1)
            indptr = np.cumsum(indptr)

            csr_matrices[(int(d), int(r))] = CSR(
                (jnp.asarray(sorted_data, dtype=dtype),
                 jnp.asarray(sorted_cols, dtype=jnp.int32),
                 jnp.asarray(indptr, dtype=jnp.int32)),
                shape=(n_neurons, n_neurons)
            )

    # Build input CSR matrices if requested
    input_csr_matrices = None
    if include_input and input_data is not None:
        input_csr_matrices = {}
        inp_indices = input_data['indices']
        inp_weights = input_data['weights']
        inp_delays = input_data.get('delays', np.ones(len(inp_weights)))

        inp_delay_steps = nest_delay_round(inp_delays, dt)

        inp_target_indices = inp_indices[:, 0]
        inp_source_indices = inp_indices[:, 1]

        inp_target_neurons = inp_target_indices // n_receptors
        inp_receptor_types = inp_target_indices % n_receptors

        inp_unique_delays = np.unique(inp_delay_steps)
        inp_unique_receptors = np.unique(inp_receptor_types)

        for d in inp_unique_delays:
            for r in inp_unique_receptors:
                mask = (inp_delay_steps == d) & (inp_receptor_types == r)
                if not np.any(mask):
                    continue

                row_indices = inp_target_neurons[mask]
                col_indices = inp_source_indices[mask]
                data = inp_weights[mask].astype(
                    np.float32 if dtype == jnp.float32 else np.float64
                )

                nnz = len(data)
                if nnz == 0:
                    continue

                # Sort by row for CSR construction
                sort_idx = np.argsort(row_indices)
                sorted_rows = row_indices[sort_idx]
                sorted_cols = col_indices[sort_idx]
                sorted_data = data[sort_idx]

                # Compute indptr
                indptr = np.zeros(n_neurons + 1, dtype=np.int32)
                np.add.at(indptr[1:], sorted_rows, 1)
                indptr = np.cumsum(indptr)

                input_csr_matrices[(int(d), int(r))] = CSR(
                    (jnp.asarray(sorted_data, dtype=dtype),
                     jnp.asarray(sorted_cols, dtype=jnp.int32),
                     jnp.asarray(indptr, dtype=jnp.int32)),
                    shape=(n_neurons, n_inputs)
                )

    return Connection(
        csr_matrices=csr_matrices,
        num_receptors=n_receptors,
        max_delay=max_delay,
        num_neurons=n_neurons,
        input_csr_matrices=input_csr_matrices,
        num_inputs=n_inputs,
    )
