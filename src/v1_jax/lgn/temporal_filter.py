"""Temporal filtering for LGN model.

This module implements temporal filtering operations for the LGN preprocessing,
including depthwise convolution with temporal kernels.

Corresponding TF implementation: lgn_model/lgn.py:45-53, 320-329
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


def temporal_filter(
    spatial_responses: jnp.ndarray,
    temporal_kernels: jnp.ndarray,
) -> jnp.ndarray:
    """Apply temporal filtering using depthwise convolution.

    Convolves spatial responses with neuron-specific temporal kernels.
    This is a causal filter (only uses past values).

    Args:
        spatial_responses: Spatial responses of shape (T, n_neurons)
        temporal_kernels: Temporal kernels of shape (n_neurons, kernel_length)

    Returns:
        Filtered responses of shape (T, n_neurons)

    Note:
        Corresponding TF: temporal_filter function (lgn.py:45-53)

        TF implementation:
        - Pads input: (1, T, 1, n) -> (1, T + kernel_length - 1, 1, n)
        - Uses depthwise_conv2d with kernel shape (kernel_length, 1, n, 1)
        - Output: (1, T, 1, n) -> (T, n)
    """
    T, n_neurons = spatial_responses.shape
    kernel_length = temporal_kernels.shape[1]

    # Pad temporal dimension (causal padding at the beginning)
    # Shape: (T, n_neurons) -> (T + kernel_length - 1, n_neurons)
    padded = jnp.pad(
        spatial_responses,
        ((kernel_length - 1, 0), (0, 0)),
        mode='constant',
        constant_values=0.0,
    )

    # Reshape for conv_general_dilated
    # We'll use lax.conv_general_dilated_local (depthwise)
    # But JAX doesn't have a direct depthwise 1D conv with per-channel kernels
    # We'll implement this efficiently using vmap

    # Transpose kernels: (n_neurons, kernel_length) -> (kernel_length, n_neurons)
    # and reverse for convolution (temporal kernels are already reversed in TF impl)
    kernel_transposed = jnp.transpose(temporal_kernels)  # (kernel_length, n_neurons)

    # Implement depthwise convolution using scan for efficiency
    # Alternative: use einsum or manual loop

    # For each time step, compute convolution output
    def convolve_single_neuron(spatial_resp, kernel):
        """Convolve a single neuron's response with its kernel."""
        # spatial_resp: (T + kernel_length - 1,)
        # kernel: (kernel_length,)
        # Output: (T,)
        return jnp.convolve(spatial_resp, kernel[::-1], mode='valid')

    # Use vmap over neurons
    filtered = jax.vmap(convolve_single_neuron, in_axes=(1, 0), out_axes=1)(
        padded, temporal_kernels
    )

    return filtered


def temporal_filter_scan(
    spatial_responses: jnp.ndarray,
    temporal_kernels: jnp.ndarray,
) -> jnp.ndarray:
    """Apply temporal filtering using lax.scan (more memory efficient).

    This implementation uses scan instead of full convolution,
    which can be more memory efficient for very long sequences.

    Args:
        spatial_responses: Spatial responses of shape (T, n_neurons)
        temporal_kernels: Temporal kernels of shape (n_neurons, kernel_length)

    Returns:
        Filtered responses of shape (T, n_neurons)
    """
    T, n_neurons = spatial_responses.shape
    kernel_length = temporal_kernels.shape[1]

    # Pad at the beginning for causal filtering
    padded = jnp.pad(
        spatial_responses,
        ((kernel_length - 1, 0), (0, 0)),
        mode='constant',
        constant_values=0.0,
    )

    # Extract sliding windows and compute dot product
    def step_fn(carry, t):
        # Window from t to t + kernel_length
        window = lax.dynamic_slice(
            padded,
            (t, 0),
            (kernel_length, n_neurons)
        )
        # Dot product with temporal kernels (element-wise then sum over time)
        # window: (kernel_length, n_neurons)
        # temporal_kernels: (n_neurons, kernel_length) -> transposed: (kernel_length, n_neurons)
        output = jnp.sum(window * jnp.transpose(temporal_kernels), axis=0)
        return carry, output

    _, filtered = lax.scan(step_fn, None, jnp.arange(T))

    return filtered


def temporal_filter_fft(
    spatial_responses: jnp.ndarray,
    temporal_kernels: jnp.ndarray,
) -> jnp.ndarray:
    """Apply temporal filtering using FFT (fast for long sequences).

    For very long temporal sequences, FFT-based convolution can be faster.

    Args:
        spatial_responses: Spatial responses of shape (T, n_neurons)
        temporal_kernels: Temporal kernels of shape (n_neurons, kernel_length)

    Returns:
        Filtered responses of shape (T, n_neurons)
    """
    T, n_neurons = spatial_responses.shape
    kernel_length = temporal_kernels.shape[1]

    # FFT size (power of 2 for efficiency)
    fft_size = 1
    while fft_size < T + kernel_length - 1:
        fft_size *= 2

    # FFT of input (zero-padded)
    input_fft = jnp.fft.rfft(spatial_responses, n=fft_size, axis=0)

    # FFT of kernels (reversed for convolution)
    # Note: temporal_kernels are already stored in the correct order
    kernel_padded = jnp.zeros((fft_size, n_neurons), dtype=jnp.float32)
    kernel_padded = kernel_padded.at[:kernel_length, :].set(jnp.transpose(temporal_kernels))
    kernel_fft = jnp.fft.rfft(kernel_padded, axis=0)

    # Multiply in frequency domain
    result_fft = input_fft * kernel_fft

    # Inverse FFT
    result = jnp.fft.irfft(result_fft, n=fft_size, axis=0)

    # Extract valid region (causal: take last T samples starting from kernel_length-1)
    return result[kernel_length - 1:kernel_length - 1 + T, :]


def transfer_function(x: jnp.ndarray) -> jnp.ndarray:
    """LGN transfer function (rectified linear).

    Args:
        x: Input firing rate (can be negative due to subtraction)

    Returns:
        Rectified firing rate (non-negative)

    Note:
        Corresponding TF: transfer_function (lgn.py:56-58)
    """
    return jnp.maximum(x, 0.0)


def compute_firing_rates(
    dom_spatial_responses: jnp.ndarray,
    non_dom_spatial_responses: jnp.ndarray,
    dom_temporal_kernels: jnp.ndarray,
    non_dom_temporal_kernels: jnp.ndarray,
    dom_amplitude: jnp.ndarray,
    non_dom_amplitude: jnp.ndarray,
    spontaneous_rates: jnp.ndarray,
    is_composite: jnp.ndarray,
) -> jnp.ndarray:
    """Compute LGN firing rates from spatial responses.

    Applies temporal filtering and combines dominant/non-dominant subunits.

    Args:
        dom_spatial_responses: Dominant spatial responses (T, n_neurons)
        non_dom_spatial_responses: Non-dominant spatial responses (T, n_neurons)
        dom_temporal_kernels: Dominant temporal kernels (n_neurons, kernel_length)
        non_dom_temporal_kernels: Non-dominant temporal kernels (n_neurons, kernel_length)
        dom_amplitude: Amplitude for dominant subunit (n_neurons,)
        non_dom_amplitude: Amplitude for non-dominant subunit (n_neurons,)
        spontaneous_rates: Spontaneous firing rates (n_neurons,)
        is_composite: Whether neuron is composite ON-OFF type (n_neurons,)

    Returns:
        Firing rates of shape (T, n_neurons)

    Note:
        Corresponding TF: LGN.firing_rates_from_spatial (lgn.py:320-329)
    """
    # Apply temporal filtering
    dom_filtered = temporal_filter(dom_spatial_responses, dom_temporal_kernels)
    non_dom_filtered = temporal_filter(non_dom_spatial_responses, non_dom_temporal_kernels)

    # Compute firing rates for single-subunit cells
    # firing_rates = ReLU(dom_filtered * amplitude + spontaneous)
    single_rates = transfer_function(
        dom_filtered * dom_amplitude + spontaneous_rates
    )

    # Compute firing rates for composite (ON-OFF) cells
    # multi_rates = ReLU(dom) + ReLU(non_dom)
    multi_rates = single_rates + transfer_function(
        non_dom_filtered * non_dom_amplitude + spontaneous_rates
    )

    # Combine based on cell type
    firing_rates = single_rates * (1 - is_composite) + multi_rates * is_composite

    return firing_rates


class TemporalFilter:
    """Temporal filter for LGN preprocessing.

    Handles temporal convolution and firing rate computation.

    Attributes:
        dom_temporal_kernels: Temporal kernels for dominant subunit
        non_dom_temporal_kernels: Temporal kernels for non-dominant subunit
        dom_amplitude: Amplitude for dominant subunit
        non_dom_amplitude: Amplitude for non-dominant subunit
        spontaneous_rates: Spontaneous firing rates
        is_composite: Whether each neuron is composite type
    """

    def __init__(
        self,
        dom_temporal_kernels: np.ndarray,
        non_dom_temporal_kernels: np.ndarray,
        dom_amplitude: np.ndarray,
        non_dom_amplitude: np.ndarray,
        spontaneous_rates: np.ndarray,
        is_composite: np.ndarray,
    ):
        """Initialize temporal filter.

        Args:
            dom_temporal_kernels: Shape (n_neurons, kernel_length)
            non_dom_temporal_kernels: Shape (n_neurons, kernel_length)
            dom_amplitude: Shape (n_neurons,)
            non_dom_amplitude: Shape (n_neurons,)
            spontaneous_rates: Shape (n_neurons,)
            is_composite: Shape (n_neurons,)
        """
        self.dom_temporal_kernels = jnp.array(dom_temporal_kernels)
        self.non_dom_temporal_kernels = jnp.array(non_dom_temporal_kernels)
        self.dom_amplitude = jnp.array(dom_amplitude)
        self.non_dom_amplitude = jnp.array(non_dom_amplitude)
        self.spontaneous_rates = jnp.array(spontaneous_rates)
        self.is_composite = jnp.array(is_composite)

    def __call__(
        self,
        dom_spatial_responses: jnp.ndarray,
        non_dom_spatial_responses: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute firing rates from spatial responses.

        Args:
            dom_spatial_responses: Dominant spatial responses (T, n_neurons)
            non_dom_spatial_responses: Non-dominant spatial responses (T, n_neurons)

        Returns:
            Firing rates of shape (T, n_neurons)
        """
        return compute_firing_rates(
            dom_spatial_responses,
            non_dom_spatial_responses,
            self.dom_temporal_kernels,
            self.non_dom_temporal_kernels,
            self.dom_amplitude,
            self.non_dom_amplitude,
            self.spontaneous_rates,
            self.is_composite,
        )

    def filter_dominant_only(
        self,
        dom_spatial_responses: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply temporal filtering to dominant subunit only.

        Useful for debugging or simpler cell types.

        Args:
            dom_spatial_responses: Dominant spatial responses (T, n_neurons)

        Returns:
            Filtered responses (T, n_neurons)
        """
        return temporal_filter(dom_spatial_responses, self.dom_temporal_kernels)
