"""LGN model for preprocessing visual stimuli.

This module implements the LGN (Lateral Geniculate Nucleus) preprocessing
model for converting visual stimuli into firing rates that serve as input
to the V1 cortical model.

The LGN model applies:
1. Spatial filtering (Gaussian convolution at neuron locations)
2. Temporal filtering (convolution with temporal kernels)
3. Transfer function (rectified linear)

Corresponding TF implementation: lgn_model/lgn.py:86-329
"""

import os
import pickle as pkl
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .spatial_filter import (
    SpatialFilter,
    bilinear_select,
    create_gaussian_kernel_trimmed,
    gaussian_conv2d,
)
from .temporal_filter import (
    TemporalFilter,
    compute_firing_rates,
    temporal_filter,
    transfer_function,
)


@dataclass
class LGNParams:
    """Parameters for LGN neurons.

    Attributes:
        x: x-coordinates of neurons (n_neurons,)
        y: y-coordinates of neurons (n_neurons,)
        non_dominant_x: x-coordinates for non-dominant subunit (n_neurons,)
        non_dominant_y: y-coordinates for non-dominant subunit (n_neurons,)
        spatial_sizes: Spatial filter sizes (n_neurons,)
        dom_amplitude: Amplitude for dominant subunit (n_neurons,)
        non_dom_amplitude: Amplitude for non-dominant subunit (n_neurons,)
        spontaneous_rates: Spontaneous firing rates (n_neurons,)
        is_composite: Boolean flags for ON-OFF composite cells (n_neurons,)
        dom_temporal_kernels: Temporal kernels for dominant subunit (n_neurons, kernel_length)
        non_dom_temporal_kernels: Temporal kernels for non-dominant subunit (n_neurons, kernel_length)
        model_id: Cell type identifiers (list of strings)
    """
    x: np.ndarray
    y: np.ndarray
    non_dominant_x: np.ndarray
    non_dominant_y: np.ndarray
    spatial_sizes: np.ndarray
    dom_amplitude: np.ndarray
    non_dom_amplitude: np.ndarray
    spontaneous_rates: np.ndarray
    is_composite: np.ndarray
    dom_temporal_kernels: np.ndarray
    non_dom_temporal_kernels: np.ndarray
    model_id: list

    @property
    def n_neurons(self) -> int:
        """Number of LGN neurons."""
        return len(self.x)

    @property
    def kernel_length(self) -> int:
        """Length of temporal kernels."""
        return self.dom_temporal_kernels.shape[1]


def load_lgn_params(
    lgn_data_path: str,
    cache_path: Optional[str] = None,
) -> LGNParams:
    """Load LGN parameters from CSV and cached temporal kernels.

    Args:
        lgn_data_path: Path to lgn_full_col_cells_3.csv
        cache_path: Path to cached temporal kernels (temporal_kernels.pkl)
                   If None, looks in same directory as lgn_data_path

    Returns:
        LGNParams containing all neuron parameters

    Note:
        Corresponding TF: LGN.__init__ (lgn.py:86-263)
    """
    # Load CSV data
    d = pd.read_csv(lgn_data_path, delimiter=' ')

    spatial_sizes = d['spatial_size'].to_numpy()
    model_id = d['model_id'].to_list()
    x = d['x'].to_numpy()
    y = d['y'].to_numpy()

    # Determine amplitude from cell type
    amplitude = np.array([1. if 'ON' in a else -1. for a in model_id])
    non_dom_amplitude = np.zeros_like(amplitude)
    is_composite = np.array(['ON' in a and 'OFF' in a for a in model_id]).astype(float)

    # Initialize non-dominant coordinates
    non_dominant_x = np.zeros_like(x)
    non_dominant_y = np.zeros_like(y)

    # Load temporal kernels from cache
    if cache_path is None:
        root_path = os.path.dirname(lgn_data_path)
        cache_path = os.path.join(root_path, 'temporal_kernels.pkl')

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Temporal kernels cache not found at {cache_path}. "
            "Please run the TensorFlow LGN model first to generate this cache."
        )

    with open(cache_path, 'rb') as f:
        cached = pkl.load(f)

    dom_temporal_kernels = cached['dom_temporal_kernels']
    non_dom_temporal_kernels = cached['non_dom_temporal_kernels']
    non_dominant_x = cached['non_dominant_x']
    non_dominant_y = cached['non_dominant_y']
    amplitude = cached['amplitude']
    non_dom_amplitude = cached['non_dom_amplitude']
    spontaneous_rates = cached['spontaneous_firing_rates']

    # Normalize coordinates to movie dimensions
    # Following TF: x = x * 239 / 240, y = y * 119 / 120
    x_normalized = x * 239 / 240
    y_normalized = y * 119 / 120

    # Clip to valid range
    x_normalized[np.floor(x_normalized) < 0] = 0.
    y_normalized[np.floor(y_normalized) < 0] = 0.

    non_dominant_x_normalized = non_dominant_x * 239 / 240
    non_dominant_y_normalized = non_dominant_y * 119 / 120
    non_dominant_x_normalized[np.floor(non_dominant_x_normalized) < 0] = 0.
    non_dominant_y_normalized[np.floor(non_dominant_y_normalized) < 0] = 0.
    non_dominant_x_normalized[np.ceil(non_dominant_x_normalized) >= 239.] = 239. - 1e-6
    non_dominant_y_normalized[np.ceil(non_dominant_y_normalized) >= 119.] = 119. - 1e-6

    return LGNParams(
        x=x_normalized.astype(np.float32),
        y=y_normalized.astype(np.float32),
        non_dominant_x=non_dominant_x_normalized.astype(np.float32),
        non_dominant_y=non_dominant_y_normalized.astype(np.float32),
        spatial_sizes=spatial_sizes.astype(np.float32),
        dom_amplitude=amplitude.astype(np.float32),
        non_dom_amplitude=non_dom_amplitude.astype(np.float32),
        spontaneous_rates=spontaneous_rates.astype(np.float32),
        is_composite=is_composite.astype(np.float32),
        dom_temporal_kernels=dom_temporal_kernels.astype(np.float32),
        non_dom_temporal_kernels=non_dom_temporal_kernels.astype(np.float32),
        model_id=model_id,
    )


class LGN:
    """LGN preprocessing model.

    Converts visual movies into LGN firing rates using spatial and
    temporal filtering.

    This is the main interface for LGN preprocessing, matching the
    TensorFlow implementation's LGN class.

    Attributes:
        params: LGN neuron parameters
        spatial_filter: Spatial filtering module
        temporal_filter: Temporal filtering module

    Example:
        >>> lgn = LGN(lgn_data_path='path/to/lgn_full_col_cells_3.csv')
        >>> movie = jnp.zeros((1000, 120, 240))  # T, H, W
        >>> spatial_dom, spatial_non_dom = lgn.spatial_response(movie)
        >>> firing_rates = lgn.firing_rates_from_spatial(spatial_dom, spatial_non_dom)
    """

    def __init__(
        self,
        lgn_data_path: Optional[str] = None,
        params: Optional[LGNParams] = None,
        movie_height: int = 120,
        movie_width: int = 240,
    ):
        """Initialize LGN model.

        Args:
            lgn_data_path: Path to LGN data CSV file. If None, uses default path.
            params: Pre-loaded LGNParams. If provided, lgn_data_path is ignored.
            movie_height: Height of input movies
            movie_width: Width of input movies

        Note:
            Either lgn_data_path or params must be provided.
        """
        if params is not None:
            self.params = params
        elif lgn_data_path is not None:
            self.params = load_lgn_params(lgn_data_path)
        else:
            # Use default path
            default_path = '/nvmessd/yinzi/lgn_full_col_cells_3.csv'
            self.params = load_lgn_params(default_path)

        self.movie_height = movie_height
        self.movie_width = movie_width

        # Store as JAX arrays for computation
        self.x = jnp.array(self.params.x)
        self.y = jnp.array(self.params.y)
        self.non_dominant_x = jnp.array(self.params.non_dominant_x)
        self.non_dominant_y = jnp.array(self.params.non_dominant_y)
        self.spatial_sizes = jnp.array(self.params.spatial_sizes)
        self.dom_amplitude = jnp.array(self.params.dom_amplitude)
        self.non_dom_amplitude = jnp.array(self.params.non_dom_amplitude)
        self.spontaneous_rates = jnp.array(self.params.spontaneous_rates)
        self.is_composite = jnp.array(self.params.is_composite)
        self.dom_temporal_kernels = jnp.array(self.params.dom_temporal_kernels)
        self.non_dom_temporal_kernels = jnp.array(self.params.non_dom_temporal_kernels)

        # Set up neuron groups for spatial filtering
        self._setup_neuron_groups()

    def _setup_neuron_groups(self):
        """Group neurons by spatial size for efficient filtering.

        Creates a list of (indices, sigma) tuples, one per spatial size bin.
        """
        spatial_sizes_np = np.array(self.params.spatial_sizes)
        spatial_range = np.arange(0, 15, 1.0)

        self.neuron_groups = []
        for i in range(len(spatial_range) - 1):
            low, high = spatial_range[i], spatial_range[i + 1]
            mask = (spatial_sizes_np >= low) & (spatial_sizes_np < high)
            indices = np.where(mask)[0]

            if len(indices) > 0:
                sigma = (low + high) / 2.0 / 3.0
                sigma = max(sigma, 0.5)
                self.neuron_groups.append((indices, sigma))

    @property
    def n_neurons(self) -> int:
        """Number of LGN neurons."""
        return self.params.n_neurons

    def spatial_response(
        self,
        movie: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute spatial responses for all neurons.

        Applies Gaussian spatial filtering and bilinear interpolation
        to extract responses at each neuron's location.

        Args:
            movie: Input movie of shape (T, H, W)

        Returns:
            Tuple of:
            - dom_spatial_responses: Dominant spatial responses (T, n_neurons)
            - non_dom_spatial_responses: Non-dominant spatial responses (T, n_neurons)

        Note:
            Corresponding TF: LGN.spatial_response (lgn.py:265-318)
        """
        T, H, W = movie.shape
        n_neurons = self.n_neurons

        all_dom_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)
        all_non_dom_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)
        neuron_order = []

        for indices, sigma in self.neuron_groups:
            # Create Gaussian kernel for this spatial size
            kernel = create_gaussian_kernel_trimmed(sigma)

            # Apply convolution
            convolved = gaussian_conv2d(movie, kernel)

            # Get dominant subunit responses
            x_sel = self.x[indices]
            y_sel = self.y[indices]
            dom_responses = bilinear_select(x_sel, y_sel, convolved)

            # Get non-dominant subunit responses
            non_dom_x_sel = self.non_dominant_x[indices]
            non_dom_y_sel = self.non_dominant_y[indices]
            non_dom_responses = bilinear_select(non_dom_x_sel, non_dom_y_sel, convolved)

            # Store results
            all_dom_responses = all_dom_responses.at[:, indices].set(dom_responses)
            all_non_dom_responses = all_non_dom_responses.at[:, indices].set(non_dom_responses)

            neuron_order.extend(indices.tolist())

        return all_dom_responses, all_non_dom_responses

    def firing_rates_from_spatial(
        self,
        dom_spatial_responses: jnp.ndarray,
        non_dom_spatial_responses: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute firing rates from spatial responses.

        Applies temporal filtering and combines dominant/non-dominant subunits.

        Args:
            dom_spatial_responses: Dominant spatial responses (T, n_neurons)
            non_dom_spatial_responses: Non-dominant spatial responses (T, n_neurons)

        Returns:
            Firing rates of shape (T, n_neurons)

        Note:
            Corresponding TF: LGN.firing_rates_from_spatial (lgn.py:320-329)
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

    def __call__(self, movie: jnp.ndarray) -> jnp.ndarray:
        """Compute LGN firing rates from movie.

        Convenience method that combines spatial_response and
        firing_rates_from_spatial.

        Args:
            movie: Input movie of shape (T, H, W)

        Returns:
            Firing rates of shape (T, n_neurons)
        """
        dom_spatial, non_dom_spatial = self.spatial_response(movie)
        return self.firing_rates_from_spatial(dom_spatial, non_dom_spatial)


def make_lgn_forward_fn(
    params: LGNParams,
    neuron_groups: list,
) -> callable:
    """Create a JIT-compilable LGN forward function.

    This creates a pure function that can be JIT compiled for efficiency.

    Args:
        params: LGN parameters
        neuron_groups: List of (indices, sigma) tuples

    Returns:
        A function that takes a movie and returns firing rates.
    """
    x = jnp.array(params.x)
    y = jnp.array(params.y)
    non_dominant_x = jnp.array(params.non_dominant_x)
    non_dominant_y = jnp.array(params.non_dominant_y)
    dom_amplitude = jnp.array(params.dom_amplitude)
    non_dom_amplitude = jnp.array(params.non_dom_amplitude)
    spontaneous_rates = jnp.array(params.spontaneous_rates)
    is_composite = jnp.array(params.is_composite)
    dom_temporal_kernels = jnp.array(params.dom_temporal_kernels)
    non_dom_temporal_kernels = jnp.array(params.non_dom_temporal_kernels)

    # Pre-compute kernels
    kernels = []
    for indices, sigma in neuron_groups:
        kernel = create_gaussian_kernel_trimmed(sigma)
        kernels.append((indices, kernel))

    def forward(movie: jnp.ndarray) -> jnp.ndarray:
        """Compute LGN firing rates from movie.

        Args:
            movie: Input movie of shape (T, H, W)

        Returns:
            Firing rates of shape (T, n_neurons)
        """
        T = movie.shape[0]
        n_neurons = params.n_neurons

        all_dom_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)
        all_non_dom_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)

        for indices, kernel in kernels:
            convolved = gaussian_conv2d(movie, kernel)

            x_sel = x[indices]
            y_sel = y[indices]
            dom_responses = bilinear_select(x_sel, y_sel, convolved)

            non_dom_x_sel = non_dominant_x[indices]
            non_dom_y_sel = non_dominant_y[indices]
            non_dom_responses = bilinear_select(non_dom_x_sel, non_dom_y_sel, convolved)

            all_dom_responses = all_dom_responses.at[:, indices].set(dom_responses)
            all_non_dom_responses = all_non_dom_responses.at[:, indices].set(non_dom_responses)

        # Apply temporal filtering and compute firing rates
        return compute_firing_rates(
            all_dom_responses,
            all_non_dom_responses,
            dom_temporal_kernels,
            non_dom_temporal_kernels,
            dom_amplitude,
            non_dom_amplitude,
            spontaneous_rates,
            is_composite,
        )

    return forward


# Convenience function for quick usage
def create_lgn_model(
    lgn_data_path: Optional[str] = None,
    movie_height: int = 120,
    movie_width: int = 240,
) -> LGN:
    """Create an LGN model with default settings.

    Args:
        lgn_data_path: Path to LGN data CSV. If None, uses default path.
        movie_height: Height of input movies
        movie_width: Width of input movies

    Returns:
        Initialized LGN model
    """
    return LGN(
        lgn_data_path=lgn_data_path,
        movie_height=movie_height,
        movie_width=movie_width,
    )
