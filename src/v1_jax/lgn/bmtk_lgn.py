"""BMTK-style LGN model for preprocessing visual stimuli.

This module implements the BMTK (Brain Modeling Toolkit) style LGN preprocessing
model with support for multiple cell types and biologically accurate parameters.

Key differences from the simplified LGN:
1. Cell type-specific spontaneous firing rates from experimental data
2. Cosine bump temporal filters with weights, kpeaks, and delays
3. Two-subunit cells (sONsOFF, sONtOFF) with balanced amplitudes
4. Per-cell-type parameter lookup

Reference implementation: lgn_functions.py from BMTK simulations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .spatial_filter import (
    bilinear_select,
    create_gaussian_kernel_trimmed,
    gaussian_conv2d,
)
from .temporal_filter import transfer_function


# =============================================================================
# Cell Type Configuration
# =============================================================================

@dataclass
class CellTypeParams:
    """Parameters for a specific LGN cell type.

    Attributes:
        cell_type: Cell type name (sON, sOFF, tOFF)
        temporal_freq: Temporal frequency label (TF1, TF2, TF4, TF8, TF15)
        spontaneous_rate: Spontaneous firing rate from experimental data
        max_rate_on: Maximum ON response rate (for composite cells)
        max_rate_off: Maximum OFF response rate (for composite cells)
        amplitude: Base amplitude (1.0 for ON, -1.0 for OFF)
    """
    cell_type: str
    temporal_freq: str
    spontaneous_rate: float
    max_rate_on: float = 0.0
    max_rate_off: float = 0.0
    amplitude: float = 1.0


# Experimental spontaneous firing rates by cell type and temporal frequency
# Data from get_data_metrics_for_each_subclass() in lgnmodel
CELL_TYPE_PARAMS: Dict[str, CellTypeParams] = {
    # sON cells
    'sON_TF1': CellTypeParams('sON', 'TF1', spontaneous_rate=4.0, amplitude=1.0),
    'sON_TF2': CellTypeParams('sON', 'TF2', spontaneous_rate=4.0, amplitude=1.0),
    'sON_TF4': CellTypeParams('sON', 'TF4', spontaneous_rate=4.0, amplitude=1.0),
    'sON_TF8': CellTypeParams('sON', 'TF8', spontaneous_rate=4.0, amplitude=1.0),
    'sON_TF15': CellTypeParams('sON', 'TF15', spontaneous_rate=4.0, amplitude=1.0),

    # sOFF cells
    'sOFF_TF1': CellTypeParams('sOFF', 'TF1', spontaneous_rate=3.5, amplitude=-1.0),
    'sOFF_TF2': CellTypeParams('sOFF', 'TF2', spontaneous_rate=3.5, amplitude=-1.0),
    'sOFF_TF4': CellTypeParams('sOFF', 'TF4', spontaneous_rate=3.5, amplitude=-1.0),
    'sOFF_TF8': CellTypeParams('sOFF', 'TF8', spontaneous_rate=3.5, amplitude=-1.0),
    'sOFF_TF15': CellTypeParams('sOFF', 'TF15', spontaneous_rate=3.5, amplitude=-1.0),

    # tOFF cells
    'tOFF_TF1': CellTypeParams('tOFF', 'TF1', spontaneous_rate=5.0, amplitude=-1.0),
    'tOFF_TF2': CellTypeParams('tOFF', 'TF2', spontaneous_rate=5.0, amplitude=-1.0),
    'tOFF_TF4': CellTypeParams('tOFF', 'TF4', spontaneous_rate=5.0, amplitude=-1.0),
    'tOFF_TF8': CellTypeParams('tOFF', 'TF8', spontaneous_rate=5.0, amplitude=-1.0),
    'tOFF_TF15': CellTypeParams('tOFF', 'TF15', spontaneous_rate=5.0, amplitude=-1.0),

    # Two-subunit cells (composite ON-OFF)
    'sONsOFF_001': CellTypeParams(
        'sONsOFF', '001',
        spontaneous_rate=4.0,
        max_rate_on=21.0,
        max_rate_off=35.0,
        amplitude=1.0,
    ),
    'sONtOFF_001': CellTypeParams(
        'sONtOFF', '001',
        spontaneous_rate=5.5,
        max_rate_on=31.0,
        max_rate_off=46.0,
        amplitude=1.0,
    ),
}


# =============================================================================
# Cosine Bump Temporal Filter
# =============================================================================

def cosine_bump_kernel(
    weights: Tuple[float, float],
    kpeaks: Tuple[float, float],
    delays: Tuple[float, float],
    dt: float = 1.0,
    kernel_length: int = 250,
) -> np.ndarray:
    """Generate a cosine bump temporal filter kernel.

    The cosine bump filter is a sum of two cosine bumps with different
    weights, peak times, and delays.

    Args:
        weights: Tuple of (weight0, weight1) for two bumps
        kpeaks: Tuple of (kpeak0, kpeak1) - peak times (half-width of bump)
        delays: Tuple of (delay0, delay1) - delays for bumps (center time)
        dt: Time step in ms
        kernel_length: Length of kernel in timesteps

    Returns:
        Temporal kernel array of shape (kernel_length,)

    Reference:
        TemporalFilterCosineBump from lgnmodel/temporalfilter.py
    """
    t = np.arange(kernel_length) * dt

    kernel = np.zeros(kernel_length, dtype=np.float32)

    for w, k, d in zip(weights, kpeaks, delays):
        if np.isnan(w) or np.isnan(k) or np.isnan(d):
            continue
        # Skip if kpeak is zero or very small (would cause division by zero)
        if k <= 1e-6:
            continue
        # Cosine bump: w * (1 + cos(pi * (t - d) / k)) / 2 for |t - d| <= k
        # Zero otherwise
        t_shifted = t - d
        mask = np.abs(t_shifted) <= k
        bump = np.zeros_like(t)
        bump[mask] = w * (1 + np.cos(np.pi * t_shifted[mask] / k)) / 2
        kernel += bump

    return kernel.astype(np.float32)


def create_temporal_kernel_from_params(
    kpeaks_dom: Tuple[float, float],
    weights_dom: Tuple[float, float],
    delays_dom: Tuple[float, float],
    kpeaks_non_dom: Optional[Tuple[float, float]] = None,
    weights_non_dom: Optional[Tuple[float, float]] = None,
    delays_non_dom: Optional[Tuple[float, float]] = None,
    dt: float = 1.0,
    kernel_length: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create dominant and non-dominant temporal kernels.

    Args:
        kpeaks_dom: Peak times for dominant subunit
        weights_dom: Weights for dominant subunit
        delays_dom: Delays for dominant subunit
        kpeaks_non_dom: Peak times for non-dominant subunit (optional)
        weights_non_dom: Weights for non-dominant subunit (optional)
        delays_non_dom: Delays for non-dominant subunit (optional)
        dt: Time step in ms
        kernel_length: Length of kernel

    Returns:
        Tuple of (dom_kernel, non_dom_kernel)
    """
    dom_kernel = cosine_bump_kernel(weights_dom, kpeaks_dom, delays_dom, dt, kernel_length)

    if kpeaks_non_dom is not None and not np.any(np.isnan(kpeaks_non_dom)):
        non_dom_kernel = cosine_bump_kernel(
            weights_non_dom, kpeaks_non_dom, delays_non_dom, dt, kernel_length
        )
    else:
        non_dom_kernel = np.zeros(kernel_length, dtype=np.float32)

    return dom_kernel, non_dom_kernel


def compute_temporal_kernel_sum(kernel: np.ndarray) -> float:
    """Compute the sum (integral) of a temporal kernel.

    Used for amplitude balancing in two-subunit cells.

    Args:
        kernel: Temporal kernel array

    Returns:
        Sum of the kernel
    """
    return float(np.sum(kernel))


# =============================================================================
# BMTK LGN Parameters
# =============================================================================

@dataclass
class BMTKLGNParams:
    """BMTK-style LGN parameters with per-cell-type information.

    Attributes:
        n_neurons: Number of LGN neurons
        x: X positions of neurons
        y: Y positions of neurons
        non_dominant_x: X positions of non-dominant subunits
        non_dominant_y: Y positions of non-dominant subunits
        spatial_sizes: Receptive field sizes
        model_ids: Model ID strings for each neuron
        cell_type_indices: Indices into CELL_TYPE_PARAMS for each neuron
        dom_amplitude: Dominant subunit amplitudes
        non_dom_amplitude: Non-dominant subunit amplitudes
        spontaneous_rates: Spontaneous firing rates
        is_composite: Boolean mask for composite (two-subunit) cells
        dom_temporal_kernels: Dominant temporal kernels (n_neurons, kernel_length)
        non_dom_temporal_kernels: Non-dominant temporal kernels
    """
    n_neurons: int
    x: np.ndarray
    y: np.ndarray
    non_dominant_x: np.ndarray
    non_dominant_y: np.ndarray
    spatial_sizes: np.ndarray
    model_ids: List[str]
    dom_amplitude: np.ndarray
    non_dom_amplitude: np.ndarray
    spontaneous_rates: np.ndarray
    is_composite: np.ndarray
    dom_temporal_kernels: np.ndarray
    non_dom_temporal_kernels: np.ndarray


def load_bmtk_lgn_params(
    csv_path: str,
    movie_height: int = 120,
    movie_width: int = 240,
    kernel_length: int = 250,
    dt: float = 1.0,
) -> BMTKLGNParams:
    """Load BMTK-style LGN parameters from CSV file.

    This function loads the LGN cell data and computes:
    1. Per-cell-type spontaneous firing rates
    2. Cosine bump temporal kernels
    3. Balanced amplitudes for two-subunit cells

    Args:
        csv_path: Path to LGN CSV file (lgn_full_col_cells_3.csv)
        movie_height: Height of input movies
        movie_width: Width of input movies
        kernel_length: Length of temporal kernels
        dt: Time step in ms

    Returns:
        BMTKLGNParams object with all parameters
    """
    # Load CSV data
    df = pd.read_csv(csv_path, sep=r'\s+')
    n_neurons = len(df)

    # Extract basic parameters
    x = df['x'].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)
    spatial_sizes = df['spatial_size'].values.astype(np.float32)
    model_ids = df['model_id'].tolist()

    # Compute non-dominant positions (with spatial separation for composite cells)
    non_dominant_x = np.copy(x)
    non_dominant_y = np.copy(y)

    if 'tuning_angle' in df.columns and 'sf_sep' in df.columns:
        tuning_angles = df['tuning_angle'].values
        sf_sep = df['sf_sep'].values

        # For composite cells, compute non-dominant position
        valid_mask = ~np.isnan(tuning_angles) & ~np.isnan(sf_sep)
        angles_rad = np.deg2rad(tuning_angles[valid_mask])
        non_dominant_x[valid_mask] = x[valid_mask] + sf_sep[valid_mask] * np.cos(angles_rad)
        non_dominant_y[valid_mask] = y[valid_mask] + sf_sep[valid_mask] * np.sin(angles_rad)

    # Initialize arrays
    dom_amplitude = np.ones(n_neurons, dtype=np.float32)
    non_dom_amplitude = np.zeros(n_neurons, dtype=np.float32)
    spontaneous_rates = np.zeros(n_neurons, dtype=np.float32)
    is_composite = np.zeros(n_neurons, dtype=np.float32)
    dom_temporal_kernels = np.zeros((n_neurons, kernel_length), dtype=np.float32)
    non_dom_temporal_kernels = np.zeros((n_neurons, kernel_length), dtype=np.float32)

    # Process each neuron
    for i in range(n_neurons):
        model_id = model_ids[i]

        # Get cell type parameters
        if model_id in CELL_TYPE_PARAMS:
            cell_params = CELL_TYPE_PARAMS[model_id]
        else:
            # Try to infer from model_id pattern
            cell_params = CellTypeParams('unknown', 'unknown', spontaneous_rate=4.0)

        spontaneous_rates[i] = cell_params.spontaneous_rate

        # Extract temporal filter parameters
        kpeaks_dom = (df['kpeaks_dom_0'].iloc[i], df['kpeaks_dom_1'].iloc[i])
        weights_dom = (df['weight_dom_0'].iloc[i], df['weight_dom_1'].iloc[i])
        delays_dom = (df['delay_dom_0'].iloc[i], df['delay_dom_1'].iloc[i])

        kpeaks_non_dom = None
        weights_non_dom = None
        delays_non_dom = None

        if 'kpeaks_non_dom_0' in df.columns:
            kpeaks_non_dom = (df['kpeaks_non_dom_0'].iloc[i], df['kpeaks_non_dom_1'].iloc[i])
            weights_non_dom = (df['weight_non_dom_0'].iloc[i], df['weight_non_dom_1'].iloc[i])
            delays_non_dom = (df['delay_non_dom_0'].iloc[i], df['delay_non_dom_1'].iloc[i])

        # Generate temporal kernels
        dom_kernel, non_dom_kernel = create_temporal_kernel_from_params(
            kpeaks_dom, weights_dom, delays_dom,
            kpeaks_non_dom, weights_non_dom, delays_non_dom,
            dt, kernel_length,
        )

        dom_temporal_kernels[i] = dom_kernel
        non_dom_temporal_kernels[i] = non_dom_kernel

        # Handle different cell types
        if model_id == 'sONsOFF_001':
            is_composite[i] = 1.0

            # Compute balanced amplitudes
            spont = cell_params.spontaneous_rate
            max_ron = cell_params.max_rate_on
            max_roff = cell_params.max_rate_off

            sON_sum = compute_temporal_kernel_sum(non_dom_kernel)
            sOFF_sum = compute_temporal_kernel_sum(dom_kernel)

            amp_on = 1.0
            if sOFF_sum != 0 and max_ron != 0:
                amp_off = -(max_roff / max_ron) * (sON_sum / sOFF_sum) * amp_on
                amp_off -= (spont * (max_roff - max_ron)) / (max_ron * sOFF_sum)
            else:
                amp_off = -1.0

            dom_amplitude[i] = amp_off  # Dominant is OFF
            non_dom_amplitude[i] = amp_on  # Non-dominant is ON
            spontaneous_rates[i] = 0.5 * spont  # Half spont for each subunit

        elif model_id == 'sONtOFF_001':
            is_composite[i] = 1.0

            # Compute balanced amplitudes
            spont = cell_params.spontaneous_rate
            max_ron = cell_params.max_rate_on
            max_roff = cell_params.max_rate_off

            sON_sum = compute_temporal_kernel_sum(non_dom_kernel)
            tOFF_sum = compute_temporal_kernel_sum(dom_kernel)

            amp_on = 1.0
            if tOFF_sum != 0 and max_ron != 0:
                amp_off = -0.7 * (max_roff / max_ron) * (sON_sum / tOFF_sum) * amp_on
                amp_off -= (spont * (max_roff - max_ron)) / (max_ron * tOFF_sum)
            else:
                amp_off = -1.0

            dom_amplitude[i] = amp_off  # Dominant is tOFF
            non_dom_amplitude[i] = amp_on  # Non-dominant is sON
            spontaneous_rates[i] = 0.5 * spont  # Half spont for each subunit

        else:
            # Single-subunit cell
            dom_amplitude[i] = cell_params.amplitude
            non_dom_amplitude[i] = 0.0

    return BMTKLGNParams(
        n_neurons=n_neurons,
        x=x,
        y=y,
        non_dominant_x=non_dominant_x,
        non_dominant_y=non_dominant_y,
        spatial_sizes=spatial_sizes,
        model_ids=model_ids,
        dom_amplitude=dom_amplitude,
        non_dom_amplitude=non_dom_amplitude,
        spontaneous_rates=spontaneous_rates,
        is_composite=is_composite,
        dom_temporal_kernels=dom_temporal_kernels,
        non_dom_temporal_kernels=non_dom_temporal_kernels,
    )


# =============================================================================
# BMTK LGN Model
# =============================================================================

def bmtk_transfer_function(x: jnp.ndarray, spontaneous: jnp.ndarray) -> jnp.ndarray:
    """BMTK transfer function: Heaviside(x + spont) * (x + spont).

    This is equivalent to ReLU(x + spont), which is the standard LGN
    transfer function.

    Args:
        x: Input signal (can be negative)
        spontaneous: Spontaneous firing rate

    Returns:
        Rectified output
    """
    return jnp.maximum(x + spontaneous, 0.0)


def get_neuron_groups_by_size(spatial_sizes: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """Group neurons by spatial size for efficient spatial filtering.

    Args:
        spatial_sizes: Array of spatial sizes for each neuron

    Returns:
        List of (indices, sigma) tuples where indices are neurons with
        similar spatial sizes and sigma is the Gaussian sigma.
    """
    # Round to 1 decimal place for grouping
    size_groups = {}
    for i, size in enumerate(spatial_sizes):
        key = round(size, 1)
        if key not in size_groups:
            size_groups[key] = []
        size_groups[key].append(i)

    groups = []
    for size, indices in sorted(size_groups.items()):
        sigma = size / 3.0  # Convert spatial size to Gaussian sigma
        groups.append((np.array(indices), sigma))

    return groups


class BMTKLGN:
    """BMTK-style LGN preprocessing model.

    This model provides biologically accurate LGN preprocessing with:
    - Cell type-specific spontaneous firing rates
    - Cosine bump temporal filters
    - Two-subunit cells with balanced amplitudes

    Example:
        >>> lgn = BMTKLGN(csv_path='path/to/lgn_full_col_cells_3.csv')
        >>> movie = jnp.zeros((1000, 120, 240))  # T, H, W
        >>> firing_rates = lgn(movie)
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        params: Optional[BMTKLGNParams] = None,
        data_dir: Optional[str] = None,
        movie_height: int = 120,
        movie_width: int = 240,
        kernel_length: int = 250,
    ):
        """Initialize BMTK LGN model.

        Args:
            csv_path: Path to LGN CSV file
            params: Pre-loaded BMTKLGNParams (if provided, csv_path is ignored)
            data_dir: Data directory for file discovery
            movie_height: Height of input movies
            movie_width: Width of input movies
            kernel_length: Length of temporal kernels
        """
        import os

        if params is not None:
            self.params = params
        elif csv_path is not None:
            self.params = load_bmtk_lgn_params(
                csv_path, movie_height, movie_width, kernel_length
            )
        elif data_dir is not None:
            # Try to find the CSV file in data_dir
            candidate_paths = [
                os.path.join(data_dir, 'lgn_full_col_cells_3.csv'),
                os.path.join(data_dir, 'lgn', 'lgn_full_col_cells_3.csv'),
            ]
            for path in candidate_paths:
                if os.path.exists(path):
                    self.params = load_bmtk_lgn_params(
                        path, movie_height, movie_width, kernel_length
                    )
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find lgn_full_col_cells_3.csv in {data_dir}"
                )
        else:
            raise ValueError("Must provide either csv_path, params, or data_dir")

        self.movie_height = movie_height
        self.movie_width = movie_width

        # Convert to JAX arrays
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
        self.neuron_groups = get_neuron_groups_by_size(
            np.array(self.params.spatial_sizes)
        )

    @property
    def n_neurons(self) -> int:
        """Number of LGN neurons."""
        return self.params.n_neurons

    def spatial_response(
        self,
        movie: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute spatial responses for all neurons.

        Args:
            movie: Input movie of shape (T, H, W)

        Returns:
            Tuple of (dom_spatial, non_dom_spatial) each of shape (T, n_neurons)
        """
        T, H, W = movie.shape
        n_neurons = self.n_neurons

        all_dom_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)
        all_non_dom_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)

        for indices, sigma in self.neuron_groups:
            # Create and apply Gaussian kernel
            kernel = create_gaussian_kernel_trimmed(sigma)
            convolved = gaussian_conv2d(movie, kernel)

            # Sample at dominant positions
            x_sel = self.x[indices]
            y_sel = self.y[indices]
            dom_responses = bilinear_select(x_sel, y_sel, convolved)

            # Sample at non-dominant positions
            non_dom_x_sel = self.non_dominant_x[indices]
            non_dom_y_sel = self.non_dominant_y[indices]
            non_dom_responses = bilinear_select(non_dom_x_sel, non_dom_y_sel, convolved)

            all_dom_responses = all_dom_responses.at[:, indices].set(dom_responses)
            all_non_dom_responses = all_non_dom_responses.at[:, indices].set(non_dom_responses)

        return all_dom_responses, all_non_dom_responses

    def temporal_filter(
        self,
        spatial_responses: jnp.ndarray,
        temporal_kernels: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply temporal filtering using convolution.

        Args:
            spatial_responses: Shape (T, n_neurons)
            temporal_kernels: Shape (n_neurons, kernel_length)

        Returns:
            Filtered responses of shape (T, n_neurons)
        """
        T, n_neurons = spatial_responses.shape
        kernel_length = temporal_kernels.shape[1]

        # Causal padding
        padded = jnp.pad(
            spatial_responses,
            ((kernel_length - 1, 0), (0, 0)),
            mode='constant',
        )

        # Convolve each neuron with its kernel
        def convolve_single(spatial_resp, kernel):
            return jnp.convolve(spatial_resp, kernel[::-1], mode='valid')

        filtered = jax.vmap(convolve_single, in_axes=(1, 0), out_axes=1)(
            padded, temporal_kernels
        )

        return filtered

    def firing_rates_from_spatial(
        self,
        dom_spatial: jnp.ndarray,
        non_dom_spatial: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute firing rates from spatial responses.

        Args:
            dom_spatial: Dominant spatial responses (T, n_neurons)
            non_dom_spatial: Non-dominant spatial responses (T, n_neurons)

        Returns:
            Firing rates of shape (T, n_neurons)
        """
        # Apply temporal filtering
        dom_filtered = self.temporal_filter(dom_spatial, self.dom_temporal_kernels)
        non_dom_filtered = self.temporal_filter(non_dom_spatial, self.non_dom_temporal_kernels)

        # Single-subunit: ReLU(dom * amplitude + spont)
        single_rates = transfer_function(
            dom_filtered * self.dom_amplitude + self.spontaneous_rates
        )

        # Composite: ReLU(dom) + ReLU(non_dom)
        multi_rates = single_rates + transfer_function(
            non_dom_filtered * self.non_dom_amplitude + self.spontaneous_rates
        )

        # Combine based on cell type
        rates = single_rates * (1 - self.is_composite) + multi_rates * self.is_composite

        return rates

    def __call__(self, movie: jnp.ndarray) -> jnp.ndarray:
        """Compute LGN firing rates from movie.

        Args:
            movie: Input movie of shape (T, H, W)

        Returns:
            Firing rates of shape (T, n_neurons)
        """
        dom_spatial, non_dom_spatial = self.spatial_response(movie)
        return self.firing_rates_from_spatial(dom_spatial, non_dom_spatial)

    def get_cell_type_statistics(self) -> Dict[str, int]:
        """Get count of each cell type.

        Returns:
            Dictionary mapping model_id to count
        """
        from collections import Counter
        return dict(Counter(self.params.model_ids))


def create_bmtk_lgn_model(
    csv_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    movie_height: int = 120,
    movie_width: int = 240,
) -> BMTKLGN:
    """Create a BMTK LGN model with default settings.

    Args:
        csv_path: Path to LGN CSV file
        data_dir: Data directory containing LGN files
        movie_height: Height of input movies
        movie_width: Width of input movies

    Returns:
        Initialized BMTKLGN model
    """
    return BMTKLGN(
        csv_path=csv_path,
        data_dir=data_dir,
        movie_height=movie_height,
        movie_width=movie_width,
    )
