"""LGN parameter loading utilities.

This module provides flexible loading of LGN parameters from various sources,
including CSV files and cached pickle files.

Reference: TensorFlow implementation lgn_model/lgn.py
"""

import os
import pickle as pkl
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


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


def find_lgn_files(
    data_dir: str,
    lgn_csv_name: str = 'lgn_full_col_cells_3.csv',
    temporal_cache_name: str = 'temporal_kernels.pkl',
) -> Tuple[str, str]:
    """Find LGN data files in the given directory or parent.

    Args:
        data_dir: Base data directory (e.g., GLIF_network path)
        lgn_csv_name: Name of the LGN parameters CSV file
        temporal_cache_name: Name of the temporal kernels cache file

    Returns:
        Tuple of (lgn_csv_path, temporal_cache_path)

    Raises:
        FileNotFoundError: If required files cannot be found
    """
    # Search paths in order of priority
    search_paths = [
        data_dir,
        os.path.dirname(data_dir),  # Parent directory
        os.path.join(data_dir, 'lgn_model'),
        '/nvmessd/yinzi',  # Default location
    ]

    lgn_csv_path = None
    for path in search_paths:
        candidate = os.path.join(path, lgn_csv_name)
        if os.path.exists(candidate):
            lgn_csv_path = candidate
            break

    if lgn_csv_path is None:
        raise FileNotFoundError(
            f"LGN CSV file '{lgn_csv_name}' not found in {search_paths}"
        )

    # Find temporal kernels cache
    temporal_cache_path = None
    cache_search_paths = [
        os.path.dirname(lgn_csv_path),
        data_dir,
        os.path.join(os.path.dirname(data_dir), 'Training-data-driven-V1-model', 'lgn_model'),
    ]

    for path in cache_search_paths:
        candidate = os.path.join(path, temporal_cache_name)
        if os.path.exists(candidate):
            temporal_cache_path = candidate
            break

    if temporal_cache_path is None:
        raise FileNotFoundError(
            f"Temporal kernels cache '{temporal_cache_name}' not found. "
            "Please run the TensorFlow LGN model first to generate this cache, "
            "or create a symlink to an existing cache file."
        )

    return lgn_csv_path, temporal_cache_path


def load_lgn_params_from_dir(
    data_dir: str,
    movie_height: int = 120,
    movie_width: int = 240,
) -> LGNParams:
    """Load LGN parameters from a data directory.

    Automatically finds and loads LGN parameters from CSV and cached files.

    Args:
        data_dir: Path to data directory (e.g., GLIF_network)
        movie_height: Height of input movies for coordinate normalization
        movie_width: Width of input movies for coordinate normalization

    Returns:
        LGNParams containing all neuron parameters
    """
    lgn_csv_path, temporal_cache_path = find_lgn_files(data_dir)
    return load_lgn_params(lgn_csv_path, temporal_cache_path, movie_height, movie_width)


def load_lgn_params(
    lgn_csv_path: str,
    temporal_cache_path: Optional[str] = None,
    movie_height: int = 120,
    movie_width: int = 240,
) -> LGNParams:
    """Load LGN parameters from CSV and cached temporal kernels.

    Args:
        lgn_csv_path: Path to lgn_full_col_cells_3.csv
        temporal_cache_path: Path to cached temporal kernels (temporal_kernels.pkl)
                            If None, looks in same directory as lgn_csv_path
        movie_height: Height of input movies for coordinate normalization
        movie_width: Width of input movies for coordinate normalization

    Returns:
        LGNParams containing all neuron parameters

    Note:
        Corresponding TF: LGN.__init__ (lgn.py:86-263)
    """
    # Load CSV data
    d = pd.read_csv(lgn_csv_path, delimiter=' ')

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
    if temporal_cache_path is None:
        root_path = os.path.dirname(lgn_csv_path)
        temporal_cache_path = os.path.join(root_path, 'temporal_kernels.pkl')

    if not os.path.exists(temporal_cache_path):
        raise FileNotFoundError(
            f"Temporal kernels cache not found at {temporal_cache_path}. "
            "Please run the TensorFlow LGN model first to generate this cache."
        )

    with open(temporal_cache_path, 'rb') as f:
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
    x_normalized = x * (movie_width - 1) / movie_width
    y_normalized = y * (movie_height - 1) / movie_height

    # Clip to valid range
    x_normalized[np.floor(x_normalized) < 0] = 0.
    y_normalized[np.floor(y_normalized) < 0] = 0.

    non_dominant_x_normalized = non_dominant_x * (movie_width - 1) / movie_width
    non_dominant_y_normalized = non_dominant_y * (movie_height - 1) / movie_height
    non_dominant_x_normalized[np.floor(non_dominant_x_normalized) < 0] = 0.
    non_dominant_y_normalized[np.floor(non_dominant_y_normalized) < 0] = 0.
    non_dominant_x_normalized[np.ceil(non_dominant_x_normalized) >= movie_width - 1] = movie_width - 1 - 1e-6
    non_dominant_y_normalized[np.ceil(non_dominant_y_normalized) >= movie_height - 1] = movie_height - 1 - 1e-6

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


def get_neuron_groups(
    spatial_sizes: np.ndarray,
    spatial_range: np.ndarray = None,
) -> list:
    """Group neurons by spatial size for efficient filtering.

    Creates a list of (indices, sigma) tuples, one per spatial size bin.

    Args:
        spatial_sizes: Spatial filter size for each neuron
        spatial_range: Bin edges for grouping (default: 0-15 in steps of 1)

    Returns:
        List of (indices, sigma) tuples for each non-empty bin
    """
    if spatial_range is None:
        spatial_range = np.arange(0, 16, 1.0)

    neuron_groups = []
    for i in range(len(spatial_range) - 1):
        low, high = spatial_range[i], spatial_range[i + 1]
        mask = (spatial_sizes >= low) & (spatial_sizes < high)
        indices = np.where(mask)[0]

        if len(indices) > 0:
            # Compute sigma as average of bin edges divided by 3
            # This matches TF: sigma = np.round(np.mean(spatial_range[i:i+2])) / 3.
            sigma = (low + high) / 2.0 / 3.0
            sigma = max(sigma, 0.5)  # Minimum sigma
            neuron_groups.append((indices, sigma))

    return neuron_groups
