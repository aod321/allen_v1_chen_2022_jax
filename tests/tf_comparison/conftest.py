"""Fixtures for TensorFlow comparison tests.

These fixtures provide synchronized random inputs and shared parameters
for comparing JAX and TensorFlow implementations.
"""

import os
import sys
import pytest
import numpy as np

# Add TF source to path for importing
TF_SOURCE_PATH = '/nvmessd/yinzi/Training-data-driven-V1-model'
if TF_SOURCE_PATH not in sys.path:
    sys.path.insert(0, TF_SOURCE_PATH)

# Skip all tests in this directory if TensorFlow is not available
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

if not HAS_TF:
    pytest.skip("TensorFlow not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def tf_lgn():
    """Load TensorFlow LGN model."""
    from lgn_model.lgn import LGN
    return LGN(lgn_data_path='/nvmessd/yinzi/lgn_full_col_cells_3.csv')


@pytest.fixture(scope="module")
def jax_lgn():
    """Load JAX LGN model."""
    from v1_jax.lgn import LGN
    return LGN(lgn_data_path='/nvmessd/yinzi/lgn_full_col_cells_3.csv')


@pytest.fixture
def synchronized_movie(global_seed):
    """Generate synchronized test movie for JAX/TF comparison."""
    np.random.seed(global_seed)
    # Small movie for quick tests
    T, H, W = 100, 120, 240
    movie = np.random.randn(T, H, W).astype(np.float32) * 0.1
    return {
        'numpy': movie,
        'shape': (T, H, W),
    }


@pytest.fixture
def synchronized_small_movie(global_seed):
    """Smaller movie for faster tests."""
    np.random.seed(global_seed)
    T, H, W = 50, 120, 240
    movie = np.random.randn(T, H, W).astype(np.float32) * 0.1
    return {
        'numpy': movie,
        'shape': (T, H, W),
    }


class TFComparisonPrecision:
    """Precision tolerances for TF comparison tests."""

    # Spatial filtering (conv2d + interpolation)
    RTOL_SPATIAL = 1e-4
    ATOL_SPATIAL = 1e-5

    # Temporal filtering (1D convolution)
    RTOL_TEMPORAL = 1e-4
    ATOL_TEMPORAL = 1e-5

    # Full LGN forward (accumulated error)
    RTOL_LGN_FULL = 1e-3
    ATOL_LGN_FULL = 1e-4


@pytest.fixture
def tf_precision():
    """Provide TF comparison precision config."""
    return TFComparisonPrecision()
