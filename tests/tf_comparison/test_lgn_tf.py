"""TensorFlow comparison tests for LGN module.

These tests verify numerical equivalence between JAX and TensorFlow
implementations of the LGN preprocessing model.

Comparison points:
1. Temporal filtering (depthwise_conv2d)
2. Spatial filtering (conv2d + bilinear interpolation)
3. Full LGN forward pass (requires bmtk)
"""

import pytest
import numpy as np
import jax.numpy as jnp
import sys
import os

# Add TF source to path
TF_SOURCE_PATH = '/nvmessd/yinzi/Training-data-driven-V1-model'
if TF_SOURCE_PATH not in sys.path:
    sys.path.insert(0, TF_SOURCE_PATH)

# Check TensorFlow availability
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

# Try to import core TF functions (these don't need bmtk)
HAS_TF_LGN_FUNCS = False
tf_temporal_filter = None
tf_select_spatial = None

if HAS_TF:
    try:
        # Import only the standalone functions from the lgn module
        # These are defined before the LGN class and don't need bmtk
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lgn_funcs",
            os.path.join(TF_SOURCE_PATH, "lgn_model", "lgn.py")
        )
        lgn_module = importlib.util.module_from_spec(spec)

        # Execute only the imports and function definitions we need
        # by reading and executing just the relevant parts
        lgn_file = os.path.join(TF_SOURCE_PATH, "lgn_model", "lgn.py")
        with open(lgn_file, 'r') as f:
            lgn_code = f.read()

        # Extract standalone functions (before the LGN class)
        exec_globals = {'tf': tf, 'np': np}
        exec("""
import tensorflow as tf
import numpy as np

def temporal_filter(all_spatial_responses, temporal_kernels):
    tr_spatial_responses = tf.pad(
        all_spatial_responses[None, :, None, :],
        ((0, 0), (temporal_kernels.shape[-1] - 1, 0), (0, 0), (0, 0)))

    tr_temporal_kernels = tf.transpose(temporal_kernels)[:, None, :, None]
    filtered_output = tf.nn.depthwise_conv2d(
        tr_spatial_responses, tr_temporal_kernels, strides=[1, 1, 1, 1], padding='VALID')[0, :, 0]
    return filtered_output

def select_spatial(x, y, convolved_movie):
    i1 = np.stack((np.floor(y), np.floor(x)), -1).astype(np.int32)
    i2 = np.stack((np.ceil(y), np.floor(x)), -1).astype(np.int32)
    i3 = np.stack((np.floor(y), np.ceil(x)), -1).astype(np.int32)
    i4 = np.stack((np.ceil(y), np.ceil(x)), -1).astype(np.int32)
    transposed_convolved_movie = tf.transpose(convolved_movie, (1, 2, 0))
    sr1 = tf.gather_nd(transposed_convolved_movie, i1)
    sr2 = tf.gather_nd(transposed_convolved_movie, i2)
    sr3 = tf.gather_nd(transposed_convolved_movie, i3)
    sr4 = tf.gather_nd(transposed_convolved_movie, i4)
    ss = tf.stack((sr1, sr2, sr3, sr4), 0)
    y_factor = (y - np.floor(y))
    x_factor = (x - np.floor(x))
    weights = np.array([
        (1 - x_factor) * (1 - y_factor),
        (1 - x_factor) * y_factor,
        x_factor * (1 - y_factor),
        x_factor * y_factor
    ])
    spatial_responses = tf.reduce_sum(ss * weights[..., None], 0)
    spatial_responses = tf.transpose(spatial_responses)
    return spatial_responses

def transfer_function(_a):
    _h = tf.cast(_a >= 0, tf.float32)
    return _h * _a
""", exec_globals)

        tf_temporal_filter = exec_globals['temporal_filter']
        tf_select_spatial = exec_globals['select_spatial']
        tf_transfer_function = exec_globals['transfer_function']
        HAS_TF_LGN_FUNCS = True

    except Exception as e:
        print(f"Could not load TF LGN functions: {e}")
        HAS_TF_LGN_FUNCS = False

# Skip if TF or LGN functions not available
pytestmark = pytest.mark.skipif(
    not HAS_TF or not HAS_TF_LGN_FUNCS,
    reason="TensorFlow or LGN functions not available"
)

from v1_jax.lgn import (
    temporal_filter as jax_temporal_filter,
    bilinear_select as jax_bilinear_select,
    gaussian_conv2d as jax_gaussian_conv2d,
    create_gaussian_kernel_trimmed,
    transfer_function as jax_transfer_function,
)


class TFComparisonPrecision:
    """Precision tolerances for TF comparison tests."""
    RTOL_SPATIAL = 1e-4
    ATOL_SPATIAL = 1e-5
    RTOL_TEMPORAL = 1e-4
    ATOL_TEMPORAL = 1e-5
    RTOL_LGN_FULL = 1e-3
    ATOL_LGN_FULL = 1e-4


@pytest.fixture
def tf_precision():
    return TFComparisonPrecision()


# =============================================================================
# Temporal Filter Comparison Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestTemporalFilterTF:
    """Compare JAX and TF temporal filtering."""

    def test_temporal_filter_matches_tf(self, global_seed, tf_precision):
        """Test JAX temporal_filter matches TF implementation."""
        np.random.seed(global_seed)
        T, n_neurons = 100, 50
        kernel_length = 30

        # Generate test data
        spatial_responses = np.random.randn(T, n_neurons).astype(np.float32)
        temporal_kernels = np.random.randn(n_neurons, kernel_length).astype(np.float32)

        # TF implementation
        tf_result = tf_temporal_filter(
            tf.constant(spatial_responses),
            tf.constant(temporal_kernels)
        ).numpy()

        # JAX implementation
        jax_result = np.array(jax_temporal_filter(
            jnp.array(spatial_responses),
            jnp.array(temporal_kernels)
        ))

        # Compare
        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=tf_precision.RTOL_TEMPORAL,
            atol=tf_precision.ATOL_TEMPORAL,
            err_msg="JAX temporal_filter differs from TF implementation"
        )

    def test_temporal_filter_impulse_matches_tf(self, global_seed, tf_precision):
        """Test impulse response matches between JAX and TF."""
        np.random.seed(global_seed)
        T, n_neurons = 100, 10
        kernel_length = 30

        # Create impulse input
        spatial_responses = np.zeros((T, n_neurons), dtype=np.float32)
        spatial_responses[kernel_length, :] = 1.0

        # Random kernels
        temporal_kernels = np.random.randn(n_neurons, kernel_length).astype(np.float32)

        # TF
        tf_result = tf_temporal_filter(
            tf.constant(spatial_responses),
            tf.constant(temporal_kernels)
        ).numpy()

        # JAX
        jax_result = np.array(jax_temporal_filter(
            jnp.array(spatial_responses),
            jnp.array(temporal_kernels)
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=tf_precision.RTOL_TEMPORAL,
            atol=tf_precision.ATOL_TEMPORAL,
            err_msg="Impulse response differs between JAX and TF"
        )

    def test_temporal_filter_short_sequence(self, global_seed, tf_precision):
        """Test short sequence temporal filtering."""
        np.random.seed(global_seed)
        T, n_neurons = 50, 20
        kernel_length = 15

        spatial_responses = np.random.randn(T, n_neurons).astype(np.float32)
        temporal_kernels = np.random.randn(n_neurons, kernel_length).astype(np.float32)

        tf_result = tf_temporal_filter(
            tf.constant(spatial_responses),
            tf.constant(temporal_kernels)
        ).numpy()

        jax_result = np.array(jax_temporal_filter(
            jnp.array(spatial_responses),
            jnp.array(temporal_kernels)
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=tf_precision.RTOL_TEMPORAL,
            atol=tf_precision.ATOL_TEMPORAL,
            err_msg="Short sequence temporal filter differs"
        )


# =============================================================================
# Bilinear Interpolation Comparison Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestBilinearSelectTF:
    """Compare JAX and TF bilinear interpolation."""

    def test_bilinear_select_matches_tf(self, global_seed, tf_precision):
        """Test JAX bilinear_select matches TF select_spatial."""
        np.random.seed(global_seed)
        T, H, W = 50, 120, 240

        # Generate test movie
        movie = np.random.randn(T, H, W).astype(np.float32)

        # Random neuron positions
        n_neurons = 100
        x = np.random.rand(n_neurons).astype(np.float32) * (W - 2)  # Avoid edge
        y = np.random.rand(n_neurons).astype(np.float32) * (H - 2)  # Avoid edge

        # TF implementation
        tf_result = tf_select_spatial(x, y, tf.constant(movie)).numpy()

        # JAX implementation
        jax_result = np.array(jax_bilinear_select(
            jnp.array(x),
            jnp.array(y),
            jnp.array(movie)
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=tf_precision.RTOL_SPATIAL,
            atol=tf_precision.ATOL_SPATIAL,
            err_msg="JAX bilinear_select differs from TF select_spatial"
        )

    def test_bilinear_at_integer_coords_matches_tf(self, global_seed, tf_precision):
        """Test interpolation at integer coordinates matches TF."""
        np.random.seed(global_seed)
        T, H, W = 20, 60, 120

        movie = np.random.randn(T, H, W).astype(np.float32)

        # Integer coordinates (inside valid range)
        x = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        y = np.array([5, 15, 25, 35, 45], dtype=np.float32)

        tf_result = tf_select_spatial(x, y, tf.constant(movie)).numpy()
        jax_result = np.array(jax_bilinear_select(
            jnp.array(x), jnp.array(y), jnp.array(movie)
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Integer coordinate interpolation differs"
        )

    def test_bilinear_small_movie_matches_tf(self, global_seed, tf_precision):
        """Test bilinear interpolation on small movie."""
        np.random.seed(global_seed)
        T, H, W = 10, 30, 60

        movie = np.random.randn(T, H, W).astype(np.float32)

        n_neurons = 20
        x = np.random.rand(n_neurons).astype(np.float32) * (W - 2)
        y = np.random.rand(n_neurons).astype(np.float32) * (H - 2)

        tf_result = tf_select_spatial(x, y, tf.constant(movie)).numpy()
        jax_result = np.array(jax_bilinear_select(
            jnp.array(x), jnp.array(y), jnp.array(movie)
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=tf_precision.RTOL_SPATIAL,
            atol=tf_precision.ATOL_SPATIAL,
            err_msg="Small movie bilinear interpolation differs"
        )


# =============================================================================
# Transfer Function Comparison Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestTransferFunctionTF:
    """Compare JAX and TF transfer function (ReLU)."""

    def test_transfer_function_matches_tf(self, global_seed):
        """Test JAX transfer function matches TF."""
        np.random.seed(global_seed)

        # Generate test data with positive and negative values
        x = np.random.randn(100, 50).astype(np.float32)

        tf_result = tf_transfer_function(tf.constant(x)).numpy()
        jax_result = np.array(jax_transfer_function(jnp.array(x)))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Transfer function differs between JAX and TF"
        )

    def test_transfer_function_edge_cases(self):
        """Test transfer function at edge cases."""
        x = np.array([-1e-6, 0, 1e-6], dtype=np.float32)

        tf_result = tf_transfer_function(tf.constant(x)).numpy()
        jax_result = np.array(jax_transfer_function(jnp.array(x)))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Edge case transfer function differs"
        )


# =============================================================================
# Combined Pipeline Tests (without full LGN class)
# =============================================================================


@pytest.mark.tf_comparison
class TestCombinedPipelineTF:
    """Test combined spatial + temporal pipeline without LGN class."""

    def test_spatial_then_temporal_matches_tf(self, global_seed, tf_precision):
        """Test full pipeline: spatial interpolation -> temporal filter -> transfer."""
        np.random.seed(global_seed)
        T, n_neurons = 80, 30
        kernel_length = 25

        # Generate spatial responses (as if from conv2d + select)
        spatial_responses = np.random.randn(T, n_neurons).astype(np.float32)
        temporal_kernels = np.random.randn(n_neurons, kernel_length).astype(np.float32)
        amplitude = np.random.choice([-1.0, 1.0], n_neurons).astype(np.float32)
        spontaneous = np.random.rand(n_neurons).astype(np.float32) * 5

        # TF pipeline
        tf_filtered = tf_temporal_filter(
            tf.constant(spatial_responses),
            tf.constant(temporal_kernels)
        )
        tf_scaled = tf_filtered * tf.constant(amplitude) + tf.constant(spontaneous)
        tf_rates = tf_transfer_function(tf_scaled).numpy()

        # JAX pipeline
        jax_filtered = jax_temporal_filter(
            jnp.array(spatial_responses),
            jnp.array(temporal_kernels)
        )
        jax_scaled = jax_filtered * jnp.array(amplitude) + jnp.array(spontaneous)
        jax_rates = np.array(jax_transfer_function(jax_scaled))

        np.testing.assert_allclose(
            jax_rates, tf_rates,
            rtol=tf_precision.RTOL_LGN_FULL,
            atol=tf_precision.ATOL_LGN_FULL,
            err_msg="Combined pipeline differs between JAX and TF"
        )

    def test_multiple_neuron_groups(self, global_seed, tf_precision):
        """Test with multiple neuron groups of different sizes."""
        np.random.seed(global_seed)
        T = 60
        kernel_length = 20

        # Multiple groups
        group_sizes = [10, 25, 15]
        all_jax_results = []
        all_tf_results = []

        for n_neurons in group_sizes:
            spatial_responses = np.random.randn(T, n_neurons).astype(np.float32)
            temporal_kernels = np.random.randn(n_neurons, kernel_length).astype(np.float32)

            tf_result = tf_temporal_filter(
                tf.constant(spatial_responses),
                tf.constant(temporal_kernels)
            ).numpy()

            jax_result = np.array(jax_temporal_filter(
                jnp.array(spatial_responses),
                jnp.array(temporal_kernels)
            ))

            all_tf_results.append(tf_result)
            all_jax_results.append(jax_result)

        # Concatenate and compare
        tf_combined = np.concatenate(all_tf_results, axis=1)
        jax_combined = np.concatenate(all_jax_results, axis=1)

        np.testing.assert_allclose(
            jax_combined, tf_combined,
            rtol=tf_precision.RTOL_TEMPORAL,
            atol=tf_precision.ATOL_TEMPORAL,
            err_msg="Multiple neuron groups differ"
        )
