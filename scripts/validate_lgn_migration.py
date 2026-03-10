#!/usr/bin/env python3
"""End-to-end validation script for LGN migration.

This script validates the JAX LGN implementation against the original
TensorFlow implementation by comparing outputs on the same input data.

Usage:
    uv run python scripts/validate_lgn_migration.py
"""

import sys
import os
import time
import pickle as pkl
import numpy as np
import pandas as pd

# Default data paths
LGN_DATA_PATH = '/nvmessd/yinzi/lgn_full_col_cells_3.csv'
TEMPORAL_KERNELS_PATH = '/nvmessd/yinzi/Training-data-driven-V1-model/lgn_model/temporal_kernels.pkl'


# ============================================================================
# Standalone TF functions (copied from original lgn.py to avoid bmtk dependency)
# ============================================================================

def tf_temporal_filter(all_spatial_responses, temporal_kernels):
    """TensorFlow implementation of temporal filtering."""
    import tensorflow as tf

    tr_spatial_responses = tf.pad(
        all_spatial_responses[None, :, None, :],
        ((0, 0), (temporal_kernels.shape[-1] - 1, 0), (0, 0), (0, 0)))

    tr_temporal_kernels = tf.transpose(temporal_kernels)[:, None, :, None]
    filtered_output = tf.nn.depthwise_conv2d(
        tr_spatial_responses, tr_temporal_kernels, strides=[1, 1, 1, 1], padding='VALID')[0, :, 0]
    return filtered_output


def tf_select_spatial(x, y, convolved_movie):
    """TensorFlow implementation of bilinear interpolation."""
    import tensorflow as tf

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


def tf_transfer_function(_a):
    """TensorFlow implementation of transfer function (ReLU)."""
    import tensorflow as tf
    _h = tf.cast(_a >= 0, tf.float32)
    return _h * _a


class SimpleTFLGN:
    """Simplified TF LGN model that loads pre-computed temporal kernels."""

    def __init__(self, lgn_csv_path, temporal_kernels_path):
        import tensorflow as tf

        # Load CSV data
        d = pd.read_csv(lgn_csv_path, delimiter=' ')
        self.spatial_sizes = d['spatial_size'].to_numpy()
        model_id = d['model_id'].to_numpy()

        # Parse ON/OFF cell types
        self.is_composite = np.array(
            [a.count('ON') > 0 and a.count('OFF') > 0 for a in model_id]
        ).astype(np.float32)

        # Receptive field positions
        self.x = d['x'].to_numpy() * 239 / 240
        self.y = d['y'].to_numpy() * 119 / 120
        self.x[np.floor(self.x) < 0] = 0.
        self.y[np.floor(self.y) < 0] = 0.

        # Load cached temporal kernels
        with open(temporal_kernels_path, 'rb') as f:
            loaded = pkl.load(f)

        self.dom_temporal_kernels = loaded['dom_temporal_kernels']
        self.non_dom_temporal_kernels = loaded['non_dom_temporal_kernels']
        self.non_dominant_x = loaded['non_dominant_x'] * 239 / 240
        self.non_dominant_y = loaded['non_dominant_y'] * 119 / 120
        self.amplitude = loaded['amplitude']
        self.non_dom_amplitude = loaded['non_dom_amplitude']
        self.spontaneous_firing_rates = loaded['spontaneous_firing_rates']

        # Clip coordinates
        self.non_dominant_x[np.floor(self.non_dominant_x) < 0] = 0.
        self.non_dominant_y[np.floor(self.non_dominant_y) < 0] = 0.
        self.non_dominant_x[np.ceil(self.non_dominant_x) >= 239.] = 239.
        self.non_dominant_y[np.ceil(self.non_dominant_y) >= 119.] = 119.

    def spatial_response(self, movie):
        """Compute spatial response for all LGN cells."""
        import tensorflow as tf
        from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter

        d_spatial = 1.
        spatial_range = np.arange(0, 15, d_spatial)

        x_range = np.arange(-50, 51)
        y_range = np.arange(-50, 51)

        all_spatial_responses = []
        all_non_dom_responses = []
        neuron_ids = []

        for i in range(len(spatial_range) - 1):
            sel = np.logical_and(
                self.spatial_sizes < spatial_range[i + 1],
                self.spatial_sizes >= spatial_range[i]
            )
            if np.sum(sel) <= 0:
                continue

            neuron_ids.extend(np.where(sel)[0])

            # Construct Gaussian spatial filter
            sigma = np.round(np.mean(spatial_range[i:i+2])) / 3.0
            original_filter = GaussianSpatialFilter(
                translate=(0., 0.), sigma=(sigma, sigma), origin=(0., 0.)
            )
            kernel = original_filter.get_kernel(x_range, y_range, amplitude=1.).full()
            nonzero_inds = np.where(np.abs(kernel) > 1e-9)
            rm, rM = nonzero_inds[0].min(), nonzero_inds[0].max()
            cm, cM = nonzero_inds[1].min(), nonzero_inds[1].max()
            kernel = kernel[rm:rM + 1, cm:cM + 1]
            gaussian_filter = kernel[..., None, None]

            # Apply convolution
            convolved_movie = tf.nn.conv2d(
                movie, gaussian_filter, strides=[1, 1], padding='SAME'
            )[..., 0]

            # Select at RF positions
            spatial = tf_select_spatial(self.x[sel], self.y[sel], convolved_movie)
            non_dom_spatial = tf_select_spatial(
                self.non_dominant_x[sel], self.non_dominant_y[sel], convolved_movie
            )

            all_spatial_responses.append(spatial)
            all_non_dom_responses.append(non_dom_spatial)

        # Concatenate and reorder
        neuron_ids = np.array(neuron_ids)
        all_spatial_responses = tf.concat(all_spatial_responses, axis=1)
        all_non_dom_responses = tf.concat(all_non_dom_responses, axis=1)

        # Sort by original neuron order
        sorted_indices = np.argsort(neuron_ids)
        all_spatial_responses = tf.gather(all_spatial_responses, sorted_indices, axis=1)
        all_non_dom_responses = tf.gather(all_non_dom_responses, sorted_indices, axis=1)

        return all_spatial_responses, all_non_dom_responses

    def firing_rates_from_spatial(self, dom_spatial, non_dom_spatial):
        """Compute firing rates from spatial responses."""
        import tensorflow as tf

        dom_filtered = tf_temporal_filter(dom_spatial, self.dom_temporal_kernels)
        non_dom_filtered = tf_temporal_filter(non_dom_spatial, self.non_dom_temporal_kernels)

        firing_rates = tf_transfer_function(
            dom_filtered * self.amplitude + self.spontaneous_firing_rates
        )
        multi_firing_rates = firing_rates + tf_transfer_function(
            non_dom_filtered * self.non_dom_amplitude + self.spontaneous_firing_rates
        )
        firing_rates = firing_rates * (1 - self.is_composite) + multi_firing_rates * self.is_composite

        return firing_rates


def load_tensorflow_lgn():
    """Load the TensorFlow LGN model."""
    import tensorflow as tf

    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')

    return SimpleTFLGN(LGN_DATA_PATH, TEMPORAL_KERNELS_PATH)


def load_jax_lgn():
    """Load the JAX LGN model."""
    import jax
    # Use CPU for fair comparison
    jax.config.update('jax_platform_name', 'cpu')

    from v1_jax.data.lgn_model import LGNModel

    return LGNModel(lgn_data_dir='/nvmessd/yinzi/GLIF_network')


def create_test_movie(duration: int = 100, height: int = 120, width: int = 240, seed: int = 42):
    """Create a random test movie.

    Args:
        duration: Number of time steps
        height: Movie height
        width: Movie width
        seed: Random seed

    Returns:
        Movie array of shape (duration, height, width, 1) for TF
        and (duration, height, width) for JAX
    """
    np.random.seed(seed)
    movie = np.random.randn(duration, height, width).astype(np.float32) * 0.1
    return movie


def test_spatial_response(tf_lgn, jax_lgn, movie, rtol=1e-4, atol=1e-5):
    """Compare spatial responses between TF and JAX."""
    import tensorflow as tf
    import jax.numpy as jnp

    print("\n" + "=" * 60)
    print("Testing Spatial Response")
    print("=" * 60)

    # TF expects (T, H, W, 1)
    movie_tf = tf.constant(movie[..., None])

    # JAX expects (T, H, W, 1)
    movie_jax = movie[..., None]

    # TF spatial response
    start_time = time.time()
    tf_dom, tf_non_dom = tf_lgn.spatial_response(movie_tf)
    tf_time = time.time() - start_time
    tf_dom = tf_dom.numpy()
    tf_non_dom = tf_non_dom.numpy()

    # JAX spatial response
    start_time = time.time()
    jax_dom, jax_non_dom = jax_lgn.spatial_response(movie_jax)
    jax_time = time.time() - start_time
    jax_dom = np.array(jax_dom)
    jax_non_dom = np.array(jax_non_dom)

    print(f"TF shape: dom={tf_dom.shape}, non_dom={tf_non_dom.shape}")
    print(f"JAX shape: dom={jax_dom.shape}, non_dom={jax_non_dom.shape}")
    print(f"TF time: {tf_time:.3f}s, JAX time: {jax_time:.3f}s")

    # Compare dominant
    dom_diff = np.abs(jax_dom - tf_dom)
    print(f"\nDominant spatial response:")
    print(f"  Max abs diff: {dom_diff.max():.6e}")
    print(f"  Mean abs diff: {dom_diff.mean():.6e}")
    print(f"  Relative max diff: {(dom_diff / (np.abs(tf_dom) + 1e-8)).max():.6e}")

    # Compare non-dominant
    non_dom_diff = np.abs(jax_non_dom - tf_non_dom)
    print(f"\nNon-dominant spatial response:")
    print(f"  Max abs diff: {non_dom_diff.max():.6e}")
    print(f"  Mean abs diff: {non_dom_diff.mean():.6e}")

    try:
        np.testing.assert_allclose(jax_dom, tf_dom, rtol=rtol, atol=atol)
        print("\n✓ Dominant spatial responses MATCH")
    except AssertionError as e:
        print(f"\n✗ Dominant spatial responses DIFFER: {e}")
        return False, (tf_dom, tf_non_dom), (jax_dom, jax_non_dom)

    try:
        np.testing.assert_allclose(jax_non_dom, tf_non_dom, rtol=rtol, atol=atol)
        print("✓ Non-dominant spatial responses MATCH")
    except AssertionError as e:
        print(f"✗ Non-dominant spatial responses DIFFER: {e}")
        return False, (tf_dom, tf_non_dom), (jax_dom, jax_non_dom)

    return True, (tf_dom, tf_non_dom), (jax_dom, jax_non_dom)


def test_firing_rates(tf_lgn, jax_lgn, tf_spatial, jax_spatial, rtol=1e-3, atol=1e-4):
    """Compare firing rates between TF and JAX."""
    import tensorflow as tf
    import jax.numpy as jnp

    print("\n" + "=" * 60)
    print("Testing Firing Rates (from spatial)")
    print("=" * 60)

    tf_dom, tf_non_dom = tf_spatial
    jax_dom, jax_non_dom = jax_spatial

    # TF firing rates
    start_time = time.time()
    tf_rates = tf_lgn.firing_rates_from_spatial(
        tf.constant(tf_dom),
        tf.constant(tf_non_dom)
    ).numpy()
    tf_time = time.time() - start_time

    # JAX firing rates
    start_time = time.time()
    jax_rates = np.array(jax_lgn.firing_rates_from_spatial(
        jnp.array(jax_dom),
        jnp.array(jax_non_dom)
    ))
    jax_time = time.time() - start_time

    print(f"TF shape: {tf_rates.shape}")
    print(f"JAX shape: {jax_rates.shape}")
    print(f"TF time: {tf_time:.3f}s, JAX time: {jax_time:.3f}s")

    # Compare
    diff = np.abs(jax_rates - tf_rates)
    print(f"\nFiring rates comparison:")
    print(f"  Max abs diff: {diff.max():.6e}")
    print(f"  Mean abs diff: {diff.mean():.6e}")
    print(f"  TF range: [{tf_rates.min():.4f}, {tf_rates.max():.4f}]")
    print(f"  JAX range: [{jax_rates.min():.4f}, {jax_rates.max():.4f}]")

    # Non-zero firing rate statistics
    tf_nonzero = tf_rates[tf_rates > 0]
    jax_nonzero = jax_rates[jax_rates > 0]
    print(f"  TF non-zero count: {len(tf_nonzero)} ({len(tf_nonzero)/tf_rates.size*100:.1f}%)")
    print(f"  JAX non-zero count: {len(jax_nonzero)} ({len(jax_nonzero)/jax_rates.size*100:.1f}%)")

    try:
        np.testing.assert_allclose(jax_rates, tf_rates, rtol=rtol, atol=atol)
        print("\n✓ Firing rates MATCH")
        return True, tf_rates, jax_rates
    except AssertionError as e:
        print(f"\n✗ Firing rates DIFFER: {e}")
        return False, tf_rates, jax_rates


def test_full_pipeline(tf_lgn, jax_lgn, movie, rtol=1e-3, atol=1e-4):
    """Test the full LGN pipeline end-to-end."""
    import tensorflow as tf
    import jax.numpy as jnp

    print("\n" + "=" * 60)
    print("Testing Full Pipeline (end-to-end)")
    print("=" * 60)

    # TF full pipeline
    movie_tf = tf.constant(movie[..., None])

    start_time = time.time()
    tf_spatial = tf_lgn.spatial_response(movie_tf)
    tf_rates = tf_lgn.firing_rates_from_spatial(*tf_spatial).numpy()
    tf_time = time.time() - start_time

    # JAX full pipeline
    start_time = time.time()
    jax_rates = jax_lgn.process_movie(movie[..., None])
    jax_time = time.time() - start_time

    print(f"TF output shape: {tf_rates.shape}")
    print(f"JAX output shape: {jax_rates.shape}")
    print(f"TF time: {tf_time:.3f}s, JAX time: {jax_time:.3f}s")
    print(f"Speedup: {tf_time/jax_time:.2f}x")

    # Compare
    diff = np.abs(jax_rates - tf_rates)
    print(f"\nFull pipeline comparison:")
    print(f"  Max abs diff: {diff.max():.6e}")
    print(f"  Mean abs diff: {diff.mean():.6e}")
    print(f"  Max relative diff: {(diff / (np.abs(tf_rates) + 1e-8)).max():.6e}")

    try:
        np.testing.assert_allclose(jax_rates, tf_rates, rtol=rtol, atol=atol)
        print("\n✓ Full pipeline MATCHES")
        return True
    except AssertionError as e:
        print(f"\n✗ Full pipeline DIFFERS: {e}")
        return False


def main():
    """Run all LGN validation tests."""
    print("=" * 60)
    print("LGN Migration Validation")
    print("=" * 60)

    # Load models
    print("\nLoading TensorFlow LGN model...")
    tf_lgn = load_tensorflow_lgn()
    print(f"  Loaded {tf_lgn.dom_temporal_kernels.shape[0]} neurons")

    print("\nLoading JAX LGN model...")
    jax_lgn = load_jax_lgn()
    print(f"  Loaded {jax_lgn.n_cells} neurons")

    # Create test movie
    print("\nCreating test movie...")
    movie = create_test_movie(duration=100)
    print(f"  Shape: {movie.shape}")

    # Test spatial response
    spatial_ok, tf_spatial, jax_spatial = test_spatial_response(
        tf_lgn, jax_lgn, movie, rtol=1e-4, atol=1e-5
    )

    # Test firing rates
    rates_ok, tf_rates, jax_rates = test_firing_rates(
        tf_lgn, jax_lgn, tf_spatial, jax_spatial, rtol=1e-3, atol=1e-4
    )

    # Test full pipeline
    pipeline_ok = test_full_pipeline(
        tf_lgn, jax_lgn, movie, rtol=1e-3, atol=1e-4
    )

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"  Spatial Response: {'✓ PASS' if spatial_ok else '✗ FAIL'}")
    print(f"  Firing Rates: {'✓ PASS' if rates_ok else '✗ FAIL'}")
    print(f"  Full Pipeline: {'✓ PASS' if pipeline_ok else '✗ FAIL'}")

    if spatial_ok and rates_ok and pipeline_ok:
        print("\n✓ All validation tests PASSED")
        return 0
    else:
        print("\n✗ Some validation tests FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
