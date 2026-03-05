"""Tests for LGN preprocessing module.

This module tests:
1. Spatial filtering (Gaussian convolution, bilinear interpolation)
2. Temporal filtering (depthwise convolution)
3. Full LGN forward pass

Test levels:
- Unit: Individual functions
- Integration: Combined operations
- TF comparison: Numerical equivalence with TensorFlow (requires TF)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from v1_jax.lgn import (
    # Spatial
    create_gaussian_kernel,
    create_gaussian_kernel_trimmed,
    gaussian_conv2d,
    bilinear_select,
    SpatialFilter,
    # Temporal
    temporal_filter,
    temporal_filter_scan,
    temporal_filter_fft,
    transfer_function,
    compute_firing_rates,
    TemporalFilter,
    # Model
    LGNParams,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_movie(global_seed):
    """Small test movie for quick tests."""
    np.random.seed(global_seed)
    return np.random.randn(100, 120, 240).astype(np.float32) * 0.1


@pytest.fixture
def tiny_movie(global_seed):
    """Tiny test movie for unit tests."""
    np.random.seed(global_seed)
    return np.random.randn(50, 60, 120).astype(np.float32) * 0.1


@pytest.fixture
def mock_lgn_params(global_seed):
    """Mock LGN parameters for testing without loading real data."""
    np.random.seed(global_seed)
    n_neurons = 100
    kernel_length = 700

    return LGNParams(
        x=np.random.rand(n_neurons).astype(np.float32) * 238,
        y=np.random.rand(n_neurons).astype(np.float32) * 118,
        non_dominant_x=np.random.rand(n_neurons).astype(np.float32) * 238,
        non_dominant_y=np.random.rand(n_neurons).astype(np.float32) * 118,
        spatial_sizes=np.random.rand(n_neurons).astype(np.float32) * 14,
        dom_amplitude=np.random.choice([-1.0, 1.0], n_neurons).astype(np.float32),
        non_dom_amplitude=np.zeros(n_neurons, dtype=np.float32),
        spontaneous_rates=np.random.rand(n_neurons).astype(np.float32) * 5,
        is_composite=np.zeros(n_neurons, dtype=np.float32),
        dom_temporal_kernels=np.random.randn(n_neurons, kernel_length).astype(np.float32) * 0.01,
        non_dom_temporal_kernels=np.zeros((n_neurons, kernel_length), dtype=np.float32),
        model_id=['sON_TF1'] * n_neurons,
    )


# =============================================================================
# Spatial Filter Tests
# =============================================================================


class TestGaussianKernel:
    """Tests for Gaussian kernel creation."""

    def test_gaussian_kernel_shape(self):
        """Test kernel has correct shape."""
        sigma = 2.0
        kernel = create_gaussian_kernel(sigma, size=51)
        assert kernel.shape == (51, 51)

    def test_gaussian_kernel_normalized(self):
        """Test kernel is normalized to sum to 1."""
        sigma = 2.0
        kernel = create_gaussian_kernel(sigma, size=51)
        assert jnp.allclose(jnp.sum(kernel), 1.0, atol=1e-5)

    def test_gaussian_kernel_symmetric(self):
        """Test kernel is symmetric."""
        sigma = 2.0
        kernel = create_gaussian_kernel(sigma, size=51)
        assert jnp.allclose(kernel, kernel.T, atol=1e-6)
        assert jnp.allclose(kernel, jnp.flip(kernel, axis=0), atol=1e-6)
        assert jnp.allclose(kernel, jnp.flip(kernel, axis=1), atol=1e-6)

    def test_gaussian_kernel_center_max(self):
        """Test center of kernel is maximum."""
        sigma = 2.0
        kernel = create_gaussian_kernel(sigma, size=51)
        center = kernel.shape[0] // 2
        assert kernel[center, center] == jnp.max(kernel)

    def test_gaussian_kernel_trimmed(self):
        """Test trimmed kernel."""
        sigma = 1.5
        kernel = create_gaussian_kernel_trimmed(sigma)
        # Should be 2D array
        assert kernel.ndim == 2
        # Should be normalized
        assert jnp.allclose(jnp.sum(kernel), 1.0, atol=1e-5)


class TestGaussianConv2D:
    """Tests for Gaussian convolution."""

    def test_conv2d_shape_preserved(self, tiny_movie):
        """Test output shape matches input (SAME padding)."""
        movie = jnp.array(tiny_movie)
        kernel = create_gaussian_kernel(1.5, size=15)
        output = gaussian_conv2d(movie, kernel)
        assert output.shape == movie.shape

    def test_conv2d_constant_input(self):
        """Test convolution of constant image gives same constant."""
        movie = jnp.ones((10, 60, 120), dtype=jnp.float32) * 5.0
        kernel = create_gaussian_kernel(2.0, size=15)
        output = gaussian_conv2d(movie, kernel)
        # Center pixels should be very close to 5.0
        # (edges may differ due to padding)
        center = output[:, 20:40, 40:80]
        assert jnp.allclose(center, 5.0, atol=1e-4)

    def test_conv2d_jit_compatible(self, tiny_movie):
        """Test convolution can be JIT compiled."""
        movie = jnp.array(tiny_movie)
        kernel = create_gaussian_kernel(1.5, size=15)

        @jax.jit
        def conv_fn(m):
            return gaussian_conv2d(m, kernel)

        output = conv_fn(movie)
        assert output.shape == movie.shape


class TestBilinearSelect:
    """Tests for bilinear interpolation."""

    def test_bilinear_integer_coords(self):
        """Test interpolation at integer coordinates."""
        # Create a simple test movie
        T, H, W = 10, 60, 120
        movie = jnp.zeros((T, H, W), dtype=jnp.float32)
        # Set specific values
        movie = movie.at[:, 30, 60].set(1.0)

        x = jnp.array([60.0])
        y = jnp.array([30.0])

        result = bilinear_select(x, y, movie)
        assert result.shape == (T, 1)
        assert jnp.allclose(result[:, 0], 1.0, atol=1e-5)

    def test_bilinear_fractional_coords(self):
        """Test interpolation at fractional coordinates."""
        T, H, W = 1, 4, 4
        # Create checkerboard pattern
        movie = jnp.array([[[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]]], dtype=jnp.float32)

        # Interpolate at center of 4 pixels
        x = jnp.array([0.5])
        y = jnp.array([0.5])

        result = bilinear_select(x, y, movie)
        # Should be average of 4 corners: (0+1+1+0)/4 = 0.5
        assert jnp.allclose(result[0, 0], 0.5, atol=1e-5)

    def test_bilinear_output_shape(self, tiny_movie):
        """Test output shape is (T, n_neurons)."""
        movie = jnp.array(tiny_movie)
        n_neurons = 50
        x = jnp.linspace(10, 100, n_neurons)
        y = jnp.linspace(10, 50, n_neurons)

        result = bilinear_select(x, y, movie)
        assert result.shape == (movie.shape[0], n_neurons)

    def test_bilinear_jit_compatible(self, tiny_movie):
        """Test bilinear interpolation can be JIT compiled."""
        movie = jnp.array(tiny_movie)
        x = jnp.array([50.0, 60.0, 70.0])
        y = jnp.array([25.0, 30.0, 35.0])

        @jax.jit
        def select_fn(m):
            return bilinear_select(x, y, m)

        result = select_fn(movie)
        assert result.shape == (movie.shape[0], 3)


# =============================================================================
# Temporal Filter Tests
# =============================================================================


class TestTemporalFilter:
    """Tests for temporal filtering."""

    def test_temporal_filter_shape(self, global_seed):
        """Test output shape matches input."""
        np.random.seed(global_seed)
        T, n_neurons = 100, 50
        kernel_length = 30

        spatial_responses = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))

        output = temporal_filter(spatial_responses, kernels)
        assert output.shape == (T, n_neurons)

    def test_temporal_filter_causal(self, global_seed):
        """Test filter is causal (output depends only on past inputs)."""
        np.random.seed(global_seed)
        T, n_neurons = 50, 10
        kernel_length = 20

        # Create input with impulse at t=30
        spatial_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)
        spatial_responses = spatial_responses.at[30, :].set(1.0)

        # Simple averaging kernel
        kernels = jnp.ones((n_neurons, kernel_length), dtype=jnp.float32) / kernel_length

        output = temporal_filter(spatial_responses, kernels)

        # Before impulse, output should be near zero
        assert jnp.allclose(output[:30, :], 0.0, atol=1e-5)

    def test_temporal_filter_impulse_response(self, global_seed):
        """Test impulse response matches kernel."""
        np.random.seed(global_seed)
        T, n_neurons = 100, 5
        kernel_length = 20

        # Create input with impulse at t=kernel_length
        spatial_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)
        spatial_responses = spatial_responses.at[kernel_length, 0].set(1.0)

        # Create a simple triangular kernel for one neuron
        kernels = jnp.zeros((n_neurons, kernel_length), dtype=jnp.float32)
        kernels = kernels.at[0, :].set(jnp.linspace(0, 1, kernel_length))

        output = temporal_filter(spatial_responses, kernels)

        # Output should contain the (reversed) kernel starting at t=kernel_length
        expected = jnp.flip(kernels[0, :])
        actual = output[kernel_length:kernel_length + kernel_length, 0]
        assert jnp.allclose(actual, expected, atol=1e-5)

    def test_temporal_filter_implementations_match(self, global_seed):
        """Test different implementations give same results."""
        np.random.seed(global_seed)
        T, n_neurons = 80, 20
        kernel_length = 30

        spatial_responses = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))

        output_default = temporal_filter(spatial_responses, kernels)
        output_scan = temporal_filter_scan(spatial_responses, kernels)

        assert jnp.allclose(output_default, output_scan, atol=1e-4)

    def test_temporal_filter_jit_compatible(self, global_seed):
        """Test temporal filter can be JIT compiled."""
        np.random.seed(global_seed)
        T, n_neurons = 50, 20
        kernel_length = 15

        spatial_responses = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))

        @jax.jit
        def filter_fn(s, k):
            return temporal_filter(s, k)

        output = filter_fn(spatial_responses, kernels)
        assert output.shape == (T, n_neurons)


class TestTransferFunction:
    """Tests for LGN transfer function."""

    def test_transfer_function_relu(self):
        """Test transfer function is ReLU."""
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert jnp.allclose(transfer_function(x), expected)

    def test_transfer_function_preserves_positive(self, global_seed):
        """Test positive values are preserved."""
        np.random.seed(global_seed)
        x = np.abs(np.random.randn(100).astype(np.float32))
        x_jax = jnp.array(x)
        assert jnp.allclose(transfer_function(x_jax), x_jax, atol=1e-6)


# =============================================================================
# Full LGN Model Tests
# =============================================================================


class TestComputeFiringRates:
    """Tests for combined firing rate computation."""

    def test_compute_firing_rates_shape(self, global_seed):
        """Test output shape."""
        np.random.seed(global_seed)
        T, n_neurons = 100, 50
        kernel_length = 30

        dom_spatial = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        non_dom_spatial = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        dom_kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))
        non_dom_kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))
        dom_amplitude = jnp.array(np.random.choice([-1.0, 1.0], n_neurons).astype(np.float32))
        non_dom_amplitude = jnp.zeros(n_neurons, dtype=jnp.float32)
        spontaneous = jnp.array(np.random.rand(n_neurons).astype(np.float32) * 5)
        is_composite = jnp.zeros(n_neurons, dtype=jnp.float32)

        output = compute_firing_rates(
            dom_spatial, non_dom_spatial,
            dom_kernels, non_dom_kernels,
            dom_amplitude, non_dom_amplitude,
            spontaneous, is_composite,
        )

        assert output.shape == (T, n_neurons)

    def test_compute_firing_rates_non_negative(self, global_seed):
        """Test firing rates are non-negative."""
        np.random.seed(global_seed)
        T, n_neurons = 50, 30
        kernel_length = 20

        dom_spatial = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        non_dom_spatial = jnp.zeros((T, n_neurons), dtype=jnp.float32)
        dom_kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))
        non_dom_kernels = jnp.zeros((n_neurons, kernel_length), dtype=jnp.float32)
        dom_amplitude = jnp.ones(n_neurons, dtype=jnp.float32)
        non_dom_amplitude = jnp.zeros(n_neurons, dtype=jnp.float32)
        spontaneous = jnp.array(np.random.rand(n_neurons).astype(np.float32) * 5)
        is_composite = jnp.zeros(n_neurons, dtype=jnp.float32)

        output = compute_firing_rates(
            dom_spatial, non_dom_spatial,
            dom_kernels, non_dom_kernels,
            dom_amplitude, non_dom_amplitude,
            spontaneous, is_composite,
        )

        assert jnp.all(output >= 0)

    def test_compute_firing_rates_jit_compatible(self, global_seed):
        """Test firing rates computation can be JIT compiled."""
        np.random.seed(global_seed)
        T, n_neurons = 50, 20
        kernel_length = 15

        dom_spatial = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        non_dom_spatial = jnp.zeros((T, n_neurons), dtype=jnp.float32)
        dom_kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))
        non_dom_kernels = jnp.zeros((n_neurons, kernel_length), dtype=jnp.float32)
        dom_amplitude = jnp.ones(n_neurons, dtype=jnp.float32)
        non_dom_amplitude = jnp.zeros(n_neurons, dtype=jnp.float32)
        spontaneous = jnp.array(np.random.rand(n_neurons).astype(np.float32) * 5)
        is_composite = jnp.zeros(n_neurons, dtype=jnp.float32)

        @jax.jit
        def compute_fn():
            return compute_firing_rates(
                dom_spatial, non_dom_spatial,
                dom_kernels, non_dom_kernels,
                dom_amplitude, non_dom_amplitude,
                spontaneous, is_composite,
            )

        output = compute_fn()
        assert output.shape == (T, n_neurons)


# =============================================================================
# Gradient Tests
# =============================================================================


class TestLGNGradients:
    """Tests for gradient flow through LGN operations."""

    def test_gaussian_conv_gradient(self, tiny_movie):
        """Test gradients flow through Gaussian convolution."""
        movie = jnp.array(tiny_movie)
        kernel = create_gaussian_kernel(1.5, size=15)

        def loss_fn(m):
            out = gaussian_conv2d(m, kernel)
            return jnp.sum(out ** 2)

        grad = jax.grad(loss_fn)(movie)
        assert grad.shape == movie.shape
        assert not jnp.any(jnp.isnan(grad))

    def test_bilinear_gradient(self, tiny_movie):
        """Test gradients flow through bilinear interpolation."""
        movie = jnp.array(tiny_movie)
        x = jnp.array([50.0, 60.0, 70.0])
        y = jnp.array([25.0, 30.0, 35.0])

        def loss_fn(m):
            out = bilinear_select(x, y, m)
            return jnp.sum(out ** 2)

        grad = jax.grad(loss_fn)(movie)
        assert grad.shape == movie.shape
        assert not jnp.any(jnp.isnan(grad))

    def test_temporal_filter_gradient(self, global_seed):
        """Test gradients flow through temporal filter."""
        np.random.seed(global_seed)
        T, n_neurons = 50, 20
        kernel_length = 15

        spatial_responses = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))

        def loss_fn(s):
            out = temporal_filter(s, kernels)
            return jnp.sum(out ** 2)

        grad = jax.grad(loss_fn)(spatial_responses)
        assert grad.shape == spatial_responses.shape
        assert not jnp.any(jnp.isnan(grad))


# =============================================================================
# Class Interface Tests
# =============================================================================


class TestTemporalFilterClass:
    """Tests for TemporalFilter class."""

    def test_temporal_filter_class_init(self, global_seed):
        """Test TemporalFilter initialization."""
        np.random.seed(global_seed)
        n_neurons = 50
        kernel_length = 30

        tf = TemporalFilter(
            dom_temporal_kernels=np.random.randn(n_neurons, kernel_length).astype(np.float32),
            non_dom_temporal_kernels=np.zeros((n_neurons, kernel_length), dtype=np.float32),
            dom_amplitude=np.ones(n_neurons, dtype=np.float32),
            non_dom_amplitude=np.zeros(n_neurons, dtype=np.float32),
            spontaneous_rates=np.random.rand(n_neurons).astype(np.float32) * 5,
            is_composite=np.zeros(n_neurons, dtype=np.float32),
        )

        assert tf.dom_temporal_kernels.shape == (n_neurons, kernel_length)

    def test_temporal_filter_class_call(self, global_seed):
        """Test TemporalFilter.__call__."""
        np.random.seed(global_seed)
        T, n_neurons = 100, 50
        kernel_length = 30

        tf = TemporalFilter(
            dom_temporal_kernels=np.random.randn(n_neurons, kernel_length).astype(np.float32),
            non_dom_temporal_kernels=np.zeros((n_neurons, kernel_length), dtype=np.float32),
            dom_amplitude=np.ones(n_neurons, dtype=np.float32),
            non_dom_amplitude=np.zeros(n_neurons, dtype=np.float32),
            spontaneous_rates=np.random.rand(n_neurons).astype(np.float32) * 5,
            is_composite=np.zeros(n_neurons, dtype=np.float32),
        )

        dom_spatial = jnp.array(np.random.randn(T, n_neurons).astype(np.float32))
        non_dom_spatial = jnp.zeros((T, n_neurons), dtype=jnp.float32)

        output = tf(dom_spatial, non_dom_spatial)
        assert output.shape == (T, n_neurons)


class TestLGNParamsDataclass:
    """Tests for LGNParams dataclass."""

    def test_lgn_params_properties(self, mock_lgn_params):
        """Test LGNParams properties."""
        params = mock_lgn_params
        assert params.n_neurons == 100
        assert params.kernel_length == 700

    def test_lgn_params_shapes(self, mock_lgn_params):
        """Test LGNParams array shapes are consistent."""
        params = mock_lgn_params
        n = params.n_neurons
        k = params.kernel_length

        assert params.x.shape == (n,)
        assert params.y.shape == (n,)
        assert params.spatial_sizes.shape == (n,)
        assert params.dom_amplitude.shape == (n,)
        assert params.dom_temporal_kernels.shape == (n, k)


# =============================================================================
# Integration Tests
# =============================================================================


class TestLGNIntegration:
    """Integration tests combining multiple operations."""

    def test_full_pipeline_shapes(self, global_seed):
        """Test full spatial + temporal pipeline shapes."""
        np.random.seed(global_seed)
        T, H, W = 50, 60, 120
        n_neurons = 30
        kernel_length = 20

        # Create mock data
        movie = jnp.array(np.random.randn(T, H, W).astype(np.float32) * 0.1)
        x = jnp.array(np.random.rand(n_neurons).astype(np.float32) * (W - 1))
        y = jnp.array(np.random.rand(n_neurons).astype(np.float32) * (H - 1))
        kernels = jnp.array(np.random.randn(n_neurons, kernel_length).astype(np.float32))

        # Apply spatial filtering
        sigma = 1.5
        gaussian_kernel = create_gaussian_kernel_trimmed(sigma)
        convolved = gaussian_conv2d(movie, gaussian_kernel)
        spatial_responses = bilinear_select(x, y, convolved)

        assert spatial_responses.shape == (T, n_neurons)

        # Apply temporal filtering
        temporal_responses = temporal_filter(spatial_responses, kernels)
        assert temporal_responses.shape == (T, n_neurons)

        # Apply transfer function
        firing_rates = transfer_function(temporal_responses)
        assert firing_rates.shape == (T, n_neurons)
        assert jnp.all(firing_rates >= 0)
