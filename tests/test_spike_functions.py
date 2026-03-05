"""Tests for spike functions with custom gradients.

Tests forward pass, backward pass (gradients), and JIT compilation
of spike_gauss and other spike functions.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from v1_jax.nn.spike_functions import (
    spike_gauss,
    gauss_pseudo,
    pseudo_derivative,
    spike_piecewise,
    spike_sigmoid,
)


class TestGaussPseudo:
    """Tests for Gaussian pseudo-derivative."""

    def test_shape_preservation(self):
        """Output shape matches input shape."""
        v = jnp.ones((32, 100))
        result = gauss_pseudo(v, sigma=0.28, amplitude=0.5)
        assert result.shape == v.shape

    def test_peak_at_zero(self):
        """Gradient is maximum at v=0."""
        v = jnp.array([0.0])
        result = gauss_pseudo(v, sigma=0.28, amplitude=0.5)
        assert np.isclose(result[0], 0.5, rtol=1e-5)

    def test_symmetry(self):
        """Gradient is symmetric around v=0."""
        v_pos = jnp.array([0.5])
        v_neg = jnp.array([-0.5])
        result_pos = gauss_pseudo(v_pos, sigma=0.28, amplitude=0.5)
        result_neg = gauss_pseudo(v_neg, sigma=0.28, amplitude=0.5)
        np.testing.assert_allclose(result_pos, result_neg, rtol=1e-5)

    def test_decay_with_distance(self):
        """Gradient decays with distance from zero."""
        v_near = jnp.array([0.1])
        v_far = jnp.array([1.0])
        result_near = gauss_pseudo(v_near, sigma=0.28, amplitude=0.5)
        result_far = gauss_pseudo(v_far, sigma=0.28, amplitude=0.5)
        assert result_near > result_far

    def test_sigma_effect(self):
        """Larger sigma gives wider gradient."""
        v = jnp.array([0.5])
        result_narrow = gauss_pseudo(v, sigma=0.1, amplitude=0.5)
        result_wide = gauss_pseudo(v, sigma=1.0, amplitude=0.5)
        assert result_wide > result_narrow


class TestSpikeGaussForward:
    """Tests for spike_gauss forward pass."""

    def test_threshold_behavior(self, small_voltage_data):
        """v > 0 -> 1, v <= 0 -> 0."""
        v = jnp.array(small_voltage_data)
        spikes = spike_gauss(v, sigma=0.28, amplitude=0.5)

        # Check positive values give 1
        assert jnp.all(spikes[v > 0] == 1.0)

        # Check non-positive values give 0
        assert jnp.all(spikes[v <= 0] == 0.0)

    def test_output_binary(self, small_voltage_data):
        """Output is binary (0 or 1)."""
        v = jnp.array(small_voltage_data)
        spikes = spike_gauss(v, sigma=0.28, amplitude=0.5)

        unique_vals = jnp.unique(spikes)
        assert len(unique_vals) <= 2
        assert jnp.all((spikes == 0.0) | (spikes == 1.0))

    def test_shape_preservation(self, small_voltage_data):
        """Output shape matches input."""
        v = jnp.array(small_voltage_data)
        spikes = spike_gauss(v, sigma=0.28, amplitude=0.5)
        assert spikes.shape == v.shape

    def test_dtype_float32(self, small_voltage_data):
        """Output is float32."""
        v = jnp.array(small_voltage_data)
        spikes = spike_gauss(v, sigma=0.28, amplitude=0.5)
        assert spikes.dtype == jnp.float32

    def test_boundary_values(self):
        """Test exact boundary behavior."""
        v = jnp.array([-1e-6, 0.0, 1e-6])
        spikes = spike_gauss(v, sigma=0.28, amplitude=0.5)

        assert spikes[0] == 0.0  # Negative
        assert spikes[1] == 0.0  # Zero
        assert spikes[2] == 1.0  # Positive


class TestSpikeGaussGradient:
    """Tests for spike_gauss backward pass (gradients)."""

    def test_gradient_exists(self, small_voltage_data):
        """Gradient computation doesn't raise errors."""
        v = jnp.array(small_voltage_data[:10, :10])

        def loss_fn(x):
            return jnp.sum(spike_gauss(x, sigma=0.28, amplitude=0.5))

        grad = jax.grad(loss_fn)(v)
        assert grad.shape == v.shape
        assert not jnp.any(jnp.isnan(grad))

    def test_gradient_matches_pseudo(self, small_voltage_data):
        """Gradient equals Gaussian pseudo-derivative."""
        v = jnp.array(small_voltage_data[:10, :10])
        sigma, amplitude = 0.28, 0.5

        def loss_fn(x):
            return jnp.sum(spike_gauss(x, sigma, amplitude))

        grad = jax.grad(loss_fn)(v)
        expected = gauss_pseudo(v, sigma, amplitude)

        np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-6)

    def test_gradient_at_boundary(self):
        """Gradient is maximum at v=0."""
        v = jnp.array([0.0, 1e-6, -1e-6, 0.1, -0.1])
        sigma, amplitude = 0.28, 0.5

        def loss_fn(x):
            return jnp.sum(spike_gauss(x, sigma, amplitude))

        grad = jax.grad(loss_fn)(v)

        # Gradient at v=0 should be approximately amplitude
        assert np.isclose(grad[0], amplitude, rtol=1e-3)

    def test_gradient_non_negative(self, small_voltage_data):
        """Gaussian pseudo-derivative is always non-negative."""
        v = jnp.array(small_voltage_data[:10, :10])

        def loss_fn(x):
            return jnp.sum(spike_gauss(x, sigma=0.28, amplitude=0.5))

        grad = jax.grad(loss_fn)(v)
        assert jnp.all(grad >= 0)

    def test_gradient_chain_rule(self):
        """Gradient flows through chain of operations."""
        v = jnp.array([[0.1, -0.1], [0.5, -0.5]])

        def loss_fn(x):
            spikes = spike_gauss(x, sigma=0.28, amplitude=0.5)
            return jnp.sum(spikes * 2.0)  # Scale by 2

        grad = jax.grad(loss_fn)(v)
        expected = 2.0 * gauss_pseudo(v, 0.28, 0.5)

        np.testing.assert_allclose(grad, expected, rtol=1e-5)


class TestSpikeGaussJIT:
    """Tests for JIT compilation of spike_gauss."""

    def test_jit_forward(self, small_voltage_data):
        """JIT-compiled forward pass matches eager."""
        v = jnp.array(small_voltage_data)

        eager_result = spike_gauss(v, sigma=0.28, amplitude=0.5)

        @jax.jit
        def jit_fn(x):
            return spike_gauss(x, sigma=0.28, amplitude=0.5)

        jit_result = jit_fn(v)

        np.testing.assert_array_equal(eager_result, jit_result)

    def test_jit_gradient(self, small_voltage_data):
        """JIT-compiled gradient matches eager."""
        v = jnp.array(small_voltage_data[:100, :100])

        def loss_fn(x):
            return jnp.sum(spike_gauss(x, sigma=0.28, amplitude=0.5))

        eager_grad = jax.grad(loss_fn)(v)
        jit_grad = jax.jit(jax.grad(loss_fn))(v)

        np.testing.assert_allclose(eager_grad, jit_grad, rtol=1e-5)

    def test_jit_vmap(self, small_voltage_data):
        """Works with vmap for batched operations."""
        v = jnp.array(small_voltage_data[:10, :100])

        @jax.jit
        @jax.vmap
        def batched_fn(x):
            return spike_gauss(x, sigma=0.28, amplitude=0.5)

        result = batched_fn(v)
        assert result.shape == v.shape


class TestNumericalGradient:
    """Tests comparing analytical gradient with expected pseudo-derivative."""

    def test_analytical_gradient_is_pseudo_derivative(self):
        """Analytical gradient equals Gaussian pseudo-derivative (not numerical).

        Note: Numerical gradient of Heaviside function is 0 everywhere,
        so we verify against the expected Gaussian pseudo-derivative instead.
        """
        np.random.seed(42)
        v = np.random.randn(5, 5).astype(np.float32) * 0.5
        v_jax = jnp.array(v)
        sigma, amplitude = 0.28, 0.5

        # Analytical gradient via custom_vjp
        analytical_grad = jax.grad(
            lambda x: jnp.sum(spike_gauss(x, sigma, amplitude))
        )(v_jax)

        # Expected: Gaussian pseudo-derivative
        expected_grad = gauss_pseudo(v_jax, sigma, amplitude)

        np.testing.assert_allclose(
            analytical_grad, expected_grad, rtol=1e-5, atol=1e-6
        )


class TestSpikePiecewise:
    """Tests for piecewise linear spike function."""

    def test_forward_pass(self):
        """Forward pass is binary threshold."""
        v = jnp.array([-1.0, 0.0, 0.5, 1.0])
        spikes = spike_piecewise(v, dampening_factor=0.3)
        expected = jnp.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(spikes, expected)

    def test_gradient_shape(self):
        """Gradient has correct shape."""
        v = jnp.ones((10, 10))

        def loss_fn(x):
            return jnp.sum(spike_piecewise(x, dampening_factor=0.3))

        grad = jax.grad(loss_fn)(v)
        assert grad.shape == v.shape


class TestSpikeSigmoid:
    """Tests for sigmoid-based spike function."""

    def test_forward_pass(self):
        """Forward pass is binary threshold."""
        v = jnp.array([-1.0, 0.0, 0.5, 1.0])
        spikes = spike_sigmoid(v, beta=10.0)
        expected = jnp.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(spikes, expected)

    def test_gradient_shape(self):
        """Gradient has correct shape."""
        v = jnp.ones((10, 10))

        def loss_fn(x):
            return jnp.sum(spike_sigmoid(x, beta=10.0))

        grad = jax.grad(loss_fn)(v)
        assert grad.shape == v.shape

    def test_beta_effect(self):
        """Higher beta gives sharper gradient at threshold."""
        # Test at v=0 where gradient is maximum
        v_zero = jnp.array([0.0])

        def loss_fn(x, beta):
            return jnp.sum(spike_sigmoid(x, beta=beta))

        grad_low = jax.grad(lambda x: loss_fn(x, 1.0))(v_zero)
        grad_high = jax.grad(lambda x: loss_fn(x, 10.0))(v_zero)

        # Higher beta gives larger gradient at v=0 (sharper peak)
        # At v=0: grad = beta * 0.25 (since sigmoid(0) = 0.5)
        assert grad_high[0] > grad_low[0]

        # But at positions away from threshold, gradient decays faster with high beta
        v_away = jnp.array([0.5])
        grad_away_low = jax.grad(lambda x: loss_fn(x, 1.0))(v_away)
        grad_away_high = jax.grad(lambda x: loss_fn(x, 100.0))(v_away)

        # With very high beta, gradient decays to near-zero away from threshold
        assert grad_away_high[0] < grad_away_low[0]
