"""Tests for loss functions.

Tests Huber quantile loss, spike rate distribution loss,
and classification losses.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from v1_jax.training.loss_functions import (
    huber_quantile_loss,
    spike_rate_distribution_loss,
    sparse_categorical_crossentropy,
    weighted_crossentropy,
    binary_crossentropy,
    mean_squared_error,
)


class TestHuberQuantileLoss:
    """Tests for Huber quantile loss function."""

    def test_shape_preservation(self):
        """Output shape matches input shape."""
        u = jnp.ones((100,))
        tau = jnp.linspace(0.01, 0.99, 100)
        result = huber_quantile_loss(u, tau, kappa=0.002)
        assert result.shape == u.shape

    def test_zero_residual(self):
        """Loss is zero when residual is zero."""
        u = jnp.zeros((10,))
        tau = jnp.linspace(0.1, 0.9, 10)
        result = huber_quantile_loss(u, tau, kappa=0.002)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_quadratic_region(self):
        """Uses quadratic loss for small residuals."""
        # |u| < kappa should use quadratic
        u = jnp.array([0.001])  # < kappa=0.002
        tau = jnp.array([0.5])
        kappa = 0.002

        result = huber_quantile_loss(u, tau, kappa)

        # Expected: |tau - indicator| / (2*kappa) * u^2
        # u > 0, so indicator = 0, tau_weight = 0.5
        expected = 0.5 / (2 * kappa) * (0.001 ** 2)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)

    def test_linear_region(self):
        """Uses linear loss for large residuals."""
        # |u| > kappa should use linear
        u = jnp.array([0.1])  # > kappa=0.002
        tau = jnp.array([0.5])
        kappa = 0.002

        result = huber_quantile_loss(u, tau, kappa)

        # Expected: |tau - indicator| * (|u| - 0.5*kappa)
        expected = 0.5 * (0.1 - 0.5 * kappa)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)

    def test_asymmetry(self):
        """Loss is asymmetric based on tau (quantile level)."""
        u_pos = jnp.array([0.1])
        u_neg = jnp.array([-0.1])
        tau = jnp.array([0.8])  # High quantile

        loss_pos = huber_quantile_loss(u_pos, tau, kappa=0.002)
        loss_neg = huber_quantile_loss(u_neg, tau, kappa=0.002)

        # For high tau, positive residuals (underestimation) penalized more
        assert loss_pos > loss_neg

    def test_non_negative(self):
        """Loss is always non-negative."""
        u = jnp.array(np.random.randn(100).astype(np.float32))
        tau = jnp.linspace(0.01, 0.99, 100)
        result = huber_quantile_loss(u, tau, kappa=0.002)
        assert jnp.all(result >= 0)


class TestSpikeRateDistributionLoss:
    """Tests for spike rate distribution matching loss."""

    def test_output_scalar(self, jax_key):
        """Output is a scalar."""
        spikes = jnp.ones((4, 100, 50)) * 0.02
        target_rate = jnp.linspace(0.01, 0.05, 50)

        result = spike_rate_distribution_loss(spikes, target_rate, jax_key)

        assert result.shape == ()

    def test_zero_loss_perfect_match(self, jax_key):
        """Loss is low when distribution matches target."""
        n_neurons = 50
        target_rate = jnp.linspace(0.01, 0.05, n_neurons)

        # Create spikes that match target distribution
        np.random.seed(42)
        spikes = np.zeros((4, 100, n_neurons), dtype=np.float32)
        for i in range(n_neurons):
            spikes[:, :, i] = (
                np.random.rand(4, 100) < target_rate[i]
            ).astype(np.float32)
        spikes = jnp.array(spikes)

        result = spike_rate_distribution_loss(spikes, target_rate, jax_key)

        # Loss should be relatively small (not exactly zero due to finite samples)
        assert result < 0.1

    def test_high_loss_mismatch(self, jax_key):
        """Loss is high when distribution mismatches target."""
        n_neurons = 50
        target_rate = jnp.linspace(0.01, 0.05, n_neurons)

        # Create all-ones spikes (high firing rate)
        spikes = jnp.ones((4, 100, n_neurons))

        result_high = spike_rate_distribution_loss(spikes, target_rate, jax_key)

        # Create matched spikes
        np.random.seed(42)
        matched_spikes = np.zeros((4, 100, n_neurons), dtype=np.float32)
        for i in range(n_neurons):
            matched_spikes[:, :, i] = (
                np.random.rand(4, 100) < target_rate[i]
            ).astype(np.float32)
        matched_spikes = jnp.array(matched_spikes)

        result_matched = spike_rate_distribution_loss(
            matched_spikes, target_rate, jax_key
        )

        assert result_high > result_matched

    def test_reproducibility(self, jax_key):
        """Same key gives same result."""
        spikes = jnp.ones((4, 100, 50)) * 0.1
        target_rate = jnp.linspace(0.01, 0.05, 50)

        result1 = spike_rate_distribution_loss(spikes, target_rate, jax_key)
        result2 = spike_rate_distribution_loss(spikes, target_rate, jax_key)

        np.testing.assert_equal(result1, result2)


class TestSparseCategoricalCrossentropy:
    """Tests for sparse categorical cross-entropy."""

    def test_perfect_prediction(self):
        """Loss is low for correct predictions with high confidence."""
        # Create logits with high confidence
        logits = jnp.array([[10.0, -10.0], [-10.0, 10.0]])
        labels = jnp.array([0, 1])

        result = sparse_categorical_crossentropy(logits, labels, from_logits=True)

        # Loss should be very small for confident correct predictions
        assert jnp.all(result < 0.01)

    def test_wrong_prediction(self):
        """Loss is high for incorrect predictions."""
        logits = jnp.array([[10.0, -10.0], [-10.0, 10.0]])
        labels = jnp.array([1, 0])  # Wrong labels

        result = sparse_categorical_crossentropy(logits, labels, from_logits=True)

        # Loss should be high for confident wrong predictions
        assert jnp.all(result > 10.0)

    def test_shape(self):
        """Output has batch dimension."""
        batch_size = 16
        num_classes = 10
        logits = jnp.ones((batch_size, num_classes))
        labels = jnp.zeros((batch_size,), dtype=jnp.int32)

        result = sparse_categorical_crossentropy(logits, labels, from_logits=True)

        assert result.shape == (batch_size,)

    def test_non_negative(self):
        """Cross-entropy is always non-negative."""
        np.random.seed(42)
        logits = jnp.array(np.random.randn(100, 10).astype(np.float32))
        labels = jnp.array(np.random.randint(0, 10, 100).astype(np.int32))

        result = sparse_categorical_crossentropy(logits, labels, from_logits=True)

        assert jnp.all(result >= 0)

    def test_gradient_exists(self):
        """Gradients can be computed."""
        logits = jnp.ones((4, 10))
        labels = jnp.array([0, 1, 2, 3])

        def loss_fn(x):
            return jnp.mean(
                sparse_categorical_crossentropy(x, labels, from_logits=True)
            )

        grad = jax.grad(loss_fn)(logits)
        assert grad.shape == logits.shape
        assert not jnp.any(jnp.isnan(grad))


class TestWeightedCrossentropy:
    """Tests for weighted cross-entropy."""

    def test_uniform_weights(self):
        """With uniform weights, equals unweighted mean."""
        logits = jnp.ones((4, 10))
        labels = jnp.array([0, 1, 2, 3])
        weights = jnp.ones((4,))

        result = weighted_crossentropy(logits, labels, weights, from_logits=True)
        unweighted = jnp.mean(
            sparse_categorical_crossentropy(logits, labels, from_logits=True)
        )

        np.testing.assert_allclose(result, unweighted, rtol=1e-5)

    def test_zero_weight_ignored(self):
        """Zero-weighted samples don't contribute."""
        logits = jnp.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
        labels = jnp.array([0, 1, 1])  # Third is wrong
        weights = jnp.array([1.0, 1.0, 0.0])  # Ignore third sample

        result = weighted_crossentropy(logits, labels, weights, from_logits=True)

        # Should be low because only correct predictions are weighted
        assert result < 0.1


class TestBinaryCrossentropy:
    """Tests for binary cross-entropy."""

    def test_perfect_prediction(self):
        """Loss is low for correct predictions."""
        logits = jnp.array([10.0, -10.0])
        labels = jnp.array([1.0, 0.0])

        result = binary_crossentropy(logits, labels, from_logits=True)

        assert jnp.all(result < 0.01)

    def test_shape(self):
        """Output matches input shape."""
        logits = jnp.ones((10, 5))
        labels = jnp.ones((10, 5))

        result = binary_crossentropy(logits, labels, from_logits=True)

        assert result.shape == logits.shape


class TestMeanSquaredError:
    """Tests for MSE loss."""

    def test_zero_error(self):
        """MSE is zero when predictions equal targets."""
        pred = jnp.ones((4, 10))
        target = jnp.ones((4, 10))

        result = mean_squared_error(pred, target)

        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_positive(self):
        """MSE is always non-negative."""
        np.random.seed(42)
        pred = jnp.array(np.random.randn(100, 10).astype(np.float32))
        target = jnp.array(np.random.randn(100, 10).astype(np.float32))

        result = mean_squared_error(pred, target)

        assert jnp.all(result >= 0)

    def test_shape(self):
        """Output has batch dimension."""
        pred = jnp.ones((16, 10))
        target = jnp.zeros((16, 10))

        result = mean_squared_error(pred, target)

        assert result.shape == (16,)


# =============================================================================
# Garrett Firing Rates Tests
# =============================================================================


class TestGarrettFiringRates:
    """Tests for Garrett firing rate loading and interpolation."""

    def test_load_garrett_firing_rates(self):
        """Test loading Garrett firing rates from file."""
        from v1_jax.data.network_loader import load_garrett_firing_rates

        try:
            rates = load_garrett_firing_rates('/nvmessd/yinzi/GLIF_network')

            # Full network has 51978 neurons
            assert rates.shape == (51978,)
            assert rates.dtype == np.float32

            # Rates should be sorted (for quantile matching)
            assert np.all(np.diff(rates) >= 0)

            # Rates should be non-negative
            assert rates.min() >= 0
        except FileNotFoundError:
            pytest.skip("Garrett firing rates file not available")

    def test_load_garrett_with_interpolation(self):
        """Test loading with interpolation to different neuron count."""
        from v1_jax.data.network_loader import load_garrett_firing_rates

        try:
            n_neurons = 1000
            rates = load_garrett_firing_rates(
                '/nvmessd/yinzi/GLIF_network',
                n_neurons=n_neurons
            )

            assert rates.shape == (n_neurons,)
            assert rates.dtype == np.float32

            # Interpolated rates should still be sorted
            assert np.all(np.diff(rates) >= 0)
        except FileNotFoundError:
            pytest.skip("Garrett firing rates file not available")

    def test_rate_distribution_regularizer(self):
        """Test SpikeRateDistributionRegularizer integration."""
        from v1_jax.training.regularizers import SpikeRateDistributionRegularizer

        n_neurons = 100
        target_rates = jnp.linspace(0.01, 0.05, n_neurons)

        reg = SpikeRateDistributionRegularizer(
            target_rates=target_rates,
            rate_cost=1.0,
        )

        # Test with mock spikes
        spikes = jnp.ones((4, 100, n_neurons)) * 0.02
        key = jax.random.PRNGKey(42)

        loss = reg(spikes, key)

        assert loss.shape == ()
        assert float(loss) >= 0
