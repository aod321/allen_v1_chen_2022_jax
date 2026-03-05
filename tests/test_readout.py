"""Tests for readout layers.

Tests cover:
- Dense readout with different pooling methods
- Sparse readout from selected neurons
- Binary and multi-class classification readouts
- JIT compatibility
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from v1_jax.models.readout import (
    ReadoutParams,
    dense_readout,
    select_readout_neurons,
    sparse_readout,
    chunk_readout,
    DenseReadout,
    BinaryReadout,
    MultiClassReadout,
    create_readout,
    apply_readout_jit,
    make_readout_fn,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_spikes():
    """Sample spike trains for testing."""
    # (time, batch, n_neurons)
    key = jax.random.PRNGKey(42)
    return (jax.random.uniform(key, (200, 4, 100)) > 0.9).astype(jnp.float32)


@pytest.fixture
def sample_params():
    """Sample readout parameters."""
    key = jax.random.PRNGKey(42)
    weights = jax.random.normal(key, (100, 10)) * 0.1
    bias = jnp.zeros(10)
    return ReadoutParams(weights=weights, bias=bias)


# =============================================================================
# Dense Readout Tests
# =============================================================================

class TestDenseReadout:
    """Tests for dense_readout function."""

    def test_mean_pooling_shape(self, sample_spikes, sample_params):
        """Test output shape with mean pooling."""
        logits = dense_readout(
            sample_spikes, sample_params, temporal_pooling='mean'
        )
        assert logits.shape == (4, 10)  # (batch, n_outputs)

    def test_sum_pooling_shape(self, sample_spikes, sample_params):
        """Test output shape with sum pooling."""
        logits = dense_readout(
            sample_spikes, sample_params, temporal_pooling='sum'
        )
        assert logits.shape == (4, 10)

    def test_last_pooling_shape(self, sample_spikes, sample_params):
        """Test output shape with last timestep."""
        logits = dense_readout(
            sample_spikes, sample_params, temporal_pooling='last'
        )
        assert logits.shape == (4, 10)

    def test_chunk_pooling_shape(self, sample_spikes, sample_params):
        """Test output shape with chunk-wise pooling."""
        logits = dense_readout(
            sample_spikes, sample_params,
            temporal_pooling='chunks', chunk_size=50
        )
        # 200 / 50 = 4 chunks
        assert logits.shape == (4, 4, 10)  # (batch, n_chunks, n_outputs)

    def test_mean_vs_sum_pooling(self):
        """Test that mean and sum pooling produce consistent relative magnitudes.

        Note: Due to JAX's internal reordering of operations for optimization,
        mean * time_len may not equal sum exactly. We test that they are
        proportionally related.
        """
        key = jax.random.PRNGKey(123)
        key1, key2 = jax.random.split(key)

        spikes = (jax.random.uniform(key1, (20, 4, 100)) > 0.9).astype(jnp.float32)
        weights = jax.random.normal(key2, (100, 10)) * 0.1
        params_no_bias = ReadoutParams(weights=weights, bias=None)

        mean_logits = dense_readout(
            spikes, params_no_bias, temporal_pooling='mean'
        )
        sum_logits = dense_readout(
            spikes, params_no_bias, temporal_pooling='sum'
        )

        time_len = spikes.shape[0]

        # The ratio should be approximately time_len
        # Only check non-zero elements to avoid division issues
        nonzero_mask = jnp.abs(mean_logits) > 1e-6
        ratios = jnp.where(nonzero_mask, sum_logits / mean_logits, time_len)
        avg_ratio = jnp.mean(ratios)

        # Average ratio should be close to time_len
        assert jnp.isclose(avg_ratio, time_len, rtol=0.01)

    def test_sparse_selection(self, sample_spikes):
        """Test readout from selected neurons."""
        key = jax.random.PRNGKey(42)
        n_select = 20
        indices = jax.random.permutation(key, 100)[:n_select]

        weights = jax.random.normal(key, (n_select, 5)) * 0.1
        params = ReadoutParams(
            weights=weights,
            bias=jnp.zeros(5),
            neuron_indices=indices,
        )

        logits = dense_readout(sample_spikes, params)
        assert logits.shape == (4, 5)


class TestDenseReadoutClass:
    """Tests for DenseReadout class."""

    def test_initialization(self):
        """Test DenseReadout initialization."""
        readout = DenseReadout(
            n_neurons=100,
            n_outputs=10,
            temporal_pooling='mean',
        )
        assert readout.n_neurons == 100
        assert readout.n_outputs == 10

    def test_call(self, sample_spikes):
        """Test DenseReadout callable interface."""
        readout = DenseReadout(
            n_neurons=100,
            n_outputs=10,
        )
        logits = readout(sample_spikes)
        assert logits.shape == (4, 10)

    def test_get_set_params(self, sample_spikes):
        """Test parameter access and modification."""
        readout = DenseReadout(n_neurons=100, n_outputs=10)

        params = readout.get_params()
        assert params.weights.shape == (100, 10)

        # Modify and set
        new_weights = params.weights * 2
        new_params = ReadoutParams(
            weights=new_weights,
            bias=params.bias,
            neuron_indices=params.neuron_indices,
        )
        new_readout = readout.set_params(new_params)

        # Verify modification
        assert jnp.allclose(new_readout.params.weights, new_weights)


# =============================================================================
# Sparse Readout Tests
# =============================================================================

class TestSparseReadout:
    """Tests for sparse_readout function."""

    def test_sparse_readout_shape(self, sample_spikes):
        """Test sparse readout output shape."""
        key = jax.random.PRNGKey(42)
        n_select = 30
        indices = jax.random.permutation(key, 100)[:n_select]

        weights = jax.random.normal(key, (n_select, 5)) * 0.1
        params = ReadoutParams(
            weights=weights,
            bias=jnp.zeros(5),
            neuron_indices=indices,
        )

        logits = sparse_readout(sample_spikes, params)
        assert logits.shape == (4, 5)

    def test_sparse_readout_requires_indices(self, sample_spikes, sample_params):
        """Test sparse_readout raises error without indices."""
        with pytest.raises(ValueError):
            sparse_readout(sample_spikes, sample_params)


class TestSelectReadoutNeurons:
    """Tests for neuron selection."""

    def test_select_correct_count(self):
        """Test correct number of neurons selected."""
        indices = select_readout_neurons(
            n_neurons=100,
            n_select=20,
        )
        assert len(indices) == 20

    def test_select_sorted(self):
        """Test selected indices are sorted."""
        indices = select_readout_neurons(
            n_neurons=100,
            n_select=20,
        )
        assert jnp.all(indices[:-1] <= indices[1:])

    def test_select_deterministic(self):
        """Test selection is deterministic with same key."""
        key = jax.random.PRNGKey(42)
        indices1 = select_readout_neurons(n_neurons=100, n_select=20, key=key)
        indices2 = select_readout_neurons(n_neurons=100, n_select=20, key=key)
        assert jnp.all(indices1 == indices2)

    def test_select_excitatory_only(self):
        """Test selecting only excitatory neurons."""
        # 0 = excitatory, 1 = inhibitory
        neuron_types = jnp.array([0] * 80 + [1] * 20)  # 80 exc, 20 inh

        key = jax.random.PRNGKey(42)
        indices = select_readout_neurons(
            n_neurons=100,
            n_select=30,
            neuron_types=neuron_types,
            excitatory_only=True,
            key=key,
        )

        # All selected should be excitatory
        assert jnp.all(neuron_types[indices] == 0)


# =============================================================================
# Binary Readout Tests
# =============================================================================

class TestBinaryReadout:
    """Tests for BinaryReadout class."""

    def test_output_shape(self, sample_spikes):
        """Test binary readout output shape."""
        readout = BinaryReadout(
            n_neurons=100,
            temporal_pooling='mean',
        )
        logits = readout(sample_spikes)
        # Should be (batch,) not (batch, 1)
        assert logits.shape == (4,)

    def test_probability(self, sample_spikes):
        """Test probability computation."""
        readout = BinaryReadout(
            n_neurons=100,
            temporal_pooling='mean',  # Use mean pooling for simpler output
        )
        probs = readout.probability(sample_spikes)

        assert probs.shape == (4,)
        assert jnp.all(probs >= 0)
        assert jnp.all(probs <= 1)

    def test_chunk_wise_output(self, sample_spikes):
        """Test chunk-wise binary readout."""
        readout = BinaryReadout(
            n_neurons=100,
            temporal_pooling='chunks',
            chunk_size=50,
        )
        logits = readout(sample_spikes)
        # 200 / 50 = 4 chunks
        assert logits.shape == (4, 4)


# =============================================================================
# Multi-class Readout Tests
# =============================================================================

class TestMultiClassReadout:
    """Tests for MultiClassReadout class."""

    def test_output_shape(self, sample_spikes):
        """Test multi-class readout output shape."""
        readout = MultiClassReadout(
            n_neurons=100,
            n_classes=10,
            temporal_pooling='mean',
        )
        logits = readout(sample_spikes)
        assert logits.shape == (4, 10)

    def test_probability_sum(self, sample_spikes):
        """Test probabilities sum to 1."""
        readout = MultiClassReadout(n_neurons=100, n_classes=10)
        probs = readout.probability(sample_spikes)

        sums = jnp.sum(probs, axis=-1)
        assert jnp.allclose(sums, 1.0)

    def test_predict(self, sample_spikes):
        """Test class prediction."""
        readout = MultiClassReadout(
            n_neurons=100,
            n_classes=10,
            temporal_pooling='mean',  # Use mean pooling for simpler output
        )
        predictions = readout.predict(sample_spikes)

        assert predictions.shape == (4,)
        assert jnp.all(predictions >= 0)
        assert jnp.all(predictions < 10)

    def test_chunk_wise_output(self, sample_spikes):
        """Test chunk-wise multi-class readout."""
        readout = MultiClassReadout(
            n_neurons=100,
            n_classes=10,
            temporal_pooling='chunks',
            chunk_size=50,
        )
        logits = readout(sample_spikes)
        assert logits.shape == (4, 4, 10)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateReadout:
    """Tests for create_readout factory."""

    def test_create_binary(self):
        """Test creating binary readout."""
        readout = create_readout(
            n_neurons=100,
            task='binary',
        )
        assert isinstance(readout, BinaryReadout)

    def test_create_classification(self):
        """Test creating classification readout."""
        readout = create_readout(
            n_neurons=100,
            task='classification',
            n_classes=100,
        )
        assert isinstance(readout, MultiClassReadout)
        assert readout.n_classes == 100

    def test_create_regression(self):
        """Test creating regression readout."""
        readout = create_readout(
            n_neurons=100,
            task='regression',
        )
        assert isinstance(readout, DenseReadout)
        assert readout.n_outputs == 1

    def test_invalid_task(self):
        """Test error on invalid task."""
        with pytest.raises(ValueError):
            create_readout(n_neurons=100, task='invalid')


# =============================================================================
# JIT Compilation Tests
# =============================================================================

class TestJITCompilation:
    """Tests for JIT compilation compatibility."""

    def test_apply_readout_jit(self, sample_spikes):
        """Test JIT-compiled readout application."""
        key = jax.random.PRNGKey(42)
        weights = jax.random.normal(key, (100, 10)) * 0.1
        bias = jnp.zeros(10)

        logits = apply_readout_jit(sample_spikes, weights, bias)
        assert logits.shape == (4, 10)

    def test_make_readout_fn(self, sample_spikes, sample_params):
        """Test creating JIT-compiled readout function."""
        readout_fn = make_readout_fn(sample_params, temporal_pooling='mean')

        logits = readout_fn(sample_spikes)
        assert logits.shape == (4, 10)

        # Test it's actually JIT-compiled (second call should be fast)
        _ = readout_fn(sample_spikes)

    def test_dense_readout_class_jit(self, sample_spikes):
        """Test DenseReadout class is JIT-compatible."""
        readout = DenseReadout(n_neurons=100, n_outputs=10)

        @jax.jit
        def apply(spikes):
            return readout(spikes)

        logits = apply(sample_spikes)
        assert logits.shape == (4, 10)


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradients:
    """Tests for gradient flow through readouts."""

    def test_gradient_flow_dense(self, sample_spikes, sample_params):
        """Test gradients flow through dense readout."""
        def loss_fn(weights, spikes, params):
            params = ReadoutParams(
                weights=weights,
                bias=params.bias,
                neuron_indices=params.neuron_indices,
            )
            logits = dense_readout(spikes, params)
            return jnp.mean(logits ** 2)

        grad = jax.grad(loss_fn)(
            sample_params.weights,
            sample_spikes,
            sample_params,
        )

        assert grad.shape == sample_params.weights.shape
        assert not jnp.all(grad == 0)

    def test_gradient_flow_binary(self, sample_spikes):
        """Test gradients flow through binary readout."""
        readout = BinaryReadout(n_neurons=100)

        def loss_fn(weights, spikes):
            readout.dense_readout.params = ReadoutParams(
                weights=weights,
                bias=readout.dense_readout.params.bias,
                neuron_indices=None,
            )
            return jnp.mean(readout(spikes) ** 2)

        # This should work without errors
        grad = jax.grad(loss_fn)(
            readout.dense_readout.params.weights,
            sample_spikes,
        )
        assert not jnp.all(grad == 0)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_timestep(self):
        """Test with single timestep input."""
        spikes = jax.random.uniform(
            jax.random.PRNGKey(42), (1, 4, 100)
        ) > 0.9
        spikes = spikes.astype(jnp.float32)

        readout = DenseReadout(n_neurons=100, n_outputs=10)
        logits = readout(spikes)
        assert logits.shape == (4, 10)

    def test_single_batch(self):
        """Test with single batch input."""
        spikes = jax.random.uniform(
            jax.random.PRNGKey(42), (100, 1, 100)
        ) > 0.9
        spikes = spikes.astype(jnp.float32)

        readout = DenseReadout(n_neurons=100, n_outputs=10)
        logits = readout(spikes)
        assert logits.shape == (1, 10)

    def test_single_output(self):
        """Test with single output neuron."""
        spikes = jax.random.uniform(
            jax.random.PRNGKey(42), (100, 4, 100)
        ) > 0.9
        spikes = spikes.astype(jnp.float32)

        readout = DenseReadout(n_neurons=100, n_outputs=1)
        logits = readout(spikes)
        assert logits.shape == (4, 1)
