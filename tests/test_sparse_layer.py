"""Tests for sparse layer implementations.

Tests cover:
- BCOO sparse matrix operations
- Input layer processing
- Recurrent connectivity with delays
- JIT compilation compatibility
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

from v1_jax.nn.sparse_layer import (
    SparseConnectivity,
    sparse_matmul_bcoo,
    sparse_input_layer,
    create_recurrent_matmul_fn,
    InputLayer,
    RecurrentLayer,
    prepare_recurrent_connectivity,
    prepare_input_connectivity,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_sparse_config():
    """Small sparse connectivity for quick tests."""
    return {
        'n_neurons': 50,
        'n_receptors': 4,
        'n_inputs': 100,
        'max_delay': 5,
        'batch_size': 4,
        'seq_len': 10,
        'sparsity': 0.1,  # 10% connectivity
    }


@pytest.fixture
def random_connectivity(small_sparse_config, global_seed):
    """Generate random sparse connectivity with unique indices."""
    np.random.seed(global_seed)
    cfg = small_sparse_config

    n_targets = cfg['n_neurons'] * cfg['n_receptors']
    n_sources = cfg['n_inputs']
    n_connections = int(n_targets * n_sources * cfg['sparsity'])

    # Generate unique indices by sampling without replacement
    # Create all possible indices and sample from them
    all_indices = []
    for t in range(n_targets):
        for s in range(n_sources):
            all_indices.append((t, s))

    sampled_indices = np.random.choice(
        len(all_indices), size=min(n_connections, len(all_indices)), replace=False
    )
    indices = np.array([all_indices[i] for i in sampled_indices], dtype=np.int64)

    # Random weights (mix of positive and negative)
    weights = np.random.randn(len(indices)).astype(np.float32) * 0.1

    return {
        'indices': indices,
        'weights': weights,
        'shape': (n_targets, n_sources),
    }


@pytest.fixture
def random_recurrent(small_sparse_config, global_seed):
    """Generate random recurrent connectivity with delays."""
    np.random.seed(global_seed)
    cfg = small_sparse_config

    n_targets = cfg['n_neurons'] * cfg['n_receptors']
    n_sources = cfg['n_neurons'] * cfg['max_delay']
    n_connections = int(cfg['n_neurons'] * cfg['n_neurons'] * cfg['sparsity'])

    # Random indices (pre-delay encoding)
    target_idx = np.random.randint(0, n_targets, size=n_connections)
    source_idx = np.random.randint(0, cfg['n_neurons'], size=n_connections)
    indices = np.stack([target_idx, source_idx], axis=-1).astype(np.int64)

    # Random weights
    weights = np.random.randn(n_connections).astype(np.float32) * 0.1

    # Random delays
    delays = np.random.randint(1, cfg['max_delay'] + 1, size=n_connections).astype(np.float32)

    return {
        'indices': indices,
        'weights': weights,
        'delays': delays,
    }


# =============================================================================
# SparseConnectivity Tests
# =============================================================================


class TestSparseConnectivity:
    """Tests for SparseConnectivity class."""

    def test_from_arrays(self, random_connectivity):
        """Test creating SparseConnectivity from arrays."""
        conn = SparseConnectivity.from_arrays(
            random_connectivity['indices'],
            random_connectivity['weights'],
            random_connectivity['shape'],
        )

        assert conn.indices.shape == random_connectivity['indices'].shape
        assert conn.weights.shape == random_connectivity['weights'].shape
        assert conn.shape == random_connectivity['shape']

    def test_to_bcoo(self, random_connectivity):
        """Test conversion to BCOO format."""
        conn = SparseConnectivity.from_arrays(
            random_connectivity['indices'],
            random_connectivity['weights'],
            random_connectivity['shape'],
        )

        bcoo = conn.to_bcoo()

        assert isinstance(bcoo, sparse.BCOO)
        assert bcoo.shape == random_connectivity['shape']

    def test_bcoo_matmul(self, random_connectivity, small_sparse_config):
        """Test sparse matrix-vector multiplication."""
        cfg = small_sparse_config
        conn = SparseConnectivity.from_arrays(
            random_connectivity['indices'],
            random_connectivity['weights'],
            random_connectivity['shape'],
        )

        x = jnp.ones((cfg['batch_size'], cfg['n_inputs']))

        # Test sparse matmul
        result = sparse_matmul_bcoo(conn, x, transpose_x=True)

        assert result.shape == (cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors'])

    def test_bcoo_vs_dense(self, small_sparse_config):
        """Test that sparse matmul matches dense matmul."""
        cfg = small_sparse_config

        # Create a small, simple connectivity for exact comparison
        # Use a deterministic pattern to avoid numerical precision issues
        n_targets = 10
        n_sources = 5

        # Simple diagonal-like pattern
        indices = np.array([
            [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
            [5, 0], [6, 1], [7, 2], [8, 3], [9, 4],
        ], dtype=np.int64)
        weights = np.ones(10, dtype=np.float32) * 0.5

        conn = SparseConnectivity.from_arrays(
            indices, weights, (n_targets, n_sources)
        )

        x = jnp.ones((2, n_sources))

        # Sparse result
        sparse_result = sparse_matmul_bcoo(conn, x, transpose_x=True)

        # Dense result for comparison
        bcoo = conn.to_bcoo()
        dense_w = bcoo.todense()
        dense_result = (dense_w @ x.T).T

        # These should match exactly for simple patterns
        assert jnp.allclose(sparse_result, dense_result, rtol=1e-5, atol=1e-6)


# =============================================================================
# Input Layer Tests
# =============================================================================


class TestInputLayer:
    """Tests for InputLayer class."""

    def test_input_layer_creation(self, random_connectivity, small_sparse_config):
        """Test InputLayer creation."""
        cfg = small_sparse_config
        bkg_weights = np.zeros(cfg['n_neurons'] * cfg['n_receptors'], dtype=np.float32)

        layer = InputLayer(
            indices=random_connectivity['indices'],
            weights=random_connectivity['weights'],
            dense_shape=random_connectivity['shape'],
            bkg_weights=bkg_weights,
        )

        assert layer.n_targets == cfg['n_neurons'] * cfg['n_receptors']
        assert layer.n_inputs == cfg['n_inputs']

    def test_input_layer_call(self, random_connectivity, small_sparse_config, jax_key):
        """Test InputLayer forward pass."""
        cfg = small_sparse_config
        bkg_weights = np.ones(cfg['n_neurons'] * cfg['n_receptors'], dtype=np.float32) * 0.01

        layer = InputLayer(
            indices=random_connectivity['indices'],
            weights=random_connectivity['weights'],
            dense_shape=random_connectivity['shape'],
            bkg_weights=bkg_weights,
        )

        inputs = jnp.ones((cfg['batch_size'], cfg['seq_len'], cfg['n_inputs'])) * 0.1

        output = layer(inputs, key=jax_key)

        assert output.shape == (cfg['batch_size'], cfg['seq_len'], cfg['n_neurons'] * cfg['n_receptors'])

    def test_input_layer_jit(self, random_connectivity, small_sparse_config, jax_key):
        """Test that InputLayer can be JIT compiled."""
        cfg = small_sparse_config
        bkg_weights = np.zeros(cfg['n_neurons'] * cfg['n_receptors'], dtype=np.float32)

        layer = InputLayer(
            indices=random_connectivity['indices'],
            weights=random_connectivity['weights'],
            dense_shape=random_connectivity['shape'],
            bkg_weights=bkg_weights,
        )

        @jax.jit
        def forward(inputs, key):
            return layer(inputs, key=key)

        inputs = jnp.ones((cfg['batch_size'], cfg['seq_len'], cfg['n_inputs'])) * 0.1

        # Should compile and run
        output = forward(inputs, jax_key)
        assert output.shape == (cfg['batch_size'], cfg['seq_len'], cfg['n_neurons'] * cfg['n_receptors'])


# =============================================================================
# Recurrent Layer Tests
# =============================================================================


class TestRecurrentLayer:
    """Tests for RecurrentLayer class."""

    def test_prepare_recurrent_connectivity(self, random_recurrent, small_sparse_config):
        """Test recurrent connectivity preparation with delays."""
        cfg = small_sparse_config

        indices, weights, shape = prepare_recurrent_connectivity(
            random_recurrent['indices'],
            random_recurrent['weights'],
            random_recurrent['delays'],
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['max_delay'],
        )

        # Shape should include delay dimension
        assert shape == (cfg['n_receptors'] * cfg['n_neurons'], cfg['n_neurons'] * cfg['max_delay'])

        # Source indices should be in extended range
        assert indices[:, 1].max() < cfg['n_neurons'] * cfg['max_delay']

    def test_recurrent_layer_creation(self, random_recurrent, small_sparse_config):
        """Test RecurrentLayer creation."""
        cfg = small_sparse_config

        indices, weights, shape = prepare_recurrent_connectivity(
            random_recurrent['indices'],
            random_recurrent['weights'],
            random_recurrent['delays'],
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['max_delay'],
        )

        layer = RecurrentLayer(
            indices=indices,
            weights=weights,
            dense_shape=shape,
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
        )

        assert layer.n_neurons == cfg['n_neurons']
        assert layer.n_receptors == cfg['n_receptors']

    def test_recurrent_layer_call(self, random_recurrent, small_sparse_config):
        """Test RecurrentLayer forward pass."""
        cfg = small_sparse_config

        indices, weights, shape = prepare_recurrent_connectivity(
            random_recurrent['indices'],
            random_recurrent['weights'],
            random_recurrent['delays'],
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['max_delay'],
        )

        layer = RecurrentLayer(
            indices=indices,
            weights=weights,
            dense_shape=shape,
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
        )

        # Create spike buffer
        z_buf = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['max_delay']))

        output = layer(z_buf)

        assert output.shape == (cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors'])

    def test_recurrent_matmul_with_spikes(self, random_recurrent, small_sparse_config):
        """Test recurrent matmul produces output from spikes."""
        cfg = small_sparse_config

        indices, weights, shape = prepare_recurrent_connectivity(
            random_recurrent['indices'],
            random_recurrent['weights'],
            random_recurrent['delays'],
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['max_delay'],
        )

        layer = RecurrentLayer(
            indices=indices,
            weights=weights,
            dense_shape=shape,
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
        )

        # Create spike buffer with some spikes
        z_buf = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['max_delay']))
        # Add spikes in first delay slot
        z_buf = z_buf.at[:, :cfg['n_neurons']].set(1.0)

        output = layer(z_buf)

        # Should have non-zero output due to connectivity
        assert jnp.abs(output).sum() > 0

    def test_recurrent_layer_jit(self, random_recurrent, small_sparse_config):
        """Test that RecurrentLayer can be JIT compiled."""
        cfg = small_sparse_config

        indices, weights, shape = prepare_recurrent_connectivity(
            random_recurrent['indices'],
            random_recurrent['weights'],
            random_recurrent['delays'],
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['max_delay'],
        )

        layer = RecurrentLayer(
            indices=indices,
            weights=weights,
            dense_shape=shape,
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
        )

        @jax.jit
        def forward(z_buf):
            return layer(z_buf)

        z_buf = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['max_delay']))

        # Should compile and run
        output = forward(z_buf)
        assert output.shape == (cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors'])


# =============================================================================
# Delay Encoding Tests
# =============================================================================


class TestDelayEncoding:
    """Tests for synaptic delay encoding."""

    def test_delay_encoding_range(self, small_sparse_config, global_seed):
        """Test that delay encoding produces correct index range."""
        np.random.seed(global_seed)
        cfg = small_sparse_config

        n_connections = 100
        target_idx = np.random.randint(0, cfg['n_neurons'] * cfg['n_receptors'], size=n_connections)
        source_idx = np.random.randint(0, cfg['n_neurons'], size=n_connections)
        indices = np.stack([target_idx, source_idx], axis=-1).astype(np.int64)
        weights = np.random.randn(n_connections).astype(np.float32)

        # All delays = 1 (minimum)
        delays_min = np.ones(n_connections)
        _, _, shape_min = prepare_recurrent_connectivity(
            indices, weights, delays_min,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
        )

        # All delays = max_delay
        delays_max = np.ones(n_connections) * cfg['max_delay']
        new_indices, _, _ = prepare_recurrent_connectivity(
            indices, weights, delays_max,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
        )

        # Max source index should be near n_neurons * max_delay
        assert new_indices[:, 1].max() < cfg['n_neurons'] * cfg['max_delay']
        assert new_indices[:, 1].min() >= cfg['n_neurons'] * (cfg['max_delay'] - 1)

    def test_delay_clipping(self, small_sparse_config, global_seed):
        """Test that delays are clipped to valid range."""
        np.random.seed(global_seed)
        cfg = small_sparse_config

        n_connections = 50
        target_idx = np.random.randint(0, cfg['n_neurons'] * cfg['n_receptors'], size=n_connections)
        source_idx = np.random.randint(0, cfg['n_neurons'], size=n_connections)
        indices = np.stack([target_idx, source_idx], axis=-1).astype(np.int64)
        weights = np.random.randn(n_connections).astype(np.float32)

        # Delays outside valid range
        delays = np.array([0, -1, 100, 200] + [2] * (n_connections - 4), dtype=np.float32)

        new_indices, _, _ = prepare_recurrent_connectivity(
            indices, weights, delays,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
        )

        # All indices should be valid
        assert new_indices[:, 1].min() >= 0
        assert new_indices[:, 1].max() < cfg['n_neurons'] * cfg['max_delay']


# =============================================================================
# Voltage Scaling Tests
# =============================================================================


class TestVoltageScaling:
    """Tests for voltage-based weight scaling."""

    def test_input_weight_scaling(self, random_connectivity, small_sparse_config):
        """Test input weight scaling by voltage."""
        cfg = small_sparse_config

        # Create neuron types and voltage scale
        n_types = 4
        node_type_ids = np.random.randint(0, n_types, size=cfg['n_neurons'])
        voltage_scale = np.array([20.0, 25.0, 18.0, 22.0], dtype=np.float32)

        indices, weights_scaled, shape = prepare_input_connectivity(
            random_connectivity['indices'],
            random_connectivity['weights'],
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['n_inputs'],
            voltage_scale=voltage_scale,
            node_type_ids=node_type_ids,
        )

        # Weights should be scaled
        assert weights_scaled.shape == random_connectivity['weights'].shape

    def test_recurrent_weight_scaling(self, random_recurrent, small_sparse_config):
        """Test recurrent weight scaling by voltage."""
        cfg = small_sparse_config

        n_types = 4
        node_type_ids = np.random.randint(0, n_types, size=cfg['n_neurons'])
        voltage_scale = np.array([20.0, 25.0, 18.0, 22.0], dtype=np.float32)

        indices, weights_scaled, shape = prepare_recurrent_connectivity(
            random_recurrent['indices'],
            random_recurrent['weights'],
            random_recurrent['delays'],
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['max_delay'],
            voltage_scale=voltage_scale,
            node_type_ids=node_type_ids,
        )

        # Weights should be scaled
        assert weights_scaled.shape == random_recurrent['weights'].shape


# =============================================================================
# Integration Tests
# =============================================================================


class TestSparseLayerIntegration:
    """Integration tests combining input and recurrent layers."""

    def test_full_forward_pass(self, random_connectivity, random_recurrent, small_sparse_config, jax_key):
        """Test full forward pass through both layers."""
        cfg = small_sparse_config

        # Create input layer
        bkg_weights = np.zeros(cfg['n_neurons'] * cfg['n_receptors'], dtype=np.float32)
        input_layer = InputLayer(
            indices=random_connectivity['indices'],
            weights=random_connectivity['weights'],
            dense_shape=random_connectivity['shape'],
            bkg_weights=bkg_weights,
        )

        # Create recurrent layer
        rec_indices, rec_weights, rec_shape = prepare_recurrent_connectivity(
            random_recurrent['indices'],
            random_recurrent['weights'],
            random_recurrent['delays'],
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['max_delay'],
        )
        rec_layer = RecurrentLayer(
            indices=rec_indices,
            weights=rec_weights,
            dense_shape=rec_shape,
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
        )

        # Create inputs
        inputs = jnp.ones((cfg['batch_size'], cfg['seq_len'], cfg['n_inputs'])) * 0.1
        z_buf = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['max_delay']))

        # Forward pass
        input_current = input_layer(inputs, key=jax_key)
        rec_current = rec_layer(z_buf)

        # Both should produce valid output
        assert input_current.shape == (cfg['batch_size'], cfg['seq_len'], cfg['n_neurons'] * cfg['n_receptors'])
        assert rec_current.shape == (cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors'])

    def test_gradient_flow(self, random_connectivity, small_sparse_config, jax_key):
        """Test gradient flow through sparse layers."""
        cfg = small_sparse_config

        bkg_weights = np.zeros(cfg['n_neurons'] * cfg['n_receptors'], dtype=np.float32)
        input_layer = InputLayer(
            indices=random_connectivity['indices'],
            weights=random_connectivity['weights'],
            dense_shape=random_connectivity['shape'],
            bkg_weights=bkg_weights,
        )

        def loss_fn(inputs):
            output = input_layer(inputs, key=jax_key)
            return jnp.sum(output)

        inputs = jnp.ones((cfg['batch_size'], cfg['seq_len'], cfg['n_inputs'])) * 0.1
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(inputs)

        # Gradients should exist and be finite
        assert grads.shape == inputs.shape
        assert jnp.all(jnp.isfinite(grads))
