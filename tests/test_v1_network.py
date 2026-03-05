"""Tests for V1 network module.

Tests cover:
- Network initialization
- State management
- Forward pass
- Integration with GLIF3 and sparse layers
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

from v1_jax.models.v1_network import (
    V1NetworkConfig,
    V1NetworkState,
    V1NetworkOutput,
    V1Network,
    v1_network_step,
    make_v1_forward_fn,
    make_v1_step_fn,
)
from v1_jax.nn.glif3_cell import GLIF3Cell, GLIF3Params, GLIF3State
from v1_jax.nn.sparse_layer import InputLayer, RecurrentLayer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_network_config():
    """Small network configuration for testing."""
    return V1NetworkConfig(
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        max_delay=3,
        input_weight_scale=1.0,
        recurrent_weight_scale=1.0,
    )


@pytest.fixture
def mock_network_data():
    """Create mock network data for testing."""
    n_neurons = 100
    n_receptors = 4
    n_inputs = 50
    n_synapses = 500
    n_input_synapses = 200
    n_types = 5

    # Node parameters per type
    node_params = {
        'V_th': np.random.uniform(-50, -40, n_types).astype(np.float32),
        'E_L': np.random.uniform(-70, -60, n_types).astype(np.float32),
        'V_reset': np.random.uniform(-70, -65, n_types).astype(np.float32),
        'C_m': np.random.uniform(50, 100, n_types).astype(np.float32),
        'g': np.random.uniform(3, 8, n_types).astype(np.float32),
        't_ref': np.random.uniform(1, 4, n_types).astype(np.float32),
        'tau_syn': np.random.uniform(2, 10, (n_types, n_receptors)).astype(np.float32),
        'k': np.random.uniform(0.001, 0.01, (n_types, 2)).astype(np.float32),
        'asc_amps': np.random.uniform(-1, 1, (n_types, 2)).astype(np.float32),
    }

    # Node type assignments
    node_type_ids = np.random.randint(0, n_types, n_neurons)

    # Synapse data
    target_ids = np.random.randint(0, n_neurons * n_receptors, n_synapses)
    source_ids = np.random.randint(0, n_neurons, n_synapses)
    indices = np.stack([target_ids, source_ids], axis=1)
    weights = np.random.randn(n_synapses).astype(np.float32) * 0.1
    delays = np.random.uniform(0.5, 3, n_synapses).astype(np.float32)

    network = {
        'n_nodes': n_neurons,
        'node_params': node_params,
        'node_type_ids': node_type_ids,
        'synapses': {
            'indices': indices,
            'weights': weights,
            'delays': delays,
            'dense_shape': (n_neurons * n_receptors, n_neurons),
        },
    }

    # Input population
    input_target_ids = np.random.randint(0, n_neurons * n_receptors, n_input_synapses)
    input_source_ids = np.random.randint(0, n_inputs, n_input_synapses)
    input_indices = np.stack([input_target_ids, input_source_ids], axis=1)
    input_weights = np.random.randn(n_input_synapses).astype(np.float32) * 0.1

    input_pop = {
        'indices': input_indices,
        'weights': input_weights,
        'n_inputs': n_inputs,
    }

    return network, input_pop


@pytest.fixture
def mock_v1_network(mock_network_data, small_network_config):
    """Create a mock V1 network for testing."""
    network, input_pop = mock_network_data
    config = small_network_config

    # Create GLIF3 parameters
    glif3_params, metadata = GLIF3Cell.from_network(
        network,
        dt=config.dt,
        gauss_std=config.gauss_std,
        dampening_factor=config.dampening_factor,
        max_delay=config.max_delay,
    )

    n_neurons = metadata['n_neurons']
    n_receptors = metadata['n_receptors']
    n_inputs = input_pop['n_inputs']

    # Create mock input layer
    input_layer = InputLayer(
        indices=input_pop['indices'],
        weights=input_pop['weights'],
        dense_shape=(n_neurons * n_receptors, n_inputs),
        bkg_weights=np.ones(n_neurons * n_receptors, dtype=np.float32),
    )

    # Create mock recurrent layer
    max_delay = config.max_delay
    rec_indices = network['synapses']['indices'].copy()
    rec_indices[:, 1] = rec_indices[:, 1] + n_neurons * (
        np.clip(np.round(network['synapses']['delays']).astype(int), 1, max_delay) - 1
    )

    recurrent_layer = RecurrentLayer(
        indices=rec_indices,
        weights=network['synapses']['weights'],
        dense_shape=(n_neurons * n_receptors, n_neurons * max_delay),
        n_neurons=n_neurons,
        n_receptors=n_receptors,
        max_delay=max_delay,
    )

    return V1Network(
        glif3_params=glif3_params,
        input_layer=input_layer,
        recurrent_layer=recurrent_layer,
        metadata=metadata,
        config=config,
    )


# =============================================================================
# Configuration Tests
# =============================================================================

class TestV1NetworkConfig:
    """Tests for V1NetworkConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = V1NetworkConfig()
        assert config.dt == 1.0
        assert config.gauss_std == 0.5
        assert config.max_delay == 5
        assert config.use_dale_law is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = V1NetworkConfig(
            dt=0.5,
            gauss_std=0.3,
            max_delay=10,
        )
        assert config.dt == 0.5
        assert config.gauss_std == 0.3
        assert config.max_delay == 10


# =============================================================================
# State Tests
# =============================================================================

class TestV1NetworkState:
    """Tests for V1NetworkState management."""

    def test_init_state_shape(self, mock_v1_network):
        """Test initial state shapes."""
        batch_size = 8
        state = mock_v1_network.init_state(batch_size)

        assert isinstance(state, V1NetworkState)
        assert isinstance(state.glif3_state, GLIF3State)
        assert state.step == 0

        # Check GLIF3 state shapes
        n_neurons = mock_v1_network.n_neurons
        n_receptors = mock_v1_network.n_receptors
        max_delay = mock_v1_network.max_delay

        assert state.glif3_state.v.shape == (batch_size, n_neurons)
        assert state.glif3_state.z_buf.shape == (batch_size, n_neurons * max_delay)
        assert state.glif3_state.psc.shape == (batch_size, n_neurons * n_receptors)

    def test_init_state_random(self, mock_v1_network):
        """Test random state initialization."""
        batch_size = 4
        key = jax.random.PRNGKey(42)

        state = mock_v1_network.init_state(batch_size, key=key, random=True)

        # Random state should have non-zero values
        assert jnp.any(state.glif3_state.v != 0)
        assert jnp.any(state.glif3_state.z_buf != 0)

    def test_state_deterministic(self, mock_v1_network):
        """Test deterministic state with same key."""
        batch_size = 4
        key = jax.random.PRNGKey(42)

        state1 = mock_v1_network.init_state(batch_size, key=key, random=True)
        state2 = mock_v1_network.init_state(batch_size, key=key, random=True)

        assert jnp.allclose(state1.glif3_state.v, state2.glif3_state.v)


# =============================================================================
# Forward Pass Tests
# =============================================================================

class TestV1NetworkForward:
    """Tests for V1 network forward pass."""

    def test_forward_shape(self, mock_v1_network):
        """Test output shapes from forward pass."""
        batch_size = 4
        seq_len = 100
        n_inputs = mock_v1_network.n_inputs
        n_neurons = mock_v1_network.n_neurons

        # Create input
        inputs = jnp.zeros((seq_len, batch_size, n_inputs))

        # Initialize state
        state = mock_v1_network.init_state(batch_size)

        # Run forward
        output = mock_v1_network(inputs, state)

        assert isinstance(output, V1NetworkOutput)
        assert output.spikes.shape == (seq_len, batch_size, n_neurons)
        assert output.voltages.shape == (seq_len, batch_size, n_neurons)

    def test_forward_spikes_binary(self, mock_v1_network):
        """Test spikes are binary."""
        batch_size = 2
        seq_len = 50
        n_inputs = mock_v1_network.n_inputs

        inputs = jax.random.normal(
            jax.random.PRNGKey(42),
            (seq_len, batch_size, n_inputs),
        ) * 0.1

        state = mock_v1_network.init_state(batch_size)
        output = mock_v1_network(inputs, state)

        # Spikes should be 0 or 1
        assert jnp.all((output.spikes == 0) | (output.spikes == 1))

    def test_forward_state_update(self, mock_v1_network):
        """Test state is updated after forward pass."""
        batch_size = 2
        seq_len = 50
        n_inputs = mock_v1_network.n_inputs

        inputs = jnp.zeros((seq_len, batch_size, n_inputs))

        state0 = mock_v1_network.init_state(batch_size)
        output = mock_v1_network(inputs, state0)

        # Step counter should be updated
        assert output.final_state.step == seq_len

        # State should be different (in general)
        assert not jnp.allclose(
            state0.glif3_state.v,
            output.final_state.glif3_state.v,
        )

    def test_forward_gradient_flow(self, mock_v1_network):
        """Test gradients flow through forward pass."""
        batch_size = 2
        seq_len = 20
        n_inputs = mock_v1_network.n_inputs

        inputs = jax.random.normal(
            jax.random.PRNGKey(42),
            (seq_len, batch_size, n_inputs),
        ) * 0.1

        state = mock_v1_network.init_state(batch_size)

        def loss_fn(network, inputs, state):
            output = network(inputs, state)
            return jnp.mean(output.spikes)

        grad = jax.grad(loss_fn, argnums=1)(mock_v1_network, inputs, state)
        assert grad.shape == inputs.shape
        assert not jnp.all(grad == 0)


# =============================================================================
# Single Step Tests
# =============================================================================

class TestV1NetworkStep:
    """Tests for single step updates."""

    def test_step_shape(self, mock_v1_network):
        """Test single step output shapes."""
        batch_size = 4
        n_inputs = mock_v1_network.n_inputs
        n_neurons = mock_v1_network.n_neurons

        inputs = jnp.zeros((batch_size, n_inputs))
        state = mock_v1_network.init_state(batch_size)

        new_state, spikes, voltages = v1_network_step(
            mock_v1_network, state, inputs
        )

        assert spikes.shape == (batch_size, n_neurons)
        assert voltages.shape == (batch_size, n_neurons)
        assert new_state.step == state.step + 1

    def test_step_consistency(self, mock_v1_network):
        """Test step results are consistent with unrolled forward."""
        batch_size = 2
        n_inputs = mock_v1_network.n_inputs

        # Single timestep input
        inputs = jax.random.normal(
            jax.random.PRNGKey(42),
            (batch_size, n_inputs),
        ) * 0.1

        state = mock_v1_network.init_state(batch_size)

        # Via step function
        _, spikes_step, _ = v1_network_step(mock_v1_network, state, inputs)

        # Via forward with seq_len=1
        inputs_seq = inputs[None, :, :]  # (1, batch, n_inputs)
        output = mock_v1_network(inputs_seq, state)
        spikes_fwd = output.spikes[0]

        assert jnp.allclose(spikes_step, spikes_fwd)


# =============================================================================
# JIT Compilation Tests
# =============================================================================

class TestV1NetworkJIT:
    """Tests for JIT compilation."""

    def test_forward_jit(self, mock_v1_network):
        """Test forward pass is JIT-compilable."""
        forward_fn = make_v1_forward_fn(mock_v1_network)

        batch_size = 2
        seq_len = 20
        n_inputs = mock_v1_network.n_inputs

        inputs = jnp.zeros((seq_len, batch_size, n_inputs))
        state = mock_v1_network.init_state(batch_size)

        # Should compile and run
        output = forward_fn(inputs, state)
        assert output.spikes.shape[0] == seq_len

    def test_step_jit(self, mock_v1_network):
        """Test step function is JIT-compilable."""
        step_fn = make_v1_step_fn(mock_v1_network)

        batch_size = 2
        n_inputs = mock_v1_network.n_inputs

        inputs = jnp.zeros((batch_size, n_inputs))
        state = mock_v1_network.init_state(batch_size)

        # Should compile and run
        new_state, spikes, voltages = step_fn(state, inputs)
        assert spikes.shape[0] == batch_size


# =============================================================================
# Trainable Parameters Tests
# =============================================================================

class TestTrainableParams:
    """Tests for trainable parameter access."""

    def test_get_trainable_params(self, mock_v1_network):
        """Test getting trainable parameters."""
        params = mock_v1_network.get_trainable_params()

        assert 'input_weights' in params
        assert 'recurrent_weights' in params
        assert params['input_weights'].ndim == 1
        assert params['recurrent_weights'].ndim == 1

    def test_apply_trainable_params(self, mock_v1_network):
        """Test applying updated parameters."""
        params = mock_v1_network.get_trainable_params()

        # Modify weights
        new_params = {
            'input_weights': params['input_weights'] * 2,
            'recurrent_weights': params['recurrent_weights'] * 2,
        }

        new_network = mock_v1_network.apply_trainable_params(new_params)

        # Check weights are updated
        new_network_params = new_network.get_trainable_params()
        assert jnp.allclose(
            new_network_params['input_weights'],
            new_params['input_weights'],
        )


# =============================================================================
# Dale's Law Tests
# =============================================================================

class TestDaleLaw:
    """Tests for Dale's law constraint."""

    def test_dale_constraint_excitatory(self, mock_v1_network):
        """Test Dale's law keeps excitatory weights positive."""
        params = mock_v1_network.get_trainable_params()

        # Create weights that violate Dale's law
        bad_weights = params['input_weights'] * -1  # Flip signs

        new_params = {
            'input_weights': bad_weights,
            'recurrent_weights': params['recurrent_weights'],
        }

        new_network = mock_v1_network.apply_trainable_params(
            new_params, use_dale_law=True
        )

        # Get constrained weights
        constrained = new_network.get_trainable_params()['input_weights']

        # Originally positive weights should remain positive
        original_positive = params['input_weights'] >= 0
        assert jnp.all(constrained[original_positive] >= 0)
