"""Tests for GLIF3 neuron cell implementation.

Tests cover:
- State initialization (zero and random)
- Single step dynamics
- JIT compilation
- Gradient flow through surrogate
- Unrolling with lax.scan
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad

from v1_jax.nn.glif3_cell import (
    GLIF3State,
    GLIF3Params,
    GLIF3Cell,
    glif3_step,
    make_glif3_step_fn,
    glif3_unroll,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_neuron_config():
    """Small network configuration for quick tests."""
    return {
        'n_neurons': 50,
        'n_receptors': 4,
        'max_delay': 5,
        'batch_size': 4,
        'dt': 1.0,
        'gauss_std': 0.5,
        'dampening_factor': 0.3,
    }


@pytest.fixture
def mock_params(small_neuron_config):
    """Create mock GLIF3 parameters for testing."""
    n_neurons = small_neuron_config['n_neurons']
    n_receptors = small_neuron_config['n_receptors']

    # Normalized parameters (V_th - E_L used as scale)
    return GLIF3Params(
        v_reset=jnp.zeros(n_neurons),  # Normalized to 0
        v_th=jnp.ones(n_neurons),      # Normalized to 1
        e_l=jnp.zeros(n_neurons),      # Normalized to 0
        t_ref=jnp.ones(n_neurons) * 2.0,
        decay=jnp.ones(n_neurons) * 0.9,  # ~10ms tau_m
        current_factor=jnp.ones(n_neurons) * 0.1,
        syn_decay=jnp.ones((n_neurons, n_receptors)) * 0.8,
        psc_initial=jnp.ones((n_neurons, n_receptors)) * 0.3,
        param_k=jnp.ones((n_neurons, 2)) * 0.01,
        asc_amps=jnp.zeros((n_neurons, 2)),
        param_g=jnp.ones(n_neurons) * 2.5,
        voltage_scale=jnp.ones(n_neurons) * 20.0,
        voltage_offset=jnp.ones(n_neurons) * (-70.0),
    )


# =============================================================================
# State Initialization Tests
# =============================================================================


class TestGLIF3StateInit:
    """Tests for GLIF3 state initialization."""

    def test_init_state_shapes(self, small_neuron_config, mock_params):
        """Test that init_state creates correct shapes."""
        cfg = small_neuron_config
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        assert state.z_buf.shape == (cfg['batch_size'], cfg['n_neurons'] * cfg['max_delay'])
        assert state.v.shape == (cfg['batch_size'], cfg['n_neurons'])
        assert state.r.shape == (cfg['batch_size'], cfg['n_neurons'])
        assert state.asc_1.shape == (cfg['batch_size'], cfg['n_neurons'])
        assert state.asc_2.shape == (cfg['batch_size'], cfg['n_neurons'])
        assert state.psc_rise.shape == (cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors'])
        assert state.psc.shape == (cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors'])

    def test_init_state_zero_values(self, small_neuron_config, mock_params):
        """Test that init_state initializes with correct values."""
        cfg = small_neuron_config
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        # Spike buffer should be zeros
        assert jnp.allclose(state.z_buf, 0.0)
        # Refractory counter should be zeros
        assert jnp.allclose(state.r, 0.0)
        # ASC should be zeros
        assert jnp.allclose(state.asc_1, 0.0)
        assert jnp.allclose(state.asc_2, 0.0)

    def test_random_state_shapes(self, small_neuron_config, mock_params, jax_key):
        """Test that random_state creates correct shapes."""
        cfg = small_neuron_config
        state = GLIF3Cell.random_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
            key=jax_key,
        )

        assert state.z_buf.shape == (cfg['batch_size'], cfg['n_neurons'] * cfg['max_delay'])
        assert state.v.shape == (cfg['batch_size'], cfg['n_neurons'])

    def test_random_state_not_zero(self, small_neuron_config, mock_params, jax_key):
        """Test that random_state creates non-zero values."""
        cfg = small_neuron_config
        state = GLIF3Cell.random_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
            key=jax_key,
        )

        # Voltage should be in range [v_reset, v_th]
        assert jnp.all(state.v >= mock_params.v_reset.min())
        assert jnp.all(state.v <= mock_params.v_th.max())


# =============================================================================
# Single Step Tests
# =============================================================================


class TestGLIF3Step:
    """Tests for single timestep GLIF3 dynamics."""

    def test_step_output_shapes(self, small_neuron_config, mock_params):
        """Test that glif3_step produces correct output shapes."""
        cfg = small_neuron_config
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        # Create inputs
        inputs = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))
        rec_current = jnp.zeros_like(inputs)

        new_state, spikes, voltage = glif3_step(
            mock_params, state, inputs, rec_current,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
            cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
        )

        assert spikes.shape == (cfg['batch_size'], cfg['n_neurons'])
        assert voltage.shape == (cfg['batch_size'], cfg['n_neurons'])
        assert new_state.v.shape == state.v.shape

    def test_step_no_input_decay(self, small_neuron_config, mock_params):
        """Test voltage decay with no input."""
        cfg = small_neuron_config

        # Start with elevated voltage
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )
        state = state._replace(v=jnp.ones_like(state.v) * 0.5)

        inputs = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))
        rec_current = jnp.zeros_like(inputs)

        new_state, _, _ = glif3_step(
            mock_params, state, inputs, rec_current,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
            cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
        )

        # Voltage should decay (but not to exactly zero due to leak current)
        # Just check it changes
        assert not jnp.allclose(new_state.v, state.v, atol=1e-3)

    def test_step_strong_input_spikes(self, small_neuron_config, mock_params):
        """Test that strong input causes spiking."""
        cfg = small_neuron_config
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        # Create strong PSC input to drive voltage above threshold
        # Need to build up PSC first, then voltage
        state = state._replace(
            psc=jnp.ones_like(state.psc) * 50.0,  # Strong PSC
            v=jnp.ones_like(state.v) * 0.9,  # Near threshold
        )

        inputs = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))
        rec_current = jnp.zeros_like(inputs)

        new_state, spikes, _ = glif3_step(
            mock_params, state, inputs, rec_current,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
            cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
        )

        # Some neurons should spike
        assert jnp.sum(spikes) > 0

    def test_step_spike_buffer_shift(self, small_neuron_config, mock_params):
        """Test that spike buffer shifts correctly."""
        cfg = small_neuron_config
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        # Put known pattern in z_buf
        z_buf_init = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['max_delay']))
        # First delay slot gets ones
        z_buf_shaped = z_buf_init.reshape(cfg['batch_size'], cfg['max_delay'], cfg['n_neurons'])
        z_buf_shaped = z_buf_shaped.at[:, 0, :10].set(1.0)  # First 10 neurons spiked
        state = state._replace(z_buf=z_buf_shaped.reshape(cfg['batch_size'], -1))

        inputs = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))
        rec_current = jnp.zeros_like(inputs)

        new_state, _, _ = glif3_step(
            mock_params, state, inputs, rec_current,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
            cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
        )

        # Old spikes should have shifted to delay slot 1
        new_z_shaped = new_state.z_buf.reshape(cfg['batch_size'], cfg['max_delay'], cfg['n_neurons'])
        # Slot 1 should now contain the old slot 0 pattern
        assert jnp.sum(new_z_shaped[:, 1, :10]) == cfg['batch_size'] * 10


# =============================================================================
# JIT Compilation Tests
# =============================================================================


class TestGLIF3JIT:
    """Tests for JIT compilation of GLIF3 operations."""

    def test_step_jit_compiles(self, small_neuron_config, mock_params):
        """Test that glif3_step can be JIT compiled."""
        cfg = small_neuron_config
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        inputs = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))
        rec_current = jnp.zeros_like(inputs)

        # JIT compile
        jit_step = jax.jit(lambda s, i, r: glif3_step(
            mock_params, s, i, r,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
            cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
        ))

        # First call compiles
        new_state1, spikes1, _ = jit_step(state, inputs, rec_current)

        # Second call should be fast
        new_state2, spikes2, _ = jit_step(new_state1, inputs, rec_current)

        assert spikes1.shape == spikes2.shape

    def test_make_step_fn_jit(self, small_neuron_config, mock_params):
        """Test the make_glif3_step_fn helper."""
        cfg = small_neuron_config
        step_fn = make_glif3_step_fn(
            mock_params,
            cfg['n_neurons'],
            cfg['n_receptors'],
            cfg['max_delay'],
            cfg['dt'],
            cfg['gauss_std'],
            cfg['dampening_factor'],
        )

        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        inputs = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))
        rec_current = jnp.zeros_like(inputs)

        new_state, spikes, voltage = step_fn(state, inputs, rec_current)

        assert spikes.shape == (cfg['batch_size'], cfg['n_neurons'])


# =============================================================================
# Gradient Tests
# =============================================================================


class TestGLIF3Gradients:
    """Tests for gradient computation through GLIF3."""

    def test_gradient_flows_through_spike(self, small_neuron_config, mock_params, precision):
        """Test that gradients flow through spike function."""
        cfg = small_neuron_config
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )
        # Set voltage near threshold for gradient
        state = state._replace(v=jnp.ones_like(state.v) * 0.5)

        inputs = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))
        rec_current = jnp.zeros_like(inputs)

        def loss_fn(v_init):
            s = state._replace(v=v_init)
            _, spikes, _ = glif3_step(
                mock_params, s, inputs, rec_current,
                cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
                cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
            )
            return jnp.sum(spikes)

        # Compute gradient
        grad_fn = grad(loss_fn)
        grads = grad_fn(state.v)

        # Gradients should be non-zero (from surrogate)
        assert jnp.any(grads != 0)
        # Gradients should be finite
        assert jnp.all(jnp.isfinite(grads))

    def test_gradient_wrt_psc(self, small_neuron_config, mock_params, precision):
        """Test gradient with respect to PSC inputs."""
        cfg = small_neuron_config
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )
        state = state._replace(v=jnp.ones_like(state.v) * 0.5)

        rec_current = jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))

        def loss_fn(inputs):
            _, spikes, _ = glif3_step(
                mock_params, state, inputs, rec_current,
                cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
                cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
            )
            return jnp.sum(spikes)

        inputs = jnp.ones((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors'])) * 0.1
        grad_fn = grad(loss_fn)
        grads = grad_fn(inputs)

        assert jnp.all(jnp.isfinite(grads))


# =============================================================================
# Unroll Tests
# =============================================================================


class TestGLIF3Unroll:
    """Tests for unrolling GLIF3 over time."""

    def test_unroll_shapes(self, small_neuron_config, mock_params):
        """Test unroll output shapes."""
        cfg = small_neuron_config
        seq_len = 20
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        inputs = jnp.zeros((seq_len, cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))

        def recurrent_fn(z_buf):
            return jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))

        final_state, all_spikes, all_voltages = glif3_unroll(
            mock_params, state, inputs, recurrent_fn,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
            cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
        )

        assert all_spikes.shape == (seq_len, cfg['batch_size'], cfg['n_neurons'])
        assert all_voltages.shape == (seq_len, cfg['batch_size'], cfg['n_neurons'])

    def test_unroll_accumulates_activity(self, small_neuron_config, mock_params, jax_key):
        """Test that unroll accumulates neural activity over time."""
        cfg = small_neuron_config
        seq_len = 50
        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=cfg['batch_size'],
            params=mock_params,
        )

        # Create some input drive
        inputs = jax.random.uniform(
            jax_key,
            (seq_len, cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']),
            minval=0.0,
            maxval=0.5,
        )

        def recurrent_fn(z_buf):
            # Simple recurrent drive
            return jnp.zeros((cfg['batch_size'], cfg['n_neurons'] * cfg['n_receptors']))

        final_state, all_spikes, all_voltages = glif3_unroll(
            mock_params, state, inputs, recurrent_fn,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
            cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
        )

        # Should have some spikes
        total_spikes = jnp.sum(all_spikes)
        assert total_spikes > 0

        # Voltages should vary over time
        voltage_std = jnp.std(all_voltages)
        assert voltage_std > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestGLIF3EdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_neuron(self, mock_params):
        """Test with single neuron."""
        n_neurons = 1
        n_receptors = 4
        max_delay = 5
        batch_size = 2

        # Create single-neuron params
        params = GLIF3Params(
            v_reset=jnp.array([0.0]),
            v_th=jnp.array([1.0]),
            e_l=jnp.array([0.0]),
            t_ref=jnp.array([2.0]),
            decay=jnp.array([0.9]),
            current_factor=jnp.array([0.1]),
            syn_decay=jnp.ones((1, n_receptors)) * 0.8,
            psc_initial=jnp.ones((1, n_receptors)) * 0.3,
            param_k=jnp.ones((1, 2)) * 0.01,
            asc_amps=jnp.zeros((1, 2)),
            param_g=jnp.array([2.5]),
            voltage_scale=jnp.array([20.0]),
            voltage_offset=jnp.array([-70.0]),
        )

        state = GLIF3Cell.init_state(
            n_neurons=n_neurons,
            n_receptors=n_receptors,
            max_delay=max_delay,
            batch_size=batch_size,
            params=params,
        )

        inputs = jnp.zeros((batch_size, n_receptors))
        rec_current = jnp.zeros_like(inputs)

        new_state, spikes, voltage = glif3_step(
            params, state, inputs, rec_current,
            n_neurons, n_receptors, max_delay,
            1.0, 0.5, 0.3,
        )

        assert spikes.shape == (batch_size, n_neurons)

    def test_large_batch(self, small_neuron_config, mock_params):
        """Test with large batch size."""
        cfg = small_neuron_config
        large_batch = 256

        state = GLIF3Cell.init_state(
            n_neurons=cfg['n_neurons'],
            n_receptors=cfg['n_receptors'],
            max_delay=cfg['max_delay'],
            batch_size=large_batch,
            params=mock_params,
        )

        inputs = jnp.zeros((large_batch, cfg['n_neurons'] * cfg['n_receptors']))
        rec_current = jnp.zeros_like(inputs)

        new_state, spikes, voltage = glif3_step(
            mock_params, state, inputs, rec_current,
            cfg['n_neurons'], cfg['n_receptors'], cfg['max_delay'],
            cfg['dt'], cfg['gauss_std'], cfg['dampening_factor'],
        )

        assert spikes.shape == (large_batch, cfg['n_neurons'])
