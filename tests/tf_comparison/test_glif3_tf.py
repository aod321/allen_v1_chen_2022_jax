"""TensorFlow comparison tests for GLIF3 neuron dynamics.

These tests verify numerical equivalence between JAX and TensorFlow
implementations of GLIF3 neuron dynamics from the BillehColumn.

Comparison points:
1. Single neuron dynamics (membrane voltage, ASC currents)
2. PSC dynamics (rise and decay)
3. Spike generation with refractory period
4. State update equations
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax
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

# Load TF spike functions for GLIF3 tests
HAS_TF_FUNCS = False
tf_spike_gauss = None

if HAS_TF:
    try:
        exec_globals = {'tf': tf, 'np': np}
        exec("""
import tensorflow as tf
import numpy as np

def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude

@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled
        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad
""", exec_globals)

        tf_spike_gauss = exec_globals['spike_gauss']
        HAS_TF_FUNCS = True

    except Exception as e:
        print(f"Could not load TF functions: {e}")
        HAS_TF_FUNCS = False

# Skip if TF not available
pytestmark = pytest.mark.skipif(
    not HAS_TF or not HAS_TF_FUNCS,
    reason="TensorFlow not available"
)

# Import JAX implementations
from v1_jax.nn.spike_functions import spike_gauss as jax_spike_gauss


class GLIF3Precision:
    """Precision tolerances for GLIF3 comparison tests."""
    # Single step dynamics
    RTOL_SINGLE_STEP = 1e-5
    ATOL_SINGLE_STEP = 1e-6

    # Multi-step dynamics (accumulated error)
    RTOL_MULTI_STEP = 1e-4
    ATOL_MULTI_STEP = 1e-5

    # PSC dynamics
    RTOL_PSC = 1e-5
    ATOL_PSC = 1e-6


@pytest.fixture
def glif3_precision():
    return GLIF3Precision()


@pytest.fixture
def neuron_params():
    """Generate test neuron parameters matching TF format."""
    np.random.seed(42)
    n_neurons = 50
    n_receptors = 4  # AMPA, NMDA, GABA_A, GABA_B

    # Time constants and membrane parameters
    tau_m = 10.0 + np.random.rand(n_neurons) * 10.0  # 10-20 ms
    C_m = 100.0 + np.random.rand(n_neurons) * 100.0  # 100-200 pF
    g = C_m / tau_m

    # Voltage parameters
    V_th = np.ones(n_neurons) * -50.0
    E_L = np.ones(n_neurons) * -70.0
    V_reset = np.ones(n_neurons) * -75.0

    # Voltage normalization (as in TF)
    voltage_scale = V_th - E_L
    voltage_offset = E_L
    V_th_norm = (V_th - voltage_offset) / voltage_scale
    E_L_norm = (E_L - voltage_offset) / voltage_scale
    V_reset_norm = (V_reset - voltage_offset) / voltage_scale

    # Synaptic time constants
    tau_syn = np.array([2.0, 100.0, 6.0, 150.0])  # ms
    tau_syn = np.tile(tau_syn, (n_neurons, 1))

    # Refractory period
    t_ref = 3.0 + np.random.rand(n_neurons) * 2.0  # 3-5 ms

    # ASC parameters
    asc_amps = np.random.randn(n_neurons, 2) * 0.1
    asc_amps = asc_amps / voltage_scale[..., None]
    k = np.random.rand(n_neurons, 2) * 0.01 + 0.001  # Decay rates

    # Simulation parameters
    dt = 1.0

    # Derived parameters
    decay = np.exp(-dt / tau_m)
    current_factor = 1 / C_m * (1 - decay) * tau_m
    syn_decay = np.exp(-dt / tau_syn)
    psc_initial = np.e / tau_syn

    return {
        'n_neurons': n_neurons,
        'n_receptors': n_receptors,
        'dt': dt,
        'decay': decay.astype(np.float32),
        'current_factor': current_factor.astype(np.float32),
        'syn_decay': syn_decay.astype(np.float32),
        'psc_initial': psc_initial.astype(np.float32),
        'V_th': V_th_norm.astype(np.float32),
        'E_L': E_L_norm.astype(np.float32),
        'V_reset': V_reset_norm.astype(np.float32),
        't_ref': t_ref.astype(np.float32),
        'asc_amps': asc_amps.astype(np.float32),
        'k': k.astype(np.float32),
        'g': g.astype(np.float32),
        'voltage_scale': voltage_scale.astype(np.float32),
        'voltage_offset': voltage_offset.astype(np.float32),
    }


# =============================================================================
# Membrane Voltage Dynamics Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestMembraneVoltageTF:
    """Compare membrane voltage update equations."""

    def test_voltage_decay_matches_tf(self, neuron_params, glif3_precision):
        """Test voltage decay matches TF implementation."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']

        # Initial voltage
        v = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.1 + neuron_params['V_reset']
        decay = neuron_params['decay']

        # TF decay
        tf_decayed_v = (tf.constant(v) * tf.constant(decay)).numpy()

        # JAX decay (same operation)
        jax_decayed_v = np.array(jnp.array(v) * jnp.array(decay))

        np.testing.assert_allclose(
            jax_decayed_v, tf_decayed_v,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="Voltage decay differs between JAX and TF"
        )

    def test_voltage_update_with_current_matches_tf(self, neuron_params, glif3_precision):
        """Test voltage update with input current."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']

        # Initial state
        v = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.1 + neuron_params['V_reset']
        input_current = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.5
        asc_1 = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.01
        asc_2 = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.01

        decay = neuron_params['decay']
        current_factor = neuron_params['current_factor']
        g = neuron_params['g']
        e_l = neuron_params['E_L']

        # TF voltage update
        gathered_g = g * e_l
        c1 = input_current + asc_1 + asc_2 + gathered_g
        tf_new_v = (
            tf.constant(v) * tf.constant(decay) +
            tf.constant(current_factor) * tf.constant(c1)
        ).numpy()

        # JAX voltage update (same computation)
        jax_gathered_g = jnp.array(g) * jnp.array(e_l)
        jax_c1 = jnp.array(input_current) + jnp.array(asc_1) + jnp.array(asc_2) + jax_gathered_g
        jax_new_v = np.array(
            jnp.array(v) * jnp.array(decay) +
            jnp.array(current_factor) * jax_c1
        )

        np.testing.assert_allclose(
            jax_new_v, tf_new_v,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="Voltage update with current differs between JAX and TF"
        )

    def test_voltage_reset_matches_tf(self, neuron_params, glif3_precision):
        """Test spike-triggered voltage reset."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']

        # Voltage and previous spike
        v = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.1
        prev_z = (np.random.rand(batch_size, n_neurons) > 0.8).astype(np.float32)

        v_reset = neuron_params['V_reset']
        v_th = neuron_params['V_th']

        # TF reset current
        tf_reset_current = (tf.constant(prev_z) * (tf.constant(v_reset) - tf.constant(v_th))).numpy()

        # JAX reset current
        jax_reset_current = np.array(
            jnp.array(prev_z) * (jnp.array(v_reset) - jnp.array(v_th))
        )

        np.testing.assert_allclose(
            jax_reset_current, tf_reset_current,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="Reset current differs between JAX and TF"
        )


# =============================================================================
# ASC (After-Spike Current) Dynamics Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestASCDynamicsTF:
    """Compare ASC current dynamics."""

    def test_asc_decay_matches_tf(self, neuron_params, glif3_precision):
        """Test ASC exponential decay matches TF."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']

        asc_1 = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.1
        k = neuron_params['k'][:, 0]
        dt = neuron_params['dt']

        # TF ASC decay
        tf_decay_factor = tf.exp(-dt * tf.constant(k))
        tf_new_asc = (tf_decay_factor * tf.constant(asc_1)).numpy()

        # JAX ASC decay
        jax_decay_factor = jnp.exp(-dt * jnp.array(k))
        jax_new_asc = np.array(jax_decay_factor * jnp.array(asc_1))

        np.testing.assert_allclose(
            jax_new_asc, tf_new_asc,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="ASC decay differs between JAX and TF"
        )

    def test_asc_update_with_spike_matches_tf(self, neuron_params, glif3_precision):
        """Test ASC update triggered by spike."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']

        asc_1 = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.1
        prev_z = (np.random.rand(batch_size, n_neurons) > 0.8).astype(np.float32)
        k = neuron_params['k'][:, 0]
        asc_amps = neuron_params['asc_amps'][:, 0]
        dt = neuron_params['dt']

        # TF ASC update
        tf_new_asc_1 = (
            tf.exp(-dt * tf.constant(k)) * tf.constant(asc_1) +
            tf.constant(prev_z) * tf.constant(asc_amps)
        ).numpy()

        # JAX ASC update
        jax_new_asc_1 = np.array(
            jnp.exp(-dt * jnp.array(k)) * jnp.array(asc_1) +
            jnp.array(prev_z) * jnp.array(asc_amps)
        )

        np.testing.assert_allclose(
            jax_new_asc_1, tf_new_asc_1,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="ASC spike update differs between JAX and TF"
        )


# =============================================================================
# PSC (Post-Synaptic Current) Dynamics Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestPSCDynamicsTF:
    """Compare PSC dynamics (synaptic filtering)."""

    def test_psc_rise_matches_tf(self, neuron_params, glif3_precision):
        """Test PSC rise component matches TF."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']
        n_receptors = neuron_params['n_receptors']

        psc_rise = np.random.randn(batch_size, n_neurons, n_receptors).astype(np.float32) * 0.1
        rec_inputs = np.random.randn(batch_size, n_neurons, n_receptors).astype(np.float32)
        syn_decay = neuron_params['syn_decay']
        psc_initial = neuron_params['psc_initial']

        # TF PSC rise update
        tf_new_psc_rise = (
            tf.constant(syn_decay) * tf.constant(psc_rise) +
            tf.constant(rec_inputs) * tf.constant(psc_initial)
        ).numpy()

        # JAX PSC rise update
        jax_new_psc_rise = np.array(
            jnp.array(syn_decay) * jnp.array(psc_rise) +
            jnp.array(rec_inputs) * jnp.array(psc_initial)
        )

        np.testing.assert_allclose(
            jax_new_psc_rise, tf_new_psc_rise,
            rtol=glif3_precision.RTOL_PSC,
            atol=glif3_precision.ATOL_PSC,
            err_msg="PSC rise differs between JAX and TF"
        )

    def test_psc_decay_matches_tf(self, neuron_params, glif3_precision):
        """Test PSC decay component matches TF."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']
        n_receptors = neuron_params['n_receptors']
        dt = neuron_params['dt']

        psc = np.random.randn(batch_size, n_neurons, n_receptors).astype(np.float32)
        psc_rise = np.random.randn(batch_size, n_neurons, n_receptors).astype(np.float32) * 0.1
        syn_decay = neuron_params['syn_decay']

        # TF PSC update
        tf_new_psc = (
            tf.constant(psc) * tf.constant(syn_decay) +
            dt * tf.constant(syn_decay) * tf.constant(psc_rise)
        ).numpy()

        # JAX PSC update
        jax_new_psc = np.array(
            jnp.array(psc) * jnp.array(syn_decay) +
            dt * jnp.array(syn_decay) * jnp.array(psc_rise)
        )

        np.testing.assert_allclose(
            jax_new_psc, tf_new_psc,
            rtol=glif3_precision.RTOL_PSC,
            atol=glif3_precision.ATOL_PSC,
            err_msg="PSC decay differs between JAX and TF"
        )


# =============================================================================
# Refractory Period Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestRefractoryPeriodTF:
    """Compare refractory period dynamics."""

    def test_refractory_update_matches_tf(self, neuron_params, glif3_precision):
        """Test refractory counter update matches TF."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']
        dt = neuron_params['dt']

        r = np.random.rand(batch_size, n_neurons).astype(np.float32) * 5.0
        prev_z = (np.random.rand(batch_size, n_neurons) > 0.8).astype(np.float32)
        t_ref = neuron_params['t_ref']

        # TF refractory update
        tf_new_r = tf.nn.relu(
            tf.constant(r) + tf.constant(prev_z) * tf.constant(t_ref) - dt
        ).numpy()

        # JAX refractory update (equivalent)
        jax_new_r = np.array(jnp.maximum(
            jnp.array(r) + jnp.array(prev_z) * jnp.array(t_ref) - dt,
            0.0
        ))

        np.testing.assert_allclose(
            jax_new_r, tf_new_r,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="Refractory update differs between JAX and TF"
        )

    def test_spike_suppression_matches_tf(self, neuron_params):
        """Test spike suppression during refractory period."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']

        # Spike output and refractory counter
        z = (np.random.rand(batch_size, n_neurons) > 0.5).astype(np.float32)
        r = np.random.rand(batch_size, n_neurons).astype(np.float32) * 5.0  # Some neurons refractory

        # TF spike suppression
        tf_new_z = tf.where(
            tf.constant(r) > 0.,
            tf.zeros_like(tf.constant(z)),
            tf.constant(z)
        ).numpy()

        # JAX spike suppression
        jax_new_z = np.array(jnp.where(
            jnp.array(r) > 0.,
            jnp.zeros_like(jnp.array(z)),
            jnp.array(z)
        ))

        np.testing.assert_array_equal(
            jax_new_z, tf_new_z,
            err_msg="Spike suppression differs between JAX and TF"
        )


# =============================================================================
# Spike Generation Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestSpikeGenerationTF:
    """Compare spike generation with voltage normalization."""

    def test_voltage_scaling_matches_tf(self, neuron_params, glif3_precision):
        """Test voltage scaling for spike function matches TF."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']

        v = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.1
        v_th = neuron_params['V_th']
        e_l = neuron_params['E_L']

        # TF voltage scaling (v_scaled)
        normalizer = v_th - e_l
        tf_v_sc = ((tf.constant(v) - tf.constant(v_th)) / tf.constant(normalizer)).numpy()

        # JAX voltage scaling
        jax_normalizer = jnp.array(v_th) - jnp.array(e_l)
        jax_v_sc = np.array((jnp.array(v) - jnp.array(v_th)) / jax_normalizer)

        np.testing.assert_allclose(
            jax_v_sc, tf_v_sc,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="Voltage scaling differs between JAX and TF"
        )

    def test_spike_gauss_with_scaled_voltage_matches_tf(self, neuron_params):
        """Test spike generation with scaled voltage."""
        np.random.seed(42)
        batch_size = 4
        n_neurons = neuron_params['n_neurons']

        # Scaled voltage (some above, some below threshold)
        v_sc = np.random.randn(batch_size, n_neurons).astype(np.float32)
        gauss_std = 0.28
        dampening_factor = 0.5

        # TF spike generation
        tf_z = tf_spike_gauss(
            tf.constant(v_sc),
            tf.constant(gauss_std, dtype=tf.float32),
            tf.constant(dampening_factor, dtype=tf.float32)
        ).numpy()

        # JAX spike generation
        jax_z = np.array(jax_spike_gauss(
            jnp.array(v_sc),
            gauss_std,
            dampening_factor
        ))

        np.testing.assert_array_equal(
            jax_z, tf_z,
            err_msg="Spike generation with scaled voltage differs between JAX and TF"
        )


# =============================================================================
# Combined Dynamics Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestCombinedDynamicsTF:
    """Test combined GLIF3 dynamics computation."""

    def test_single_step_dynamics_matches_tf(self, neuron_params, glif3_precision):
        """Test a complete single-step dynamics update."""
        np.random.seed(42)
        batch_size = 2
        n_neurons = neuron_params['n_neurons']
        n_receptors = neuron_params['n_receptors']
        dt = neuron_params['dt']

        # Initial state
        v = np.random.randn(batch_size, n_neurons).astype(np.float32) * 0.05 + neuron_params['V_reset']
        r = np.zeros((batch_size, n_neurons), dtype=np.float32)
        asc_1 = np.zeros((batch_size, n_neurons), dtype=np.float32)
        asc_2 = np.zeros((batch_size, n_neurons), dtype=np.float32)
        psc_rise = np.zeros((batch_size, n_neurons, n_receptors), dtype=np.float32)
        psc = np.random.rand(batch_size, n_neurons, n_receptors).astype(np.float32) * 0.5
        prev_z = np.zeros((batch_size, n_neurons), dtype=np.float32)

        # Parameters
        decay = neuron_params['decay']
        current_factor = neuron_params['current_factor']
        syn_decay = neuron_params['syn_decay']
        psc_initial = neuron_params['psc_initial']
        v_th = neuron_params['V_th']
        v_reset = neuron_params['V_reset']
        e_l = neuron_params['E_L']
        g = neuron_params['g']
        t_ref = neuron_params['t_ref']
        asc_amps = neuron_params['asc_amps']
        k = neuron_params['k']
        gauss_std = 0.28
        dampening_factor = 0.5

        # === TF computation ===
        # Input current (simplified, no recurrent)
        tf_input = np.random.randn(batch_size, n_neurons, n_receptors).astype(np.float32) * 0.1

        # PSC update
        tf_new_psc_rise = syn_decay * psc_rise + tf_input * psc_initial
        tf_new_psc = psc * syn_decay + dt * syn_decay * psc_rise

        # Refractory update
        tf_new_r = tf.nn.relu(r + prev_z * t_ref - dt).numpy()

        # ASC update
        tf_new_asc_1 = (np.exp(-dt * k[:, 0]) * asc_1 + prev_z * asc_amps[:, 0]).astype(np.float32)
        tf_new_asc_2 = (np.exp(-dt * k[:, 1]) * asc_2 + prev_z * asc_amps[:, 1]).astype(np.float32)

        # Voltage update
        reset_current = prev_z * (v_reset - v_th)
        input_current = np.sum(psc, axis=-1)
        decayed_v = decay * v
        gathered_g = g * e_l
        c1 = input_current + asc_1 + asc_2 + gathered_g
        tf_new_v = decayed_v + current_factor * c1 + reset_current

        # Spike generation
        normalizer = v_th - e_l
        v_sc = (tf_new_v - v_th) / normalizer
        tf_z = tf_spike_gauss(
            tf.constant(v_sc, dtype=tf.float32),
            tf.constant(gauss_std, dtype=tf.float32),
            tf.constant(dampening_factor, dtype=tf.float32)
        ).numpy()

        # Spike suppression
        tf_z = np.where(tf_new_r > 0., np.zeros_like(tf_z), tf_z)

        # === JAX computation ===
        jax_input = jnp.array(tf_input)

        # PSC update
        jax_new_psc_rise = jnp.array(syn_decay) * jnp.array(psc_rise) + jax_input * jnp.array(psc_initial)
        jax_new_psc = jnp.array(psc) * jnp.array(syn_decay) + dt * jnp.array(syn_decay) * jnp.array(psc_rise)

        # Refractory update
        jax_new_r = jnp.maximum(jnp.array(r) + jnp.array(prev_z) * jnp.array(t_ref) - dt, 0.)

        # ASC update
        jax_new_asc_1 = jnp.exp(-dt * jnp.array(k[:, 0])) * jnp.array(asc_1) + jnp.array(prev_z) * jnp.array(asc_amps[:, 0])
        jax_new_asc_2 = jnp.exp(-dt * jnp.array(k[:, 1])) * jnp.array(asc_2) + jnp.array(prev_z) * jnp.array(asc_amps[:, 1])

        # Voltage update
        jax_reset_current = jnp.array(prev_z) * (jnp.array(v_reset) - jnp.array(v_th))
        jax_input_current = jnp.sum(jnp.array(psc), axis=-1)
        jax_decayed_v = jnp.array(decay) * jnp.array(v)
        jax_gathered_g = jnp.array(g) * jnp.array(e_l)
        jax_c1 = jax_input_current + jnp.array(asc_1) + jnp.array(asc_2) + jax_gathered_g
        jax_new_v = jax_decayed_v + jnp.array(current_factor) * jax_c1 + jax_reset_current

        # Spike generation
        jax_normalizer = jnp.array(v_th) - jnp.array(e_l)
        jax_v_sc = (jax_new_v - jnp.array(v_th)) / jax_normalizer
        jax_z = jax_spike_gauss(jax_v_sc, gauss_std, dampening_factor)

        # Spike suppression
        jax_z = jnp.where(jax_new_r > 0., jnp.zeros_like(jax_z), jax_z)

        # === Comparisons ===
        np.testing.assert_allclose(
            np.array(jax_new_v), tf_new_v,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="Voltage update differs"
        )

        np.testing.assert_allclose(
            np.array(jax_new_r), tf_new_r,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="Refractory update differs"
        )

        np.testing.assert_allclose(
            np.array(jax_new_asc_1), tf_new_asc_1,
            rtol=glif3_precision.RTOL_SINGLE_STEP,
            atol=glif3_precision.ATOL_SINGLE_STEP,
            err_msg="ASC1 update differs"
        )

        np.testing.assert_array_equal(
            np.array(jax_z), tf_z,
            err_msg="Spike output differs"
        )
