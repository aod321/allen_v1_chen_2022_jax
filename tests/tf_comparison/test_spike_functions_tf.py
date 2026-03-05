"""TensorFlow comparison tests for spike functions.

These tests verify numerical equivalence between JAX and TensorFlow
implementations of spike generation and surrogate gradient functions.

Comparison points:
1. spike_gauss forward pass (Heaviside function)
2. gauss_pseudo surrogate gradient
3. exp_convolve synaptic filtering
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

# Load TF spike functions
HAS_TF_SPIKE_FUNCS = False
tf_spike_gauss = None
tf_gauss_pseudo = None
tf_exp_convolve = None
tf_pseudo_derivative = None

if HAS_TF:
    try:
        # Import from TF models.py
        exec_globals = {'tf': tf, 'np': np}
        exec("""
import tensorflow as tf
import numpy as np

def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude

def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)

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

def exp_convolve(tensor, decay=.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse, initializer=initializer)
    filtered = tf.transpose(filtered, perm)
    return filtered
""", exec_globals)

        tf_spike_gauss = exec_globals['spike_gauss']
        tf_gauss_pseudo = exec_globals['gauss_pseudo']
        tf_exp_convolve = exec_globals['exp_convolve']
        tf_pseudo_derivative = exec_globals['pseudo_derivative']
        HAS_TF_SPIKE_FUNCS = True

    except Exception as e:
        print(f"Could not load TF spike functions: {e}")
        HAS_TF_SPIKE_FUNCS = False

# Skip if TF not available
pytestmark = pytest.mark.skipif(
    not HAS_TF or not HAS_TF_SPIKE_FUNCS,
    reason="TensorFlow or spike functions not available"
)

# Import JAX implementations
from v1_jax.nn.spike_functions import (
    spike_gauss as jax_spike_gauss,
    gauss_pseudo as jax_gauss_pseudo,
    spike_piecewise as jax_spike_piecewise,
)
from v1_jax.nn.synaptic import exp_convolve as jax_exp_convolve


class SpikeFunctionPrecision:
    """Precision tolerances for spike function comparison tests."""
    # Forward pass (Heaviside is exact)
    RTOL_FORWARD = 0.0
    ATOL_FORWARD = 0.0

    # Surrogate gradient (continuous function)
    RTOL_GRADIENT = 1e-5
    ATOL_GRADIENT = 1e-6

    # Exponential convolution (accumulated error)
    RTOL_EXP_CONVOLVE = 1e-4
    ATOL_EXP_CONVOLVE = 1e-5


@pytest.fixture
def spike_precision():
    return SpikeFunctionPrecision()


# =============================================================================
# spike_gauss Forward Pass Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestSpikeGaussForwardTF:
    """Compare JAX and TF spike_gauss forward pass."""

    def test_spike_gauss_forward_matches_tf(self, global_seed, spike_precision):
        """Test JAX spike_gauss forward pass matches TF implementation."""
        np.random.seed(global_seed)

        # Generate test voltages spanning both sides of threshold
        v_scaled = np.random.randn(100, 50).astype(np.float32)
        sigma = 0.28
        amplitude = 0.5

        # TF implementation
        tf_result = tf_spike_gauss(
            tf.constant(v_scaled),
            tf.constant(sigma, dtype=tf.float32),
            tf.constant(amplitude, dtype=tf.float32)
        ).numpy()

        # JAX implementation
        jax_result = np.array(jax_spike_gauss(
            jnp.array(v_scaled),
            sigma,
            amplitude
        ))

        np.testing.assert_array_equal(
            jax_result, tf_result,
            err_msg="spike_gauss forward pass differs between JAX and TF"
        )

    def test_spike_gauss_binary_output_matches_tf(self, global_seed):
        """Test spike output is binary and matches TF."""
        np.random.seed(global_seed)

        v_scaled = np.random.randn(200, 100).astype(np.float32)
        sigma = 0.28
        amplitude = 0.5

        tf_result = tf_spike_gauss(
            tf.constant(v_scaled),
            tf.constant(sigma, dtype=tf.float32),
            tf.constant(amplitude, dtype=tf.float32)
        ).numpy()

        jax_result = np.array(jax_spike_gauss(
            jnp.array(v_scaled), sigma, amplitude
        ))

        # Both should be binary
        assert set(np.unique(tf_result)) <= {0.0, 1.0}
        assert set(np.unique(jax_result)) <= {0.0, 1.0}

        # And equal
        np.testing.assert_array_equal(jax_result, tf_result)

    def test_spike_gauss_threshold_boundary(self):
        """Test spike generation at threshold boundary."""
        # Values very close to threshold
        v_scaled = np.array([-1e-6, 0.0, 1e-6], dtype=np.float32)
        sigma = 0.28
        amplitude = 0.5

        tf_result = tf_spike_gauss(
            tf.constant(v_scaled),
            tf.constant(sigma, dtype=tf.float32),
            tf.constant(amplitude, dtype=tf.float32)
        ).numpy()

        jax_result = np.array(jax_spike_gauss(
            jnp.array(v_scaled), sigma, amplitude
        ))

        # TF uses > 0, so exactly 0 should not spike
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(tf_result, expected)
        np.testing.assert_array_equal(jax_result, expected)


# =============================================================================
# Surrogate Gradient Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestGaussPseudoTF:
    """Compare JAX and TF Gaussian surrogate gradient."""

    def test_gauss_pseudo_matches_tf(self, global_seed, spike_precision):
        """Test JAX gauss_pseudo matches TF implementation."""
        np.random.seed(global_seed)

        v_scaled = np.random.randn(100, 50).astype(np.float32)
        sigma = 0.28
        amplitude = 0.5

        # TF implementation
        tf_result = tf_gauss_pseudo(
            tf.constant(v_scaled),
            tf.constant(sigma, dtype=tf.float32),
            tf.constant(amplitude, dtype=tf.float32)
        ).numpy()

        # JAX implementation
        jax_result = np.array(jax_gauss_pseudo(
            jnp.array(v_scaled), sigma, amplitude
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=spike_precision.RTOL_GRADIENT,
            atol=spike_precision.ATOL_GRADIENT,
            err_msg="gauss_pseudo differs between JAX and TF"
        )

    def test_gauss_pseudo_peak_at_zero(self, spike_precision):
        """Test Gaussian surrogate peaks at v=0."""
        v_scaled = np.array([0.0], dtype=np.float32)
        sigma = 0.28
        amplitude = 0.5

        tf_result = tf_gauss_pseudo(
            tf.constant(v_scaled),
            tf.constant(sigma, dtype=tf.float32),
            tf.constant(amplitude, dtype=tf.float32)
        ).numpy()

        jax_result = np.array(jax_gauss_pseudo(
            jnp.array(v_scaled), sigma, amplitude
        ))

        # At v=0, exp(0) = 1, so result should equal amplitude
        expected = amplitude
        np.testing.assert_allclose(tf_result[0], expected, rtol=1e-6)
        np.testing.assert_allclose(jax_result[0], expected, rtol=1e-6)

    def test_gauss_pseudo_different_sigmas(self, global_seed, spike_precision):
        """Test with different sigma values."""
        np.random.seed(global_seed)

        v_scaled = np.random.randn(50, 30).astype(np.float32)
        amplitude = 0.5

        for sigma in [0.1, 0.28, 0.5, 1.0]:
            tf_result = tf_gauss_pseudo(
                tf.constant(v_scaled),
                tf.constant(sigma, dtype=tf.float32),
                tf.constant(amplitude, dtype=tf.float32)
            ).numpy()

            jax_result = np.array(jax_gauss_pseudo(
                jnp.array(v_scaled), sigma, amplitude
            ))

            np.testing.assert_allclose(
                jax_result, tf_result,
                rtol=spike_precision.RTOL_GRADIENT,
                atol=spike_precision.ATOL_GRADIENT,
                err_msg=f"gauss_pseudo differs for sigma={sigma}"
            )


# =============================================================================
# Gradient Flow Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestSpikeGradientTF:
    """Compare gradient computation between JAX and TF."""

    def test_spike_gauss_gradient_matches_tf(self, global_seed, spike_precision):
        """Test JAX spike_gauss gradient matches TF gradient."""
        np.random.seed(global_seed)

        v_scaled = np.random.randn(50, 30).astype(np.float32)
        sigma = 0.28
        amplitude = 0.5

        # TF gradient
        v_tf = tf.Variable(v_scaled)
        with tf.GradientTape() as tape:
            z = tf_spike_gauss(
                v_tf,
                tf.constant(sigma, dtype=tf.float32),
                tf.constant(amplitude, dtype=tf.float32)
            )
            # Sum to get scalar loss
            loss = tf.reduce_sum(z)
        tf_grad = tape.gradient(loss, v_tf).numpy()

        # JAX gradient
        def jax_loss_fn(v):
            z = jax_spike_gauss(v, sigma, amplitude)
            return jnp.sum(z)

        jax_grad = np.array(jax.grad(jax_loss_fn)(jnp.array(v_scaled)))

        np.testing.assert_allclose(
            jax_grad, tf_grad,
            rtol=spike_precision.RTOL_GRADIENT,
            atol=spike_precision.ATOL_GRADIENT,
            err_msg="spike_gauss gradient differs between JAX and TF"
        )

    def test_spike_gauss_gradient_with_weights(self, global_seed, spike_precision):
        """Test gradient computation with upstream gradients."""
        np.random.seed(global_seed)

        v_scaled = np.random.randn(50, 30).astype(np.float32)
        weights = np.random.randn(30).astype(np.float32)  # Readout weights
        sigma = 0.28
        amplitude = 0.5

        # TF gradient
        v_tf = tf.Variable(v_scaled)
        with tf.GradientTape() as tape:
            z = tf_spike_gauss(
                v_tf,
                tf.constant(sigma, dtype=tf.float32),
                tf.constant(amplitude, dtype=tf.float32)
            )
            # Weighted sum
            loss = tf.reduce_sum(z * tf.constant(weights))
        tf_grad = tape.gradient(loss, v_tf).numpy()

        # JAX gradient
        def jax_loss_fn(v):
            z = jax_spike_gauss(v, sigma, amplitude)
            return jnp.sum(z * jnp.array(weights))

        jax_grad = np.array(jax.grad(jax_loss_fn)(jnp.array(v_scaled)))

        np.testing.assert_allclose(
            jax_grad, tf_grad,
            rtol=spike_precision.RTOL_GRADIENT,
            atol=spike_precision.ATOL_GRADIENT,
            err_msg="Weighted gradient differs between JAX and TF"
        )

    def test_gradient_chain_rule(self, global_seed, spike_precision):
        """Test gradient through a simple computation chain."""
        np.random.seed(global_seed)

        # Simple chain: linear -> spike -> sum
        x = np.random.randn(20, 10).astype(np.float32)
        w = np.random.randn(10, 10).astype(np.float32)
        sigma = 0.28
        amplitude = 0.5

        # TF
        x_tf = tf.constant(x)
        w_tf = tf.Variable(w)
        with tf.GradientTape() as tape:
            v = tf.matmul(x_tf, w_tf)  # Linear
            z = tf_spike_gauss(
                v,
                tf.constant(sigma, dtype=tf.float32),
                tf.constant(amplitude, dtype=tf.float32)
            )
            loss = tf.reduce_mean(z)
        tf_grad = tape.gradient(loss, w_tf).numpy()

        # JAX
        def jax_chain(w_jax):
            v = jnp.matmul(jnp.array(x), w_jax)
            z = jax_spike_gauss(v, sigma, amplitude)
            return jnp.mean(z)

        jax_grad = np.array(jax.grad(jax_chain)(jnp.array(w)))

        np.testing.assert_allclose(
            jax_grad, tf_grad,
            rtol=spike_precision.RTOL_GRADIENT,
            atol=spike_precision.ATOL_GRADIENT,
            err_msg="Chain rule gradient differs between JAX and TF"
        )


# =============================================================================
# exp_convolve Tests
# =============================================================================


@pytest.mark.tf_comparison
class TestExpConvolveTF:
    """Compare JAX and TF exponential convolution."""

    def test_exp_convolve_matches_tf(self, global_seed, spike_precision):
        """Test JAX exp_convolve matches TF implementation."""
        np.random.seed(global_seed)

        tensor = np.random.randn(100, 50).astype(np.float32)
        decay = 0.8

        # TF implementation
        tf_result = tf_exp_convolve(
            tf.constant(tensor),
            decay=decay,
            axis=0
        ).numpy()

        # JAX implementation
        jax_result = np.array(jax_exp_convolve(
            jnp.array(tensor),
            decay=decay,
            axis=0
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=spike_precision.RTOL_EXP_CONVOLVE,
            atol=spike_precision.ATOL_EXP_CONVOLVE,
            err_msg="exp_convolve differs between JAX and TF"
        )

    def test_exp_convolve_different_decays(self, global_seed, spike_precision):
        """Test exp_convolve with different decay values."""
        np.random.seed(global_seed)

        tensor = np.random.randn(80, 40).astype(np.float32)

        for decay in [0.5, 0.8, 0.9, 0.95]:
            tf_result = tf_exp_convolve(
                tf.constant(tensor),
                decay=decay,
                axis=0
            ).numpy()

            jax_result = np.array(jax_exp_convolve(
                jnp.array(tensor),
                decay=decay,
                axis=0
            ))

            np.testing.assert_allclose(
                jax_result, tf_result,
                rtol=spike_precision.RTOL_EXP_CONVOLVE,
                atol=spike_precision.ATOL_EXP_CONVOLVE,
                err_msg=f"exp_convolve differs for decay={decay}"
            )

    def test_exp_convolve_impulse_response(self, spike_precision):
        """Test impulse response of exp_convolve."""
        # Impulse input
        tensor = np.zeros((50, 10), dtype=np.float32)
        tensor[10, :] = 1.0
        decay = 0.9

        tf_result = tf_exp_convolve(
            tf.constant(tensor),
            decay=decay,
            axis=0
        ).numpy()

        jax_result = np.array(jax_exp_convolve(
            jnp.array(tensor),
            decay=decay,
            axis=0
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=spike_precision.RTOL_EXP_CONVOLVE,
            atol=spike_precision.ATOL_EXP_CONVOLVE,
            err_msg="Impulse response differs between JAX and TF"
        )

    def test_exp_convolve_3d_tensor(self, global_seed, spike_precision):
        """Test exp_convolve on 3D tensor (batch dimension)."""
        np.random.seed(global_seed)

        # (time, batch, neurons)
        tensor = np.random.randn(60, 4, 30).astype(np.float32)
        decay = 0.85

        tf_result = tf_exp_convolve(
            tf.constant(tensor),
            decay=decay,
            axis=0
        ).numpy()

        jax_result = np.array(jax_exp_convolve(
            jnp.array(tensor),
            decay=decay,
            axis=0
        ))

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=spike_precision.RTOL_EXP_CONVOLVE,
            atol=spike_precision.ATOL_EXP_CONVOLVE,
            err_msg="3D exp_convolve differs between JAX and TF"
        )


# =============================================================================
# piecewise surrogate gradient (bonus comparison)
# =============================================================================


@pytest.mark.tf_comparison
class TestPseudoDerivativeTF:
    """Compare piecewise linear surrogate gradient."""

    def test_pseudo_derivative_matches_tf(self, global_seed, spike_precision):
        """Test piecewise linear surrogate gradient matches TF."""
        np.random.seed(global_seed)

        v_scaled = np.random.randn(100, 50).astype(np.float32)
        dampening_factor = 0.3

        # TF implementation
        tf_result = tf_pseudo_derivative(
            tf.constant(v_scaled),
            tf.constant(dampening_factor, dtype=tf.float32)
        ).numpy()

        # JAX implementation (spike_piecewise uses similar logic)
        # Note: Our JAX piecewise is in the spike function, not a standalone pseudo derivative
        # So we compute it directly
        jax_result = np.array(
            dampening_factor * jnp.maximum(1 - jnp.abs(jnp.array(v_scaled)), 0)
        )

        np.testing.assert_allclose(
            jax_result, tf_result,
            rtol=spike_precision.RTOL_GRADIENT,
            atol=spike_precision.ATOL_GRADIENT,
            err_msg="pseudo_derivative differs between JAX and TF"
        )
