"""Pytest configuration and fixtures for V1 JAX tests.

Provides common fixtures for:
- Random state synchronization between JAX and TensorFlow
- Network configuration loading
- Test input generation
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Optional TensorFlow import for comparison tests
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

import jax
import jax.numpy as jnp


# =============================================================================
# Random State Management
# =============================================================================


@pytest.fixture(scope="session")
def global_seed():
    """Global random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def reset_random_state(global_seed):
    """Reset random state before each test."""
    np.random.seed(global_seed)
    if HAS_TF:
        tf.random.set_seed(global_seed)


@pytest.fixture
def jax_key(global_seed):
    """Provide a fresh JAX PRNG key."""
    return jax.random.PRNGKey(global_seed)


class RandomnessController:
    """Utility for generating synchronized random tensors across frameworks."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        if HAS_TF:
            tf.random.set_seed(self.seed)
        self.jax_key = jax.random.PRNGKey(self.seed)

    def randn(self, *shape, dtype=np.float32):
        """Generate synchronized random normal tensor."""
        arr = np.random.randn(*shape).astype(dtype)
        result = {'numpy': arr, 'jax': jnp.array(arr)}
        if HAS_TF:
            result['tf'] = tf.constant(arr)
        return result

    def rand(self, *shape, dtype=np.float32):
        """Generate synchronized uniform random tensor."""
        arr = np.random.rand(*shape).astype(dtype)
        result = {'numpy': arr, 'jax': jnp.array(arr)}
        if HAS_TF:
            result['tf'] = tf.constant(arr)
        return result

    def randint(self, low, high, shape, dtype=np.int32):
        """Generate synchronized random integers."""
        arr = np.random.randint(low, high, shape).astype(dtype)
        result = {'numpy': arr, 'jax': jnp.array(arr)}
        if HAS_TF:
            result['tf'] = tf.constant(arr)
        return result

    def split_jax_key(self, n: int = 2):
        """Split JAX key and update internal state."""
        keys = jax.random.split(self.jax_key, n + 1)
        self.jax_key = keys[0]
        return keys[1:]


@pytest.fixture
def rng_controller(global_seed):
    """Provide randomness controller fixture."""
    return RandomnessController(seed=global_seed)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def small_voltage_data(global_seed):
    """Small voltage test data for spike function tests."""
    np.random.seed(global_seed)
    return np.random.randn(32, 1000).astype(np.float32)


@pytest.fixture
def spike_function_params():
    """Default parameters for spike_gauss."""
    return {'sigma': 0.28, 'amplitude': 0.5}


@pytest.fixture
def synchronized_inputs(global_seed):
    """Generate synchronized test inputs for JAX/TF comparison."""
    np.random.seed(global_seed)
    return {
        'inputs': np.random.rand(4, 100, 17400).astype(np.float32) * 0.1,
        'labels': np.random.randint(0, 2, size=(4, 2)).astype(np.int32),
        'weights': np.ones((4,), dtype=np.float32),
    }


@pytest.fixture
def small_network_inputs(global_seed):
    """Small network inputs for quick tests."""
    np.random.seed(global_seed)
    return {
        'inputs': np.random.rand(2, 50, 1000).astype(np.float32) * 0.1,
        'labels': np.random.randint(0, 2, size=(2, 2)).astype(np.int32),
    }


# =============================================================================
# Network Configuration Fixtures
# =============================================================================


@pytest.fixture
def glif3_params():
    """Default GLIF3 neuron parameters for testing."""
    return {
        'n_neurons': 100,
        'n_receptors': 4,
        'dt': 1.0,
        'v_th': -0.05,
        'v_reset': -0.07,
        'e_l': -0.07,
        'tau_m': 10.0,
        'tau_syn': np.array([2.0, 100.0, 6.0, 150.0]),
        't_ref': 2.0,
        'gauss_std': 0.28,
        'dampening_factor': 0.5,
    }


# =============================================================================
# Precision Configuration
# =============================================================================


class PrecisionConfig:
    """Numerical precision tolerances for tests."""

    # Basic operations
    RTOL_BASIC = 1e-5
    ATOL_BASIC = 1e-6

    # Exponential operations (higher error accumulation)
    RTOL_EXP = 1e-4
    ATOL_EXP = 1e-5

    # RNN operations (error accumulates over time)
    RTOL_RNN_SHORT = 1e-4   # < 100 steps
    RTOL_RNN_MEDIUM = 1e-3  # 100-500 steps
    RTOL_RNN_LONG = 1e-2    # > 500 steps

    # Sparse operations
    RTOL_SPARSE = 1e-4
    ATOL_SPARSE = 1e-5

    # Gradient computations
    RTOL_GRADIENT = 1e-3
    ATOL_GRADIENT = 1e-4


@pytest.fixture
def precision():
    """Provide precision configuration."""
    return PrecisionConfig()


# =============================================================================
# Markers and Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmarks")
    config.addinivalue_line("markers", "slow: Slow tests (>1min)")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "tf_comparison: Tests comparing with TensorFlow")


# Skip TF comparison tests if TensorFlow not available
def pytest_collection_modifyitems(config, items):
    if not HAS_TF:
        skip_tf = pytest.mark.skip(reason="TensorFlow not installed")
        for item in items:
            if "tf_comparison" in item.keywords:
                item.add_marker(skip_tf)
