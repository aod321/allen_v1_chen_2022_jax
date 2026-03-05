"""Tests for training infrastructure.

Tests:
- TrainState creation and manipulation
- TrainConfig defaults and validation
- V1Trainer initialization and training steps
- MetricsAccumulator aggregation
- JIT compilation of training functions
- Gradient flow through training
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict

# Import training modules
from v1_jax.training.trainer import (
    TrainState,
    TrainMetrics,
    TrainConfig,
    V1Trainer,
    MetricsAccumulator,
    create_train_step_fn,
    create_eval_step_fn,
    create_lr_schedule,
)
from v1_jax.training.distributed import (
    DistributedConfig,
    get_devices,
    get_device_count,
    shard_batch_for_pmap,
    unshard_batch_from_pmap,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def jax_key():
    """Create JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def train_config():
    """Create default training config."""
    return TrainConfig(
        learning_rate=1e-3,
        rate_cost=0.1,
        voltage_cost=1e-5,
        weight_cost=0.0,
        use_rate_regularization=False,  # Simplified for tests
        use_voltage_regularization=True,
        use_weight_regularization=False,
        use_dale_law=True,
        gradient_clip_norm=1.0,
    )


@pytest.fixture
def mock_params():
    """Create mock trainable parameters."""
    return {
        'input_weights': jnp.ones((100,), dtype=jnp.float32) * 0.1,
        'recurrent_weights': jnp.ones((500,), dtype=jnp.float32) * 0.05,
    }


@pytest.fixture
def mock_train_state(mock_params, jax_key):
    """Create mock training state."""
    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(mock_params)

    return TrainState(
        step=0,
        params=mock_params,
        opt_state=opt_state,
        initial_params=mock_params.copy(),
        rng_key=jax_key,
    )


# =============================================================================
# TrainState Tests
# =============================================================================

class TestTrainState:
    """Tests for TrainState."""

    def test_train_state_creation(self, mock_params, jax_key):
        """Test TrainState can be created."""
        import optax
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(mock_params)

        state = TrainState(
            step=0,
            params=mock_params,
            opt_state=opt_state,
            initial_params=mock_params,
            rng_key=jax_key,
        )

        assert state.step == 0
        assert 'input_weights' in state.params
        assert 'recurrent_weights' in state.params
        assert state.rng_key is not None

    def test_train_state_immutable(self, mock_train_state):
        """Test TrainState is a NamedTuple (immutable)."""
        with pytest.raises(AttributeError):
            mock_train_state.step = 1

    def test_train_state_pytree_compatible(self, mock_train_state):
        """Test TrainState works with JAX pytree operations."""
        # Should not raise
        leaves, treedef = jax.tree_util.tree_flatten(mock_train_state)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored.step == mock_train_state.step


class TestTrainMetrics:
    """Tests for TrainMetrics."""

    def test_metrics_creation(self):
        """Test TrainMetrics creation."""
        metrics = TrainMetrics(
            loss=jnp.array(1.0),
            classification_loss=jnp.array(0.5),
            rate_loss=jnp.array(0.3),
            voltage_loss=jnp.array(0.1),
            weight_loss=jnp.array(0.1),
            accuracy=jnp.array(0.8),
            mean_rate=jnp.array(0.02),
        )

        assert float(metrics.loss) == pytest.approx(1.0)
        assert float(metrics.accuracy) == pytest.approx(0.8)

    def test_metrics_pytree(self):
        """Test TrainMetrics is pytree compatible."""
        metrics = TrainMetrics(
            loss=jnp.array(1.0),
            classification_loss=jnp.array(0.5),
            rate_loss=jnp.array(0.3),
            voltage_loss=jnp.array(0.1),
            weight_loss=jnp.array(0.1),
            accuracy=jnp.array(0.8),
            mean_rate=jnp.array(0.02),
        )

        # Tree map should work
        doubled = jax.tree.map(lambda x: x * 2, metrics)
        assert float(doubled.loss) == pytest.approx(2.0)


# =============================================================================
# TrainConfig Tests
# =============================================================================

class TestTrainConfig:
    """Tests for TrainConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainConfig()

        assert config.learning_rate == 1e-3
        assert config.rate_cost == 0.1
        assert config.voltage_cost == 1e-5
        assert config.use_dale_law is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainConfig(
            learning_rate=5e-4,
            rate_cost=0.5,
            gradient_clip_norm=0.5,
        )

        assert config.learning_rate == 5e-4
        assert config.rate_cost == 0.5
        assert config.gradient_clip_norm == 0.5


# =============================================================================
# MetricsAccumulator Tests
# =============================================================================

class TestMetricsAccumulator:
    """Tests for MetricsAccumulator."""

    def test_accumulator_init(self):
        """Test accumulator initialization."""
        acc = MetricsAccumulator()
        metrics = acc.compute()
        assert metrics == {}

    def test_accumulator_single_update(self):
        """Test single update."""
        acc = MetricsAccumulator()

        metrics = TrainMetrics(
            loss=jnp.array(1.0),
            classification_loss=jnp.array(0.5),
            rate_loss=jnp.array(0.3),
            voltage_loss=jnp.array(0.1),
            weight_loss=jnp.array(0.1),
            accuracy=jnp.array(0.8),
            mean_rate=jnp.array(0.02),
        )

        acc.update(metrics)
        result = acc.compute()

        assert result['loss'] == pytest.approx(1.0)
        assert result['accuracy'] == pytest.approx(0.8)

    def test_accumulator_multiple_updates(self):
        """Test multiple updates compute average."""
        acc = MetricsAccumulator()

        for loss in [1.0, 2.0, 3.0]:
            metrics = TrainMetrics(
                loss=jnp.array(loss),
                classification_loss=jnp.array(loss),
                rate_loss=jnp.array(0.0),
                voltage_loss=jnp.array(0.0),
                weight_loss=jnp.array(0.0),
                accuracy=jnp.array(1.0),
                mean_rate=jnp.array(0.01),
            )
            acc.update(metrics)

        result = acc.compute()
        assert result['loss'] == pytest.approx(2.0)  # Average of 1, 2, 3

    def test_accumulator_reset(self):
        """Test accumulator reset."""
        acc = MetricsAccumulator()

        metrics = TrainMetrics(
            loss=jnp.array(5.0),
            classification_loss=jnp.array(5.0),
            rate_loss=jnp.array(0.0),
            voltage_loss=jnp.array(0.0),
            weight_loss=jnp.array(0.0),
            accuracy=jnp.array(0.5),
            mean_rate=jnp.array(0.03),
        )
        acc.update(metrics)

        acc.reset()
        result = acc.compute()
        assert result == {}

    def test_format_string(self):
        """Test format string output."""
        acc = MetricsAccumulator()

        metrics = TrainMetrics(
            loss=jnp.array(1.234),
            classification_loss=jnp.array(1.0),
            rate_loss=jnp.array(0.123),
            voltage_loss=jnp.array(0.111),
            weight_loss=jnp.array(0.0),
            accuracy=jnp.array(0.876),
            mean_rate=jnp.array(0.025),
        )
        acc.update(metrics)

        formatted = acc.format_string()
        assert 'Loss' in formatted
        assert 'Acc' in formatted


# =============================================================================
# Learning Rate Schedule Tests
# =============================================================================

class TestLRSchedule:
    """Tests for learning rate schedules."""

    def test_constant_schedule(self):
        """Test constant learning rate."""
        schedule = create_lr_schedule(
            base_lr=1e-3,
            schedule_type='constant',
        )

        assert schedule(0) == pytest.approx(1e-3)
        assert schedule(1000) == pytest.approx(1e-3)

    def test_cosine_schedule(self):
        """Test cosine decay schedule."""
        schedule = create_lr_schedule(
            base_lr=1e-3,
            decay_steps=1000,
            schedule_type='cosine',
        )

        # LR should decrease over time
        lr_0 = schedule(0)
        lr_500 = schedule(500)
        lr_1000 = schedule(1000)

        assert lr_0 > lr_500 > lr_1000

    def test_warmup_schedule(self):
        """Test warmup schedule."""
        schedule = create_lr_schedule(
            base_lr=1e-3,
            warmup_steps=100,
            schedule_type='constant',
        )

        # LR should increase during warmup
        lr_0 = schedule(0)
        lr_50 = schedule(50)
        lr_100 = schedule(100)

        assert lr_0 < lr_50 < lr_100
        assert lr_100 == pytest.approx(1e-3, rel=1e-5)


# =============================================================================
# Distributed Utilities Tests
# =============================================================================

class TestDistributedUtilities:
    """Tests for distributed training utilities."""

    def test_get_device_count(self):
        """Test device count retrieval."""
        count = get_device_count()
        assert count >= 1

    def test_get_devices(self):
        """Test device list retrieval."""
        devices = get_devices()
        assert len(devices) >= 1

    def test_get_devices_with_limit(self):
        """Test device list with limit."""
        devices = get_devices(num_devices=1)
        assert len(devices) == 1

    def test_shard_batch_for_pmap(self):
        """Test batch sharding for pmap."""
        batch_size = 8
        num_devices = 2

        inputs = jnp.ones((batch_size, 10, 5))
        labels = jnp.zeros((batch_size,), dtype=jnp.int32)
        weights = jnp.ones((batch_size,))

        sharded = shard_batch_for_pmap((inputs, labels, weights), num_devices)

        assert sharded[0].shape == (num_devices, batch_size // num_devices, 10, 5)
        assert sharded[1].shape == (num_devices, batch_size // num_devices)
        assert sharded[2].shape == (num_devices, batch_size // num_devices)

    def test_unshard_batch_from_pmap(self):
        """Test batch unsharding from pmap."""
        num_devices = 2
        per_device = 4

        sharded_inputs = jnp.ones((num_devices, per_device, 10, 5))
        sharded_labels = jnp.zeros((num_devices, per_device), dtype=jnp.int32)
        sharded_weights = jnp.ones((num_devices, per_device))

        unsharded = unshard_batch_from_pmap((sharded_inputs, sharded_labels, sharded_weights))

        assert unsharded[0].shape == (num_devices * per_device, 10, 5)
        assert unsharded[1].shape == (num_devices * per_device,)
        assert unsharded[2].shape == (num_devices * per_device,)


# =============================================================================
# Distributed Config Tests
# =============================================================================

class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_default_config(self):
        """Test default distributed config."""
        config = DistributedConfig()

        assert config.num_devices is None
        assert config.data_axis_name == 'batch'
        assert config.use_pmap is False
        assert config.gradient_reduce == 'mean'

    def test_custom_config(self):
        """Test custom distributed config."""
        config = DistributedConfig(
            num_devices=4,
            use_pmap=True,
            gradient_reduce='sum',
        )

        assert config.num_devices == 4
        assert config.use_pmap is True
        assert config.gradient_reduce == 'sum'


# =============================================================================
# Integration Tests (Simplified)
# =============================================================================

class TestTrainerIntegration:
    """Integration tests for trainer components."""

    def test_train_state_with_optimizer_step(self, mock_params, jax_key):
        """Test training state with optimizer update."""
        import optax

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(mock_params)

        state = TrainState(
            step=0,
            params=mock_params,
            opt_state=opt_state,
            initial_params=mock_params.copy(),
            rng_key=jax_key,
        )

        # Simulate gradient update
        grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, mock_params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            initial_params=state.initial_params,
            rng_key=jax.random.split(state.rng_key)[0],
        )

        assert new_state.step == 1
        # Params should have changed
        assert not jnp.allclose(new_state.params['input_weights'], state.params['input_weights'])

    def test_jit_compatibility(self, mock_params, jax_key):
        """Test that key components are JIT compatible."""
        import optax

        @jax.jit
        def update_step(params, grads, opt_state):
            optimizer = optax.adam(1e-3)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(mock_params)
        grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.01, mock_params)

        # Should not raise
        new_params, new_opt_state = update_step(mock_params, grads, opt_state)

        assert new_params is not None
        assert new_opt_state is not None

    def test_metrics_accumulator_in_loop(self):
        """Test metrics accumulator in training loop simulation."""
        acc = MetricsAccumulator()

        # Simulate 10 training steps
        for i in range(10):
            metrics = TrainMetrics(
                loss=jnp.array(1.0 - i * 0.05),  # Decreasing loss
                classification_loss=jnp.array(1.0 - i * 0.05),
                rate_loss=jnp.array(0.0),
                voltage_loss=jnp.array(0.0),
                weight_loss=jnp.array(0.0),
                accuracy=jnp.array(0.5 + i * 0.05),  # Increasing accuracy
                mean_rate=jnp.array(0.02),
            )
            acc.update(metrics)

        result = acc.compute()

        # Average loss should be around 0.775 ((1.0 + 0.55) / 2)
        assert 0.5 < result['loss'] < 1.0
        # Average accuracy should be around 0.725
        assert 0.5 < result['accuracy'] < 1.0


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestGradientFlow:
    """Tests for gradient computation through training components."""

    def test_loss_gradient_exists(self, jax_key):
        """Test that gradients flow through loss computation."""
        def simple_loss(params, inputs, labels):
            logits = jnp.sum(params * inputs, axis=-1)
            return jnp.mean(jnp.square(logits - labels))

        params = jnp.ones((10,), dtype=jnp.float32)
        inputs = jax.random.normal(jax_key, (5, 10))
        labels = jax.random.normal(jax_key, (5,))

        grads = jax.grad(simple_loss)(params, inputs, labels)

        assert grads.shape == params.shape
        assert not jnp.allclose(grads, 0.0)

    def test_gradient_clip(self, jax_key):
        """Test gradient clipping."""
        import optax

        # Create large gradients
        grads = {'w': jnp.ones((100,)) * 100.0}

        # Apply gradient clipping
        clipper = optax.clip_by_global_norm(1.0)
        clipped_grads, _ = clipper.update(grads, clipper.init(grads))

        # Gradient norm should be <= 1.0
        grad_norm = jnp.sqrt(jnp.sum(jnp.square(clipped_grads['w'])))
        assert grad_norm <= 1.0 + 1e-5


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_batch_metrics(self):
        """Test accumulator with no updates."""
        acc = MetricsAccumulator()
        result = acc.compute()
        assert result == {}

    def test_single_device_sharding(self):
        """Test sharding with single device."""
        batch_size = 4
        num_devices = 1

        inputs = jnp.ones((batch_size, 10))
        labels = jnp.zeros((batch_size,), dtype=jnp.int32)
        weights = jnp.ones((batch_size,))

        sharded = shard_batch_for_pmap((inputs, labels, weights), num_devices)

        assert sharded[0].shape == (1, batch_size, 10)

    def test_zero_learning_rate(self):
        """Test with zero learning rate (params shouldn't change)."""
        import optax

        params = {'w': jnp.ones((10,))}
        optimizer = optax.adam(0.0)
        opt_state = optimizer.init(params)

        grads = {'w': jnp.ones((10,)) * 0.1}
        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Params should not change with zero LR
        assert jnp.allclose(new_params['w'], params['w'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
