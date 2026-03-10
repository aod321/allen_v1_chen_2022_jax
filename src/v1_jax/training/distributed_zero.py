"""ZeRO-style optimizer state sharding for memory-efficient distributed training.

Implements ZeRO-2 (Zero Redundancy Optimizer Stage 2) which shards:
- Optimizer states across devices
- Gradients across devices (using reduce-scatter)

This significantly reduces per-device memory usage, allowing larger batch sizes
or models to fit in GPU memory.

Reference:
    ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
    https://arxiv.org/abs/1910.02054
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax

from .trainer import TrainState, TrainMetrics, V1Trainer, TrainConfig
from ..models.v1_network import V1NetworkState, V1NetworkOutput


# =============================================================================
# ZeRO-2 Configuration
# =============================================================================

@dataclass
class ZeROConfig:
    """Configuration for ZeRO optimizer sharding.

    Attributes:
        num_devices: Number of devices to use (None for all available)
        partition_optimizer_state: Shard optimizer states (ZeRO-1)
        partition_gradients: Shard gradients (ZeRO-2)
        partition_parameters: Shard parameters (ZeRO-3, not implemented)
        data_axis_name: Axis name for data parallelism
        gradient_reduce: How to reduce gradients ('mean' or 'sum')
    """
    num_devices: Optional[int] = None
    partition_optimizer_state: bool = True
    partition_gradients: bool = True
    partition_parameters: bool = False  # ZeRO-3, not implemented yet
    data_axis_name: str = 'data'
    gradient_reduce: str = 'mean'


# =============================================================================
# Sharded Training State
# =============================================================================

class ZeROTrainState(NamedTuple):
    """Training state with ZeRO-style sharding.

    Attributes:
        step: Current training step
        params: Replicated model parameters (all devices have full copy)
        opt_state_shard: Sharded optimizer state (each device has 1/N)
        initial_params: Initial parameters for stiff regularization
        rng_key: Random key

    Note: param_shard_info is NOT stored in state as it contains Python scalars
    that cannot be pmap'd. Instead it's stored in ZeRO2Trainer.
    """
    step: int
    params: Dict[str, Array]
    opt_state_shard: Any  # Sharded optimizer state
    initial_params: Dict[str, Array]
    rng_key: Array


# =============================================================================
# ZeRO-2 Trainer
# =============================================================================

class ZeRO2Trainer:
    """Distributed trainer implementing ZeRO-2 optimization.

    ZeRO-2 partitions optimizer states and gradients across devices:
    - Each device stores 1/N of optimizer states
    - Gradients are reduce-scattered (each device gets 1/N)
    - Parameters are replicated (all-gathered after updates)

    Memory savings:
    - Optimizer states: N times less per device
    - Gradients: N times less per device
    - Parameters: Same (replicated for compute efficiency)

    Example:
        >>> config = ZeROConfig(num_devices=8)
        >>> zero_trainer = ZeRO2Trainer(trainer, config)
        >>> state = zero_trainer.init_state(rng_key)
        >>> for batch in dataloader:
        ...     state, metrics = zero_trainer.train_step(state, batch)
    """

    def __init__(
        self,
        trainer: V1Trainer,
        config: Optional[ZeROConfig] = None,
    ):
        """Initialize ZeRO-2 trainer.

        Args:
            trainer: Base V1Trainer instance
            config: ZeRO configuration
        """
        self.trainer = trainer
        self.config = config or ZeROConfig()

        # Get devices
        self.devices = jax.devices()
        if self.config.num_devices is not None:
            self.devices = self.devices[:self.config.num_devices]
        self.num_devices = len(self.devices)

        # Create mesh for data parallelism
        self.mesh = Mesh(
            np.array(self.devices),
            axis_names=(self.config.data_axis_name,)
        )

        # Create shardings
        self.replicated = NamedSharding(self.mesh, P())
        self.sharded = NamedSharding(self.mesh, P(self.config.data_axis_name))

        # Precompute shard boundaries
        self._setup_sharding_info()

    def _setup_sharding_info(self):
        """Setup information for parameter/gradient sharding."""
        # Get a sample of parameters to determine shapes
        sample_params = self.trainer.network.get_trainable_params()

        # Compute global shard info (stored as class attributes for pmap closure)
        self.param_names = sorted(sample_params.keys())
        self.param_shapes = {name: sample_params[name].shape for name in self.param_names}
        self.param_sizes = {name: sample_params[name].size for name in self.param_names}
        self.total_size = sum(self.param_sizes.values())
        self.shard_size = (self.total_size + self.num_devices - 1) // self.num_devices
        self.padded_size = self.shard_size * self.num_devices

        # For each parameter, compute which indices go to which device
        self.param_shard_info = {}
        for name, param in sample_params.items():
            total_size = param.size
            shard_size = (total_size + self.num_devices - 1) // self.num_devices
            self.param_shard_info[name] = {
                'shape': param.shape,
                'total_size': total_size,
                'shard_size': shard_size,
                'dtype': param.dtype,
            }

    def _flatten_params(self, params: Dict[str, Array]) -> Array:
        """Flatten parameter dict into a single array.

        Args:
            params: Dictionary of parameters

        Returns:
            1D array of all parameters concatenated
        """
        flat_list = []
        for name in sorted(params.keys()):
            flat_list.append(params[name].reshape(-1))
        return jnp.concatenate(flat_list)

    def _unflatten_params(self, flat: Array, template: Dict[str, Array]) -> Dict[str, Array]:
        """Unflatten 1D array back into parameter dict.

        Args:
            flat: 1D array of parameters
            template: Template dict with shapes

        Returns:
            Dictionary of parameters
        """
        result = {}
        offset = 0
        for name in sorted(template.keys()):
            shape = template[name].shape
            size = template[name].size
            result[name] = flat[offset:offset + size].reshape(shape)
            offset += size
        return result

    def _shard_for_device(
        self,
        flat_params: Array,
        device_idx: int,
    ) -> Array:
        """Get the shard of parameters for a specific device.

        Args:
            flat_params: Flattened parameters
            device_idx: Device index

        Returns:
            Shard of parameters for this device
        """
        total_size = flat_params.shape[0]
        shard_size = (total_size + self.num_devices - 1) // self.num_devices
        start = device_idx * shard_size
        end = min(start + shard_size, total_size)

        # Pad if necessary to ensure uniform shard sizes
        shard = flat_params[start:end]
        if shard.shape[0] < shard_size:
            shard = jnp.pad(shard, (0, shard_size - shard.shape[0]))

        return shard

    def init_state(self, rng_key: Array) -> ZeROTrainState:
        """Initialize ZeRO-sharded training state.

        Args:
            rng_key: Random key

        Returns:
            ZeROTrainState with sharded optimizer states
        """
        # Initialize base state
        base_state = self.trainer.init_train_state(rng_key)

        # Flatten parameters for sharding
        flat_params = self._flatten_params(base_state.params)

        # Create sharded optimizer state
        # Each device gets a slice of the optimizer state
        def create_shard_opt_state(device_idx):
            # Create dummy params for this shard
            shard_params = {'shard': jnp.zeros(self.shard_size, dtype=flat_params.dtype)}
            return self.trainer.optimizer.init(shard_params)

        # Initialize optimizer state for each shard
        opt_state_shards = [create_shard_opt_state(i) for i in range(self.num_devices)]

        # Stack the shards with device dimension
        opt_state_shard = jax.tree.map(
            lambda *xs: jnp.stack(xs),
            *opt_state_shards
        )

        return ZeROTrainState(
            step=0,
            params=base_state.params,
            opt_state_shard=opt_state_shard,
            initial_params=base_state.initial_params,
            rng_key=rng_key,
        )

    def create_train_step_fn(
        self,
        readout_fn: Callable,
    ) -> Callable:
        """Create ZeRO-2 training step function.

        The training step:
        1. Forward + backward pass (replicated params)
        2. Reduce-scatter gradients (each device gets 1/N of grads)
        3. Each device updates its shard of optimizer state
        4. All-gather updated parameters

        Args:
            readout_fn: Readout function (spikes -> predictions)

        Returns:
            Training step function
        """
        trainer = self.trainer
        num_devices = self.num_devices
        axis_name = self.config.data_axis_name

        # Use pre-computed shard info from class attributes (captured in closure)
        param_names = self.param_names
        param_shapes = self.param_shapes
        param_sizes = self.param_sizes
        total_size = self.total_size
        shard_size = self.shard_size
        padded_size = self.padded_size

        @partial(jax.pmap, axis_name=axis_name)
        def train_step_pmap(
            state: ZeROTrainState,
            inputs: Array,
            labels: Array,
            sample_weights: Array,
            network_state: V1NetworkState,
        ) -> Tuple[ZeROTrainState, V1NetworkOutput, TrainMetrics]:
            """ZeRO-2 training step (pmapped)."""
            # Get device index
            device_idx = jax.lax.axis_index(axis_name)

            # Split key for this step
            rng_key, new_key = jax.random.split(state.rng_key)

            # === Step 1: Forward + Backward (with replicated params) ===
            grad_fn = jax.value_and_grad(trainer._compute_loss, has_aux=True)
            (loss, (output, metrics)), grads = grad_fn(
                state.params,
                state.initial_params,
                inputs,
                labels,
                sample_weights,
                network_state,
                readout_fn,
                rng_key,
            )

            # === Step 2: Reduce-scatter gradients ===
            # Flatten gradients using pre-computed param_names
            flat_list = [grads[name].reshape(-1) for name in param_names]
            flat_grads = jnp.concatenate(flat_list)

            # Pad to padded_size (static computation)
            flat_grads = jnp.pad(flat_grads, (0, padded_size - total_size))

            # Reshape for reduce-scatter: (num_devices, shard_size)
            grad_chunks = flat_grads.reshape(num_devices, shard_size)

            # Reduce-scatter: psum then each device takes its slice
            grad_sum = jax.lax.psum(grad_chunks, axis_name=axis_name)
            my_grad_shard = grad_sum[device_idx] / num_devices

            # === Step 3: Update optimizer state shard ===
            my_opt_state = jax.tree.map(lambda x: x[device_idx], state.opt_state_shard)

            grad_dict = {'shard': my_grad_shard}
            param_shard = {'shard': jnp.zeros(shard_size, dtype=my_grad_shard.dtype)}

            updates, new_opt_state = trainer.optimizer.update(
                grad_dict, my_opt_state, param_shard
            )
            update_shard = updates['shard']

            # === Step 4: All-gather updates and apply to params ===
            all_updates = jax.lax.all_gather(update_shard, axis_name=axis_name)
            flat_updates = all_updates.reshape(-1)[:total_size]

            # Unflatten to param dict using pre-computed shapes
            param_updates = {}
            offset = 0
            for name in param_names:
                size = param_sizes[name]
                shape = param_shapes[name]
                param_updates[name] = flat_updates[offset:offset + size].reshape(shape)
                offset += size

            new_params = optax.apply_updates(state.params, param_updates)

            # Update the sharded opt state
            new_opt_state_shard = jax.tree.map(
                lambda old, new: old.at[device_idx].set(new),
                state.opt_state_shard,
                new_opt_state,
            )

            # === Sync metrics ===
            synced_metrics = TrainMetrics(
                loss=jax.lax.pmean(metrics.loss, axis_name=axis_name),
                classification_loss=jax.lax.pmean(metrics.classification_loss, axis_name=axis_name),
                rate_loss=jax.lax.pmean(metrics.rate_loss, axis_name=axis_name),
                voltage_loss=jax.lax.pmean(metrics.voltage_loss, axis_name=axis_name),
                weight_loss=jax.lax.pmean(metrics.weight_loss, axis_name=axis_name),
                accuracy=jax.lax.pmean(metrics.accuracy, axis_name=axis_name),
                mean_rate=jax.lax.pmean(metrics.mean_rate, axis_name=axis_name),
            )

            new_state = ZeROTrainState(
                step=state.step + 1,
                params=new_params,
                opt_state_shard=new_opt_state_shard,
                initial_params=state.initial_params,
                rng_key=new_key,
            )

            return new_state, output, synced_metrics

        return train_step_pmap

    def create_eval_step_fn(
        self,
        readout_fn: Callable,
    ) -> Callable:
        """Create ZeRO-2 evaluation step function.

        Args:
            readout_fn: Readout function

        Returns:
            Evaluation step function
        """
        trainer = self.trainer
        axis_name = self.config.data_axis_name

        @partial(jax.pmap, axis_name=axis_name)
        def eval_step_pmap(
            state: ZeROTrainState,
            inputs: Array,
            labels: Array,
            sample_weights: Array,
            network_state: V1NetworkState,
        ) -> Tuple[V1NetworkOutput, TrainMetrics]:
            """Evaluation step (no gradient/optimizer updates needed)."""
            rng_key, _ = jax.random.split(state.rng_key)

            _, (output, metrics) = trainer._compute_loss(
                state.params,
                state.initial_params,
                inputs,
                labels,
                sample_weights,
                network_state,
                readout_fn,
                rng_key,
            )

            # Sync metrics
            synced_metrics = TrainMetrics(
                loss=jax.lax.pmean(metrics.loss, axis_name=axis_name),
                classification_loss=jax.lax.pmean(metrics.classification_loss, axis_name=axis_name),
                rate_loss=jax.lax.pmean(metrics.rate_loss, axis_name=axis_name),
                voltage_loss=jax.lax.pmean(metrics.voltage_loss, axis_name=axis_name),
                weight_loss=jax.lax.pmean(metrics.weight_loss, axis_name=axis_name),
                accuracy=jax.lax.pmean(metrics.accuracy, axis_name=axis_name),
                mean_rate=jax.lax.pmean(metrics.mean_rate, axis_name=axis_name),
            )

            return output, synced_metrics

        return eval_step_pmap

    def replicate_state(self, state: ZeROTrainState) -> ZeROTrainState:
        """Replicate state across devices for pmap.

        Args:
            state: ZeRO training state

        Returns:
            Replicated state
        """
        # opt_state_shard is already shaped (num_devices, ...), replicate the full thing
        # Each device will use its index to access its own shard
        return ZeROTrainState(
            step=jax.device_put_replicated(state.step, self.devices),
            params=jax.device_put_replicated(state.params, self.devices),
            opt_state_shard=jax.device_put_replicated(state.opt_state_shard, self.devices),
            initial_params=jax.device_put_replicated(state.initial_params, self.devices),
            rng_key=jax.device_put_replicated(state.rng_key, self.devices),
        )

    def unreplicate_state(self, state: ZeROTrainState) -> ZeROTrainState:
        """Get single-device copy of replicated state.

        Args:
            state: Replicated state

        Returns:
            Unreplicated state (from first device)
        """
        return ZeROTrainState(
            step=int(state.step[0]),
            params=jax.tree.map(lambda x: x[0], state.params),
            opt_state_shard=state.opt_state_shard,  # Keep sharded
            initial_params=jax.tree.map(lambda x: x[0], state.initial_params),
            rng_key=state.rng_key[0],
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_zero2_trainer(
    trainer: V1Trainer,
    config: Optional[ZeROConfig] = None,
) -> ZeRO2Trainer:
    """Create a ZeRO-2 distributed trainer.

    Args:
        trainer: Base V1Trainer instance
        config: ZeRO configuration

    Returns:
        ZeRO2Trainer instance
    """
    return ZeRO2Trainer(trainer, config)


# =============================================================================
# Memory Estimation Utilities
# =============================================================================

def estimate_memory_savings(
    param_count: int,
    num_devices: int,
    dtype_bytes: int = 4,  # float32
) -> Dict[str, float]:
    """Estimate memory savings from ZeRO-2.

    Args:
        param_count: Number of model parameters
        num_devices: Number of devices
        dtype_bytes: Bytes per parameter (4 for float32)

    Returns:
        Dict with memory estimates in GB
    """
    # Model parameters (replicated)
    param_memory = param_count * dtype_bytes / 1e9

    # Adam optimizer states: 2x params (momentum + variance)
    opt_state_per_device_baseline = 2 * param_count * dtype_bytes / 1e9
    opt_state_per_device_zero2 = opt_state_per_device_baseline / num_devices

    # Gradients
    grad_per_device_baseline = param_count * dtype_bytes / 1e9
    grad_per_device_zero2 = grad_per_device_baseline / num_devices

    return {
        'param_memory_gb': param_memory,
        'opt_state_baseline_gb': opt_state_per_device_baseline,
        'opt_state_zero2_gb': opt_state_per_device_zero2,
        'grad_baseline_gb': grad_per_device_baseline,
        'grad_zero2_gb': grad_per_device_zero2,
        'total_baseline_per_device_gb': param_memory + opt_state_per_device_baseline + grad_per_device_baseline,
        'total_zero2_per_device_gb': param_memory + opt_state_per_device_zero2 + grad_per_device_zero2,
        'savings_factor': (opt_state_per_device_baseline + grad_per_device_baseline) /
                         (opt_state_per_device_zero2 + grad_per_device_zero2),
    }
