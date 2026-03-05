"""Distributed training infrastructure for V1 model.

Provides multi-GPU training support using JAX's pmap and modern sharding APIs.

Supports:
- Data parallelism via jax.pmap (legacy)
- Device sharding via jax.sharding (modern)
- Gradient synchronization across devices
- Distributed metrics aggregation

Reference: Chen et al., Science Advances 2022
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

from .trainer import TrainState, TrainMetrics, V1Trainer
from ..models.v1_network import V1NetworkState, V1NetworkOutput


# =============================================================================
# Distributed Configuration
# =============================================================================

@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Attributes:
        num_devices: Number of devices to use (None for all available)
        data_axis_name: Axis name for data parallelism
        use_pmap: Use legacy pmap (False for modern sharding)
        gradient_reduce: How to reduce gradients ('mean' or 'sum')
    """
    num_devices: Optional[int] = None
    data_axis_name: str = 'batch'
    use_pmap: bool = False
    gradient_reduce: str = 'mean'


# =============================================================================
# Device Management
# =============================================================================

def get_devices(num_devices: Optional[int] = None) -> list:
    """Get available JAX devices.

    Args:
        num_devices: Number of devices to use (None for all)

    Returns:
        List of JAX devices
    """
    devices = jax.devices()
    if num_devices is not None:
        devices = devices[:num_devices]
    return devices


def get_device_count() -> int:
    """Get total number of available devices."""
    return jax.device_count()


def get_local_device_count() -> int:
    """Get number of devices on current host."""
    return jax.local_device_count()


# =============================================================================
# Sharding-based Distribution (Modern API)
# =============================================================================

class ShardedTrainer:
    """Distributed trainer using JAX's modern sharding API.

    Uses NamedSharding for explicit device placement and automatic
    gradient synchronization.

    Example:
        >>> config = DistributedConfig(num_devices=8)
        >>> sharded_trainer = ShardedTrainer(trainer, config)
        >>> state = sharded_trainer.init_state(rng_key)
        >>> for batch in dataloader:
        ...     state, metrics = sharded_trainer.train_step(state, batch)
    """

    def __init__(
        self,
        trainer: V1Trainer,
        config: Optional[DistributedConfig] = None,
    ):
        """Initialize sharded trainer.

        Args:
            trainer: Base V1Trainer instance
            config: Distributed configuration
        """
        self.trainer = trainer
        self.config = config or DistributedConfig()

        # Get devices
        self.devices = get_devices(self.config.num_devices)
        self.num_devices = len(self.devices)

        # Create mesh for data parallelism
        self.mesh = Mesh(np.array(self.devices), axis_names=(self.config.data_axis_name,))

        # Create shardings
        self.replicated = NamedSharding(self.mesh, PartitionSpec())
        self.sharded_batch = NamedSharding(self.mesh, PartitionSpec(self.config.data_axis_name))

    def shard_params(self, params: Dict[str, Array]) -> Dict[str, Array]:
        """Replicate parameters across devices.

        Args:
            params: Model parameters

        Returns:
            Replicated parameters
        """
        return jax.device_put(params, self.replicated)

    def shard_batch(
        self,
        inputs: Array,
        labels: Array,
        weights: Array,
    ) -> Tuple[Array, Array, Array]:
        """Shard batch data across devices.

        Args:
            inputs: Input tensor (batch, ...)
            labels: Label tensor (batch,)
            weights: Sample weights (batch,)

        Returns:
            Tuple of sharded tensors
        """
        return (
            jax.device_put(inputs, self.sharded_batch),
            jax.device_put(labels, self.sharded_batch),
            jax.device_put(weights, self.sharded_batch),
        )

    def init_state(self, rng_key: Array) -> TrainState:
        """Initialize replicated training state.

        Args:
            rng_key: Random key

        Returns:
            Replicated TrainState
        """
        state = self.trainer.init_train_state(rng_key)

        # Replicate state across devices
        return TrainState(
            step=state.step,
            params=self.shard_params(state.params),
            opt_state=jax.device_put(state.opt_state, self.replicated),
            initial_params=self.shard_params(state.initial_params),
            rng_key=jax.device_put(state.rng_key, self.replicated),
        )

    def create_train_step_fn(
        self,
        readout_fn: Callable,
    ) -> Callable:
        """Create sharded training step function.

        Args:
            readout_fn: Readout function (spikes -> predictions)

        Returns:
            JIT-compiled sharded train step
        """
        @jax.jit
        def train_step(
            state: TrainState,
            inputs: Array,
            labels: Array,
            sample_weights: Array,
            network_state: V1NetworkState,
        ) -> Tuple[TrainState, V1NetworkOutput, TrainMetrics]:
            # The sharding constraints are implicit via device_put
            return self.trainer.train_step(
                state, inputs, labels, sample_weights, network_state, readout_fn
            )

        return train_step


# =============================================================================
# Pmap-based Distribution (Legacy API)
# =============================================================================

class PmapTrainer:
    """Distributed trainer using JAX's pmap API.

    Uses pmap for explicit data parallelism with manual gradient reduction.
    This is the legacy approach but provides more explicit control.

    Example:
        >>> config = DistributedConfig(use_pmap=True)
        >>> pmap_trainer = PmapTrainer(trainer, config)
        >>> state = pmap_trainer.init_state(rng_key)
        >>> for batch in dataloader:
        ...     state, metrics = pmap_trainer.train_step(state, batch)
    """

    def __init__(
        self,
        trainer: V1Trainer,
        config: Optional[DistributedConfig] = None,
    ):
        """Initialize pmap trainer.

        Args:
            trainer: Base V1Trainer instance
            config: Distributed configuration
        """
        self.trainer = trainer
        self.config = config or DistributedConfig()
        self.axis_name = self.config.data_axis_name

        # Get devices
        self.devices = get_devices(self.config.num_devices)
        self.num_devices = len(self.devices)

    def replicate(self, pytree: Any) -> Any:
        """Replicate pytree across devices.

        Args:
            pytree: JAX pytree to replicate

        Returns:
            Replicated pytree
        """
        return jax.device_put_replicated(pytree, self.devices)

    def unreplicate(self, pytree: Any) -> Any:
        """Get single device copy of replicated pytree.

        Args:
            pytree: Replicated pytree

        Returns:
            Unreplicated pytree (from first device)
        """
        return jax.tree.map(lambda x: x[0], pytree)

    def init_state(self, rng_key: Array) -> TrainState:
        """Initialize replicated training state.

        Args:
            rng_key: Random key

        Returns:
            Replicated TrainState
        """
        # Create different keys for each device
        keys = jax.random.split(rng_key, self.num_devices)

        # Initialize states for each device
        states = []
        for i, key in enumerate(keys):
            state = self.trainer.init_train_state(key)
            states.append(state)

        # Stack states
        return jax.tree.map(lambda *xs: jnp.stack(xs), *states)

    def create_train_step_fn(
        self,
        readout_fn: Callable,
    ) -> Callable:
        """Create pmap training step function.

        Args:
            readout_fn: Readout function (spikes -> predictions)

        Returns:
            Pmap-ed train step function
        """
        def single_device_step(
            state: TrainState,
            inputs: Array,
            labels: Array,
            sample_weights: Array,
            network_state: V1NetworkState,
        ) -> Tuple[TrainState, V1NetworkOutput, TrainMetrics]:
            return self.trainer.train_step(
                state, inputs, labels, sample_weights, network_state, readout_fn
            )

        # Pmap with gradient synchronization
        @partial(jax.pmap, axis_name=self.axis_name)
        def pmap_train_step(
            state: TrainState,
            inputs: Array,
            labels: Array,
            sample_weights: Array,
            network_state: V1NetworkState,
        ) -> Tuple[TrainState, V1NetworkOutput, TrainMetrics]:
            new_state, output, metrics = single_device_step(
                state, inputs, labels, sample_weights, network_state
            )

            # Synchronize metrics across devices
            synced_metrics = TrainMetrics(
                loss=jax.lax.pmean(metrics.loss, axis_name=self.axis_name),
                classification_loss=jax.lax.pmean(metrics.classification_loss, axis_name=self.axis_name),
                rate_loss=jax.lax.pmean(metrics.rate_loss, axis_name=self.axis_name),
                voltage_loss=jax.lax.pmean(metrics.voltage_loss, axis_name=self.axis_name),
                weight_loss=jax.lax.pmean(metrics.weight_loss, axis_name=self.axis_name),
                accuracy=jax.lax.pmean(metrics.accuracy, axis_name=self.axis_name),
                mean_rate=jax.lax.pmean(metrics.mean_rate, axis_name=self.axis_name),
            )

            return new_state, output, synced_metrics

        return pmap_train_step

    def create_eval_step_fn(
        self,
        readout_fn: Callable,
    ) -> Callable:
        """Create pmap evaluation step function.

        Args:
            readout_fn: Readout function

        Returns:
            Pmap-ed eval step function
        """
        @partial(jax.pmap, axis_name=self.axis_name)
        def pmap_eval_step(
            state: TrainState,
            inputs: Array,
            labels: Array,
            sample_weights: Array,
            network_state: V1NetworkState,
        ) -> Tuple[V1NetworkOutput, TrainMetrics]:
            output, metrics = self.trainer.eval_step(
                state, inputs, labels, sample_weights, network_state, readout_fn
            )

            # Synchronize metrics
            synced_metrics = TrainMetrics(
                loss=jax.lax.pmean(metrics.loss, axis_name=self.axis_name),
                classification_loss=jax.lax.pmean(metrics.classification_loss, axis_name=self.axis_name),
                rate_loss=jax.lax.pmean(metrics.rate_loss, axis_name=self.axis_name),
                voltage_loss=jax.lax.pmean(metrics.voltage_loss, axis_name=self.axis_name),
                weight_loss=jax.lax.pmean(metrics.weight_loss, axis_name=self.axis_name),
                accuracy=jax.lax.pmean(metrics.accuracy, axis_name=self.axis_name),
                mean_rate=jax.lax.pmean(metrics.mean_rate, axis_name=self.axis_name),
            )

            return output, synced_metrics

        return pmap_eval_step


# =============================================================================
# Data Sharding Utilities
# =============================================================================

def shard_batch_for_pmap(
    batch: Tuple[Array, ...],
    num_devices: int,
) -> Tuple[Array, ...]:
    """Shard batch across devices for pmap.

    Args:
        batch: Tuple of batch tensors (inputs, labels, weights)
        num_devices: Number of devices

    Returns:
        Sharded batch tensors with leading device dimension
    """
    def reshape_for_pmap(x):
        batch_size = x.shape[0]
        per_device = batch_size // num_devices
        return x.reshape(num_devices, per_device, *x.shape[1:])

    return tuple(reshape_for_pmap(x) for x in batch)


def unshard_batch_from_pmap(
    batch: Tuple[Array, ...],
) -> Tuple[Array, ...]:
    """Unshard batch from pmap format.

    Args:
        batch: Tuple of sharded tensors with device dimension

    Returns:
        Unsharded batch tensors
    """
    def reshape_from_pmap(x):
        return x.reshape(-1, *x.shape[2:])

    return tuple(reshape_from_pmap(x) for x in batch)


# =============================================================================
# Gradient Synchronization
# =============================================================================

def sync_gradients(
    grads: Dict[str, Array],
    axis_name: str = 'batch',
    reduce: str = 'mean',
) -> Dict[str, Array]:
    """Synchronize gradients across devices.

    Args:
        grads: Gradient pytree
        axis_name: Pmap axis name
        reduce: Reduction method ('mean' or 'sum')

    Returns:
        Synchronized gradients
    """
    if reduce == 'mean':
        return jax.lax.pmean(grads, axis_name=axis_name)
    elif reduce == 'sum':
        return jax.lax.psum(grads, axis_name=axis_name)
    else:
        raise ValueError(f"Unknown reduce method: {reduce}")


def sync_batch_stats(
    stats: Dict[str, Array],
    axis_name: str = 'batch',
) -> Dict[str, Array]:
    """Synchronize batch statistics across devices.

    Useful for batch normalization or running mean/variance.

    Args:
        stats: Statistics pytree
        axis_name: Pmap axis name

    Returns:
        Synchronized statistics
    """
    return jax.lax.pmean(stats, axis_name=axis_name)


# =============================================================================
# Distributed Trainer Factory
# =============================================================================

def create_distributed_trainer(
    trainer: V1Trainer,
    config: Optional[DistributedConfig] = None,
) -> Union[ShardedTrainer, PmapTrainer]:
    """Create appropriate distributed trainer.

    Args:
        trainer: Base V1Trainer instance
        config: Distributed configuration

    Returns:
        ShardedTrainer or PmapTrainer based on config
    """
    config = config or DistributedConfig()

    if config.use_pmap:
        return PmapTrainer(trainer, config)
    else:
        return ShardedTrainer(trainer, config)


# =============================================================================
# Multi-Host Support
# =============================================================================

def initialize_multi_host(
    coordinator_address: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
):
    """Initialize JAX for multi-host training.

    Must be called before any other JAX operations in multi-host setup.

    Args:
        coordinator_address: Address of coordinator (host:port)
        num_processes: Total number of processes
        process_id: This process's ID
    """
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )


def is_main_process() -> bool:
    """Check if this is the main process (process_id == 0)."""
    return jax.process_index() == 0


def get_process_count() -> int:
    """Get total number of processes."""
    return jax.process_count()


def get_process_index() -> int:
    """Get this process's index."""
    return jax.process_index()


# =============================================================================
# All-reduce Operations
# =============================================================================

def all_reduce_mean(x: Array) -> Array:
    """All-reduce with mean across all devices."""
    return jax.lax.pmean(x, axis_name='batch')


def all_reduce_sum(x: Array) -> Array:
    """All-reduce with sum across all devices."""
    return jax.lax.psum(x, axis_name='batch')


def all_gather(x: Array) -> Array:
    """Gather data from all devices."""
    return jax.lax.all_gather(x, axis_name='batch')


def broadcast_from_main(x: Array) -> Array:
    """Broadcast value from main device to all devices."""
    # Get values from device 0 and broadcast
    return jax.lax.pbroadcast(x, source=0, axis_name='batch')
