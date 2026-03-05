#!/usr/bin/env python3
"""Training script for V1 cortical network model.

Usage:
    python scripts/train.py --config config.json
    python scripts/train.py --data_dir /path/to/data --results_dir /path/to/results

Reference: Chen et al., Science Advances 2022
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from v1_jax.models import V1Network, V1NetworkConfig, V1NetworkState
from v1_jax.models.readout import MultiClassReadout, BinaryReadout
from v1_jax.data.network_loader import load_billeh, cached_load_billeh
from v1_jax.data.stim_generator import create_drifting_grating_batch, create_classification_labels
from v1_jax.training.trainer import (
    V1Trainer,
    TrainConfig,
    TrainState,
    TrainMetrics,
    MetricsAccumulator,
    create_train_step_fn,
    create_eval_step_fn,
)
from v1_jax.training.distributed import (
    DistributedConfig,
    create_distributed_trainer,
    get_device_count,
    is_main_process,
)
from v1_jax.utils.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
)


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Combines all sub-configurations for training.
    """
    # Paths
    data_dir: str = ''
    results_dir: str = ''
    restore_from: str = ''

    # Task
    task_name: str = 'garrett'  # garrett, evidence, vcd_grating, ori_diff, 10class

    # Network
    neurons: int = 51978
    n_input: int = 17400
    core_only: bool = False
    max_delay: int = 5
    input_weight_scale: float = 1.0

    # Training
    n_epochs: int = 100
    batch_size: int = 2
    seq_len: int = 600
    steps_per_epoch: int = 100
    val_steps: int = 20

    # Optimizer
    learning_rate: float = 1e-3
    rate_cost: float = 0.1
    voltage_cost: float = 1e-5
    weight_cost: float = 0.0
    gradient_clip_norm: float = 1.0

    # Network parameters
    dampening_factor: float = 0.5
    gauss_std: float = 0.28
    use_dale_law: bool = True
    use_decoded_noise: bool = True
    noise_scale: Tuple[float, float] = (2.0, 2.0)

    # Distributed
    num_devices: Optional[int] = None
    use_pmap: bool = False

    # Checkpointing
    save_interval: int = 1  # Save every N epochs
    max_checkpoints: int = 10

    # Misc
    seed: int = 42
    max_time: float = -1  # Max training hours (-1 for unlimited)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Data Loading
# =============================================================================

def load_target_firing_rates(data_dir: str, n_neurons: int, seed: int) -> Array:
    """Load and interpolate target firing rates.

    Args:
        data_dir: Data directory path
        n_neurons: Number of neurons
        seed: Random seed

    Returns:
        Target firing rates array (n_neurons,)
    """
    rates_path = os.path.join(data_dir, 'garrett_firing_rates.pkl')

    if os.path.exists(rates_path):
        with open(rates_path, 'rb') as f:
            firing_rates = pickle.load(f)

        sorted_rates = np.sort(firing_rates)
        percentiles = (np.arange(len(firing_rates)) + 1).astype(np.float32) / len(firing_rates)

        rng = np.random.RandomState(seed=seed)
        x_rand = rng.uniform(size=n_neurons)
        target_rates = np.sort(np.interp(x_rand, percentiles, sorted_rates))

        return jnp.array(target_rates, dtype=jnp.float32)
    else:
        # Default target rate if file not found
        print(f"Warning: {rates_path} not found, using default target rates")
        return jnp.full(n_neurons, 0.02, dtype=jnp.float32)


def create_data_iterator(
    config: ExperimentConfig,
    is_training: bool = True,
    key: Optional[Array] = None,
) -> Iterator[Tuple[Array, Array, Array]]:
    """Create data iterator for training/validation.

    Args:
        config: Experiment configuration
        is_training: Whether this is for training
        key: Random key

    Yields:
        Tuples of (inputs, labels, weights)
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed)

    n_steps = config.steps_per_epoch if is_training else config.val_steps

    for step in range(n_steps):
        key, subkey = jax.random.split(key)

        # Generate batch based on task
        if config.task_name == 'garrett':
            # Drifting grating discrimination
            inputs, labels = create_drifting_grating_batch(
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                n_inputs=config.n_input,
                key=subkey,
            )
        else:
            # Placeholder for other tasks
            inputs = jax.random.normal(
                subkey, (config.seq_len, config.batch_size, config.n_input)
            )
            labels = jax.random.randint(
                subkey, (config.batch_size,), 0, 2
            )

        # Default weights (all 1s)
        weights = jnp.ones(config.batch_size, dtype=jnp.float32)

        yield inputs, labels, weights


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    train_step_fn,
    state: TrainState,
    network: V1Network,
    data_iter: Iterator,
    metrics_acc: MetricsAccumulator,
    config: ExperimentConfig,
) -> TrainState:
    """Run one training epoch.

    Args:
        train_step_fn: JIT-compiled training step function
        state: Current training state
        network: V1Network instance
        data_iter: Data iterator
        metrics_acc: Metrics accumulator
        config: Experiment configuration

    Returns:
        Updated TrainState
    """
    metrics_acc.reset()

    for step, (inputs, labels, weights) in enumerate(data_iter):
        # Initialize network state for this batch
        batch_size = inputs.shape[1]
        network_state = network.init_state(batch_size)

        # Run training step
        state, output, metrics = train_step_fn(
            state, inputs, labels, weights, network_state
        )

        # Update metrics
        metrics_acc.update(metrics)

        # Print progress
        if (step + 1) % 10 == 0 or step == 0:
            time_str = datetime.datetime.now().strftime('%H:%M:%S')
            print(f'  [{time_str}] Step {step + 1}/{config.steps_per_epoch}: '
                  f'{metrics_acc.format_string()}', end='\r')

    print()  # New line after progress
    return state


def validate(
    eval_step_fn,
    state: TrainState,
    network: V1Network,
    data_iter: Iterator,
    metrics_acc: MetricsAccumulator,
    config: ExperimentConfig,
) -> Dict[str, float]:
    """Run validation.

    Args:
        eval_step_fn: JIT-compiled eval step function
        state: Current training state
        network: V1Network instance
        data_iter: Validation data iterator
        metrics_acc: Metrics accumulator
        config: Experiment configuration

    Returns:
        Validation metrics dictionary
    """
    metrics_acc.reset()

    for inputs, labels, weights in data_iter:
        batch_size = inputs.shape[1]
        network_state = network.init_state(batch_size)

        output, metrics = eval_step_fn(
            state, inputs, labels, weights, network_state
        )

        metrics_acc.update(metrics)

    return metrics_acc.compute()


# =============================================================================
# Main Training Function
# =============================================================================

def main(config: ExperimentConfig):
    """Main training function.

    Args:
        config: Experiment configuration
    """
    # Print configuration
    print("=" * 60)
    print("V1 Model Training (JAX)")
    print("=" * 60)
    print(f"Task: {config.task_name}")
    print(f"Neurons: {config.neurons}")
    print(f"Devices: {get_device_count()}")
    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.seq_len}")
    print("=" * 60)

    # Create results directory
    results_path = Path(config.results_dir) / config.task_name
    results_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save_json(results_path / 'config.json')

    # Initialize random key
    key = jax.random.PRNGKey(config.seed)
    key, network_key, train_key = jax.random.split(key, 3)

    # Load network
    print("Loading network...")
    network_config = V1NetworkConfig(
        dt=1.0,
        gauss_std=config.gauss_std,
        dampening_factor=config.dampening_factor,
        max_delay=config.max_delay,
        input_weight_scale=config.input_weight_scale,
        use_dale_law=config.use_dale_law,
        use_decoded_noise=config.use_decoded_noise,
        noise_scale=config.noise_scale,
    )

    # Load Billeh network data
    network_data, input_pop = load_billeh(config.data_dir)
    n_neurons = len(network_data['node_params']['node_type_id'])

    # Create V1 network
    network = V1Network.from_billeh(
        network_path=config.data_dir,
        config=network_config,
    )
    print(f"Network loaded: {network.n_neurons} neurons, {network.n_inputs} inputs")

    # Create readout
    n_output = 10 if config.task_name == '10class' else 2
    readout = MultiClassReadout(
        n_classes=n_output,
        pool_method='mean',
    )

    def readout_fn(spikes):
        return readout(spikes)

    # Load target firing rates
    target_rates = load_target_firing_rates(
        config.data_dir, network.n_neurons, config.seed
    )

    # Create trainer
    train_config = TrainConfig(
        learning_rate=config.learning_rate,
        rate_cost=config.rate_cost,
        voltage_cost=config.voltage_cost,
        weight_cost=config.weight_cost,
        use_rate_regularization=True,
        use_voltage_regularization=True,
        use_weight_regularization=config.weight_cost > 0,
        use_dale_law=config.use_dale_law,
        gradient_clip_norm=config.gradient_clip_norm,
    )

    trainer = V1Trainer(
        network=network,
        config=train_config,
        target_firing_rates=target_rates,
    )

    # Create distributed trainer if multiple devices
    if get_device_count() > 1 and config.num_devices != 1:
        dist_config = DistributedConfig(
            num_devices=config.num_devices,
            use_pmap=config.use_pmap,
        )
        dist_trainer = create_distributed_trainer(trainer, dist_config)
        train_state = dist_trainer.init_state(train_key)
        train_step_fn = dist_trainer.create_train_step_fn(readout_fn)
        # For eval, create eval step if available
        if hasattr(dist_trainer, 'create_eval_step_fn'):
            eval_step_fn = dist_trainer.create_eval_step_fn(readout_fn)
        else:
            eval_step_fn = create_eval_step_fn(trainer, readout_fn)
    else:
        train_state = trainer.init_train_state(train_key)
        train_step_fn = create_train_step_fn(trainer, readout_fn)
        eval_step_fn = create_eval_step_fn(trainer, readout_fn)

    # Create checkpoint manager
    ckpt_config = CheckpointConfig(
        checkpoint_dir=str(results_path / 'checkpoints'),
        max_to_keep=config.max_checkpoints,
        save_interval_steps=config.save_interval,
    )
    ckpt_manager = CheckpointManager(ckpt_config)

    # Restore if specified
    if config.restore_from:
        print(f"Restoring from {config.restore_from}...")
        train_state, _ = ckpt_manager.restore(config.restore_from)
        print(f"Restored at step {train_state.step}")

    # Metrics accumulators
    train_metrics = MetricsAccumulator()
    val_metrics = MetricsAccumulator()

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    stop = False

    for epoch in range(config.n_epochs):
        if stop:
            break

        date_str = datetime.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'\nEpoch {epoch + 1}/{config.n_epochs} @ {date_str}')

        # Create data iterators
        key, train_data_key, val_data_key = jax.random.split(key, 3)
        train_iter = create_data_iterator(config, is_training=True, key=train_data_key)
        val_iter = create_data_iterator(config, is_training=False, key=val_data_key)

        # Training
        train_state = train_epoch(
            train_step_fn, train_state, network, train_iter,
            train_metrics, config
        )

        # Validation
        val_results = validate(
            eval_step_fn, train_state, network, val_iter,
            val_metrics, config
        )

        # Print validation results
        time_str = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'  [{time_str}] Validation: {val_metrics.format_string()}')

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            if is_main_process():
                ckpt_manager.save(
                    train_state, train_config,
                    step=epoch + 1,
                    metrics=val_results,
                )
                print(f'  Checkpoint saved at epoch {epoch + 1}')

        # Check time limit
        if config.max_time > 0:
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours > config.max_time:
                print(f'\nMax time ({config.max_time}h) reached, stopping...')
                stop = True

    # Final save
    if is_main_process():
        ckpt_manager.save(
            train_state, train_config,
            step=config.n_epochs,
            metrics=val_metrics.compute(),
        )
        ckpt_manager.wait_until_finished()

        # Save final results
        final_results = {
            'train_loss': train_metrics.compute().get('loss', 0),
            'train_accuracy': train_metrics.compute().get('accuracy', 0),
            'val_loss': val_metrics.compute().get('loss', 0),
            'val_accuracy': val_metrics.compute().get('accuracy', 0),
            'total_epochs': config.n_epochs,
            'training_time_hours': (time.time() - start_time) / 3600,
        }
        with open(results_path / 'results.json', 'w') as f:
            json.dump(final_results, f, indent=2)

    print(f'\nTraining complete! Results saved to {results_path}')
    ckpt_manager.close()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train V1 cortical network model')

    # Config file
    parser.add_argument('--config', type=str, default='',
                        help='Path to config JSON file')

    # Paths
    parser.add_argument('--data_dir', type=str, default='',
                        help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--restore_from', type=str, default='',
                        help='Checkpoint to restore from')

    # Task
    parser.add_argument('--task_name', type=str, default='garrett',
                        choices=['garrett', 'evidence', 'vcd_grating', 'ori_diff', '10class'],
                        help='Task to train')

    # Network
    parser.add_argument('--neurons', type=int, default=51978,
                        help='Number of neurons')
    parser.add_argument('--max_delay', type=int, default=5,
                        help='Maximum synaptic delay')

    # Training
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=600,
                        help='Sequence length')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                        help='Steps per epoch')

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--rate_cost', type=float, default=0.1,
                        help='Spike rate regularization')
    parser.add_argument('--voltage_cost', type=float, default=1e-5,
                        help='Voltage regularization')

    # Distributed
    parser.add_argument('--num_devices', type=int, default=None,
                        help='Number of devices to use')
    parser.add_argument('--use_pmap', action='store_true',
                        help='Use pmap instead of sharding')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_time', type=float, default=-1,
                        help='Max training time in hours')

    return parser.parse_args()


def main_cli():
    """Main CLI entry point."""
    args = parse_args()

    # Load config from file or create from args
    if args.config:
        config = ExperimentConfig.from_json(args.config)
        # Override with command line args
        for key, value in vars(args).items():
            if key != 'config' and value is not None:
                if hasattr(config, key):
                    # Only override if explicitly set (not default)
                    setattr(config, key, value)
    else:
        config = ExperimentConfig(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            restore_from=args.restore_from,
            task_name=args.task_name,
            neurons=args.neurons,
            max_delay=args.max_delay,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            rate_cost=args.rate_cost,
            voltage_cost=args.voltage_cost,
            num_devices=args.num_devices,
            use_pmap=args.use_pmap,
            seed=args.seed,
            max_time=args.max_time,
        )

    main(config)


if __name__ == '__main__':
    main_cli()
