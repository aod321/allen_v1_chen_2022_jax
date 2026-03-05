#!/usr/bin/env python3
"""Training script for V1 cortical network model.

This script uses Hydra for configuration management, supporting hierarchical
configs, command-line overrides, and experiment tracking.

Usage:
    # Default configuration
    python scripts/train.py

    # Override parameters
    python scripts/train.py training.learning_rate=1e-4 training.batch_size=4

    # Switch task
    python scripts/train.py task=evidence

    # Enable wandb logging
    python scripts/train.py wandb.project=my-project wandb.entity=my-team

    # Use custom config file
    python scripts/train.py --config-path=/path/to/configs --config-name=custom

Reference:
    Chen et al., Science Advances 2022
"""

from __future__ import annotations

import datetime
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import wandb as wandb_module

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from v1_jax.models import V1Network, V1NetworkConfig, V1NetworkState
from v1_jax.models.readout import MultiClassReadout, BinaryReadout
from v1_jax.data.network_loader import load_billeh, cached_load_billeh
from v1_jax.data.stim_generator import (
    create_drifting_grating_batch,
    create_classification_labels,
)
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

    This dataclass provides type-safe access to experiment parameters.
    It can be created from Hydra's DictConfig or loaded from JSON files.

    Attributes:
        data_dir: Path to the Billeh network data directory.
        results_dir: Path to save training results and checkpoints.
        restore_from: Path to checkpoint for resuming training.
        task_name: Name of the training task.
        neurons: Number of neurons in the network.
        n_input: Number of LGN input units.
        core_only: Whether to use only core neurons.
        max_delay: Maximum synaptic delay in timesteps.
        input_weight_scale: Scaling factor for input weights.
        n_epochs: Number of training epochs.
        batch_size: Batch size per device.
        seq_len: Sequence length (timesteps).
        steps_per_epoch: Number of training steps per epoch.
        val_steps: Number of validation steps.
        learning_rate: Learning rate for optimizer.
        rate_cost: Spike rate regularization coefficient.
        voltage_cost: Voltage regularization coefficient.
        weight_cost: Weight regularization coefficient.
        gradient_clip_norm: Maximum gradient norm for clipping.
        dampening_factor: Surrogate gradient dampening factor.
        gauss_std: Width of Gaussian surrogate gradient.
        use_dale_law: Whether to enforce Dale's law.
        use_decoded_noise: Whether to add decoded noise.
        noise_scale: Noise scale tuple (input, recurrent).
        num_devices: Number of devices to use (None = all).
        use_pmap: Whether to use pmap for distributed training.
        save_interval: Save checkpoint every N epochs.
        max_checkpoints: Maximum number of checkpoints to keep.
        seed: Random seed for reproducibility.
        max_time: Maximum training time in hours (-1 = unlimited).
        wandb_project: Wandb project name (empty = disabled).
        wandb_entity: Wandb team/entity name.
        wandb_name: Wandb run name.
        wandb_group: Wandb experiment group.
        wandb_tags: Wandb tags for the run.
        wandb_notes: Wandb run notes.
    """

    # Paths
    data_dir: str = ''
    results_dir: str = './results'
    restore_from: str = ''

    # Task
    task_name: str = 'garrett'

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
    save_interval: int = 1
    max_checkpoints: int = 10

    # Misc
    seed: int = 42
    max_time: float = -1

    # Wandb (lazy loading)
    wandb_project: str = ''
    wandb_entity: str = ''
    wandb_name: str = ''
    wandb_group: str = ''
    wandb_tags: Tuple[str, ...] = ()
    wandb_notes: str = ''

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary.

        Args:
            d: Dictionary containing configuration parameters.

        Returns:
            ExperimentConfig instance.
        """
        # Handle noise_scale conversion from list to tuple
        if 'noise_scale' in d and isinstance(d['noise_scale'], list):
            d = d.copy()
            d['noise_scale'] = tuple(d['noise_scale'])
        # Handle wandb_tags conversion from list to tuple
        if 'wandb_tags' in d and isinstance(d['wandb_tags'], list):
            d = d.copy()
            d['wandb_tags'] = tuple(d['wandb_tags'])
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file.

        Returns:
            ExperimentConfig instance.
        """
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ExperimentConfig':
        """Create configuration from Hydra DictConfig.

        Args:
            cfg: Hydra DictConfig containing nested configuration.

        Returns:
            ExperimentConfig instance with flattened parameters.
        """
        # Flatten nested Hydra config into ExperimentConfig fields
        config_dict: Dict[str, Any] = {}

        # Top-level fields
        config_dict['data_dir'] = cfg.get('data_dir', '')
        config_dict['results_dir'] = cfg.get('results_dir', './results')
        config_dict['restore_from'] = cfg.get('restore_from', '')
        config_dict['seed'] = cfg.get('seed', 42)
        config_dict['max_time'] = cfg.get('max_time', -1)
        config_dict['num_devices'] = cfg.get('num_devices', None)
        config_dict['use_pmap'] = cfg.get('use_pmap', False)
        config_dict['save_interval'] = cfg.get('save_interval', 1)
        config_dict['max_checkpoints'] = cfg.get('max_checkpoints', 10)

        # Task config
        if 'task' in cfg:
            config_dict['task_name'] = cfg.task.get('name', 'garrett')

        # Network config
        if 'network' in cfg:
            net = cfg.network
            config_dict['neurons'] = net.get('neurons', 51978)
            config_dict['n_input'] = net.get('n_input', 17400)
            config_dict['core_only'] = net.get('core_only', False)
            config_dict['max_delay'] = net.get('max_delay', 5)
            config_dict['input_weight_scale'] = net.get('input_weight_scale', 1.0)
            config_dict['dampening_factor'] = net.get('dampening_factor', 0.5)
            config_dict['gauss_std'] = net.get('gauss_std', 0.28)
            config_dict['use_dale_law'] = net.get('use_dale_law', True)
            config_dict['use_decoded_noise'] = net.get('use_decoded_noise', True)
            noise_scale = net.get('noise_scale', [2.0, 2.0])
            if isinstance(noise_scale, (list, tuple)):
                config_dict['noise_scale'] = tuple(noise_scale)
            else:
                config_dict['noise_scale'] = (2.0, 2.0)

        # Training config
        if 'training' in cfg:
            train = cfg.training
            config_dict['n_epochs'] = train.get('n_epochs', 100)
            config_dict['batch_size'] = train.get('batch_size', 2)
            config_dict['seq_len'] = train.get('seq_len', 600)
            config_dict['steps_per_epoch'] = train.get('steps_per_epoch', 100)
            config_dict['val_steps'] = train.get('val_steps', 20)
            config_dict['learning_rate'] = train.get('learning_rate', 1e-3)
            config_dict['rate_cost'] = train.get('rate_cost', 0.1)
            config_dict['voltage_cost'] = train.get('voltage_cost', 1e-5)
            config_dict['weight_cost'] = train.get('weight_cost', 0.0)
            config_dict['gradient_clip_norm'] = train.get('gradient_clip_norm', 1.0)

        # Wandb config
        if 'wandb' in cfg:
            wb = cfg.wandb
            config_dict['wandb_project'] = wb.get('project', '')
            config_dict['wandb_entity'] = wb.get('entity', '')
            config_dict['wandb_name'] = wb.get('name', '')
            config_dict['wandb_group'] = wb.get('group', '')
            tags = wb.get('tags', [])
            config_dict['wandb_tags'] = tuple(tags) if tags else ()
            config_dict['wandb_notes'] = wb.get('notes', '')

        return cls(**config_dict)

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save the configuration.
        """
        # Convert path to string if it's a Path object
        path_str = str(path)
        with open(path_str, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Data Loading
# =============================================================================

def load_target_firing_rates(
    data_dir: str,
    n_neurons: int,
    seed: int,
) -> Array:
    """Load and interpolate target firing rates from experimental data.

    Loads firing rates from the Garrett et al. dataset and interpolates
    them to match the number of neurons in the network.

    Args:
        data_dir: Path to data directory containing firing rates file.
        n_neurons: Number of neurons to generate target rates for.
        seed: Random seed for reproducible interpolation.

    Returns:
        Target firing rates array with shape (n_neurons,).
    """
    rates_path = os.path.join(data_dir, 'garrett_firing_rates.pkl')

    if os.path.exists(rates_path):
        with open(rates_path, 'rb') as f:
            firing_rates = pickle.load(f)

        sorted_rates = np.sort(firing_rates)
        percentiles = (np.arange(len(firing_rates)) + 1).astype(np.float32)
        percentiles /= len(firing_rates)

        rng = np.random.RandomState(seed=seed)
        x_rand = rng.uniform(size=n_neurons)
        target_rates = np.sort(np.interp(x_rand, percentiles, sorted_rates))

        return jnp.array(target_rates, dtype=jnp.float32)
    else:
        print(f"Warning: {rates_path} not found, using default target rates")
        return jnp.full(n_neurons, 0.02, dtype=jnp.float32)


def create_data_iterator(
    config: ExperimentConfig,
    is_training: bool = True,
    key: Optional[Array] = None,
) -> Iterator[Tuple[Array, Array, Array]]:
    """Create a data iterator for training or validation.

    Generates batches of stimuli, labels, and sample weights based on
    the specified task configuration.

    Args:
        config: Experiment configuration containing task and data params.
        is_training: If True, creates training iterator; else validation.
        key: JAX random key for data generation.

    Yields:
        Tuple of (inputs, labels, weights) where:
            - inputs: Stimulus tensor of shape (seq_len, batch_size, n_input)
            - labels: Label tensor of shape (batch_size,)
            - weights: Sample weight tensor of shape (batch_size,)
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed)

    n_steps = config.steps_per_epoch if is_training else config.val_steps

    for step in range(n_steps):
        key, subkey = jax.random.split(key)

        # Generate batch based on task
        if config.task_name == 'garrett':
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

        weights = jnp.ones(config.batch_size, dtype=jnp.float32)
        yield inputs, labels, weights


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    train_step_fn: Callable,
    state: TrainState,
    network: V1Network,
    data_iter: Iterator[Tuple[Array, Array, Array]],
    metrics_acc: MetricsAccumulator,
    config: ExperimentConfig,
) -> TrainState:
    """Run one training epoch.

    Executes training steps over all batches in the data iterator,
    accumulating metrics and printing progress.

    Args:
        train_step_fn: JIT-compiled training step function.
        state: Current training state containing model parameters.
        network: V1Network instance for state initialization.
        data_iter: Iterator yielding (inputs, labels, weights) tuples.
        metrics_acc: Accumulator for tracking training metrics.
        config: Experiment configuration.

    Returns:
        Updated TrainState after processing all batches.
    """
    metrics_acc.reset()

    for step, (inputs, labels, weights) in enumerate(data_iter):
        batch_size = inputs.shape[1]
        network_state = network.init_state(batch_size)

        state, output, metrics = train_step_fn(
            state, inputs, labels, weights, network_state
        )

        metrics_acc.update(metrics)

        if (step + 1) % 10 == 0 or step == 0:
            time_str = datetime.datetime.now().strftime('%H:%M:%S')
            print(
                f'  [{time_str}] Step {step + 1}/{config.steps_per_epoch}: '
                f'{metrics_acc.format_string()}',
                end='\r'
            )

    print()
    return state


def validate(
    eval_step_fn: Callable,
    state: TrainState,
    network: V1Network,
    data_iter: Iterator[Tuple[Array, Array, Array]],
    metrics_acc: MetricsAccumulator,
    config: ExperimentConfig,
) -> Dict[str, float]:
    """Run validation on the held-out data.

    Evaluates the model on validation data without gradient computation,
    accumulating metrics across all batches.

    Args:
        eval_step_fn: JIT-compiled evaluation step function.
        state: Current training state containing model parameters.
        network: V1Network instance for state initialization.
        data_iter: Iterator yielding validation (inputs, labels, weights).
        metrics_acc: Accumulator for tracking validation metrics.
        config: Experiment configuration.

    Returns:
        Dictionary mapping metric names to their average values.
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
# Wandb Integration
# =============================================================================

def init_wandb(config: ExperimentConfig) -> Optional[ModuleType]:
    """Initialize Weights & Biases logging (lazy loading).

    Wandb is only imported and initialized when wandb_project is specified
    in the configuration. This avoids unnecessary dependencies.

    Args:
        config: Experiment configuration containing wandb settings.

    Returns:
        The wandb module if successfully initialized, None otherwise.
    """
    if not config.wandb_project:
        return None

    try:
        import wandb as _wandb
        _wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity or None,
            name=config.wandb_name or None,
            group=config.wandb_group or None,
            tags=list(config.wandb_tags) if config.wandb_tags else None,
            notes=config.wandb_notes or None,
            config=config.to_dict(),
        )
        print(f"Wandb initialized: {_wandb.run.name}")
        return _wandb
    except ImportError:
        print(
            "Warning: wandb_project specified but wandb is not installed. "
            "Install with: pip install wandb"
        )
        return None


# =============================================================================
# Main Training Function
# =============================================================================

def run_training(config: ExperimentConfig) -> None:
    """Execute the main training loop.

    This function orchestrates the complete training process including:
    - Network initialization
    - Trainer setup
    - Checkpoint management
    - Training/validation loops
    - Metrics logging

    Args:
        config: Complete experiment configuration.
    """
    # Initialize wandb (lazy loading)
    wandb = init_wandb(config)

    # Print configuration
    print("=" * 60)
    print("V1 Model Training (JAX + Hydra)")
    print("=" * 60)
    print(f"Task: {config.task_name}")
    print(f"Neurons: {config.neurons}")
    print(f"Devices: {get_device_count()}")
    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.seq_len}")
    if wandb:
        print(f"Wandb: {config.wandb_project}/{wandb.run.name}")
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

    def readout_fn(spikes: Array) -> Array:
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
        train_iter = create_data_iterator(
            config, is_training=True, key=train_data_key
        )
        val_iter = create_data_iterator(
            config, is_training=False, key=val_data_key
        )

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

        # Log to wandb
        if wandb:
            train_log = {f'train/{k}': v for k, v in train_metrics.compute().items()}
            val_log = {f'val/{k}': v for k, v in val_results.items()}
            wandb.log({**train_log, **val_log, 'epoch': epoch + 1})

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

    # Finish wandb run
    if wandb:
        wandb.finish()


# =============================================================================
# Hydra Entry Point
# =============================================================================

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Hydra main entry point.

    This function is decorated with @hydra.main and serves as the
    entry point for Hydra-based configuration management.

    Args:
        cfg: Hydra DictConfig containing the merged configuration
            from YAML files and command-line overrides.
    """
    # Print resolved config for debugging
    print("\nResolved configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Convert Hydra config to ExperimentConfig
    config = ExperimentConfig.from_hydra(cfg)

    # Run training
    run_training(config)


if __name__ == '__main__':
    main()
