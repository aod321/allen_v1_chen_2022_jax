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
from v1_jax.nn.sparse_layer import SparseFormat
from v1_jax.models.readout import (
    MultiClassReadout,
    BinaryReadout,
    L5VotingReadout,
    L5VotingConfig,
    create_l5_voting_readout,
    L5ThresholdReadout,
    L5ThresholdConfig,
    create_l5_threshold_readout,
    L5_POOL_ASSIGNMENTS,
    L5_READOUT_STRATEGY,
)
from v1_jax.data.network_loader import load_billeh, cached_load_billeh, load_garrett_firing_rates
from v1_jax.data.stim_generator import (
    create_drifting_grating_batch,
    create_classification_labels,
)
from v1_jax.data.mnist_loader import MNISTDataLoader, MNISTConfig
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

    # Memory optimization
    use_gradient_checkpointing: bool = False
    checkpoint_every_n_steps: int = 50

    # Sparse matrix optimization
    sparse_format: SparseFormat = "bcsr"

    # Distributed
    num_devices: Optional[int] = None
    use_pmap: bool = False

    # Checkpointing
    save_interval: int = 1
    max_checkpoints: int = 10

    # L5 Readout (biologically realistic)
    use_l5_readout: bool = False  # Use L5 pyramidal cell voting readout
    localized_readout: bool = True  # Use spatially localized L5 neurons
    n_readout_populations: int = 15  # Number of L5 readout populations
    readout_neurons_per_class: int = 16  # L5 neurons per class
    readout_class_offset: int = 5  # Offset into readout populations for classes

    # Misc
    seed: int = 42
    max_time: float = -1
    profile: bool = False  # Enable detailed timing profiling
    use_mixed_precision: bool = False  # Use bfloat16 mixed precision
    use_zero2: bool = False  # Use ZeRO-2 optimizer sharding

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
        config_dict['profile'] = cfg.get('profile', False)
        config_dict['use_mixed_precision'] = cfg.get('use_mixed_precision', False)

        # Task config
        if 'task' in cfg:
            config_dict['task_name'] = cfg.task.get('name', 'garrett')
            # L5 readout config (can be nested under task.readout)
            if 'readout' in cfg.task:
                readout = cfg.task.readout
                config_dict['use_l5_readout'] = readout.get('use_l5_readout', False)
                config_dict['localized_readout'] = readout.get('localized_readout', True)
                config_dict['n_readout_populations'] = readout.get('n_readout_populations', 15)
                config_dict['readout_neurons_per_class'] = readout.get('neurons_per_class', 16)
                config_dict['readout_class_offset'] = readout.get('class_offset', 5)

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
            # Memory optimization
            config_dict['use_gradient_checkpointing'] = net.get('use_gradient_checkpointing', False)
            config_dict['checkpoint_every_n_steps'] = net.get('checkpoint_every_n_steps', 50)
            # Sparse matrix optimization
            config_dict['sparse_format'] = net.get('sparse_format', 'bcsr')

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
            config_dict['use_zero2'] = train.get('use_zero2', False)

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
    seed: int = 42,
) -> Array:
    """Load and interpolate target firing rates from experimental data.

    Loads firing rates from the Garrett et al. dataset and interpolates
    them to match the number of neurons in the network.

    This is a thin wrapper around load_garrett_firing_rates for backward
    compatibility with the training script.

    Args:
        data_dir: Path to data directory containing firing rates file.
        n_neurons: Number of neurons to generate target rates for.
        seed: Random seed (not used, kept for API compatibility).

    Returns:
        Target firing rates array with shape (n_neurons,).
    """
    try:
        target_rates = load_garrett_firing_rates(data_dir, n_neurons=n_neurons)
        return jnp.array(target_rates, dtype=jnp.float32)
    except FileNotFoundError:
        print(f"Warning: garrett_firing_rates.pkl not found in {data_dir}, using default target rates")
        return jnp.full(n_neurons, 0.02, dtype=jnp.float32)


class GarrettDataLoader:
    """Data loader for Garrett task using pre-computed LGN firing rates.

    This matches the TF implementation which loads pre-computed LGN responses
    from many_small_stimuli.pkl.

    TF data flow:
    1. rates shape: (n_images, 1000, n_lgn) - 1000ms of LGN response per image
    2. For each image presentation:
       - _pause = rates[0, -50:]  # last 50ms as gray baseline
       - _im = rates[img_idx, 50:im_slice+delay]  # skip first 50ms gray
       - _seq = concat(_pause, _im)  # total: 50 + im_slice + delay - 50 = im_slice + delay
    3. seq_len = 600ms = 2 * (im_slice + delay) = 2 * 300ms = 2 image presentations
    4. sample_poisson: p = 1 - exp(-rate/1000), then z = p * 1.3 (current_input=True)
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        seq_len: int = 600,
        im_slice: int = 100,
        delay: int = 200,
        p_reappear: float = 0.1,  # TF default is 0.1, not 0.5
        n_images: int = 8,  # TF default is 8
        is_training: bool = True,
        current_input: bool = True,
        seed: int = 42,
    ):
        """Initialize Garrett data loader.

        Args:
            data_dir: Path to GLIF_network directory
            batch_size: Batch size
            seq_len: Sequence length (must be divisible by (im_slice + delay))
            im_slice: Image presentation duration (ms), default 100
            delay: Delay period after image (ms), default 200
            p_reappear: Probability of same image reappearing (TF default: 0.1)
            n_images: Number of images to use (TF default: 8)
            is_training: Use training or validation data
            current_input: Use rate-coded input (True) or binary spikes (False)
            seed: Random seed
        """
        # Load pre-computed LGN firing rates
        if is_training:
            stim_path = os.path.join(data_dir, '..', 'many_small_stimuli.pkl')
        else:
            stim_path = os.path.join(data_dir, '..', 'alternate_small_stimuli.pkl')

        with open(stim_path, 'rb') as f:
            data = pickle.load(f)

        # Convert dict to array: (n_images, time, n_lgn)
        self.rates = np.stack(list(data.values()), axis=0).astype(np.float32)
        self.n_total_images = self.rates.shape[0]
        self.n_images = min(n_images, self.n_total_images)
        self.n_lgn = self.rates.shape[-1]

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.im_slice = im_slice
        self.delay = delay
        self.p_reappear = p_reappear
        self.current_input = current_input

        # Each image presentation is im_slice + delay ms
        self.presentation_len = im_slice + delay
        self.n_presentations = seq_len // self.presentation_len

        # Chunk structure (for labels)
        self.chunk_size = 50  # ms
        self.n_chunks_per_presentation = self.presentation_len // self.chunk_size
        self.n_chunks_total = seq_len // self.chunk_size

        # TF uses pre_chunks=4 for garrett task
        self.pre_chunks = 4
        self.resp_chunks = 1

        self.rng = np.random.RandomState(seed)

        print(f"Loaded {self.n_total_images} images, using {self.n_images}")
        print(f"Rates shape: {self.rates.shape}")
        print(f"Presentation length: {self.presentation_len}ms, {self.n_presentations} presentations per sequence")

    def _sample_poisson(self, rates: np.ndarray) -> np.ndarray:
        """Convert firing rates to input current (matching TF implementation).

        Args:
            rates: LGN firing rates (Hz)

        Returns:
            Input current values
        """
        # Assuming dt = 1 ms, convert Hz to probability
        p = 1 - np.exp(-rates / 1000.0)

        if self.current_input:
            # Match TF: _z = _p * 1.3
            return (p * 1.3).astype(np.float32)
        else:
            # Sample binary spikes
            return (self.rng.uniform(size=p.shape) < p).astype(np.float32)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of data for the Garrett task (vectorized).

        Matches TF's generate_data_set_continuing function.

        Returns:
            Tuple of (inputs, labels, weights) where:
                - inputs: (seq_len, batch_size, n_lgn)
                - labels: (batch_size,) binary change detection labels
                - weights: (batch_size,) sample weights
        """
        # Vectorized: generate all random numbers at once
        # Shape: (batch_size, n_presentations)
        change_probs = self.rng.uniform(size=(self.batch_size, self.n_presentations))
        changes = change_probs > self.p_reappear
        changes[:, 0] = False  # First presentation cannot be a change

        # Generate initial image indices for all batches
        current_indices = self.rng.randint(0, self.n_images, size=self.batch_size)

        # Pre-allocate image indices array: (batch_size, n_presentations)
        img_indices = np.zeros((self.batch_size, self.n_presentations), dtype=np.int32)
        img_indices[:, 0] = current_indices

        # Generate new indices where changes occur
        for pres in range(1, self.n_presentations):
            # Where change happens, sample new different index
            change_mask = changes[:, pres]
            new_indices = self.rng.randint(0, self.n_images, size=self.batch_size)
            # Resample where new == old (ensure different image)
            same_mask = (new_indices == img_indices[:, pres - 1]) & change_mask
            while np.any(same_mask) and self.n_images > 1:
                new_indices[same_mask] = self.rng.randint(0, self.n_images, size=np.sum(same_mask))
                same_mask = (new_indices == img_indices[:, pres - 1]) & change_mask

            img_indices[:, pres] = np.where(change_mask, new_indices, img_indices[:, pres - 1])

        # Labels: whether last presentation had a change
        labels = changes[:, -1].astype(np.int32)

        # Pre-compute pause segment (same for all)
        pause = self.rates[0, -50:]  # (50, n_lgn)

        # Gather all image segments at once
        # img_indices: (batch_size, n_presentations)
        # self.rates: (n_images, time, n_lgn)
        # We need: (batch_size, n_presentations, 250, n_lgn)
        im_segments = self.rates[img_indices, 50:self.im_slice + self.delay]  # (batch, n_pres, 250, n_lgn)

        # Build full sequence: pause + im_segment for each presentation
        # pause: (50, n_lgn) -> broadcast to (batch, n_pres, 50, n_lgn)
        pause_expanded = np.broadcast_to(pause, (self.batch_size, self.n_presentations, 50, self.n_lgn))

        # Concatenate pause and image: (batch, n_pres, 300, n_lgn)
        segments = np.concatenate([pause_expanded, im_segments], axis=2)

        # Reshape to (batch, seq_len, n_lgn)
        inputs = segments.reshape(self.batch_size, self.seq_len, self.n_lgn)

        # Transpose to (seq_len, batch, n_lgn)
        inputs = np.transpose(inputs, (1, 0, 2))

        # Convert firing rates to model input
        inputs = self._sample_poisson(inputs)

        weights = np.ones(self.batch_size, dtype=np.float32)

        # Return numpy arrays - conversion to JAX happens in training loop
        return inputs, labels, weights


def create_data_iterator(
    config: ExperimentConfig,
    is_training: bool = True,
    key: Optional[Array] = None,
    data_loader: Optional[Any] = None,
) -> Iterator[Tuple[Array, Array, Array]]:
    """Create a data iterator for training or validation.

    Generates batches of stimuli, labels, and sample weights based on
    the specified task configuration.

    Args:
        config: Experiment configuration containing task and data params.
        is_training: If True, creates training iterator; else validation.
        key: JAX random key for data generation.
        data_loader: Optional pre-initialized data loader (for Garrett/MNIST task).

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
        if config.task_name == 'garrett' and data_loader is not None:
            # Use pre-loaded LGN data (matches TF implementation)
            inputs, labels, weights = data_loader.sample_batch()
        elif config.task_name == 'garrett':
            # Fallback to synthetic data if loader not available
            inputs, labels = create_drifting_grating_batch(
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                n_inputs=config.n_input,
                key=subkey,
            )
            weights = jnp.ones(config.batch_size, dtype=jnp.float32)
        elif config.task_name == 'mnist' and data_loader is not None:
            # MNIST digit classification
            inputs, labels, weights = data_loader.sample_batch()
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
    dist_trainer: Optional[Any] = None,
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
        dist_trainer: Optional distributed trainer for pmap mode.

    Returns:
        Updated TrainState after processing all batches.
    """
    from collections import deque

    metrics_acc.reset()
    use_pmap = dist_trainer is not None and config.use_pmap
    num_devices = dist_trainer.num_devices if dist_trainer else 1
    devices = dist_trainer.devices if dist_trainer else None

    def prepare_numpy_batch(batch_data):
        """Prepare batch using numpy only (can run in thread)."""
        inputs, labels, weights = batch_data
        if use_pmap:
            batch_size = inputs.shape[1]
            batch_per_device = batch_size // num_devices
            inputs = inputs.reshape(inputs.shape[0], num_devices, batch_per_device, *inputs.shape[2:])
            inputs = np.transpose(inputs, (1, 0, 2) + tuple(range(3, inputs.ndim)))
            labels = labels.reshape(num_devices, batch_per_device)
            weights = weights.reshape(num_devices, batch_per_device)
        return inputs, labels, weights

    def to_device(inputs, labels, weights):
        """Transfer to GPU (must run in main thread)."""
        if use_pmap:
            # inputs shape before sharding: (devices, time, batch_per_device, n_input)
            batch_per_device = inputs.shape[2]  # Get batch_per_device BEFORE sharding
            inputs = jax.device_put_sharded(list(inputs), devices)
            labels = jax.device_put_sharded(list(labels), devices)
            weights = jax.device_put_sharded(list(weights), devices)
            network_state = network.init_state(batch_per_device)
            network_state = jax.device_put_replicated(network_state, devices)
        else:
            inputs = jnp.asarray(inputs)
            labels = jnp.asarray(labels)
            weights = jnp.asarray(weights)
            batch_size = inputs.shape[1]
            network_state = network.init_state(batch_size)
        return inputs, labels, weights, network_state

    # Streaming mode: generate batches on-the-fly (memory efficient for large steps_per_epoch)
    # Profiling accumulators
    profile_enabled = getattr(config, 'profile', False)
    if profile_enabled:
        times_transfer = []
        times_compute = []
        times_total = []
        print('  [PROFILE] Profiling enabled - timing each step')

    for step, batch_data in enumerate(data_iter):
        if step >= config.steps_per_epoch:
            break
        inputs, labels, weights = prepare_numpy_batch(batch_data)
        if profile_enabled:
            t_start = time.time()

        # Transfer to GPU
        inputs, labels, weights, network_state = to_device(inputs, labels, weights)

        if profile_enabled:
            # Block until transfer complete
            jax.block_until_ready(inputs)
            t_transfer = time.time()

        # Run training step
        state, output, metrics = train_step_fn(
            state, inputs, labels, weights, network_state
        )

        if profile_enabled:
            # Block until compute complete (metrics may be dict or NamedTuple)
            if hasattr(metrics, 'loss'):
                jax.block_until_ready(metrics.loss)
            elif isinstance(metrics, dict):
                jax.block_until_ready(metrics['loss'])
            else:
                jax.block_until_ready(metrics)
            t_compute = time.time()
            times_transfer.append(t_transfer - t_start)
            times_compute.append(t_compute - t_transfer)
            times_total.append(t_compute - t_start)

        # For pmap, metrics are already synced, just take first device
        if use_pmap:
            metrics = jax.tree.map(lambda x: x[0], metrics)

        metrics_acc.update(metrics)

        if (step + 1) % 10 == 0 or step == 0:
            time_str = datetime.datetime.now().strftime('%H:%M:%S')
            print(
                f'  [{time_str}] Step {step + 1}/{config.steps_per_epoch}: '
                f'{metrics_acc.format_string()}',
                end='\r'
            )

    print()

    # Print profiling summary
    if profile_enabled and len(times_total) > 0:
        # Skip first step (JIT compilation)
        if len(times_total) > 1:
            times_transfer = times_transfer[1:]
            times_compute = times_compute[1:]
            times_total = times_total[1:]

        print('\n  ========== PROFILING SUMMARY ==========')
        print(f'  Steps profiled: {len(times_total)} (excluding JIT warmup)')
        print(f'  Data transfer:  {np.mean(times_transfer)*1000:.1f}ms ± {np.std(times_transfer)*1000:.1f}ms')
        print(f'  Compute (fwd+bwd): {np.mean(times_compute)*1000:.1f}ms ± {np.std(times_compute)*1000:.1f}ms')
        print(f'  Total per step: {np.mean(times_total)*1000:.1f}ms ± {np.std(times_total)*1000:.1f}ms')
        print(f'  Transfer %:     {np.mean(times_transfer)/np.mean(times_total)*100:.1f}%')
        print(f'  Compute %:      {np.mean(times_compute)/np.mean(times_total)*100:.1f}%')
        print('  =========================================\n')

    return state


def validate(
    eval_step_fn: Callable,
    state: TrainState,
    network: V1Network,
    data_iter: Iterator[Tuple[Array, Array, Array]],
    metrics_acc: MetricsAccumulator,
    config: ExperimentConfig,
    dist_trainer: Optional[Any] = None,
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
        dist_trainer: Optional distributed trainer for pmap mode.

    Returns:
        Dictionary mapping metric names to their average values.
    """
    metrics_acc.reset()
    use_pmap = dist_trainer is not None and config.use_pmap
    num_devices = dist_trainer.num_devices if dist_trainer else 1
    devices = dist_trainer.devices if dist_trainer else None

    def prepare_numpy_batch(batch_data):
        inputs, labels, weights = batch_data
        if use_pmap:
            batch_size = inputs.shape[1]
            batch_per_device = batch_size // num_devices
            inputs = inputs.reshape(inputs.shape[0], num_devices, batch_per_device, *inputs.shape[2:])
            inputs = np.transpose(inputs, (1, 0, 2) + tuple(range(3, inputs.ndim)))
            labels = labels.reshape(num_devices, batch_per_device)
            weights = weights.reshape(num_devices, batch_per_device)
        return inputs, labels, weights

    def to_device(inputs, labels, weights):
        """Transfer to GPU."""
        if use_pmap:
            # inputs shape before sharding: (devices, time, batch_per_device, n_input)
            batch_per_device = inputs.shape[2]  # Get batch_per_device BEFORE sharding
            inputs = jax.device_put_sharded(list(inputs), devices)
            labels = jax.device_put_sharded(list(labels), devices)
            weights = jax.device_put_sharded(list(weights), devices)
            network_state = network.init_state(batch_per_device)
            network_state = jax.device_put_replicated(network_state, devices)
        else:
            inputs = jnp.asarray(inputs)
            labels = jnp.asarray(labels)
            weights = jnp.asarray(weights)
            batch_size = inputs.shape[1]
            network_state = network.init_state(batch_size)
        return inputs, labels, weights, network_state

    # Pre-generate all batches
    all_batches = [prepare_numpy_batch(batch_data) for batch_data in data_iter]

    for inputs, labels, weights in all_batches:
        inputs, labels, weights, network_state = to_device(inputs, labels, weights)

        output, metrics = eval_step_fn(
            state, inputs, labels, weights, network_state
        )

        if use_pmap:
            metrics = jax.tree.map(lambda x: x[0], metrics)

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

    # Check batch_size for pmap mode
    num_devices = get_device_count()
    if config.use_pmap and num_devices > 1:
        if config.batch_size % num_devices != 0:
            old_batch_size = config.batch_size
            # Round up to nearest multiple of num_devices
            config = ExperimentConfig(**{
                **config.to_dict(),
                'batch_size': ((config.batch_size + num_devices - 1) // num_devices) * num_devices
            })
            print(f"Warning: batch_size adjusted from {old_batch_size} to {config.batch_size} for pmap (must be divisible by {num_devices})")
        if config.batch_size < num_devices:
            config = ExperimentConfig(**{
                **config.to_dict(),
                'batch_size': num_devices
            })
            print(f"Warning: batch_size increased to {num_devices} (minimum 1 sample per device)")

    # Print configuration
    print("=" * 60)
    print("V1 Model Training (JAX + Hydra)")
    print("=" * 60)
    print(f"Task: {config.task_name}")
    print(f"Neurons: {config.neurons}")
    print(f"Devices: {num_devices}")
    print(f"Batch size: {config.batch_size}" + (f" ({config.batch_size // num_devices} per device)" if config.use_pmap and num_devices > 1 else ""))
    print(f"Sequence length: {config.seq_len}")
    if wandb:
        print(f"Wandb: {config.wandb_project}/{wandb.run.name}")
    print("=" * 60)

    # Create results directory (must be absolute for checkpoint manager)
    results_path = (Path(config.results_dir) / config.task_name).resolve()
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
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        sparse_format=config.sparse_format,
    )

    # Load Billeh network data with target firing rates
    # If using L5 readout, also load L5 pyramidal neuron indices
    input_pop, network_data, bkg_weights, target_rates_loaded = load_billeh(
        n_input=config.n_input,
        n_neurons=config.neurons,
        core_only=config.core_only,
        data_dir=config.data_dir,
        seed=config.seed,
        use_dale_law=config.use_dale_law,
        load_target_rates=True,  # Load Garrett firing rates
        localized_readout=config.use_l5_readout and config.localized_readout,
        n_readout_populations=config.n_readout_populations,
        readout_neurons_per_class=config.readout_neurons_per_class,
    )
    n_neurons = network_data['n_nodes']

    # Load decoded noise data for use_decoded_noise mode
    noise_data = None
    if config.use_decoded_noise:
        noise_path = os.path.join(config.data_dir, 'additive_noise.mat')
        if os.path.exists(noise_path):
            from scipy.io import loadmat
            tmp = loadmat(noise_path)
            # Convert to JAX array for JIT compatibility
            noise_data = jnp.array(tmp['additive_noise'].reshape(-1).astype(np.float32))
            print(f"Loaded decoded noise: shape {noise_data.shape}")
        else:
            print(f"Warning: additive_noise.mat not found at {noise_path}")
            config = config.__class__(**{**config.__dict__, 'use_decoded_noise': False})

    # Create V1 network using pre-loaded data
    network = V1Network.from_billeh(
        network_path=config.data_dir,
        config=network_config,
        bkg_weights=bkg_weights,
        network_data=network_data,
        input_pop=input_pop,
        noise_data=noise_data,
    )
    print(f"Network loaded: {network.n_neurons} neurons, {network.n_inputs} inputs")

    # Create readout
    n_output = 10 if config.task_name in ('10class', 'mnist') else 2
    chunk_size = 50  # Match TF down_sample parameter (50ms chunks)

    if config.use_l5_readout:
        # L5 pyramidal cell readout (biologically realistic)
        # Reference: Chen et al., Science Advances 2022

        # Determine strategy and pool assignment based on task
        task_key = config.task_name
        strategy = L5_READOUT_STRATEGY.get(task_key, 'competition')
        pools = L5_POOL_ASSIGNMENTS.get(task_key, list(range(5, 5 + n_output)))

        print(f"Using L5 readout: strategy={strategy}, pools={pools}")

        if strategy == 'threshold':
            # Single-pool threshold readout for binary tasks
            # Output: [threshold, firing_rate] -> softmax -> binary decision
            threshold_config = L5ThresholdConfig(
                threshold=0.01,  # r₀ from paper
                temporal_pooling='chunks',
                chunk_size=chunk_size,
            )
            readout = create_l5_threshold_readout(
                network_data=network_data,
                pool_index=pools[0],
                config=threshold_config,
            )
            n_output = 2  # Binary classification

            print(f"  Threshold readout: pool {pools[0]}, r₀=0.01")
            pool_key = f'localized_readout_neuron_ids_{pools[0]}'
            if pool_key in network_data:
                print(f"  Pool size: {len(network_data[pool_key])} L5 neurons")

        else:
            # Multi-pool competition readout
            # Output: [rate_0, rate_1, ...] -> softmax -> argmax
            n_classes = len(pools)
            voting_config = L5VotingConfig(
                n_classes=n_classes,
                neurons_per_class=config.readout_neurons_per_class,
                temporal_pooling='chunks',
                chunk_size=chunk_size,
            )
            readout = create_l5_voting_readout(
                network_data=network_data,
                n_classes=n_classes,
                config=voting_config,
                class_offset=pools[0],  # First pool index
            )
            n_output = n_classes

            print(f"  Competition readout: {n_classes} pools")
            for i, pool_idx in enumerate(pools):
                pool_key = f'localized_readout_neuron_ids_{pool_idx}'
                if pool_key in network_data:
                    print(f"    Class {i} (pool {pool_idx}): {len(network_data[pool_key])} L5 neurons")

        # L5 readout has no trainable parameters
        readout_params = None

        # Readout config for trainer
        readout_config = {
            'temporal_pooling': 'chunks',
            'chunk_size': chunk_size,
            'apply_softmax': False,  # Softmax applied during loss computation
            'use_l5_readout': True,
            'strategy': strategy,
        }

        def readout_fn(spikes: Array) -> Array:
            return readout(spikes)
    else:
        # Standard linear readout (trainable)
        # Use chunk-wise temporal pooling with softmax to match TF implementation
        readout = MultiClassReadout(
            n_neurons=network.n_neurons,
            n_classes=n_output,
            temporal_pooling='chunks',
            chunk_size=chunk_size,
            apply_softmax=True,  # Apply softmax like TF does
        )

        # Extract readout params for training
        readout_params = {
            'weights': readout.dense_readout.params.weights,
            'bias': readout.dense_readout.params.bias,
        }

        # Readout config for trainer
        readout_config = {
            'temporal_pooling': 'chunks',
            'chunk_size': chunk_size,
            'apply_softmax': True,
            'use_l5_readout': False,
        }

        def readout_fn(spikes: Array) -> Array:
            return readout(spikes)

    # Use target firing rates from load_billeh (already loaded and interpolated)
    if target_rates_loaded is not None:
        target_rates = jnp.array(target_rates_loaded, dtype=jnp.float32)
    else:
        # Fallback to manual loading (for backward compatibility)
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
        readout_config=readout_config,
    )

    # Create distributed trainer if multiple devices
    dist_trainer = None
    if get_device_count() > 1 and config.num_devices != 1:
        dist_config = DistributedConfig(
            num_devices=config.num_devices,
            use_pmap=config.use_pmap,
            use_zero2=config.use_zero2,
        )
        dist_trainer = create_distributed_trainer(trainer, dist_config)
        train_state = dist_trainer.init_state(train_key, readout_params=readout_params)
        train_step_fn = dist_trainer.create_train_step_fn(readout_fn)
        if hasattr(dist_trainer, 'create_eval_step_fn'):
            eval_step_fn = dist_trainer.create_eval_step_fn(readout_fn)
        else:
            eval_step_fn = create_eval_step_fn(trainer, readout_fn)

        # For ZeRO-2, replicate the state for pmap
        if config.use_zero2:
            train_state = dist_trainer.replicate_state(train_state)
    else:
        train_state = trainer.init_train_state(train_key, readout_params=readout_params)
        train_step_fn = create_train_step_fn(trainer, readout_fn)
        eval_step_fn = create_eval_step_fn(trainer, readout_fn)

    # Create checkpoint manager
    ckpt_config = CheckpointConfig(
        checkpoint_dir=str(results_path / 'checkpoints'),
        max_to_keep=config.max_checkpoints,
        save_interval_steps=config.save_interval,
    )
    ckpt_manager = CheckpointManager(ckpt_config)

    # Restore from checkpoint
    start_epoch = 0
    if config.restore_from:
        print(f"Restoring from {config.restore_from}...")
        train_state, _ = ckpt_manager.restore(config.restore_from)
        # restore_from is expected to be the epoch number
        start_epoch = int(config.restore_from)
        print(f"Restored from epoch {start_epoch}, will continue from epoch {start_epoch + 1}")
        # Re-init optimizer state since checkpoint loses Optax NamedTuple structure
        print("Re-initializing optimizer state...")
        if config.use_pmap and dist_trainer is not None:
            opt_state = dist_trainer.trainer.optimizer.init(train_state.params)
        else:
            opt_state = trainer.optimizer.init(train_state.params)
        train_state = TrainState(
            step=train_state.step,
            params=train_state.params,
            opt_state=opt_state,
            initial_params=train_state.initial_params,
            rng_key=train_state.rng_key,
        )
        # Replicate for pmap if needed
        if config.use_pmap and dist_trainer is not None:
            train_state = dist_trainer.replicate(train_state)
    else:
        # Auto-restore from latest checkpoint if exists
        # latest_step is the epoch number (checkpoints are saved with step=epoch)
        latest_epoch = ckpt_manager.latest_step
        if latest_epoch is not None:
            print(f"Auto-restoring from latest checkpoint (epoch {latest_epoch})...")
            train_state, _ = ckpt_manager.restore(step=latest_epoch)
            start_epoch = latest_epoch
            print(f"Restored from epoch {start_epoch}, will continue from epoch {start_epoch + 1}")
            # Re-init optimizer state since checkpoint loses Optax NamedTuple structure
            print("Re-initializing optimizer state...")
            if config.use_pmap and dist_trainer is not None:
                opt_state = dist_trainer.trainer.optimizer.init(train_state.params)
            else:
                opt_state = trainer.optimizer.init(train_state.params)
            train_state = TrainState(
                step=train_state.step,
                params=train_state.params,
                opt_state=opt_state,
                initial_params=train_state.initial_params,
                rng_key=train_state.rng_key,
            )
            # Replicate for pmap if needed
            if config.use_pmap and dist_trainer is not None:
                train_state = dist_trainer.replicate(train_state)

    # Metrics accumulators
    train_metrics = MetricsAccumulator()
    val_metrics = MetricsAccumulator()

    # Initialize data loaders based on task
    train_loader = None
    val_loader = None
    if config.task_name == 'garrett':
        print("\nInitializing Garrett data loaders...")
        train_loader = GarrettDataLoader(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            is_training=True,
            seed=config.seed,
        )
        val_loader = GarrettDataLoader(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            is_training=False,
            seed=config.seed + 1000,
        )
    elif config.task_name == 'mnist':
        print("\nInitializing MNIST data loaders...")
        mnist_config = MNISTConfig(
            seq_len=config.seq_len,
            pre_delay=50,
            im_slice=100,
            post_delay=config.seq_len - 150,  # Fill remaining time
            intensity=1.0,
            current_input=True,
        )
        train_loader = MNISTDataLoader(
            n_inputs=config.n_input,
            batch_size=config.batch_size,
            config=mnist_config,
            is_training=True,
            seed=config.seed,
        )
        val_loader = MNISTDataLoader(
            n_inputs=config.n_input,
            batch_size=config.batch_size,
            config=mnist_config,
            is_training=False,
            seed=config.seed + 1000,
        )

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    stop = False

    for epoch in range(start_epoch, config.n_epochs):
        if stop:
            break

        date_str = datetime.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'\nEpoch {epoch + 1}/{config.n_epochs} @ {date_str}')

        # Create data iterators
        key, train_data_key, val_data_key = jax.random.split(key, 3)
        train_iter = create_data_iterator(
            config, is_training=True, key=train_data_key, data_loader=train_loader
        )
        val_iter = create_data_iterator(
            config, is_training=False, key=val_data_key, data_loader=val_loader
        )

        # Training
        train_state = train_epoch(
            train_step_fn, train_state, network, train_iter,
            train_metrics, config, dist_trainer
        )

        # Validation
        val_results = validate(
            eval_step_fn, train_state, network, val_iter,
            val_metrics, config, dist_trainer
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
