"""Checkpoint management for V1 model training.

Provides save/restore functionality for model parameters, optimizer state,
and training state using orbax-checkpoint.

Reference: Chen et al., Science Advances 2022
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import orbax.checkpoint as ocp

from ..training.trainer import TrainState, TrainConfig


# =============================================================================
# Checkpoint Configuration
# =============================================================================

@dataclass
class CheckpointConfig:
    """Configuration for checkpointing.

    Attributes:
        checkpoint_dir: Directory to save checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        save_interval_steps: Save checkpoint every N steps
        keep_every_n_checkpoints: Keep every Nth checkpoint permanently
        async_save: Whether to save asynchronously
    """
    checkpoint_dir: str
    max_to_keep: int = 5
    save_interval_steps: int = 1000
    keep_every_n_checkpoints: int = 10
    async_save: bool = True


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manager for saving and restoring training checkpoints.

    Handles:
    - Saving/restoring TrainState (params, optimizer state, step)
    - Saving/restoring TrainConfig
    - Managing checkpoint retention policy
    - Async saving for minimal training interruption

    Example:
        >>> ckpt_manager = CheckpointManager(CheckpointConfig(checkpoint_dir='./ckpts'))
        >>> # Save
        >>> ckpt_manager.save(train_state, train_config, step=1000)
        >>> # Restore latest
        >>> train_state, train_config = ckpt_manager.restore_latest()
        >>> # Restore specific step
        >>> train_state, train_config = ckpt_manager.restore(step=500)
    """

    def __init__(self, config: CheckpointConfig):
        """Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration
        """
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create orbax checkpoint manager
        options = ocp.CheckpointManagerOptions(
            max_to_keep=config.max_to_keep,
            save_interval_steps=config.save_interval_steps,
            keep_period=config.keep_every_n_checkpoints,
        )

        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=options,
        )

    def save(
        self,
        train_state: TrainState,
        train_config: Optional[TrainConfig] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Save checkpoint.

        Args:
            train_state: Current training state
            train_config: Training configuration (saved as JSON alongside)
            step: Step number (defaults to train_state.step)
            metrics: Optional metrics to save with checkpoint

        Returns:
            True if checkpoint was saved
        """
        if step is None:
            step = int(train_state.step)

        # Prepare checkpoint data
        ckpt_data = {
            'step': int(train_state.step),
            'params': {k: np.array(v) for k, v in train_state.params.items()},
            'opt_state': jax.tree.map(np.array, train_state.opt_state),
            'initial_params': {k: np.array(v) for k, v in train_state.initial_params.items()},
            'rng_key': np.array(train_state.rng_key),
        }

        # Save checkpoint
        self.manager.save(step, args=ocp.args.StandardSave(ckpt_data))

        # Save config as JSON if provided
        if train_config is not None:
            config_path = self.checkpoint_dir / f'config_{step}.json'
            with open(config_path, 'w') as f:
                json.dump(asdict(train_config), f, indent=2)

        # Save metrics if provided
        if metrics is not None:
            metrics_path = self.checkpoint_dir / f'metrics_{step}.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        return True

    def restore(
        self,
        step: Optional[int] = None,
        params_only: bool = False,
    ) -> tuple[TrainState, Optional[TrainConfig]]:
        """Restore checkpoint.

        Args:
            step: Step to restore (None for latest)
            params_only: If True, only restore parameters (not optimizer state)

        Returns:
            Tuple of (TrainState, TrainConfig or None)
        """
        if step is None:
            step = self.manager.latest_step()

        if step is None:
            raise ValueError("No checkpoints found")

        # Restore checkpoint data
        ckpt_data = self.manager.restore(step)

        # Reconstruct TrainState
        train_state = TrainState(
            step=int(ckpt_data['step']),
            params={k: jnp.array(v) for k, v in ckpt_data['params'].items()},
            opt_state=jax.tree.map(jnp.array, ckpt_data['opt_state']) if not params_only else None,
            initial_params={k: jnp.array(v) for k, v in ckpt_data['initial_params'].items()},
            rng_key=jnp.array(ckpt_data['rng_key']),
        )

        # Try to restore config
        train_config = None
        config_path = self.checkpoint_dir / f'config_{step}.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                train_config = TrainConfig(**config_dict)

        return train_state, train_config

    def restore_latest(
        self,
        params_only: bool = False,
    ) -> tuple[TrainState, Optional[TrainConfig]]:
        """Restore latest checkpoint.

        Args:
            params_only: If True, only restore parameters

        Returns:
            Tuple of (TrainState, TrainConfig or None)
        """
        return self.restore(step=None, params_only=params_only)

    def restore_params_only(
        self,
        step: Optional[int] = None,
    ) -> Dict[str, Array]:
        """Restore only model parameters.

        Useful for inference or fine-tuning.

        Args:
            step: Step to restore (None for latest)

        Returns:
            Dictionary of parameters
        """
        train_state, _ = self.restore(step=step, params_only=True)
        return train_state.params

    @property
    def latest_step(self) -> Optional[int]:
        """Get the latest checkpoint step."""
        return self.manager.latest_step()

    @property
    def all_steps(self) -> list[int]:
        """Get all available checkpoint steps."""
        return list(self.manager.all_steps())

    def wait_until_finished(self):
        """Wait for any async operations to complete."""
        self.manager.wait_until_finished()

    def close(self):
        """Close the checkpoint manager."""
        self.manager.close()


# =============================================================================
# Simple Save/Load Functions
# =============================================================================

def save_params(
    params: Dict[str, Array],
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save model parameters to a file.

    Simple function for saving just the parameters without full training state.

    Args:
        params: Dictionary of parameters
        path: Path to save file
        metadata: Optional metadata to include
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'params': {k: np.array(v) for k, v in params.items()},
    }

    if metadata is not None:
        save_dict['metadata'] = metadata

    np.savez(path, **save_dict)


def load_params(
    path: Union[str, Path],
) -> tuple[Dict[str, Array], Optional[Dict[str, Any]]]:
    """Load model parameters from a file.

    Args:
        path: Path to saved file

    Returns:
        Tuple of (params dict, metadata or None)
    """
    data = np.load(path, allow_pickle=True)

    params = {k: jnp.array(v) for k, v in data['params'].item().items()}

    metadata = None
    if 'metadata' in data:
        metadata = data['metadata'].item()

    return params, metadata


# =============================================================================
# TensorFlow Checkpoint Conversion
# =============================================================================

def convert_tf_checkpoint(
    tf_checkpoint_path: str,
    output_path: str,
    variable_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Array]:
    """Convert TensorFlow checkpoint to JAX format.

    Useful for loading pre-trained TensorFlow models.

    Args:
        tf_checkpoint_path: Path to TF checkpoint
        output_path: Path to save converted checkpoint
        variable_mapping: Optional mapping from TF variable names to JAX names

    Returns:
        Dictionary of converted parameters
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required for checkpoint conversion")

    # Load TF checkpoint
    reader = tf.train.load_checkpoint(tf_checkpoint_path)

    # Get all variable names
    var_to_shape = reader.get_variable_to_shape_map()

    params = {}
    for var_name in var_to_shape:
        # Skip optimizer-related variables
        if 'Adam' in var_name or 'optimizer' in var_name.lower():
            continue

        # Get value
        value = reader.get_tensor(var_name)

        # Map variable name if mapping provided
        if variable_mapping is not None and var_name in variable_mapping:
            jax_name = variable_mapping[var_name]
        else:
            # Default mapping: clean up TF variable name
            jax_name = var_name.replace('/', '_').replace(':0', '')

        params[jax_name] = jnp.array(value)

    # Save converted checkpoint
    save_params(params, output_path, metadata={
        'source': tf_checkpoint_path,
        'conversion': 'tf_to_jax',
    })

    return params


# =============================================================================
# Checkpoint Analysis Utilities
# =============================================================================

def analyze_checkpoint(
    checkpoint_dir: Union[str, Path],
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze checkpoint contents.

    Args:
        checkpoint_dir: Checkpoint directory
        step: Step to analyze (None for latest)

    Returns:
        Analysis dictionary with parameter shapes and statistics
    """
    manager = CheckpointManager(CheckpointConfig(checkpoint_dir=str(checkpoint_dir)))
    train_state, train_config = manager.restore(step=step)

    analysis = {
        'step': train_state.step,
        'parameters': {},
        'total_params': 0,
    }

    for name, param in train_state.params.items():
        param_count = int(np.prod(param.shape))
        analysis['parameters'][name] = {
            'shape': list(param.shape),
            'count': param_count,
            'dtype': str(param.dtype),
            'mean': float(jnp.mean(param)),
            'std': float(jnp.std(param)),
            'min': float(jnp.min(param)),
            'max': float(jnp.max(param)),
        }
        analysis['total_params'] += param_count

    if train_config is not None:
        analysis['config'] = asdict(train_config)

    manager.close()
    return analysis


def compare_checkpoints(
    checkpoint_dir: Union[str, Path],
    step1: int,
    step2: int,
) -> Dict[str, Any]:
    """Compare two checkpoints.

    Args:
        checkpoint_dir: Checkpoint directory
        step1: First step
        step2: Second step

    Returns:
        Comparison dictionary
    """
    manager = CheckpointManager(CheckpointConfig(checkpoint_dir=str(checkpoint_dir)))

    state1, _ = manager.restore(step=step1)
    state2, _ = manager.restore(step=step2)

    comparison = {
        'steps': (step1, step2),
        'step_diff': step2 - step1,
        'parameters': {},
    }

    for name in state1.params:
        if name in state2.params:
            diff = state2.params[name] - state1.params[name]
            comparison['parameters'][name] = {
                'l2_diff': float(jnp.sqrt(jnp.sum(jnp.square(diff)))),
                'max_abs_diff': float(jnp.max(jnp.abs(diff))),
                'mean_abs_diff': float(jnp.mean(jnp.abs(diff))),
            }

    manager.close()
    return comparison
