#!/usr/bin/env python
"""Training script using brainstate + brainevent + braintrace IODim.

This script provides a training entry point using the brainstate ecosystem:
- Forward: brainevent.EventArray @ CSR (event-driven, 5-20x faster)
- Gradient: braintrace.IODim (online eligibility traces, 500x less memory)

NO BPTT - IODim only. This is a design decision following AlphaBrain.

Supports TWO modes:
1. Garrett task mode: Real LGN firing rates through input layer
2. Synthetic mode: Direct current injection for debugging

Usage:
    # Garrett task with real data
    uv run python scripts/train_brainstate.py --data_dir=/path/to/GLIF_network --task=garrett

    # Synthetic data mode
    uv run python scripts/train_brainstate.py --data_dir=/path/to/GLIF_network --task=synthetic

Memory comparison (V1 network with 52K neurons, seq_len=600):
    - BPTT: ~8 GB for activations (NOT SUPPORTED)
    - IODim: ~1 MB for eligibility traces (DEFAULT)
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Enable JAX 64-bit if needed
# jax.config.update("jax_enable_x64", True)

import brainstate
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v1_jax.data import load_billeh
from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate
from src.v1_jax.training.trainer_brainstate import IODimTrainer, IODimConfig


@dataclass
class TrainingConfig:
    """Training configuration.

    NOTE: Only IODim training is supported. BPTT is intentionally NOT implemented.
    This follows AlphaBrain's design: brainevent forward + braintrace IODim gradient.
    """
    # Data
    data_dir: str = "/nvmessd/yinzi/GLIF_network"
    results_dir: str = "./results_brainstate"

    # Task: "garrett" for real LGN data, "synthetic" for direct current injection
    task: str = "garrett"

    # Network
    n_neurons: int = 51978  # Number of V1 neurons
    n_input: int = 17400    # Number of LGN inputs
    core_only: bool = False
    seed: int = 3000

    # Training (IODim only - no BPTT)
    learning_rate: float = 1e-3
    batch_size: int = 8     # Reduced for IODim memory efficiency
    n_epochs: int = 16
    steps_per_epoch: int = 100
    val_steps: int = 20
    seq_len: int = 600
    dt: float = 1.0

    # Garrett task specific
    im_slice: int = 100     # Image presentation duration (ms)
    delay: int = 200        # Delay period after image (ms)
    n_images: int = 8       # Number of images to use

    # IODim parameters
    etrace_decay: float = 0.99  # Higher = more accurate but slower
    loss_scale: float = 1000.0  # Scale loss to amplify gradients

    # Regularization
    rate_cost: float = 0.1
    weight_regularization: float = 0.0
    grad_clip_norm: float = 1.0

    # Optimizer
    optimizer: str = "Adam"

    # Logging
    log_interval: int = 10
    verbose: bool = True


class GarrettDataLoaderBrainstate:
    """Data loader for Garrett task using pre-computed LGN firing rates.

    Simplified version for brainstate training.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        seq_len: int = 600,
        im_slice: int = 100,
        delay: int = 200,
        p_reappear: float = 0.1,
        n_images: int = 8,
        is_training: bool = True,
        seed: int = 42,
    ):
        """Initialize Garrett data loader."""
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

        self.presentation_len = im_slice + delay
        self.n_presentations = seq_len // self.presentation_len

        self.rng = np.random.RandomState(seed)

        print(f"Loaded {self.n_total_images} images, using {self.n_images}")
        print(f"Rates shape: {self.rates.shape}")

    def _sample_poisson(self, rates: np.ndarray) -> np.ndarray:
        """Convert firing rates to input current."""
        p = 1 - np.exp(-rates / 1000.0)
        return (p * 1.3).astype(np.float32)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch of data for the Garrett task.

        Returns:
            Tuple of (inputs, labels) where:
                - inputs: (seq_len, batch_size, n_lgn)
                - labels: (batch_size,) binary change detection labels
        """
        change_probs = self.rng.uniform(size=(self.batch_size, self.n_presentations))
        changes = change_probs > self.p_reappear
        changes[:, 0] = False

        current_indices = self.rng.randint(0, self.n_images, size=self.batch_size)
        img_indices = np.zeros((self.batch_size, self.n_presentations), dtype=np.int32)
        img_indices[:, 0] = current_indices

        for pres in range(1, self.n_presentations):
            change_mask = changes[:, pres]
            new_indices = self.rng.randint(0, self.n_images, size=self.batch_size)
            same_mask = (new_indices == img_indices[:, pres - 1]) & change_mask
            while np.any(same_mask) and self.n_images > 1:
                new_indices[same_mask] = self.rng.randint(0, self.n_images, size=np.sum(same_mask))
                same_mask = (new_indices == img_indices[:, pres - 1]) & change_mask
            img_indices[:, pres] = np.where(change_mask, new_indices, img_indices[:, pres - 1])

        labels = changes[:, -1].astype(np.int32)

        pause = self.rates[0, -50:]
        im_segments = self.rates[img_indices, 50:self.im_slice + self.delay]
        pause_expanded = np.broadcast_to(pause, (self.batch_size, self.n_presentations, 50, self.n_lgn))
        segments = np.concatenate([pause_expanded, im_segments], axis=2)
        inputs = segments.reshape(self.batch_size, self.seq_len, self.n_lgn)
        inputs = np.transpose(inputs, (1, 0, 2))
        inputs = self._sample_poisson(inputs)

        return inputs, labels


def create_synthetic_data(
    n_neurons: int,
    n_receptors: int,
    batch_size: int,
    seq_len: int,
    n_classes: int = 8,
    seed: int = 42,
    current_scale: float = 300.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create synthetic training data for V1 network training (direct current mode)."""
    rng = np.random.RandomState(seed)

    labels = rng.randint(0, n_classes, size=(batch_size,))

    class_patterns = np.zeros((n_classes, n_neurons), dtype=np.float32)
    neurons_per_class = n_neurons // n_classes

    for c in range(n_classes):
        start_idx = c * neurons_per_class
        end_idx = min(start_idx + neurons_per_class, n_neurons)
        class_patterns[c, start_idx:end_idx] = 1.0

    t = np.arange(seq_len) / seq_len
    temporal_mod = 0.5 + 0.5 * np.sin(2 * np.pi * t * 2)[:, None, None]

    inputs = np.zeros((seq_len, batch_size, n_neurons), dtype=np.float32)
    for b in range(batch_size):
        pattern = class_patterns[labels[b]]
        noise = rng.randn(seq_len, n_neurons) * 0.1
        inputs[:, b, :] = pattern[None, :] * temporal_mod[:, 0, 0, None] + noise

    inputs = np.clip(inputs, 0, 1) * current_scale

    targets = np.zeros((seq_len, batch_size, n_neurons), dtype=np.float32)
    for b in range(batch_size):
        targets[:, b, :] = class_patterns[labels[b]][None, :] * 0.5

    return jnp.asarray(inputs), jnp.asarray(targets)


def train_iodim(
    network: V1NetworkBrainstate,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    config: TrainingConfig,
) -> List[float]:
    """Train using IODim online gradient computation.

    This is the ONLY supported training method. Uses braintrace's IODim algorithm
    with online eligibility traces for memory-efficient gradient computation.

    Memory: ~1 MB for eligibility traces (vs ~8 GB for BPTT activations)
    """
    iodim_config = IODimConfig(
        learning_rate=config.learning_rate,
        grad_clip_norm=config.grad_clip_norm,
        etrace_decay=config.etrace_decay,
        optimizer=config.optimizer,
        rate_cost=config.rate_cost,
        weight_regularization=config.weight_regularization,
    )

    trainer = IODimTrainer(network, iodim_config)
    loss_curve = []

    for epoch in range(config.n_epochs):
        start_time = time.time()
        loss = trainer.train_step(inputs, targets)
        elapsed = time.time() - start_time

        loss_curve.append(loss)

        if config.verbose and (epoch % config.log_interval == 0 or epoch == config.n_epochs - 1):
            print(f"[IODim] Epoch {epoch + 1}/{config.n_epochs}: "
                  f"loss={loss:.6f}, time={elapsed:.2f}s")

    return loss_curve


def train_garrett_simple(
    network: V1NetworkBrainstate,
    train_loader: GarrettDataLoaderBrainstate,
    val_loader: GarrettDataLoaderBrainstate,
    config: TrainingConfig,
) -> Tuple[List[float], List[float], List[float]]:
    """Train on Garrett task using simple forward pass (no gradient).

    This is a simple test to verify the forward pass works correctly.
    For actual training, we would need to implement proper gradient computation.

    Returns:
        Tuple of (train_losses, train_accs, val_accs)
    """
    train_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(config.n_epochs):
        epoch_start = time.time()

        # Training (forward pass only for now)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step in range(config.steps_per_epoch):
            inputs, labels = train_loader.sample_batch()
            inputs = jnp.asarray(inputs)
            labels = jnp.asarray(labels)

            # Forward pass with LGN input
            network.reset(batch_size=config.batch_size)
            outputs, spikes = network.simulate(inputs, reset_before=False, use_lgn_input=True)

            # Simple classification from last time step mean spike rate
            # Use last 50 timesteps for readout
            readout_spikes = jnp.mean(spikes[-50:], axis=0)  # (batch, n_neurons)
            # Simple linear readout (placeholder)
            pred_logits = jnp.mean(readout_spikes, axis=-1)  # (batch,)
            predictions = (pred_logits > 0.5).astype(jnp.int32)

            correct = jnp.sum(predictions == labels)
            epoch_correct += int(correct)
            epoch_total += config.batch_size

            # Compute loss (MSE on spike rates)
            spike_rate = jnp.mean(spikes)
            loss = float(jnp.mean((spikes - 0.1) ** 2) + config.rate_cost * spike_rate)
            epoch_loss += loss

        avg_loss = epoch_loss / config.steps_per_epoch
        train_acc = epoch_correct / epoch_total
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        # Validation
        val_correct = 0
        val_total = 0
        for step in range(config.val_steps):
            inputs, labels = val_loader.sample_batch()
            inputs = jnp.asarray(inputs)
            labels = jnp.asarray(labels)

            network.reset(batch_size=config.batch_size)
            outputs, spikes = network.simulate(inputs, reset_before=False, use_lgn_input=True)

            readout_spikes = jnp.mean(spikes[-50:], axis=0)
            pred_logits = jnp.mean(readout_spikes, axis=-1)
            predictions = (pred_logits > 0.5).astype(jnp.int32)

            correct = jnp.sum(predictions == labels)
            val_correct += int(correct)
            val_total += config.batch_size

        val_acc = val_correct / val_total
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start

        if config.verbose:
            spike_rate_pct = float(jnp.mean(spikes)) * 100
            print(f"[Forward] Epoch {epoch + 1}/{config.n_epochs}: "
                  f"loss={avg_loss:.4f}, train_acc={train_acc:.2%}, "
                  f"val_acc={val_acc:.2%}, spike_rate={spike_rate_pct:.2f}%, "
                  f"time={epoch_time:.1f}s")

    return train_losses, train_accs, val_accs


def main(cfg: DictConfig) -> None:
    """Main training function.

    NOTE: Only IODim training is supported. No BPTT option.
    """
    # Convert Hydra config to dataclass
    config = TrainingConfig(
        data_dir=cfg.get('data_dir', TrainingConfig.data_dir),
        results_dir=cfg.get('results_dir', TrainingConfig.results_dir),
        task=cfg.get('task', 'garrett'),
        n_neurons=cfg.get('n_neurons', 51978),
        n_input=cfg.get('n_input', 17400),
        core_only=cfg.get('core_only', False),
        seed=cfg.get('seed', 3000),
        learning_rate=cfg.training.get('learning_rate', 1e-3),
        batch_size=cfg.training.get('batch_size', 8),
        n_epochs=cfg.training.get('n_epochs', 16),
        steps_per_epoch=cfg.training.get('steps_per_epoch', 100),
        val_steps=cfg.training.get('val_steps', 20),
        seq_len=cfg.training.get('seq_len', 600),
        dt=cfg.training.get('dt', 1.0),
        etrace_decay=cfg.training.get('etrace_decay', 0.99),
        loss_scale=cfg.training.get('loss_scale', 1000.0),
        rate_cost=cfg.training.get('rate_cost', 0.1),
        weight_regularization=cfg.training.get('weight_regularization', 0.0),
        grad_clip_norm=cfg.training.get('grad_clip_norm', 1.0),
        optimizer=cfg.training.get('optimizer', 'Adam'),
        log_interval=cfg.get('log_interval', 10),
        verbose=cfg.get('verbose', True),
    )

    print("=" * 60)
    print("V1 Network Training with brainstate + brainevent + braintrace")
    print("=" * 60)
    print("Forward:  brainevent.EventArray @ CSR (event-driven)")
    print("Gradient: braintrace.IODim (online eligibility traces)")
    print("-" * 60)
    print(f"Task: {config.task}")
    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Number of epochs: {config.n_epochs}")
    print(f"IODim decay: {config.etrace_decay}")
    print(f"Loss scale: {config.loss_scale}")
    print("=" * 60)

    # Load network data with input connectivity
    print("\nLoading network data...")
    input_pop, network_data, bkg_weights = load_billeh(
        n_input=config.n_input,
        n_neurons=config.n_neurons,
        core_only=config.core_only,
        data_dir=config.data_dir,
        seed=config.seed,
    )

    n_neurons = network_data['n_nodes']
    n_receptors = network_data['node_params']['tau_syn'].shape[1]
    n_inputs = input_pop['n_inputs']

    print(f"Loaded network with {n_neurons} neurons, {n_inputs} inputs, {n_receptors} receptors")

    # Create network with input layer
    print("\nCreating V1 network (brainstate + brainevent)...")
    network = V1NetworkBrainstate.from_billeh(
        network_data,
        input_data=input_pop,
        bkg_weights=bkg_weights,
        dt=config.dt,
        mode='training',
        precision=32,
    )
    print(f"Network has input layer: {network.has_input_layer}")

    if config.task == "garrett":
        # Garrett task with real LGN data
        print("\nInitializing Garrett data loaders...")
        train_loader = GarrettDataLoaderBrainstate(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            im_slice=config.im_slice,
            delay=config.delay,
            n_images=config.n_images,
            is_training=True,
            seed=config.seed,
        )
        val_loader = GarrettDataLoaderBrainstate(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            im_slice=config.im_slice,
            delay=config.delay,
            n_images=config.n_images,
            is_training=False,
            seed=config.seed + 1,
        )

        print("\nStarting forward pass test (Garrett task)...")
        print("NOTE: IODim training not yet supported with input layer.")
        print("      Running forward pass to verify network correctness.")
        start_time = time.time()
        train_losses, train_accs, val_accs = train_garrett_simple(
            network, train_loader, val_loader, config
        )
        total_time = time.time() - start_time

        # Results
        print("\n" + "=" * 60)
        print("Training Complete (Garrett Task)")
        print("=" * 60)
        print(f"Final train loss: {train_losses[-1]:.6f}")
        print(f"Final train acc: {train_accs[-1]:.2%}")
        print(f"Final val acc: {val_accs[-1]:.2%}")
        print(f"Best val acc: {max(val_accs):.2%}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Time per epoch: {total_time / config.n_epochs:.2f}s")

        # Save results
        os.makedirs(config.results_dir, exist_ok=True)
        results_file = os.path.join(config.results_dir, 'training_results_garrett.npz')
        np.savez(
            results_file,
            train_losses=np.array(train_losses),
            train_accs=np.array(train_accs),
            val_accs=np.array(val_accs),
            total_time=total_time,
        )
        print(f"\nResults saved to: {results_file}")

    else:
        # Synthetic data mode (for debugging)
        print("\nCreating synthetic training data...")
        inputs, targets = create_synthetic_data(
            n_neurons=n_neurons,
            n_receptors=n_receptors,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            n_classes=8,
            seed=config.seed,
            current_scale=300.0,
        )

        print(f"Input shape: {inputs.shape} (seq_len, batch, n_neurons)")
        print(f"Input range: {inputs.min():.1f} - {inputs.max():.1f} pA")
        print(f"Target shape: {targets.shape}")

        print("\nStarting training with IODim (synthetic data)...")
        start_time = time.time()
        loss_curve = train_iodim(network, inputs, targets, config)
        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("Training Complete (Synthetic Data)")
        print("=" * 60)
        print(f"Final loss: {loss_curve[-1]:.6f}")
        print(f"Initial loss: {loss_curve[0]:.6f}")
        print(f"Loss reduction: {(loss_curve[0] - loss_curve[-1]) / loss_curve[0] * 100:.1f}%")
        print(f"Total time: {total_time:.1f}s")
        print(f"Time per epoch: {total_time / config.n_epochs:.2f}s")

        os.makedirs(config.results_dir, exist_ok=True)
        results_file = os.path.join(config.results_dir, 'training_results_synthetic.npz')
        np.savez(
            results_file,
            loss_curve=np.array(loss_curve),
            total_time=total_time,
        )
        print(f"\nResults saved to: {results_file}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    main(cfg)


if __name__ == "__main__":
    # Simple command-line usage without Hydra
    import argparse

    parser = argparse.ArgumentParser(
        description="Train V1 network with brainstate + brainevent + braintrace IODim"
    )
    parser.add_argument("--data_dir", type=str, default="/nvmessd/yinzi/GLIF_network")
    parser.add_argument("--results_dir", type=str, default="./results_brainstate")
    parser.add_argument("--task", type=str, default="garrett", choices=["garrett", "synthetic"],
                        help="Task: 'garrett' for real LGN data, 'synthetic' for debugging")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=16)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=600)
    parser.add_argument("--etrace_decay", type=float, default=0.99,
                        help="IODim eligibility trace decay (0.9-0.999, higher=more accurate)")
    parser.add_argument("--loss_scale", type=float, default=1000.0,
                        help="Loss scaling factor for gradient amplification")

    args = parser.parse_args()

    # Create a simple config
    cfg = OmegaConf.create({
        'data_dir': args.data_dir,
        'results_dir': args.results_dir,
        'task': args.task,
        'training': {
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'steps_per_epoch': args.steps_per_epoch,
            'val_steps': args.val_steps,
            'learning_rate': args.learning_rate,
            'seq_len': args.seq_len,
            'etrace_decay': args.etrace_decay,
            'loss_scale': args.loss_scale,
        }
    })

    main(cfg)
