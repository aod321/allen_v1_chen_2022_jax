#!/usr/bin/env python
"""IODim training with Garrett data (LGN pre-computed transformation).

This script runs IODim training using real Garrett data by pre-computing
the LGN → V1 transformation outside the IODim graph.

Strategy:
1. Load Garrett LGN firing rates
2. Pre-compute LGN → V1 current transformation using input CSR matrices
3. Run IODim training with pre-computed V1 current as input

This separates the input layer from the IODim differentiable graph,
which is a valid approach for training.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from typing import Tuple, Optional

os.environ['PYTHONUNBUFFERED'] = '1'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting IODim Garrett training...")
log("Importing libraries...")

import jax
import jax.numpy as jnp
import numpy as np

import brainstate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v1_jax.data import load_billeh
from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate
from src.v1_jax.training.trainer_brainstate import IODimTrainer, IODimConfig

log(f"JAX devices: {jax.devices()}")

# Configuration
DATA_DIR = "/nvmessd/yinzi/GLIF_network"
RESULTS_DIR = "./results_brainstate"
BATCH_SIZE = 8
SEQ_LEN = 600
N_EPOCHS = 8
STEPS_PER_EPOCH = 50
VAL_STEPS = 10
LEARNING_RATE = 1e-3
ETRACE_DECAY = 0.95  # Lower decay for faster learning
LOSS_SCALE = 100.0   # Scale loss for better gradients

log(f"Config: batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, epochs={N_EPOCHS}")
log(f"        steps/epoch={STEPS_PER_EPOCH}, val_steps={VAL_STEPS}")
log(f"        lr={LEARNING_RATE}, etrace_decay={ETRACE_DECAY}")


class GarrettDataWithV1Current:
    """Data loader that pre-computes LGN → V1 transformation.

    This separates the input layer from the IODim graph.
    """

    def __init__(
        self,
        data_dir: str,
        input_csr_matrices: dict,
        bkg_weights: jnp.ndarray,
        n_neurons: int,
        n_receptors: int,
        batch_size: int,
        seq_len: int = 600,
        im_slice: int = 100,
        delay: int = 200,
        n_images: int = 8,
        is_training: bool = True,
        seed: int = 42,
    ):
        # Load LGN firing rates
        if is_training:
            stim_path = os.path.join(data_dir, '..', 'many_small_stimuli.pkl')
        else:
            stim_path = os.path.join(data_dir, '..', 'alternate_small_stimuli.pkl')

        with open(stim_path, 'rb') as f:
            data = pickle.load(f)

        self.rates = np.stack(list(data.values()), axis=0).astype(np.float32)
        self.n_total_images = self.rates.shape[0]
        self.n_images = min(n_images, self.n_total_images)
        self.n_lgn = self.rates.shape[-1]

        self.input_csr_matrices = input_csr_matrices
        self.bkg_weights = bkg_weights
        self.n_neurons = n_neurons
        self.n_receptors = n_receptors

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.im_slice = im_slice
        self.delay = delay

        self.presentation_len = im_slice + delay
        self.n_presentations = seq_len // self.presentation_len

        self.rng = np.random.RandomState(seed)

        log(f"  Loaded {self.n_total_images} images, using {self.n_images}")
        log(f"  n_lgn={self.n_lgn}, n_neurons={n_neurons}")

    def _sample_poisson(self, rates: np.ndarray) -> np.ndarray:
        p = 1 - np.exp(-rates / 1000.0)
        return (p * 1.3).astype(np.float32)

    def _transform_lgn_to_v1(self, lgn_input: np.ndarray) -> np.ndarray:
        """Pre-compute LGN → V1 transformation.

        Args:
            lgn_input: (seq_len, batch, n_lgn) LGN firing rates

        Returns:
            v1_current: (seq_len, batch, n_neurons) V1 input current
        """
        seq_len, batch_size, _ = lgn_input.shape

        # Start with background current
        bkg = self.bkg_weights.reshape(self.n_neurons, self.n_receptors)
        bkg_sum = np.sum(bkg, axis=1)  # (n_neurons,)

        v1_current = np.zeros((seq_len, batch_size, self.n_neurons), dtype=np.float32)

        # Add background
        v1_current[:] = bkg_sum[None, None, :]

        # Add LGN contribution through input CSR (simplified - no delay)
        # For IODim training, we aggregate all receptor contributions
        for key, csr in self.input_csr_matrices.items():
            delay, receptor = key
            # CSR shape: (n_neurons, n_lgn)
            # For each timestep and batch, compute CSR @ lgn
            for t in range(seq_len):
                for b in range(batch_size):
                    contrib = csr @ lgn_input[t, b]  # (n_neurons,)
                    v1_current[t, b] += np.asarray(contrib)

        return v1_current

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch with pre-computed V1 current.

        Returns:
            v1_current: (seq_len, batch, n_neurons) V1 input current
            targets: (seq_len, batch, n_neurons) target spike patterns
            labels: (batch,) classification labels
        """
        # Sample image sequence
        change_probs = self.rng.uniform(size=(self.batch_size, self.n_presentations))
        changes = change_probs > 0.1  # p_reappear
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

        # Build LGN input sequence
        pause = self.rates[0, -50:]
        im_segments = self.rates[img_indices, 50:self.im_slice + self.delay]
        pause_expanded = np.broadcast_to(pause, (self.batch_size, self.n_presentations, 50, self.n_lgn))
        segments = np.concatenate([pause_expanded, im_segments], axis=2)
        lgn_input = segments.reshape(self.batch_size, self.seq_len, self.n_lgn)
        lgn_input = np.transpose(lgn_input, (1, 0, 2))  # (seq_len, batch, n_lgn)
        lgn_input = self._sample_poisson(lgn_input)

        # Pre-compute LGN → V1 transformation
        v1_current = self._transform_lgn_to_v1(lgn_input)

        # Create target spike patterns (simple: based on last image)
        # For change detection: target higher activity for "change" trials
        targets = np.zeros((self.seq_len, self.batch_size, self.n_neurons), dtype=np.float32)
        for b in range(self.batch_size):
            if labels[b] == 1:  # Change trial
                targets[-100:, b, :1000] = 0.1  # Higher target rate for first 1000 neurons
            else:
                targets[-100:, b, :1000] = 0.05

        return v1_current, targets, labels


def train_iodim_garrett():
    """Run IODim training with Garrett data."""

    log("Loading network data...")
    input_pop, network_data, bkg_weights = load_billeh(
        n_input=17400,
        n_neurons=51978,
        core_only=False,
        data_dir=DATA_DIR,
        seed=3000,
    )

    n_neurons = network_data['n_nodes']
    n_receptors = network_data['node_params']['tau_syn'].shape[1]

    log(f"  n_neurons={n_neurons}, n_receptors={n_receptors}")

    log("Creating V1 network (without input layer for IODim)...")
    # Create network WITHOUT input layer for IODim compatibility
    network = V1NetworkBrainstate.from_billeh(
        network_data,
        input_data=None,  # No input layer - we pre-compute transformation
        bkg_weights=None,
        dt=1.0,
        mode='training',
        precision=32,
    )
    log(f"  has_input_layer={network.has_input_layer}")

    # Build input CSR matrices for pre-computation
    from src.v1_jax.nn.connectivity_brainstate import build_connection_from_billeh

    log("Building input CSR matrices for LGN → V1 transformation...")
    conn = build_connection_from_billeh(
        synapses=network_data['synapses'],
        n_neurons=n_neurons,
        n_receptors=n_receptors,
        dt=1.0,
        include_input=True,
        input_data=input_pop,
        n_inputs=input_pop['n_inputs'],
        dtype=jnp.float32,
    )

    input_csr_matrices = {}
    if conn.input_csr_matrices:
        for key, csr in conn.input_csr_matrices.items():
            # Convert JAX CSR to numpy for pre-computation
            input_csr_matrices[key] = csr

    log(f"  Input CSR groups: {len(input_csr_matrices)}")

    # Process background weights
    if bkg_weights is not None:
        voltage_scale_types = network_data['node_params']['V_th'] - network_data['node_params']['E_L']
        node_type_ids = network_data.get('node_type_ids', np.zeros(n_neurons, dtype=np.int32))
        bkg_weights_scaled = bkg_weights / np.repeat(voltage_scale_types[node_type_ids], n_receptors)
        bkg_weights_scaled = bkg_weights_scaled * 10.0
        bkg_arr = np.asarray(bkg_weights_scaled, dtype=np.float32)
    else:
        bkg_arr = np.ones(n_neurons * n_receptors, dtype=np.float32)

    log("Creating data loaders...")
    train_loader = GarrettDataWithV1Current(
        data_dir=DATA_DIR,
        input_csr_matrices=input_csr_matrices,
        bkg_weights=bkg_arr,
        n_neurons=n_neurons,
        n_receptors=n_receptors,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        is_training=True,
        seed=42,
    )

    val_loader = GarrettDataWithV1Current(
        data_dir=DATA_DIR,
        input_csr_matrices=input_csr_matrices,
        bkg_weights=bkg_arr,
        n_neurons=n_neurons,
        n_receptors=n_receptors,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        is_training=False,
        seed=43,
    )

    log("Setting up IODim trainer...")
    iodim_config = IODimConfig(
        learning_rate=LEARNING_RATE,
        grad_clip_norm=1.0,
        etrace_decay=ETRACE_DECAY,
        optimizer="Adam",
        rate_cost=0.1,
        weight_regularization=0.0,
    )

    trainer = IODimTrainer(network, iodim_config)

    log("Starting IODim training...")
    log("=" * 60)

    train_losses = []
    val_losses = []

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()

        # Training
        epoch_loss = 0.0
        for step in range(STEPS_PER_EPOCH):
            inputs, targets, labels = train_loader.sample_batch()
            inputs_jax = jnp.asarray(inputs)
            targets_jax = jnp.asarray(targets)

            loss = trainer.train_step(inputs_jax, targets_jax)
            epoch_loss += loss

            if step == 0 and epoch == 0:
                log(f"First step completed (JIT compilation)")

        avg_train_loss = epoch_loss / STEPS_PER_EPOCH
        train_losses.append(avg_train_loss)

        # Validation
        val_loss = 0.0
        for step in range(VAL_STEPS):
            inputs, targets, labels = val_loader.sample_batch()
            inputs_jax = jnp.asarray(inputs)
            targets_jax = jnp.asarray(targets)

            # Forward pass only for validation
            network.reset(batch_size=BATCH_SIZE)
            outputs, spikes = network.simulate(inputs_jax, reset_before=False)

            # Compute validation loss (MSE on spike patterns)
            spike_loss = float(jnp.mean((spikes - targets_jax) ** 2))
            rate_loss = float(jnp.mean(spikes) * 0.1)
            val_loss += spike_loss + rate_loss

        avg_val_loss = val_loss / VAL_STEPS
        val_losses.append(avg_val_loss)

        epoch_time = time.time() - epoch_start

        log(f"[IODim] Epoch {epoch + 1}/{N_EPOCHS}: "
            f"train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, "
            f"time={epoch_time:.1f}s")

    log("=" * 60)
    log("Training Complete!")
    log(f"Final train loss: {train_losses[-1]:.6f}")
    log(f"Final val loss: {val_losses[-1]:.6f}")
    log(f"Loss reduction: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, 'iodim_garrett_results.npz')
    np.savez(
        results_file,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )
    log(f"Results saved to: {results_file}")

    return train_losses, val_losses


if __name__ == "__main__":
    train_iodim_garrett()
