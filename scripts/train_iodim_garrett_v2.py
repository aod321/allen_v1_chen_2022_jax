#!/usr/bin/env python
"""IODim training with Garrett data - Efficient Version.

Pre-computes ALL LGN → V1 transformations at startup for maximum efficiency.
"""

from __future__ import annotations

import os
import sys
import time
import pickle

os.environ['PYTHONUNBUFFERED'] = '1'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting IODim Garrett training (v2 - efficient)...")

import jax
import jax.numpy as jnp
import numpy as np

log(f"JAX devices: {jax.devices()}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v1_jax.data import load_billeh
from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate
from src.v1_jax.training.trainer_brainstate import IODimTrainer, IODimConfig

# Configuration
DATA_DIR = "/nvmessd/yinzi/GLIF_network"
BATCH_SIZE = 8
SEQ_LEN = 300  # Shorter sequence for faster training
N_EPOCHS = 8
STEPS_PER_EPOCH = 30
VAL_STEPS = 5
N_IMAGES = 8  # Number of images to use
LEARNING_RATE = 1e-3
ETRACE_DECAY = 0.95

log(f"Config: batch={BATCH_SIZE}, seq_len={SEQ_LEN}, epochs={N_EPOCHS}")


def precompute_lgn_to_v1_all_images(
    lgn_rates: np.ndarray,
    input_csr_matrices: dict,
    bkg_current: np.ndarray,
    n_neurons: int,
) -> np.ndarray:
    """Pre-compute V1 input current for ALL LGN images.

    Args:
        lgn_rates: (n_images, T_max, n_lgn) LGN firing rates
        input_csr_matrices: dict of (delay, receptor) -> CSR
        bkg_current: (n_neurons,) background current
        n_neurons: number of V1 neurons

    Returns:
        v1_currents: (n_images, T_max, n_neurons) pre-computed V1 currents
    """
    n_images, T_max, n_lgn = lgn_rates.shape

    log(f"  Pre-computing LGN → V1 for {n_images} images, T={T_max}...")

    # Start with background
    v1_currents = np.zeros((n_images, T_max, n_neurons), dtype=np.float32)
    v1_currents[:] = bkg_current[None, None, :]

    # Add LGN contributions (vectorized over time, loop over images and CSR)
    start = time.time()
    for img_idx in range(n_images):
        for key, csr in input_csr_matrices.items():
            delay, receptor = key
            # CSR is (n_neurons, n_lgn), lgn is (T, n_lgn)
            # Result is (T, n_neurons)
            for t in range(T_max):
                contrib = csr @ lgn_rates[img_idx, t]
                v1_currents[img_idx, t] += np.asarray(contrib)

        if (img_idx + 1) % 10 == 0:
            log(f"    Processed {img_idx + 1}/{n_images} images...")

    log(f"  Pre-computation done in {time.time() - start:.1f}s")
    return v1_currents


def create_batch_from_precomputed(
    v1_currents: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.RandomState,
):
    """Create a training batch from pre-computed V1 currents.

    Simplified Garrett task: just random image sequences.
    """
    n_images = v1_currents.shape[0]

    # Random image indices for each batch element and time segment
    presentation_len = 150  # 100ms image + 50ms delay
    n_presentations = seq_len // presentation_len

    batch_inputs = np.zeros((seq_len, batch_size, v1_currents.shape[2]), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int32)

    for b in range(batch_size):
        current_idx = rng.randint(0, n_images)
        for pres in range(n_presentations):
            # Sample whether to change image
            if pres > 0 and rng.random() > 0.1:  # 90% chance to change
                new_idx = rng.randint(0, n_images)
                while new_idx == current_idx and n_images > 1:
                    new_idx = rng.randint(0, n_images)
                current_idx = new_idx
                if pres == n_presentations - 1:
                    labels[b] = 1  # Change on last presentation

            # Fill time segment with pre-computed V1 current
            t_start = pres * presentation_len
            t_end = min(t_start + presentation_len, seq_len)
            seg_len = t_end - t_start

            # Use pre-computed current (with some time offset for variety)
            t_offset = rng.randint(0, max(1, v1_currents.shape[1] - seg_len))
            batch_inputs[t_start:t_end, b, :] = v1_currents[current_idx, t_offset:t_offset + seg_len, :]

    # Create simple target: higher firing for change detection
    targets = np.zeros_like(batch_inputs)
    for b in range(batch_size):
        if labels[b] == 1:
            targets[-50:, b, :1000] = 0.1
        else:
            targets[-50:, b, :1000] = 0.05

    return batch_inputs, targets, labels


def main():
    # Load network
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

    # Create network (without input layer - we pre-compute)
    log("Creating V1 network...")
    network = V1NetworkBrainstate.from_billeh(
        network_data,
        input_data=None,
        bkg_weights=None,
        dt=1.0,
        mode='training',
        precision=32,
    )

    # Build input CSR matrices
    log("Building input CSR matrices...")
    from src.v1_jax.nn.connectivity_brainstate import build_connection_from_billeh

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

    input_csr_matrices = conn.input_csr_matrices if conn.input_csr_matrices else {}
    log(f"  Input CSR groups: {len(input_csr_matrices)}")

    # Process background current
    if bkg_weights is not None:
        voltage_scale = network_data['node_params']['V_th'] - network_data['node_params']['E_L']
        node_type_ids = network_data.get('node_type_ids', np.zeros(n_neurons, dtype=np.int32))
        bkg_scaled = bkg_weights / np.repeat(voltage_scale[node_type_ids], n_receptors)
        bkg_scaled = bkg_scaled * 10.0
        bkg_current = np.sum(bkg_scaled.reshape(n_neurons, n_receptors), axis=1)
    else:
        bkg_current = np.ones(n_neurons, dtype=np.float32)

    # Load LGN data
    log("Loading LGN firing rates...")
    stim_path = os.path.join(DATA_DIR, '..', 'many_small_stimuli.pkl')
    with open(stim_path, 'rb') as f:
        data = pickle.load(f)
    lgn_rates = np.stack(list(data.values()), axis=0)[:N_IMAGES].astype(np.float32)
    log(f"  LGN rates shape: {lgn_rates.shape}")

    # Convert to firing rate scale
    p = 1 - np.exp(-lgn_rates / 1000.0)
    lgn_rates = (p * 1.3).astype(np.float32)

    # Pre-compute ALL LGN → V1 transformations
    log("Pre-computing LGN → V1 transformations...")
    v1_currents = precompute_lgn_to_v1_all_images(
        lgn_rates, input_csr_matrices, bkg_current, n_neurons
    )
    log(f"  V1 currents shape: {v1_currents.shape}")

    # Setup IODim trainer
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

    # Training
    log("=" * 60)
    log("Starting IODim training...")
    log("=" * 60)

    rng = np.random.RandomState(42)
    train_losses = []

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            inputs, targets, labels = create_batch_from_precomputed(
                v1_currents, BATCH_SIZE, SEQ_LEN, rng
            )
            inputs_jax = jnp.asarray(inputs)
            targets_jax = jnp.asarray(targets)

            step_start = time.time()
            loss = trainer.train_step(inputs_jax, targets_jax)
            epoch_loss += loss

            if step == 0 and epoch == 0:
                log(f"First step JIT: {time.time() - step_start:.1f}s")

        avg_loss = epoch_loss / STEPS_PER_EPOCH
        train_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        log(f"[IODim] Epoch {epoch + 1}/{N_EPOCHS}: loss={avg_loss:.6f}, time={epoch_time:.1f}s")

    # Results
    log("=" * 60)
    log("Training Complete!")
    log(f"Initial loss: {train_losses[0]:.6f}")
    log(f"Final loss: {train_losses[-1]:.6f}")
    reduction = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    log(f"Loss reduction: {reduction:.1f}%")

    # Save
    os.makedirs("./results_brainstate", exist_ok=True)
    np.savez(
        "./results_brainstate/iodim_garrett_v2.npz",
        train_losses=np.array(train_losses),
    )
    log("Results saved!")


if __name__ == "__main__":
    main()
