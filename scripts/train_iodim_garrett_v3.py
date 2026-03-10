#!/usr/bin/env python
"""IODim training with Garrett data - Optimized Version.

Key improvements:
1. Higher learning rate (0.01 vs 0.001)
2. Short stimulus + recurrent-dominated period
3. Loss computed on post-stimulus activity only
"""

from __future__ import annotations

import os
import sys
import time
import pickle

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting IODim Garrett training (v3 - optimized)...")

import jax
import jax.numpy as jnp
import numpy as np

log(f"JAX devices: {jax.devices()}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brainstate
import braintrace

from src.v1_jax.data import load_billeh
from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate

# Optimized Configuration
DATA_DIR = "/nvmessd/yinzi/GLIF_network"
BATCH_SIZE = 4
SEQ_LEN = 200  # Total sequence length
STIMULUS_STEPS = 50  # First 50ms with LGN input
N_EPOCHS = 10
STEPS_PER_EPOCH = 20
N_IMAGES = 8
LEARNING_RATE = 0.01  # Higher LR
LOSS_SCALE = 1000.0
ETRACE_DECAY = 0.95

log(f"Config: batch={BATCH_SIZE}, seq_len={SEQ_LEN}, stimulus_steps={STIMULUS_STEPS}")
log(f"        lr={LEARNING_RATE}, loss_scale={LOSS_SCALE}, epochs={N_EPOCHS}")


def precompute_lgn_to_v1(
    lgn_rates: np.ndarray,
    input_csr_matrices: dict,
    bkg_current: np.ndarray,
    n_neurons: int,
) -> np.ndarray:
    """Pre-compute V1 input current for LGN images."""
    n_images, T_max, n_lgn = lgn_rates.shape

    log(f"  Pre-computing LGN → V1 for {n_images} images, T={T_max}...")

    v1_currents = np.zeros((n_images, T_max, n_neurons), dtype=np.float32)
    v1_currents[:] = bkg_current[None, None, :]

    start = time.time()
    for img_idx in range(n_images):
        for key, csr in input_csr_matrices.items():
            delay, receptor = key
            for t in range(T_max):
                contrib = csr @ lgn_rates[img_idx, t]
                v1_currents[img_idx, t] += np.asarray(contrib)

    log(f"  Pre-computation done in {time.time() - start:.1f}s")
    return v1_currents


def create_training_batch(
    v1_currents: np.ndarray,
    batch_size: int,
    seq_len: int,
    stimulus_steps: int,
    rng: np.random.RandomState,
):
    """Create training batch with stimulus + post-stimulus structure.

    First stimulus_steps: LGN-driven input
    Remaining steps: No external input (recurrent-dominated)
    """
    n_images, T_max, n_neurons = v1_currents.shape

    batch_inputs = np.zeros((seq_len, batch_size, n_neurons), dtype=np.float32)
    labels = rng.randint(0, 2, size=batch_size)  # Binary classification

    for b in range(batch_size):
        # Pick random image
        img_idx = rng.randint(0, n_images)
        t_offset = rng.randint(0, max(1, T_max - stimulus_steps))

        # Fill stimulus period with LGN input
        batch_inputs[:stimulus_steps, b, :] = v1_currents[
            img_idx, t_offset:t_offset + stimulus_steps, :
        ]

        # Post-stimulus period: no external input (zeros)
        # This lets recurrent dynamics dominate

    # Create targets based on labels
    # Target: specific firing pattern in subset of neurons during post-stimulus
    targets = np.zeros((seq_len - stimulus_steps, batch_size, n_neurons), dtype=np.float32)
    for b in range(batch_size):
        # Different target rate based on label
        if labels[b] == 1:
            targets[:, b, :5000] = 0.15  # Higher rate for "change" class
        else:
            targets[:, b, :5000] = 0.05  # Lower rate for "no change" class

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

    # Create network
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

    # Background current
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

    # Pre-compute LGN → V1
    log("Pre-computing LGN → V1 transformations...")
    v1_currents = precompute_lgn_to_v1(
        lgn_rates, input_csr_matrices, bkg_current, n_neurons
    )
    log(f"  V1 currents shape: {v1_currents.shape}")

    # Setup training
    log("Setting up IODim training...")

    brainstate.environ.set(dt=1.0)
    network.reset(batch_size=BATCH_SIZE)

    etrace = braintrace.IODimVjpAlgorithm(network, decay_or_rank=ETRACE_DECAY)
    sample_input = jnp.zeros((BATCH_SIZE, n_neurons))
    etrace.compile_graph(sample_input)

    trainable_weights = network.get_trainable_weights()
    initial_weights = trainable_weights.value.copy()
    train_states = {'weights': trainable_weights}

    optimizer = brainstate.optim.Adam(lr=LEARNING_RATE)
    optimizer.register_trainable_weights(train_states)

    time_idx = jnp.arange(SEQ_LEN)

    def loss_fn(outputs, targets):
        """MSE on post-stimulus activity."""
        post_stim_outputs = outputs[STIMULUS_STEPS:]
        return jnp.mean((post_stim_outputs - targets) ** 2)

    def make_loss_fn(inputs, targets):
        def loss_fn_inner():
            network.reset(batch_size=BATCH_SIZE)
            etrace.reset_state()

            def step_fn(i):
                return etrace(inputs[i])

            outputs = brainstate.compile.for_loop(step_fn, time_idx, pbar=None)
            return loss_fn(outputs, targets) * LOSS_SCALE
        return loss_fn_inner

    @brainstate.compile.jit
    def train_step(inputs, targets):
        loss_fn_inner = make_loss_fn(inputs, targets)
        grad_fn = brainstate.transform.grad(loss_fn_inner, train_states, return_value=True)
        grads, loss_val = grad_fn()

        # Gradient clipping - more aggressive
        flat_grads = [jnp.reshape(g, (-1,)) for g in grads.values()]
        all_grads = jnp.concatenate(flat_grads)
        grad_norm = jnp.linalg.norm(all_grads)
        clip_coeff = jnp.minimum(1.0 / (grad_norm + 1e-6), 1.0)  # Clip to norm 1.0
        clipped_grads = {k: clip_coeff * v for k, v in grads.items()}

        optimizer.update(clipped_grads)
        return loss_val / LOSS_SCALE, grad_norm

    # Training
    log("=" * 60)
    log("Starting IODim training (optimized)...")
    log("=" * 60)

    rng = np.random.RandomState(42)
    train_losses = []

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        for step in range(STEPS_PER_EPOCH):
            inputs, targets, labels = create_training_batch(
                v1_currents, BATCH_SIZE, SEQ_LEN, STIMULUS_STEPS, rng
            )
            inputs_jax = jnp.asarray(inputs)
            targets_jax = jnp.asarray(targets)

            step_start = time.time()
            loss, grad_norm = train_step(inputs_jax, targets_jax)
            jax.block_until_ready(loss)
            epoch_loss += float(loss)
            epoch_grad_norm += float(grad_norm)

            if step == 0 and epoch == 0:
                log(f"First step JIT: {time.time() - step_start:.1f}s")

        avg_loss = epoch_loss / STEPS_PER_EPOCH
        avg_grad_norm = epoch_grad_norm / STEPS_PER_EPOCH
        train_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        weight_change = float(jnp.abs(trainable_weights.value - initial_weights).mean())

        log(f"[IODim] Epoch {epoch + 1}/{N_EPOCHS}: loss={avg_loss:.6f}, "
            f"grad_norm={avg_grad_norm:.2e}, weight_Δ={weight_change:.4f}, time={epoch_time:.1f}s")

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
        "./results_brainstate/iodim_garrett_v3.npz",
        train_losses=np.array(train_losses),
    )
    log("Results saved!")


if __name__ == "__main__":
    main()
