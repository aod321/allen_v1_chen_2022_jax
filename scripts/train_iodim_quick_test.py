#!/usr/bin/env python
"""Quick IODim training test - verify the full pipeline works.

Fixed version: Uses short stimulus + recurrent-dominated dynamics
so that recurrent weight changes can affect output.
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import jax
import jax.numpy as jnp
import brainstate
import braintrace

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting quick IODim test (fixed version)...")

from src.v1_jax.data import load_billeh
from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate

# Test config - optimized for recurrent dynamics
BATCH_SIZE = 2
SEQ_LEN = 50  # Total sequence length
STIMULUS_STEPS = 10  # Steps with external input
N_EPOCHS = 5
STEPS_PER_EPOCH = 5
LEARNING_RATE = 0.01
LOSS_SCALE = 1000.0
ETRACE_DECAY = 0.95

log("Loading network...")
input_pop, network_data, bkg_weights = load_billeh(
    n_input=17400, n_neurons=51978, core_only=False,
    data_dir='/nvmessd/yinzi/GLIF_network', seed=3000,
)

n_neurons = network_data['n_nodes']
log(f"Network: {n_neurons} neurons")

log("Creating network...")
network = V1NetworkBrainstate.from_billeh(
    network_data, input_data=None, bkg_weights=None,
    dt=1.0, mode='training', precision=32,
)

# Reset BEFORE creating IODim
log("Resetting network state...")
network.reset(batch_size=BATCH_SIZE)

# Create IODim
log("Creating IODim model...")
etrace = braintrace.IODimVjpAlgorithm(network, decay_or_rank=ETRACE_DECAY)

# Compile with sample input
log("Compiling IODim graph...")
sample_input = jnp.zeros((BATCH_SIZE, n_neurons))
start = time.time()
etrace.compile_graph(sample_input)
log(f"Graph compiled in {time.time() - start:.2f}s")

# Get trainable weights
trainable_weights = network.get_trainable_weights()
if trainable_weights is None:
    log("ERROR: No trainable weights!")
    sys.exit(1)

train_states = {'weights': trainable_weights}
initial_weights = trainable_weights.value.copy()
log(f"Trainable weights: {initial_weights.shape}")
log(f"  Range: [{initial_weights.min():.2f}, {initial_weights.max():.2f}]")

# Setup optimizer
optimizer = brainstate.optim.Adam(lr=LEARNING_RATE)
optimizer.register_trainable_weights(train_states)

# Create input pattern: stimulus at beginning, then nothing
time_idx = jnp.arange(SEQ_LEN)

def create_stimulus_inputs(key):
    """Create input with stimulus at beginning, then nothing."""
    # Random stimulus to subset of neurons (more realistic)
    stimulus = jax.random.normal(key, (STIMULUS_STEPS, BATCH_SIZE, n_neurons)) * 500.0
    stimulus = jnp.maximum(stimulus, 0)  # Only positive

    # No input after stimulus period
    post_stimulus = jnp.zeros((SEQ_LEN - STIMULUS_STEPS, BATCH_SIZE, n_neurons))

    return jnp.concatenate([stimulus, post_stimulus], axis=0)

def create_target_rates(key):
    """Create target firing rates for post-stimulus period."""
    # Different target rates for different neuron populations
    target = jax.random.uniform(key, (BATCH_SIZE, n_neurons)) * 0.2
    # Expand to post-stimulus sequence
    return jnp.broadcast_to(target[None, :, :], (SEQ_LEN - STIMULUS_STEPS, BATCH_SIZE, n_neurons))

def loss_fn(outputs, targets):
    """MSE loss on firing rates during post-stimulus period."""
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
    # Use non-deprecated API
    grad_fn = brainstate.transform.grad(loss_fn_inner, train_states, return_value=True)
    grads, loss_val = grad_fn()

    # Gradient clipping
    flat_grads = [jnp.reshape(g, (-1,)) for g in grads.values()]
    all_grads = jnp.concatenate(flat_grads)
    grad_norm = jnp.linalg.norm(all_grads)
    clip_coeff = jnp.minimum(10.0 / (grad_norm + 1e-6), 1.0)
    clipped_grads = {k: clip_coeff * v for k, v in grads.items()}

    optimizer.update(clipped_grads)
    return loss_val / LOSS_SCALE, grad_norm

log("=" * 60)
log("Starting training...")
log(f"Config: SEQ_LEN={SEQ_LEN}, STIMULUS_STEPS={STIMULUS_STEPS}")
log(f"        LOSS_SCALE={LOSS_SCALE}, LR={LEARNING_RATE}")
log("=" * 60)

losses = []
for epoch in range(N_EPOCHS):
    epoch_start = time.time()
    epoch_loss = 0.0
    epoch_grad_norm = 0.0

    for step in range(STEPS_PER_EPOCH):
        key = jax.random.PRNGKey(epoch * 1000 + step)
        key1, key2 = jax.random.split(key)

        inputs = create_stimulus_inputs(key1)
        targets = create_target_rates(key2)

        step_start = time.time()
        loss, grad_norm = train_step(inputs, targets)
        jax.block_until_ready(loss)
        epoch_loss += float(loss)
        epoch_grad_norm += float(grad_norm)

        if step == 0 and epoch == 0:
            log(f"First step JIT: {time.time() - step_start:.1f}s")

    avg_loss = epoch_loss / STEPS_PER_EPOCH
    avg_grad_norm = epoch_grad_norm / STEPS_PER_EPOCH
    losses.append(avg_loss)
    epoch_time = time.time() - epoch_start

    # Weight change
    weight_change = jnp.abs(trainable_weights.value - initial_weights).mean()

    log(f"Epoch {epoch + 1:2d}/{N_EPOCHS}: loss={avg_loss:.6f}, grad_norm={avg_grad_norm:.2e}, weight_Δ={weight_change:.4f}, time={epoch_time:.1f}s")

log("=" * 60)
log("Training complete!")
log(f"Initial loss: {losses[0]:.6f}")
log(f"Final loss: {losses[-1]:.6f}")
log(f"Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
