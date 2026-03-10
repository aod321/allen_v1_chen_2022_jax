#!/usr/bin/env python
"""Fixed IODim training test - proper recurrent dynamics.

Key fix: Use short initial stimulus + let recurrent dynamics dominate,
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

log("Starting fixed IODim test...")

# Small network for testing
n_neurons = 100
n_receptors = 4
n_types = 5

rng = np.random.RandomState(42)

V_th_types = np.ones(n_types) * -46.0 + rng.randn(n_types) * 2
E_L_types = np.ones(n_types) * -68.0 + rng.randn(n_types) * 2
g_types = np.ones(n_types) * 2.5 + rng.rand(n_types) * 0.5
C_m_types = np.ones(n_types) * 36.0 + rng.rand(n_types) * 5
V_reset_types = E_L_types.copy()
t_ref_types = np.ones(n_types) * 3.0
asc_decay_types = np.stack([np.ones(n_types) * 0.003, np.ones(n_types) * 0.1], axis=-1)
asc_amps_types = np.stack([np.ones(n_types) * -1.7, np.ones(n_types) * -62.0], axis=-1)
tau_syn_types = np.tile([5.5, 8.5, 2.8, 5.8], (n_types, 1))
node_type_ids = rng.randint(0, n_types, size=n_neurons)

# Dense all-to-all connectivity with LARGER weights
n_connections = n_neurons * n_neurons
source_ids = np.repeat(np.arange(n_neurons), n_neurons)
target_ids = np.tile(np.arange(n_neurons), n_neurons)
receptor_types = rng.randint(0, n_receptors, size=n_connections)  # Random receptors
delays = rng.rand(n_connections) * 3 + 2  # 2-5ms delays

target_indices = target_ids * n_receptors + receptor_types
indices = np.stack([target_indices, source_ids], axis=-1).astype(np.int32)

# LARGER weights so recurrent has significant effect
weights = rng.randn(n_connections).astype(np.float32) * 50.0  # Much larger!

network_data = {
    'n_nodes': n_neurons,
    'node_params': {
        'V_th': V_th_types, 'g': g_types, 'E_L': E_L_types,
        'C_m': C_m_types, 'V_reset': V_reset_types, 't_ref': t_ref_types,
        'k': asc_decay_types, 'asc_amps': asc_amps_types, 'tau_syn': tau_syn_types,
    },
    'node_type_ids': node_type_ids,
    'synapses': {
        'indices': indices, 'weights': weights, 'delays': delays,
        'dense_shape': (n_neurons * n_receptors, n_neurons),
    },
}

log(f"Network: {n_neurons} neurons, {n_connections} connections")

brainstate.environ.set(dt=1.0)

from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate

network = V1NetworkBrainstate.from_billeh(
    network_data, dt=1.0, mode='training', precision=32,
)

# Training config
BATCH_SIZE = 2
SEQ_LEN = 30  # Longer sequence for recurrent dynamics
N_EPOCHS = 10
STEPS_PER_EPOCH = 10
LEARNING_RATE = 0.1
LOSS_SCALE = 1000.0
ETRACE_DECAY = 0.95
STIMULUS_STEPS = 5  # Only first N steps have external input

log("Resetting network...")
network.reset(batch_size=BATCH_SIZE)

log("Creating IODim model...")
etrace = braintrace.IODimVjpAlgorithm(network, decay_or_rank=ETRACE_DECAY)

log("Compiling IODim graph...")
sample_input = jnp.zeros((BATCH_SIZE, n_neurons))
start = time.time()
etrace.compile_graph(sample_input)
log(f"Graph compiled in {time.time() - start:.2f}s")

# Get trainable weights
trainable_weights = network.get_trainable_weights()
initial_weights = trainable_weights.value.copy()
log(f"Trainable weights: {initial_weights.shape}")
log(f"  Range: [{initial_weights.min():.2f}, {initial_weights.max():.2f}]")

train_states = {'weights': trainable_weights}

# Setup optimizer
optimizer = brainstate.optim.Adam(lr=LEARNING_RATE)
optimizer.register_trainable_weights(train_states)

# Create input pattern
# Strong stimulus for first STIMULUS_STEPS, then no input
time_idx = jnp.arange(SEQ_LEN)

def create_stimulus_inputs(key, pattern='random'):
    """Create input with stimulus at beginning, then nothing."""
    if pattern == 'random':
        # Random stimulus to different neurons
        stimulus = jax.random.normal(key, (STIMULUS_STEPS, BATCH_SIZE, n_neurons)) * 500.0
        stimulus = jnp.maximum(stimulus, 0)  # Only positive
    else:
        # Constant strong stimulus
        stimulus = jnp.ones((STIMULUS_STEPS, BATCH_SIZE, n_neurons)) * 1000.0

    # No input after stimulus period
    post_stimulus = jnp.zeros((SEQ_LEN - STIMULUS_STEPS, BATCH_SIZE, n_neurons))

    return jnp.concatenate([stimulus, post_stimulus], axis=0)

# Target: specific spike rate in the later part of sequence
# We want some neurons to be active, some silent
def create_target_rates(key):
    """Create target firing rates (different for each class)."""
    # Random target pattern
    target = jax.random.uniform(key, (BATCH_SIZE, n_neurons)) * 0.3
    # Expand to sequence (we only care about post-stimulus period)
    return jnp.broadcast_to(target[None, :, :], (SEQ_LEN - STIMULUS_STEPS, BATCH_SIZE, n_neurons))

def loss_fn(outputs, targets):
    """MSE loss on firing rates during post-stimulus period."""
    # Only use outputs from post-stimulus period
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

    # Gradient clipping
    flat_grads = [jnp.reshape(g, (-1,)) for g in grads.values()]
    all_grads = jnp.concatenate(flat_grads)
    grad_norm = jnp.linalg.norm(all_grads)
    clip_coeff = jnp.minimum(10.0 / (grad_norm + 1e-6), 1.0)
    clipped_grads = {k: clip_coeff * v for k, v in grads.items()}

    optimizer.update(clipped_grads)

    return loss_val / LOSS_SCALE, grad_norm

@brainstate.compile.jit
def eval_step(inputs, targets):
    """Evaluate without gradient."""
    network.reset(batch_size=BATCH_SIZE)
    etrace.reset_state()

    def step_fn(i):
        return etrace(inputs[i])

    outputs = brainstate.compile.for_loop(step_fn, time_idx, pbar=None)

    return loss_fn(outputs, targets), outputs.mean()

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

        inputs = create_stimulus_inputs(key1, pattern='random')
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

# Final evaluation
key = jax.random.PRNGKey(9999)
key1, key2 = jax.random.split(key)
final_inputs = create_stimulus_inputs(key1, pattern='random')
final_targets = create_target_rates(key2)

final_loss, final_output_mean = eval_step(final_inputs, final_targets)
log(f"Final eval - loss: {final_loss:.6f}, output_mean: {final_output_mean:.4f}")
