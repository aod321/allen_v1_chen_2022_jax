#!/usr/bin/env python3
"""Debug IODim gradient flow."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import brainstate
import braintrace

brainstate.environ.set(dt=1.0)

print("=" * 60)
print("Debugging IODim gradient flow")
print("=" * 60)

# Create a small test network
n_neurons = 100
n_receptors = 4
n_types = 5
batch_size = 2
seq_len = 10

# Generate synthetic network parameters
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

n_connections = n_neurons * 10
source_ids = rng.randint(0, n_neurons, size=n_connections)
target_ids = rng.randint(0, n_neurons, size=n_connections)
receptor_types = rng.randint(0, n_receptors, size=n_connections)
delays = rng.rand(n_connections) * 5 + 1

target_indices = target_ids * n_receptors + receptor_types
indices = np.stack([target_indices, source_ids], axis=-1).astype(np.int32)
weights = rng.randn(n_connections).astype(np.float32) * 0.5

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

# Create network
from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate

network = V1NetworkBrainstate.from_billeh(
    network_data, dt=1.0, mode='training', precision=32,
)

print(f"Network: {n_neurons} neurons, {n_connections} connections")

# Test 1: Check trainable weights
print("\n[1] Checking trainable weights")
print("-" * 60)

trainable_weights = network.get_trainable_weights()
print(f"Trainable weights shape: {trainable_weights.value.shape}")
print(f"Trainable weights range: [{trainable_weights.value.min():.4f}, {trainable_weights.value.max():.4f}]")

# Test 2: Manual gradient computation without IODim
print("\n[2] Manual gradient computation (BPTT-style)")
print("-" * 60)

def simple_loss(weights_value, inputs, targets):
    """Simple forward pass and loss without IODim."""
    # Update network weights
    original_weights = network._trainable_weights.value
    network._trainable_weights.value = weights_value

    network.reset(batch_size=inputs.shape[1])

    outputs = []
    for t in range(inputs.shape[0]):
        out = network.update(inputs[t])
        outputs.append(out)

    outputs = jnp.stack(outputs)  # (T, batch, n_neurons)

    # Restore original weights
    network._trainable_weights.value = original_weights

    return jnp.mean((outputs - targets) ** 2)

# Create test data
inputs = jnp.ones((seq_len, batch_size, n_neurons)) * 500.0
targets = jnp.ones((seq_len, batch_size, n_neurons)) * 0.3

# Compute gradient
print("Computing gradients with jax.grad...")
loss_val = simple_loss(trainable_weights.value, inputs, targets)
print(f"Loss: {loss_val:.6f}")

grad_fn = jax.grad(simple_loss)
grads = grad_fn(trainable_weights.value, inputs, targets)

print(f"Gradient stats:")
print(f"  Shape: {grads.shape}")
print(f"  Mean: {grads.mean():.8f}")
print(f"  Std: {grads.std():.8f}")
print(f"  Max: {grads.max():.8f}")
print(f"  Min: {grads.min():.8f}")
print(f"  Non-zero: {(jnp.abs(grads) > 1e-10).sum()} / {len(grads)}")

# Test 3: Check IODim wrapper
print("\n[3] Checking IODim wrapper")
print("-" * 60)

network.reset(batch_size=batch_size)

# Create IODim model
etrace_model = braintrace.IODimVjpAlgorithm(
    network,
    decay_or_rank=0.99,
)

print(f"IODim model created")

# Compile graph
sample_input = inputs[0]  # (batch, n_neurons)
etrace_model.compile_graph(sample_input)
print(f"Graph compiled with sample input shape: {sample_input.shape}")

# Test forward pass
network.reset(batch_size=batch_size)
etrace_model.reset_state()

outputs = []
for t in range(seq_len):
    out = etrace_model(inputs[t])
    outputs.append(out)
    if t < 3:
        print(f"  Step {t}: output mean = {out.mean():.6f}, max = {out.max():.6f}")

outputs_stack = jnp.stack(outputs)
print(f"Total outputs shape: {outputs_stack.shape}")
print(f"Mean output: {outputs_stack.mean():.4f}")

# Test 4: IODim gradient computation
print("\n[4] IODim gradient computation")
print("-" * 60)

train_states = {'weights': network._trainable_weights}

def iodim_loss():
    network.reset(batch_size=batch_size)
    etrace_model.reset_state()

    outputs = []
    for t in range(seq_len):
        out = etrace_model(inputs[t])
        outputs.append(out)

    outputs = jnp.stack(outputs)
    return jnp.mean((outputs - targets) ** 2)

grad_fn = brainstate.augment.grad(iodim_loss, train_states, return_value=True)
grads, loss_val = grad_fn()

print(f"IODim Loss: {loss_val:.6f}")
print(f"IODim Gradient stats:")
for name, g in grads.items():
    print(f"  {name}:")
    print(f"    Shape: {g.shape}")
    print(f"    Mean: {g.mean():.8f}")
    print(f"    Std: {g.std():.8f}")
    print(f"    Max: {g.max():.8f}")
    print(f"    Min: {g.min():.8f}")
    print(f"    Non-zero: {(jnp.abs(g) > 1e-10).sum()} / {len(g)}")

# Test 5: Compare with different input currents
print("\n[5] Gradient sensitivity to input current")
print("-" * 60)

for current in [100, 500, 1000, 2000]:
    inputs_test = jnp.ones((seq_len, batch_size, n_neurons)) * current

    def iodim_loss_current():
        network.reset(batch_size=batch_size)
        etrace_model.reset_state()

        outputs = []
        for t in range(seq_len):
            out = etrace_model(inputs_test[t])
            outputs.append(out)

        outputs = jnp.stack(outputs)
        return jnp.mean((outputs - targets) ** 2)

    grad_fn = brainstate.augment.grad(iodim_loss_current, train_states, return_value=True)
    grads, loss_val = grad_fn()

    g = grads['weights']
    print(f"  Current {current:4d} pA: loss={loss_val:.4f}, grad_mean={g.mean():.8f}, grad_max={g.max():.8f}")

# Test 6: Check weight update
print("\n[6] Testing weight update")
print("-" * 60)

initial_weights = network._trainable_weights.value.copy()

# Try different loss scales
print("\n[6a] Testing different LOSS SCALES (key to gradient magnitude)")
print("-" * 60)

for loss_scale in [1.0, 100.0, 1000.0, 10000.0]:
    inputs_test = jnp.ones((seq_len, batch_size, n_neurons)) * 500.0

    def iodim_loss_scaled():
        network.reset(batch_size=batch_size)
        etrace_model.reset_state()

        outputs = []
        for t in range(seq_len):
            out = etrace_model(inputs_test[t])
            outputs.append(out)

        outputs = jnp.stack(outputs)
        return jnp.mean((outputs - targets) ** 2) * loss_scale

    grad_fn = brainstate.augment.grad(iodim_loss_scaled, train_states, return_value=True)
    grads, loss_val = grad_fn()

    g = grads['weights']
    print(f"  Scale {loss_scale:7.0f}: loss={loss_val:.4f}, grad_mean={g.mean():.2e}, grad_max={g.max():.2e}")

print("\n[6b] Testing weight update with HIGH LOSS SCALE")
print("-" * 60)

# Reset weights
network._trainable_weights.value = initial_weights.copy()

# Use high loss scale and learning rate
lr = 1.0  # Increased
loss_scale = 10000.0  # High scale
inputs_test = jnp.ones((seq_len, batch_size, n_neurons)) * 500.0

losses = []
for step in range(5):
    weights_before = network._trainable_weights.value.copy()

    def iodim_loss_step():
        network.reset(batch_size=batch_size)
        etrace_model.reset_state()

        outputs = []
        for t in range(seq_len):
            out = etrace_model(inputs_test[t])
            outputs.append(out)

        outputs = jnp.stack(outputs)
        return jnp.mean((outputs - targets) ** 2) * loss_scale

    grad_fn = brainstate.augment.grad(iodim_loss_step, train_states, return_value=True)
    grads, loss_val = grad_fn()

    losses.append(loss_val / loss_scale)  # Unscaled loss for display

    g = grads['weights']

    # Manual SGD update
    update = lr * g
    network._trainable_weights.value = network._trainable_weights.value - update

    weight_change = jnp.abs(network._trainable_weights.value - weights_before).mean()
    total_change = jnp.abs(network._trainable_weights.value - initial_weights).mean()

    print(f"  Step {step + 1}: loss={loss_val/loss_scale:.6f}, grad_max={g.max():.2e}, step_update={weight_change:.2e}, total_change={total_change:.2e}")

print(f"\nLoss curve: {losses}")
print(f"Loss change: {losses[0] - losses[-1]:.6f}")

# Test 7: Check IODim decay parameter effect
print("\n[7] Testing different IODim decay values")
print("-" * 60)

# Reset weights
network._trainable_weights.value = initial_weights.copy()

for decay in [0.5, 0.8, 0.95, 0.99]:
    network.reset(batch_size=batch_size)

    etrace_test = braintrace.IODimVjpAlgorithm(
        network,
        decay_or_rank=decay,
    )
    etrace_test.compile_graph(inputs[0])

    train_states_test = {'weights': network._trainable_weights}

    def iodim_loss_decay():
        network.reset(batch_size=batch_size)
        etrace_test.reset_state()

        outputs = []
        for t in range(seq_len):
            out = etrace_test(inputs[t])
            outputs.append(out)

        outputs = jnp.stack(outputs)
        return jnp.mean((outputs - targets) ** 2) * 10000.0

    grad_fn = brainstate.augment.grad(iodim_loss_decay, train_states_test, return_value=True)
    grads, loss_val = grad_fn()

    g = grads['weights']
    print(f"  Decay {decay:.2f}: loss={loss_val/10000:.4f}, grad_mean={g.mean():.2e}, grad_max={g.max():.2e}, grad_nonzero={int((jnp.abs(g) > 1e-10).sum())}")

# Test 8: Check if outputs are actually learning
print("\n[8] Detailed training loop with output monitoring")
print("-" * 60)

# Reset weights
network._trainable_weights.value = initial_weights.copy()
network.reset(batch_size=batch_size)

# Re-create IODim with optimal decay
etrace_final = braintrace.IODimVjpAlgorithm(network, decay_or_rank=0.8)
etrace_final.compile_graph(inputs[0])

train_states_final = {'weights': network._trainable_weights}

# Use constant input for consistency
inputs_const = jnp.ones((seq_len, batch_size, n_neurons)) * 500.0
targets_const = jnp.ones((seq_len, batch_size, n_neurons)) * 0.05  # Lower target

loss_scale = 10000.0
lr = 0.5

for step in range(10):
    weights_before = network._trainable_weights.value.copy()

    def iodim_loss_final():
        network.reset(batch_size=batch_size)
        etrace_final.reset_state()

        outputs = []
        for t in range(seq_len):
            out = etrace_final(inputs_const[t])
            outputs.append(out)

        outputs = jnp.stack(outputs)
        return jnp.mean((outputs - targets_const) ** 2) * loss_scale

    grad_fn = brainstate.augment.grad(iodim_loss_final, train_states_final, return_value=True)
    grads, loss_val = grad_fn()

    # Also compute output mean to see if it changes
    network.reset(batch_size=batch_size)
    etrace_final.reset_state()
    outputs = []
    for t in range(seq_len):
        out = etrace_final(inputs_const[t])
        outputs.append(out)
    outputs = jnp.stack(outputs)
    output_mean = outputs.mean()

    g = grads['weights']

    # Update weights
    network._trainable_weights.value = network._trainable_weights.value - lr * g

    print(f"  Step {step+1:2d}: loss={loss_val/loss_scale:.4f}, output_mean={output_mean:.4f}, grad_max={g.max():.2e}")

print("\n" + "=" * 60)
print("Debug complete")
print("=" * 60)
