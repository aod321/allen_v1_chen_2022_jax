#!/usr/bin/env python3
"""Quick test of brainstate training pipeline.

This script tests the end-to-end training with a small network
to verify all components work together.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import brainstate

# Set environment
brainstate.environ.set(dt=1.0)

print("=" * 60)
print("Testing brainstate training pipeline")
print("=" * 60)

# Create a small synthetic network for testing
print("\n[1] Creating synthetic network data...")

n_neurons = 100
n_receptors = 4
n_types = 5
batch_size = 4
seq_len = 20

# Generate synthetic network parameters
rng = np.random.RandomState(42)

# Per-type parameters
V_th_types = np.ones(n_types) * -46.0 + rng.randn(n_types) * 2
E_L_types = np.ones(n_types) * -68.0 + rng.randn(n_types) * 2
g_types = np.ones(n_types) * 2.5 + rng.rand(n_types) * 0.5
C_m_types = np.ones(n_types) * 36.0 + rng.rand(n_types) * 5
V_reset_types = E_L_types.copy()
t_ref_types = np.ones(n_types) * 3.0

asc_decay_types = np.stack([np.ones(n_types) * 0.003, np.ones(n_types) * 0.1], axis=-1)
asc_amps_types = np.stack([np.ones(n_types) * -1.7, np.ones(n_types) * -62.0], axis=-1)
tau_syn_types = np.tile([5.5, 8.5, 2.8, 5.8], (n_types, 1))

# Type assignment
node_type_ids = rng.randint(0, n_types, size=n_neurons)

# Generate synthetic connectivity (sparse)
n_connections = n_neurons * 10  # ~10 connections per neuron
source_ids = rng.randint(0, n_neurons, size=n_connections)
target_ids = rng.randint(0, n_neurons, size=n_connections)
receptor_types = rng.randint(0, n_receptors, size=n_connections)
delays = rng.rand(n_connections) * 5 + 1  # 1-6 ms

# Create indices: target_idx = neuron_idx * n_receptors + receptor
target_indices = target_ids * n_receptors + receptor_types
indices = np.stack([target_indices, source_ids], axis=-1).astype(np.int32)

# Random weights
weights = rng.randn(n_connections).astype(np.float32) * 0.5

network_data = {
    'n_nodes': n_neurons,
    'node_params': {
        'V_th': V_th_types,
        'g': g_types,
        'E_L': E_L_types,
        'C_m': C_m_types,
        'V_reset': V_reset_types,
        't_ref': t_ref_types,
        'k': asc_decay_types,
        'asc_amps': asc_amps_types,
        'tau_syn': tau_syn_types,
    },
    'node_type_ids': node_type_ids,
    'synapses': {
        'indices': indices,
        'weights': weights,
        'delays': delays,
        'dense_shape': (n_neurons * n_receptors, n_neurons),
    },
}

print(f"Network: {n_neurons} neurons, {n_connections} connections")

# Create V1Network
print("\n[2] Creating V1NetworkBrainstate...")

from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate

network = V1NetworkBrainstate.from_billeh(
    network_data,
    dt=1.0,
    mode='training',
    precision=32,
)

print(f"Network created: {network.n_neurons} neurons, {network.n_receptors} receptors")

# Test forward pass
print("\n[3] Testing forward pass...")

network.reset(batch_size=batch_size)

# Create input with enough current to drive spiking
external_input = jnp.ones((batch_size, n_neurons)) * 300.0  # 300 pA

output = network.update(external_input)
print(f"Output shape: {output.shape}")
print(f"Output mean: {output.mean():.6f}, max: {output.max():.6f}")

# Multiple steps
outputs = []
for t in range(seq_len):
    out = network.update(external_input)
    outputs.append(out)

outputs = jnp.stack(outputs)
print(f"\nMulti-step output shape: {outputs.shape}")
print(f"Mean spike rate: {outputs.mean():.4f}")

# Test IODim training
print("\n[4] Testing IODim training...")

from src.v1_jax.training.trainer_brainstate import IODimTrainer, IODimConfig

# Create inputs and targets
inputs = jnp.ones((seq_len, batch_size, n_neurons)) * 300.0  # (T, B, N)

# Add some variation
t_idx = jnp.arange(seq_len)[:, None, None]
inputs = inputs * (0.5 + 0.5 * jnp.sin(2 * jnp.pi * t_idx / seq_len))

# Target: spike rate ~ 0.3 for all neurons
targets = jnp.ones((seq_len, batch_size, n_neurons)) * 0.3

print(f"Input shape: {inputs.shape}")
print(f"Target shape: {targets.shape}")

# Create trainer
iodim_config = IODimConfig(
    learning_rate=1e-3,
    grad_clip_norm=1.0,
    etrace_decay=0.99,
    optimizer='Adam',
)

try:
    trainer = IODimTrainer(network, iodim_config)
    print("IODimTrainer created successfully")

    # Train for a few steps
    print("\nTraining for 5 epochs...")
    for epoch in range(5):
        loss = trainer.train_step(inputs, targets)
        print(f"  Epoch {epoch + 1}: loss = {loss:.6f}")

    print("\nTraining completed successfully!")

except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
