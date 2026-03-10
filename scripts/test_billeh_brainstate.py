#!/usr/bin/env python3
"""Test brainstate training with real Billeh network data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import brainstate
import time

# Set environment
brainstate.environ.set(dt=1.0)

print("=" * 60)
print("Testing brainstate with Billeh network data")
print("=" * 60)

# Load network data
print("\n[1] Loading Billeh network data...")

from src.v1_jax.data.network_loader import load_network

data_dir = "/nvmessd/yinzi/GLIF_network"
network_data = load_network(
    path=os.path.join(data_dir, 'network_dat.pkl'),
    h5_path=os.path.join(data_dir, 'network/v1_nodes.h5'),
    data_dir=data_dir,
    core_only=True,
    n_neurons=None,  # Use all neurons
    seed=3000,
)

n_neurons = network_data['n_nodes']
n_receptors = network_data['node_params']['tau_syn'].shape[1]
n_synapses = len(network_data['synapses']['weights'])

print(f"Network: {n_neurons} neurons, {n_receptors} receptors, {n_synapses} synapses")

# Create V1Network
print("\n[2] Creating V1NetworkBrainstate...")
start_time = time.time()

from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate

network = V1NetworkBrainstate.from_billeh(
    network_data,
    dt=1.0,
    mode='training',
    precision=32,
)

build_time = time.time() - start_time
print(f"Network created in {build_time:.2f}s")
print(f"  Neurons: {network.n_neurons}")
print(f"  Receptors: {network.n_receptors}")
print(f"  CSR matrices: {len(network.connectivity.csr_matrices)}")

# Test forward pass
print("\n[3] Testing forward pass...")

batch_size = 4
seq_len = 10

network.reset(batch_size=batch_size)

# Create input with enough current to drive spiking
external_input = jnp.ones((batch_size, n_neurons)) * 200.0  # 200 pA

# Warm-up JIT compilation
print("  Warming up JIT...")
start_time = time.time()
_ = network.update(external_input)
warmup_time = time.time() - start_time
print(f"  Warm-up time: {warmup_time:.2f}s")

# Multiple steps
print(f"  Running {seq_len} steps...")
network.reset(batch_size=batch_size)

start_time = time.time()
outputs = []
for t in range(seq_len):
    out = network.update(external_input)
    outputs.append(out)

run_time = time.time() - start_time
outputs = jnp.stack(outputs)

print(f"  Runtime: {run_time:.2f}s ({run_time/seq_len*1000:.1f} ms/step)")
print(f"  Output shape: {outputs.shape}")
print(f"  Mean spike rate: {outputs.mean():.4f}")

# Test IODim training
print("\n[4] Testing IODim training...")

from src.v1_jax.training.trainer_brainstate import IODimTrainer, IODimConfig

# Create inputs and targets
inputs = jnp.ones((seq_len, batch_size, n_neurons)) * 200.0  # (T, B, N)

# Add temporal variation
t_idx = jnp.arange(seq_len)[:, None, None]
inputs = inputs * (0.5 + 0.5 * jnp.sin(2 * jnp.pi * t_idx / seq_len))

# Target: spike rate ~ 0.2 for all neurons
targets = jnp.ones((seq_len, batch_size, n_neurons)) * 0.2

print(f"  Input shape: {inputs.shape}")
print(f"  Target shape: {targets.shape}")

# Create trainer
iodim_config = IODimConfig(
    learning_rate=1e-3,
    grad_clip_norm=1.0,
    etrace_decay=0.99,
    optimizer='Adam',
)

try:
    trainer = IODimTrainer(network, iodim_config)
    print("  IODimTrainer created successfully")

    # Train for a few steps
    print("\n  Training for 3 epochs...")
    for epoch in range(3):
        start_time = time.time()
        loss = trainer.train_step(inputs, targets)
        epoch_time = time.time() - start_time
        print(f"    Epoch {epoch + 1}: loss = {loss:.6f}, time = {epoch_time:.2f}s")

    print("\n  Training completed successfully!")

except Exception as e:
    print(f"\n  Error during training: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
