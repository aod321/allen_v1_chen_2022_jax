#!/usr/bin/env python
"""Debug script to identify where brainstate training hangs."""

from __future__ import annotations

import os
import sys
import time
import pickle
from typing import Tuple

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def log(msg):
    """Print with flush to ensure immediate output."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Script starting...")
log("Importing JAX...")
import jax
import jax.numpy as jnp
log(f"JAX devices: {jax.devices()}")

log("Importing numpy...")
import numpy as np

log("Importing brainstate...")
import brainstate

log("Adding project root to path...")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log("Importing load_billeh...")
from src.v1_jax.data import load_billeh

log("Importing V1NetworkBrainstate...")
from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate

# Configuration
DATA_DIR = "/nvmessd/yinzi/GLIF_network"
BATCH_SIZE = 4  # Smaller for faster debugging
SEQ_LEN = 100   # Shorter sequence for faster debugging
N_STEPS = 2     # Just a few steps

log(f"Config: batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, n_steps={N_STEPS}")

# Load network data
log("Loading network data with load_billeh()...")
start = time.time()
input_pop, network_data, bkg_weights = load_billeh(
    n_input=17400,
    n_neurons=51978,
    core_only=False,
    data_dir=DATA_DIR,
    seed=3000,
)
log(f"Network data loaded in {time.time() - start:.1f}s")
log(f"  n_neurons: {network_data['n_nodes']}")
log(f"  n_inputs: {input_pop['n_inputs']}")
log(f"  n_receptors: {network_data['node_params']['tau_syn'].shape[1]}")

# Create network
log("Creating V1NetworkBrainstate...")
start = time.time()
network = V1NetworkBrainstate.from_billeh(
    network_data,
    input_data=input_pop,
    bkg_weights=bkg_weights,
    dt=1.0,
    mode='training',
    precision=32,
)
log(f"Network created in {time.time() - start:.1f}s")
log(f"  has_input_layer: {network.has_input_layer}")
log(f"  n_neurons: {network.n_neurons}")
log(f"  n_inputs: {network.n_inputs}")

# Load LGN data
log("Loading LGN firing rate data...")
stim_path = os.path.join(DATA_DIR, '..', 'many_small_stimuli.pkl')
start = time.time()
with open(stim_path, 'rb') as f:
    data = pickle.load(f)
rates = np.stack(list(data.values()), axis=0).astype(np.float32)
log(f"LGN data loaded in {time.time() - start:.1f}s")
log(f"  rates shape: {rates.shape}")

# Create simple input batch
log("Creating input batch...")
n_lgn = rates.shape[-1]
# Simple random selection for testing
rng = np.random.RandomState(42)
img_indices = rng.randint(0, 8, size=BATCH_SIZE)
inputs = np.zeros((SEQ_LEN, BATCH_SIZE, n_lgn), dtype=np.float32)
for b in range(BATCH_SIZE):
    inputs[:, b, :] = rates[img_indices[b], :SEQ_LEN, :]
# Convert to firing rate scale
p = 1 - np.exp(-inputs / 1000.0)
inputs = (p * 1.3).astype(np.float32)
inputs_jax = jnp.asarray(inputs)
log(f"Input shape: {inputs_jax.shape}")
log(f"Input range: {inputs_jax.min():.3f} - {inputs_jax.max():.3f}")

# Test reset
log("Testing network.reset()...")
start = time.time()
network.reset(batch_size=BATCH_SIZE)
log(f"Reset completed in {time.time() - start:.3f}s")

# Test single time step
log("Testing single update_with_lgn() call...")
log("  This may take a while due to JIT compilation...")
start = time.time()
try:
    single_input = inputs_jax[0]  # (batch, n_lgn)
    log(f"  Single input shape: {single_input.shape}")
    output = network.update_with_lgn(single_input)
    log(f"  Output shape: {output.shape}")
    log(f"  Output range: {float(output.min()):.4f} - {float(output.max()):.4f}")
    log(f"First update_with_lgn() completed in {time.time() - start:.1f}s")
except Exception as e:
    log(f"ERROR in update_with_lgn: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test a few more steps
log(f"Running {N_STEPS} more steps...")
for step in range(N_STEPS):
    start = time.time()
    output = network.update_with_lgn(inputs_jax[step + 1])
    log(f"  Step {step + 1}: {time.time() - start:.3f}s, output mean: {float(output.mean()):.4f}")

# Test full simulation
log(f"Testing simulate() with use_lgn_input=True...")
log("  Resetting network first...")
network.reset(batch_size=BATCH_SIZE)
log("  Starting simulation...")
start = time.time()
try:
    outputs, spikes = network.simulate(inputs_jax, reset_before=False, use_lgn_input=True)
    log(f"Simulation completed in {time.time() - start:.1f}s")
    log(f"  Outputs shape: {outputs.shape}")
    log(f"  Spikes shape: {spikes.shape}")
    log(f"  Mean spike rate: {float(spikes.mean()) * 100:.2f}%")
except Exception as e:
    log(f"ERROR in simulate: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

log("=" * 60)
log("SUCCESS! Brainstate forward pass works correctly.")
log("=" * 60)
