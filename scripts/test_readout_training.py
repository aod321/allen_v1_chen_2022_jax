#!/usr/bin/env python3
"""Test script to verify readout weights are being trained."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from v1_jax.models import V1Network, V1NetworkConfig
from v1_jax.models.readout import MultiClassReadout
from v1_jax.data.network_loader import load_billeh
from v1_jax.training.trainer import V1Trainer, TrainConfig
from v1_jax.data.mnist_loader import MNISTDataLoader, MNISTConfig

# Config
data_dir = '/nvmessd/yinzi/GLIF_network'
n_neurons = 10000  # Smaller for testing
n_input = 17400

# Load network
print("Loading network...")
input_pop, network_data, bkg_weights = load_billeh(
    n_input=n_input,
    n_neurons=n_neurons,
    core_only=False,
    data_dir=data_dir,
    seed=42,
    use_dale_law=True,
)

network_config = V1NetworkConfig(
    dt=1.0,
    gauss_std=0.28,
    dampening_factor=0.5,
    max_delay=5,
    input_weight_scale=1.0,
    use_dale_law=True,
    use_decoded_noise=False,
    sparse_format='bcsr',
)

network = V1Network.from_billeh(
    network_path=data_dir,
    config=network_config,
    bkg_weights=bkg_weights,
    network_data=network_data,
    input_pop=input_pop,
)
print(f"Network loaded: {network.n_neurons} neurons")

# Create readout
n_output = 10
chunk_size = 50
readout = MultiClassReadout(
    n_neurons=network.n_neurons,
    n_classes=n_output,
    temporal_pooling='chunks',
    chunk_size=chunk_size,
    apply_softmax=True,
)

readout_params = {
    'weights': readout.dense_readout.params.weights,
    'bias': readout.dense_readout.params.bias,
}

readout_config = {
    'temporal_pooling': 'chunks',
    'chunk_size': chunk_size,
    'apply_softmax': True,
}

print(f"Initial readout_weights shape: {readout_params['weights'].shape}")
print(f"Initial readout_weights mean: {jnp.mean(readout_params['weights']):.6f}")
print(f"Initial readout_weights std: {jnp.std(readout_params['weights']):.6f}")

# Create trainer
train_config = TrainConfig(
    learning_rate=1e-3,
    rate_cost=0.001,
    voltage_cost=0.0,  # Disable to focus on classification
    weight_cost=0.0,
    use_rate_regularization=False,  # Disable for testing
    use_voltage_regularization=False,
    use_weight_regularization=False,
    use_dale_law=True,
    gradient_clip_norm=1.0,
)

# Target rates (not used since rate_regularization is disabled)
target_rates = jnp.full(network.n_neurons, 0.02, dtype=jnp.float32)

trainer = V1Trainer(
    network=network,
    config=train_config,
    target_firing_rates=target_rates,
    readout_config=readout_config,
)

# Init train state
key = jax.random.PRNGKey(42)
train_state = trainer.init_train_state(key, readout_params=readout_params)

print("\nParams in train_state:")
for k in train_state.params.keys():
    print(f"  {k}: {train_state.params[k].shape}")

# Verify readout params are in state
assert 'readout_weights' in train_state.params, "readout_weights not in params!"
assert 'readout_bias' in train_state.params, "readout_bias not in params!"
print("\n✓ Readout params are in train_state.params")

# Create MNIST loader
mnist_config = MNISTConfig(
    seq_len=600,
    pre_delay=50,
    im_slice=100,
    post_delay=450,
    intensity=1.0,
    current_input=True,
)

train_loader = MNISTDataLoader(
    n_inputs=n_input,
    batch_size=4,
    config=mnist_config,
    is_training=True,
    seed=42,
)

# Get a batch
inputs, labels, weights = train_loader.sample_batch()
inputs = jnp.asarray(inputs)
labels = jnp.asarray(labels)
weights = jnp.asarray(weights)

print(f"\nBatch shapes:")
print(f"  inputs: {inputs.shape}")
print(f"  labels: {labels.shape}")
print(f"  weights: {weights.shape}")

# Init network state
network_state = network.init_state(batch_size=4)

# Define readout_fn (for compatibility, though trainer uses params directly)
def readout_fn(spikes):
    return readout(spikes)

# Do a few training steps and check if readout weights change
print("\n--- Training Steps ---")
initial_weights = train_state.params['readout_weights'].copy()
initial_bias = train_state.params['readout_bias'].copy()

for step in range(5):
    train_state, output, metrics = trainer.train_step(
        train_state, inputs, labels, weights, network_state, readout_fn
    )

    weight_change = jnp.mean(jnp.abs(train_state.params['readout_weights'] - initial_weights))
    bias_change = jnp.mean(jnp.abs(train_state.params['readout_bias'] - initial_bias))

    print(f"Step {step + 1}: Loss={float(metrics.loss):.4f}, CLoss={float(metrics.classification_loss):.4f}, "
          f"Acc={float(metrics.accuracy):.4f}, ΔW={float(weight_change):.6f}, ΔB={float(bias_change):.6f}")

# Final comparison
final_weight_diff = jnp.max(jnp.abs(train_state.params['readout_weights'] - initial_weights))
final_bias_diff = jnp.max(jnp.abs(train_state.params['readout_bias'] - initial_bias))

print(f"\nMax weight change after 5 steps: {float(final_weight_diff):.6f}")
print(f"Max bias change after 5 steps: {float(final_bias_diff):.6f}")

if final_weight_diff > 0 or final_bias_diff > 0:
    print("\n✓ Readout weights ARE being updated!")
else:
    print("\n✗ Readout weights are NOT being updated!")

# Check gradient magnitudes
print("\n--- Checking Gradient Magnitudes ---")
def compute_grads(params, inputs, labels, weights, network_state, readout_fn, rng_key):
    return trainer._compute_loss(
        params, train_state.initial_params,
        inputs, labels, weights, network_state, readout_fn, rng_key
    )

grad_fn = jax.grad(lambda p, i, l, w, ns, rf, k: compute_grads(p, i, l, w, ns, rf, k)[0], argnums=0)
grads = grad_fn(train_state.params, inputs, labels, weights, network_state, readout_fn, jax.random.PRNGKey(0))

for k, v in grads.items():
    grad_mean = float(jnp.mean(jnp.abs(v)))
    grad_max = float(jnp.max(jnp.abs(v)))
    grad_std = float(jnp.std(v))
    print(f"  {k}: mean={grad_mean:.8f}, max={grad_max:.6f}, std={grad_std:.8f}")
