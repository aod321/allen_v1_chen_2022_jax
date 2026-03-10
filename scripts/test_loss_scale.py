#!/usr/bin/env python3
"""Test loss scaling for gradient amplification."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

# Set up brainstate before other imports
import brainstate
brainstate.environ.set(dt=1.0)

import braintrace

# Import only what we need, avoiding pandas
from src.v1_jax.nn.glif3_brainstate import GLIF3Brainstate
from src.v1_jax.nn.connectivity_brainstate import (
    Connection, SynapticDelayBuffer, build_connection_from_billeh
)

# Apply JAX compatibility patches
from src.v1_jax.compat import apply_jax_compat_patches
apply_jax_compat_patches()

import brainevent
from brainevent import CSR

print('=== Testing loss scaling for gradient amplification ===')

# Create a minimal test network directly (not through V1NetworkBrainstate)
n_neurons = 30
n_receptors = 4
n_types = 2
n_connections = 200

rng = np.random.RandomState(42)

# Create neuron parameters
V_th_types = np.ones(n_types) * -46.0
E_L_types = np.ones(n_types) * -68.0
g_types = np.ones(n_types) * 2.5
C_m_types = np.ones(n_types) * 36.0
V_reset_types = E_L_types.copy()
t_ref_types = np.ones(n_types) * 3.0
asc_decay_types = np.stack([np.ones(n_types) * 0.003, np.ones(n_types) * 0.1], axis=-1)
asc_amps_types = np.stack([np.ones(n_types) * -1.7, np.ones(n_types) * -62.0], axis=-1)
tau_syn_types = np.tile([5.5, 8.5, 2.8, 5.8], (n_types, 1))

node_type_ids = rng.randint(0, n_types, size=n_neurons).astype(np.int32)

# Gather per-neuron from per-type
V_th = V_th_types[node_type_ids]
g = g_types[node_type_ids]
E_L = E_L_types[node_type_ids]
C_m = C_m_types[node_type_ids]
V_reset = E_L.copy()
t_ref = t_ref_types[node_type_ids]

# Create connectivity
source_ids = rng.randint(0, n_neurons, size=n_connections)
target_ids = rng.randint(0, n_neurons, size=n_connections)
receptor_types = rng.randint(0, n_receptors, size=n_connections)
delays = (rng.rand(n_connections) * 2 + 1).astype(np.int32)
target_indices = target_ids * n_receptors + receptor_types
indices = np.stack([target_indices, source_ids], axis=-1).astype(np.int32)
weights = rng.randn(n_connections).astype(np.float32) * 3.0

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

# Create GLIF3 neurons directly
population = GLIF3Brainstate(
    size=n_neurons,
    V_th=V_th,
    g=g,
    E_L=E_L,
    C_m=C_m,
    V_reset=V_reset,
    t_ref=t_ref,
    asc_decay=asc_decay_types,
    asc_amps=asc_amps_types,
    tau_syn=tau_syn_types,
    node_type_ids=node_type_ids,
    mode='training',
)

# Create connectivity
connectivity = build_connection_from_billeh(
    synapses=network_data['synapses'],
    n_neurons=n_neurons,
    n_receptors=n_receptors,
    dt=1.0,
    dtype=jnp.float32,
)

# Create delay buffer
delay_buffer = SynapticDelayBuffer(
    max_delay=connectivity.max_delay,
    num_neurons=n_neurons,
    num_receptors=n_receptors,
    dtype=jnp.float32,
)

# Collect trainable weights
all_weights = []
weight_info = []
current_idx = 0
for key in sorted(connectivity.csr_matrices.keys()):
    csr = connectivity.csr_matrices[key]
    weights_data = csr.data
    n_weights = len(weights_data)
    all_weights.append(weights_data)
    weight_info.append((key, current_idx, current_idx + n_weights))
    current_idx += n_weights

trainable_weights = brainstate.ParamState(jnp.concatenate(all_weights))

print(f'Network: {n_neurons} neurons, {n_connections} connections')
print(f'Trainable weights: {trainable_weights.value.shape}')

# Test data
seq_len = 20
batch_size = 1
inputs = jnp.ones((seq_len, batch_size, n_neurons)) * 200.0
targets = jnp.ones((seq_len, batch_size, n_neurons)) * 0.3


# Simple network wrapper for braintrace
class SimpleV1Wrapper(brainstate.nn.Module):
    def __init__(self, population, connectivity, delay_buffer, trainable_weights, weight_info):
        super().__init__()
        self.population = population
        self.connectivity = connectivity
        self.delay_buffer = delay_buffer
        self._trainable_weights = trainable_weights
        self._weight_info = weight_info
        self.n_neurons = population.num_neurons
        self.n_receptors = population.num_receptors

    def reset(self, batch_size=1):
        self.population.reset_state(batch_size=batch_size)
        self.delay_buffer.reset()

    def update(self, external_input):
        if external_input.ndim == 1:
            external_input = external_input[None, :]
        batch_size = external_input.shape[0]

        # Get delayed synaptic input
        delayed_input = self.delay_buffer.get_current_synaptic_input()
        syn_input = delayed_input.T
        syn_input = jnp.broadcast_to(syn_input, (batch_size, self.n_neurons, self.n_receptors))

        # Update neurons
        output = self.population.update(syn_input, x=external_input)

        # Propagate spikes
        spikes = self.population.get_spike()
        self._propagate_spikes(spikes)

        self.delay_buffer.advance_and_clear_current()

        return output

    def _propagate_spikes(self, spikes):
        if spikes.ndim > 1:
            spikes_flat = jnp.mean(spikes, axis=0)
        else:
            spikes_flat = spikes

        trainable_weights = self._trainable_weights.value

        for key, start_idx, end_idx in self._weight_info:
            delay, receptor = key
            csr = self.connectivity.csr_matrices[key]

            current_weights = trainable_weights[start_idx:end_idx]
            temp_csr = CSR(
                (current_weights, csr.indices, csr.indptr),
                shape=csr.shape
            )

            target_input = brainevent.EventArray(spikes_flat) @ temp_csr
            self.delay_buffer.add_delayed_synaptic_input(delay, receptor, target_input)


# Create network wrapper
network = SimpleV1Wrapper(population, connectivity, delay_buffer, trainable_weights, weight_info)

# Initialize IODim
network.reset(batch_size=batch_size)
etrace_model = braintrace.IODimVjpAlgorithm(network, decay_or_rank=0.99)
etrace_model.compile_graph(inputs[0])

train_states = {'weights': network._trainable_weights}
time_idx = jnp.arange(seq_len)

print()
print('Testing gradient scaling:')

# Test gradient with different scales
for scale in [1.0, 10.0, 100.0, 1000.0]:
    def loss_fn_scaled():
        network.reset(batch_size=batch_size)
        etrace_model.reset_state()

        def step_fn(i):
            return etrace_model(inputs[i])

        y_pred = brainstate.compile.for_loop(step_fn, time_idx, pbar=None)
        mse = jnp.mean((y_pred - targets) ** 2)
        return mse * scale

    grad_fn = brainstate.augment.grad(loss_fn_scaled, train_states, return_value=True)
    grads, loss_val = grad_fn()

    print(f'scale={scale:5.0f}: loss={loss_val/scale:.6f}, grad_mean={grads["weights"].mean():.8f}, grad_max={grads["weights"].max():.8f}')

print()
print('Expected: gradient should scale linearly with loss scale')
print()

# Now test actual training with different scales
print('Testing training with different loss scales:')

for loss_scale in [1.0, 1000.0, 10000.0]:
    # Reset weights
    network._trainable_weights.value = jnp.concatenate(all_weights)
    initial_weights = network._trainable_weights.value.copy()

    lr = 1e-3

    def loss_fn_train():
        network.reset(batch_size=batch_size)
        etrace_model.reset_state()

        def step_fn(i):
            return etrace_model(inputs[i])

        y_pred = brainstate.compile.for_loop(step_fn, time_idx, pbar=None)
        mse = jnp.mean((y_pred - targets) ** 2)
        return mse * loss_scale

    grad_fn = brainstate.augment.grad(loss_fn_train, train_states, return_value=True)

    losses = []
    for step in range(10):
        grads, scaled_loss = grad_fn()
        network._trainable_weights.value = network._trainable_weights.value - lr * grads['weights']
        losses.append(float(scaled_loss / loss_scale))

    weight_change = jnp.abs(network._trainable_weights.value - initial_weights).mean()
    loss_change = losses[0] - losses[-1]

    print(f'  loss_scale={loss_scale:6.0f}: loss {losses[0]:.4f} -> {losses[-1]:.4f} (change={loss_change:+.4f}), weight_change={weight_change:.6f}')

print()
print('Done!')
