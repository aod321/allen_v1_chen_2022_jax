#!/usr/bin/env python3
"""Diagnose low spike rate and non-decreasing loss."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import brainstate

brainstate.environ.set(dt=1.0)

print("=" * 60)
print("Diagnosing low spike rate and loss issues")
print("=" * 60)

# Test 1: Check GLIF3 spike rates with various currents
print("\n[1] GLIF3 spike rate vs input current")
print("-" * 60)

from src.v1_jax.nn.glif3_brainstate import GLIF3Brainstate

n_neurons = 100
batch_size = 4

# Typical GLIF3 parameters
V_th = np.ones(n_neurons) * -46.0
E_L = np.ones(n_neurons) * -68.0
g = np.ones(n_neurons) * 2.5
C_m = np.ones(n_neurons) * 36.0
V_reset = np.ones(n_neurons) * -68.0
t_ref = np.ones(n_neurons) * 3.0
asc_decay = np.array([[0.003, 0.1]])
asc_amps = np.array([[-1.7, -62.0]])
tau_syn = np.array([[5.5, 8.5, 2.8, 5.8]])
node_type_ids = np.zeros(n_neurons, dtype=np.int32)

neuron = GLIF3Brainstate(
    size=n_neurons,
    V_th=V_th, g=g, E_L=E_L, C_m=C_m, V_reset=V_reset, t_ref=t_ref,
    asc_decay=asc_decay, asc_amps=asc_amps, tau_syn=tau_syn,
    node_type_ids=node_type_ids,
    mode='training',
    precision=32,
)

print(f"voltage_scale = {neuron.voltage_scale[0]:.2f} mV")
print(f"_current_factor = {neuron._current_factor[0]:.6f}")

# Test different current levels
seq_len = 100
syn_inputs = jnp.zeros((batch_size, n_neurons, 4))

print(f"\nSpike rate over {seq_len} steps:")
print(f"{'Current (pA)':<15} {'Spike Rate':<15} {'Mean V_norm':<15}")
print("-" * 45)

for current in [0, 50, 100, 200, 300, 500, 1000, 2000]:
    neuron.reset_state(batch_size=batch_size)

    spikes = []
    voltages = []
    for t in range(seq_len):
        x = jnp.ones((batch_size, n_neurons)) * current
        out = neuron.update(syn_inputs, x=x)
        spikes.append(neuron.get_spike())
        voltages.append(neuron.V.value)

    spikes = jnp.stack(spikes)
    voltages = jnp.stack(voltages)

    # For surrogate sigmoid, spike_output > 0.5 means spike
    spike_rate = (spikes > 0.5).mean()
    mean_v = voltages.mean()

    print(f"{current:<15} {spike_rate:<15.4f} {mean_v:<15.4f}")

# Test 2: Check what current is needed to reach threshold
print("\n[2] Current required to reach threshold")
print("-" * 60)

# Single step analysis
neuron.reset_state(batch_size=1)
V_init = neuron.V.value[0, 0]
decay = neuron._decay[0]
current_factor = neuron._current_factor[0]
v_th_norm = neuron.v_th_norm[0]

print(f"V_init (normalized) = {V_init:.4f}")
print(f"v_th_norm = {v_th_norm:.4f}")
print(f"decay = {decay:.6f}")
print(f"current_factor = {current_factor:.6f}")

# Calculate minimum current to spike in 1 step
# V_new = decay * V_init + current_factor * I
# For spike: V_new >= v_th_norm
# I >= (v_th_norm - decay * V_init) / current_factor
I_min_1step = (v_th_norm - decay * V_init) / current_factor
print(f"\nMinimum current for 1-step spike: {I_min_1step:.1f} pA")

# For steady state (V_ss when V_new = V_old = V_ss)
# V_ss = decay * V_ss + current_factor * I
# V_ss * (1 - decay) = current_factor * I
# V_ss = current_factor * I / (1 - decay)
# For spike: V_ss >= v_th_norm
# I >= v_th_norm * (1 - decay) / current_factor
I_threshold = v_th_norm * (1 - decay) / current_factor
print(f"Threshold current (steady-state): {I_threshold:.1f} pA")

# Test 3: Check gradient flow
print("\n[3] Checking gradient flow")
print("-" * 60)

# Simple gradient test
def loss_fn(weights, inputs, targets):
    """Simple loss function for gradient testing."""
    neuron.reset_state(batch_size=inputs.shape[0])

    outputs = []
    for t in range(inputs.shape[1]):
        x = inputs[:, t, :] * weights  # Scale input by weights
        out = neuron.update(jnp.zeros((inputs.shape[0], n_neurons, 4)), x=x)
        outputs.append(out)

    outputs = jnp.stack(outputs, axis=1)  # (batch, T, n_neurons)
    return jnp.mean((outputs - targets) ** 2)

# Create test data
test_inputs = jnp.ones((batch_size, 20, n_neurons)) * 500.0  # High current
test_targets = jnp.ones((batch_size, 20, n_neurons)) * 0.5  # 50% spike rate target
init_weights = jnp.ones(n_neurons)

# Compute gradient
grad_fn = jax.grad(loss_fn)
grads = grad_fn(init_weights, test_inputs, test_targets)

print(f"Gradient stats:")
print(f"  mean: {grads.mean():.8f}")
print(f"  std: {grads.std():.8f}")
print(f"  max: {grads.max():.8f}")
print(f"  min: {grads.min():.8f}")
print(f"  non-zero: {(jnp.abs(grads) > 1e-10).sum()} / {len(grads)}")

# Test 4: Check surrogate gradient function
print("\n[4] Surrogate gradient behavior")
print("-" * 60)

# Test brainstate.surrogate.sigmoid
test_values = jnp.array([-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0])

print("brainstate.surrogate.sigmoid:")
for v in test_values:
    out = brainstate.surrogate.sigmoid(v)

    # Get gradient
    grad_fn = jax.grad(lambda x: brainstate.surrogate.sigmoid(x).sum())
    grad = grad_fn(jnp.array([v]))[0]

    print(f"  input={v:5.1f} -> output={float(out):.4f}, grad={grad:.4f}")

# Test 5: Check AlphaBrain reference
print("\n[5] Compare with AlphaBrain GLIF3")
print("-" * 60)

try:
    sys.path.insert(0, '/nvmessd/yinzi/AlphaBrain/AlphaBrain')
    from glif3 import GLIF3 as GLIF3_AB

    ab_neuron = GLIF3_AB(
        size=n_neurons,
        V_th=-46.0, E_L=-68.0, g=2.5, C_m=36.0, V_reset=-68.0, t_ref=3.0,
        asc_decay=(0.003, 0.1), asc_amps=(-1.7, -62.0),
        tau_syn=(5.5, 8.5, 2.8, 5.8),
        mode='training',
        precision=32,
    )

    print(f"AlphaBrain _P30 = {ab_neuron._P30:.6f}")
    print(f"Our _current_factor = {neuron._current_factor[0]:.6f}")
    print(f"Ratio = {ab_neuron._P30 / neuron._current_factor[0]:.2f}")

    # Test spike rate
    print(f"\nAlphaBrain spike rate over {seq_len} steps:")
    print(f"{'Current (pA)':<15} {'Spike Rate':<15}")
    print("-" * 30)

    for current in [0, 100, 200, 500, 1000]:
        ab_neuron.reset_state()

        spikes = []
        for t in range(seq_len):
            out = ab_neuron.update(x=current)
            spikes.append(ab_neuron.get_spike())

        spikes = jnp.stack(spikes)
        spike_rate = (spikes > 0.5).mean()
        print(f"{current:<15} {spike_rate:<15.4f}")

except Exception as e:
    print(f"Could not load AlphaBrain: {e}")

print("\n" + "=" * 60)
print("Diagnosis complete")
print("=" * 60)
