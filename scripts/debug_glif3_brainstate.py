#!/usr/bin/env python3
"""Diagnostic script to debug GLIF3Brainstate zero output issue.

This script tests the GLIF3Brainstate neuron model to understand why
the output is all zeros.
"""

import sys
sys.path.insert(0, '/nvmessd/yinzi/allen_v1_chen_2022_jax/src')

import jax
import jax.numpy as jnp
import numpy as np
import brainstate

# Set dt before any brainstate operations
brainstate.environ.set(dt=1.0)

print("=" * 60)
print("GLIF3Brainstate Diagnostic Script")
print("=" * 60)

# Test 1: Basic GLIF3Brainstate with synthetic parameters
print("\n[Test 1] Basic GLIF3Brainstate with constant current injection")
print("-" * 60)

from v1_jax.nn.glif3_brainstate import GLIF3Brainstate

# Create a small test neuron population with typical GLIF3 parameters
n_neurons = 10
batch_size = 2

# Typical GLIF3 parameters (from Allen Institute)
V_th = np.ones(n_neurons) * -46.0  # mV
E_L = np.ones(n_neurons) * -68.0   # mV
g = np.ones(n_neurons) * 2.5       # nS
C_m = np.ones(n_neurons) * 36.0    # pF
V_reset = np.ones(n_neurons) * -68.0  # mV (close to E_L)
t_ref = np.ones(n_neurons) * 3.0   # ms

# ASC parameters (2 types)
asc_decay = np.array([[0.003, 0.1]])  # (1 type, 2 asc)
asc_amps = np.array([[-1.7, -62.0]])  # (1 type, 2 asc) pA

# Synaptic time constants (4 receptor types)
tau_syn = np.array([[5.5, 8.5, 2.8, 5.8]])  # (1 type, 4 receptors) ms

node_type_ids = np.zeros(n_neurons, dtype=np.int32)

print(f"Creating GLIF3Brainstate with {n_neurons} neurons")
print(f"  V_th = {V_th[0]:.1f} mV, E_L = {E_L[0]:.1f} mV")
print(f"  voltage_scale = V_th - E_L = {V_th[0] - E_L[0]:.1f} mV")

neuron = GLIF3Brainstate(
    size=n_neurons,
    V_th=V_th,
    g=g,
    E_L=E_L,
    C_m=C_m,
    V_reset=V_reset,
    t_ref=t_ref,
    asc_decay=asc_decay,
    asc_amps=asc_amps,
    tau_syn=tau_syn,
    node_type_ids=node_type_ids,
    mode='training',
    precision=32,
)

print(f"\nNormalized parameters:")
print(f"  v_th_norm = {neuron.v_th_norm[0]:.4f}")
print(f"  v_reset_norm = {neuron.v_reset_norm[0]:.4f}")
print(f"  e_l_norm = {neuron.e_l_norm[0]:.4f}")
print(f"  voltage_scale = {neuron.voltage_scale[0]:.4f}")
print(f"  voltage_offset = {neuron.voltage_offset[0]:.4f}")

print(f"\nComputed constants:")
print(f"  _decay = {neuron._decay[0]:.6f}")
print(f"  _current_factor = {neuron._current_factor[0]:.6f}")
print(f"  tau_m = {neuron.tau_m[0]:.4f} ms")

neuron.reset_state(batch_size=batch_size)

print(f"\nInitial state:")
print(f"  V.value shape = {neuron.V.value.shape}")
print(f"  V.value[0,:3] = {neuron.V.value[0, :3]}")

# Test with zero synaptic input and various external currents
syn_inputs = jnp.zeros((batch_size, n_neurons, 4))  # 4 receptors

print("\n" + "-" * 60)
print("Testing with increasing external current (in pA):")
print("-" * 60)

for x_current in [0.0, 100.0, 500.0, 1000.0, 5000.0]:
    neuron.reset_state(batch_size=batch_size)

    # Simulate 10 steps
    outputs = []
    voltages = []
    for t in range(10):
        x = jnp.ones((batch_size, n_neurons)) * x_current
        out = neuron.update(syn_inputs, x=x)
        outputs.append(out)
        voltages.append(neuron.V.value.copy())

    outputs = jnp.stack(outputs)
    voltages = jnp.stack(voltages)

    # Get physical voltage
    V_physical = voltages * neuron.voltage_scale + neuron.voltage_offset

    print(f"\nx = {x_current:6.0f} pA:")
    print(f"  Output mean: {outputs.mean():.6f}, max: {outputs.max():.6f}")
    print(f"  V_norm final: {voltages[-1, 0, 0]:.4f}")
    print(f"  V_physical final: {V_physical[-1, 0, 0]:.2f} mV")

# Test 2: Check the surrogate gradient function
print("\n" + "=" * 60)
print("[Test 2] Surrogate gradient function behavior")
print("=" * 60)

membrane_values = jnp.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
print(f"\nspk_fun_training = {neuron.spk_fun_training}")

for m in membrane_values:
    out = neuron.spk_fun_training(m)
    print(f"  membrane_normalized = {m:5.1f} -> spike_output = {out:.6f}")

# Test 3: Compare with AlphaBrain GLIF3
print("\n" + "=" * 60)
print("[Test 3] Compare with AlphaBrain GLIF3")
print("=" * 60)

try:
    sys.path.insert(0, '/nvmessd/yinzi/AlphaBrain/AlphaBrain')
    from glif3 import GLIF3

    # Create AlphaBrain GLIF3 with same parameters
    ab_neuron = GLIF3(
        size=n_neurons,
        V_th=V_th[0],
        E_L=E_L[0],
        g=g[0],
        C_m=C_m[0],
        V_reset=V_reset[0],
        t_ref=t_ref[0],
        asc_decay=asc_decay[0],
        asc_amps=asc_amps[0],
        tau_syn=tau_syn[0],
        mode='training',
        precision=32,
    )

    ab_neuron.reset_state()

    print(f"\nAlphaBrain GLIF3 initial state:")
    print(f"  V.value shape = {ab_neuron.V.value.shape}")
    print(f"  V.value[:3] = {ab_neuron.V.value[:3]}")
    print(f"  _P33 = {ab_neuron._P33:.6f}")
    print(f"  _P30 = {ab_neuron._P30:.6f}")

    print("\n" + "-" * 60)
    print("Testing AlphaBrain GLIF3 with increasing external current:")
    print("-" * 60)

    for x_current in [0.0, 100.0, 500.0, 1000.0, 5000.0]:
        ab_neuron.reset_state()

        # Simulate 10 steps (no batch dimension needed for AlphaBrain)
        outputs = []
        voltages = []
        for t in range(10):
            out = ab_neuron.update(x=x_current)
            outputs.append(out)
            voltages.append(ab_neuron.V.value.copy())

        outputs = jnp.stack(outputs)
        voltages = jnp.stack(voltages)

        print(f"\nx = {x_current:6.0f} pA:")
        print(f"  Output mean: {outputs.mean():.6f}, max: {outputs.max():.6f}")
        print(f"  V final: {voltages[-1, 0]:.2f} mV")

except Exception as e:
    print(f"Could not load AlphaBrain GLIF3: {e}")

# Test 4: Diagnose the membrane dynamics equation
print("\n" + "=" * 60)
print("[Test 4] Diagnose membrane dynamics equation")
print("=" * 60)

# Manual step-by-step computation
print("\nManual membrane dynamics computation:")

V_init = neuron.v_reset_norm[0]
decay = neuron._decay[0]
current_factor = neuron._current_factor[0]
g_val = neuron.g[0]
e_l_norm = neuron.e_l_norm[0]

print(f"  V_init (normalized) = {V_init:.4f}")
print(f"  decay = exp(-dt/tau_m) = {decay:.6f}")
print(f"  current_factor = (1/C_m) * (1 - exp(-dt/tau_m)) * tau_m = {current_factor:.6f}")

x_current = 500.0  # pA
I_syn = 0.0
total_asc = 0.0
gathered_g_el = g_val * e_l_norm  # This should be the leak term but e_l_norm = 0!

print(f"\nWith x = {x_current} pA:")
print(f"  gathered_g_el = g * e_l_norm = {g_val:.2f} * {e_l_norm:.4f} = {gathered_g_el:.6f}")
print(f"  total_current = I_syn + total_asc + gathered_g_el + x")
print(f"                = {I_syn:.2f} + {total_asc:.2f} + {gathered_g_el:.6f} + {x_current:.2f}")
print(f"                = {I_syn + total_asc + gathered_g_el + x_current:.2f}")

total_current = I_syn + total_asc + gathered_g_el + x_current
V_new = decay * V_init + current_factor * total_current

print(f"\n  V_new = decay * V_init + current_factor * total_current")
print(f"        = {decay:.6f} * {V_init:.4f} + {current_factor:.6f} * {total_current:.2f}")
print(f"        = {decay * V_init:.6f} + {current_factor * total_current:.6f}")
print(f"        = {V_new:.6f}")

# Convert to physical voltage
V_physical = V_new * neuron.voltage_scale[0] + neuron.voltage_offset[0]
print(f"\n  V_physical = V_new * voltage_scale + voltage_offset")
print(f"             = {V_new:.6f} * {neuron.voltage_scale[0]:.2f} + {neuron.voltage_offset[0]:.2f}")
print(f"             = {V_physical:.2f} mV")

# Now check what voltage is needed to spike
print(f"\n  v_th_norm = {neuron.v_th_norm[0]:.4f}")
print(f"  Need V_new > v_th_norm to spike")
print(f"  Current V_new ({V_new:.6f}) {'>' if V_new > neuron.v_th_norm[0] else '<='} v_th_norm ({neuron.v_th_norm[0]:.4f})")

# Calculate how much current is needed
# To spike: V_new > 1.0 (normalized threshold)
# V_new = decay * V_init + current_factor * total_current
# 1.0 = decay * V_init + current_factor * total_current
# total_current = (1.0 - decay * V_init) / current_factor
required_total_current = (neuron.v_th_norm[0] - decay * V_init) / current_factor
print(f"\n  To spike, need total_current = {required_total_current:.2f}")
print(f"  (This is very high because current_factor is small in normalized space)")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
print("""
KEY FINDINGS:

1. The _current_factor converts current (pA) to voltage (mV), but our
   membrane potential is NORMALIZED (dimensionless). This mismatch means
   the external input has very little effect on the normalized voltage.

2. The 'gathered_g_el' term is zero because e_l_norm = 0 by definition
   (E_L - E_L = 0). This is actually correct for normalized dynamics.

3. The main issue is that external input 'x' should be in NORMALIZED units,
   not raw pA. We need to scale it by voltage_scale:

   x_normalized = x_pA * _current_factor / voltage_scale

   Or equivalently, modify _current_factor to output normalized voltage.

RECOMMENDED FIX:

   1. Divide _current_factor by voltage_scale to get normalized output:
      self._current_factor_norm = self._current_factor / self.voltage_scale

   2. Use this in the membrane dynamics:
      V_new = decay * V + current_factor_norm * total_current
""")
