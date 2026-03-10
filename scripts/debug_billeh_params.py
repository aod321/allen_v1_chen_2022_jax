#!/usr/bin/env python3
"""Debug Billeh network parameters to understand low spike rate."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import numpy as np
import brainstate

brainstate.environ.set(dt=1.0)

print("=" * 60)
print("Analyzing Billeh network parameters")
print("=" * 60)

# Load network data
from src.v1_jax.data.network_loader import load_network

data_dir = "/nvmessd/yinzi/GLIF_network"
network_data = load_network(
    path=os.path.join(data_dir, 'network_dat.pkl'),
    h5_path=os.path.join(data_dir, 'network/v1_nodes.h5'),
    data_dir=data_dir,
    core_only=True,
    n_neurons=None,
    seed=3000,
)

n_neurons = network_data['n_nodes']
node_params = network_data['node_params']
node_type_ids = network_data['node_type_ids']
n_types = len(node_params['V_th'])

print(f"\nNetwork: {n_neurons} neurons, {n_types} types")

# Per-type parameters
V_th = node_params['V_th']
E_L = node_params['E_L']
g = node_params['g']
C_m = node_params['C_m']

# Compute derived parameters
voltage_scale = V_th - E_L
tau_m = C_m / g

# Current factor for each type
dt = 1.0
decay = np.exp(-dt / tau_m)
current_factor_mv = (1.0 / C_m) * (1.0 - np.exp(-dt / tau_m)) * tau_m
current_factor_norm = current_factor_mv / voltage_scale

# Threshold current (steady-state)
I_threshold = 1.0 * (1 - decay) / current_factor_norm

print(f"\n{'Type':<6} {'V_th':<10} {'E_L':<10} {'V_scale':<10} {'tau_m':<10} {'I_thresh':<10}")
print("-" * 66)
for t in range(n_types):
    print(f"{t:<6} {V_th[t]:<10.2f} {E_L[t]:<10.2f} {voltage_scale[t]:<10.2f} {tau_m[t]:<10.2f} {I_threshold[t]:<10.1f}")

# Count neurons per type
type_counts = np.bincount(node_type_ids, minlength=n_types)
print(f"\nNeurons per type:")
for t in range(n_types):
    print(f"  Type {t}: {type_counts[t]} neurons ({type_counts[t]/n_neurons*100:.1f}%)")

# Gather per-neuron parameters
V_th_neurons = V_th[node_type_ids]
E_L_neurons = E_L[node_type_ids]
voltage_scale_neurons = voltage_scale[node_type_ids]
I_threshold_neurons = I_threshold[node_type_ids]

print(f"\nPer-neuron parameter distributions:")
print(f"  V_th: {V_th_neurons.min():.2f} to {V_th_neurons.max():.2f} mV, mean={V_th_neurons.mean():.2f}")
print(f"  E_L: {E_L_neurons.min():.2f} to {E_L_neurons.max():.2f} mV, mean={E_L_neurons.mean():.2f}")
print(f"  voltage_scale: {voltage_scale_neurons.min():.2f} to {voltage_scale_neurons.max():.2f} mV, mean={voltage_scale_neurons.mean():.2f}")
print(f"  I_threshold: {I_threshold_neurons.min():.1f} to {I_threshold_neurons.max():.1f} pA, mean={I_threshold_neurons.mean():.1f}")

# Calculate what fraction of neurons would spike at different input currents
print(f"\nFraction of neurons with I_threshold <= input current:")
for I_input in [50, 100, 200, 300, 500, 1000, 2000]:
    frac = (I_threshold_neurons <= I_input).mean()
    print(f"  {I_input:4d} pA: {frac*100:.1f}%")

# Test actual spike rates with V1Network
print("\n" + "=" * 60)
print("Testing actual spike rates with V1NetworkBrainstate")
print("=" * 60)

from src.v1_jax.models.v1_network_brainstate import V1NetworkBrainstate

network = V1NetworkBrainstate.from_billeh(
    network_data, dt=1.0, mode='training', precision=32,
)

batch_size = 2
seq_len = 50

print(f"\nSpike rate over {seq_len} steps with uniform external input:")
print(f"{'Current (pA)':<15} {'Spike Rate':<15} {'Active Neurons':<20}")
print("-" * 50)

for current in [100, 200, 500, 1000, 2000, 5000]:
    network.reset(batch_size=batch_size)

    spikes_all = []
    for t in range(seq_len):
        external_input = jnp.ones((batch_size, n_neurons)) * current
        output = network.update(external_input)
        spikes = network.population.get_spike()
        spikes_all.append(spikes)

    spikes_all = jnp.stack(spikes_all)  # (T, batch, n_neurons)

    # Count spikes (output > 0.5 for surrogate)
    spike_counts = (spikes_all > 0.5).sum(axis=(0, 1))  # per neuron
    spike_rate = (spikes_all > 0.5).mean()
    active_neurons = (spike_counts > 0).sum()

    print(f"{current:<15} {spike_rate:<15.4f} {active_neurons}/{n_neurons}")

# Check ASC (adaptive spike currents) impact
print("\n" + "=" * 60)
print("Checking ASC (adaptive spike currents) impact")
print("=" * 60)

asc_amps = node_params['asc_amps']  # (n_types, 2)
asc_decay = node_params['k']  # (n_types, 2)

print(f"\nASC parameters per type:")
print(f"{'Type':<6} {'ASC1_amp':<12} {'ASC2_amp':<12} {'ASC1_k':<12} {'ASC2_k':<12}")
print("-" * 54)
for t in range(n_types):
    print(f"{t:<6} {asc_amps[t,0]:<12.2f} {asc_amps[t,1]:<12.2f} {asc_decay[t,0]:<12.4f} {asc_decay[t,1]:<12.4f}")

# ASC amplitudes are typically negative (hyperpolarizing) which REDUCES excitability after spiking
print(f"\nASC1 amplitude: {asc_amps[:,0].min():.2f} to {asc_amps[:,0].max():.2f} pA")
print(f"ASC2 amplitude: {asc_amps[:,1].min():.2f} to {asc_amps[:,1].max():.2f} pA")
print("(Negative values = hyperpolarizing = spike suppression)")

print("\n" + "=" * 60)
print("Analysis complete")
print("=" * 60)
