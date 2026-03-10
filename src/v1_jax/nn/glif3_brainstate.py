"""GLIF3 neuron model using brainstate for online gradient training.

This module provides a brainstate-based implementation of the GLIF3 neuron model
compatible with braintrace's IODim/D-RTRL algorithms for memory-efficient training.

Based on AlphaBrain implementation and adapted for Billeh V1 network data.
"""

from __future__ import annotations

from typing import Union, Callable, Optional, Dict, Any

import numpy as np
import jax
import jax.numpy as jnp

import brainstate

__all__ = ['GLIF3Brainstate']


class GLIF3Brainstate(brainstate.nn.Neuron):
    """GLIF3 neuron model using brainstate for online gradient training.

    This implementation follows the Allen Institute's GLIF3 specification
    with enhanced alpha synapse functionality for multi-receptor synaptic inputs.
    Compatible with braintrace's IODim algorithm for memory-efficient training.

    The model includes:
    - Membrane dynamics with exponential Euler integration
    - Multiple adaptive spike currents (ASC) with decay and spike responses
    - Alpha-function synapses for synaptic input processing
    - Refractory period with reset mechanisms
    - braintrace-compatible arithmetic (no jnp.where)

    Parameters
    ----------
    size : int
        Number of neurons in the population.
    V_th : array-like
        Spike threshold in mV, shape (n_neurons,)
    g : array-like
        Membrane conductance in nS, shape (n_neurons,)
    E_L : array-like
        Leak reversal potential in mV, shape (n_neurons,)
    C_m : array-like
        Membrane capacitance in pF, shape (n_neurons,)
    V_reset : array-like
        Reset potential in mV, shape (n_neurons,)
    t_ref : array-like
        Refractory period in ms, shape (n_neurons,)
    asc_decay : array-like
        ASC decay rates (k values), shape (n_neurons, 2) or (n_types, 2)
    asc_amps : array-like
        ASC amplitudes in pA, shape (n_neurons, 2) or (n_types, 2)
    tau_syn : array-like
        Synaptic time constants in ms, shape (n_neurons, n_receptors) or (n_types, n_receptors)
    node_type_ids : array-like, optional
        Type ID per neuron for gathering per-type params
    spk_fun : Callable
        Spike activation function for simulation mode
    spk_fun_training : Callable
        Spike activation function for training mode (surrogate gradient)
    mode : str
        'simulation' or 'training'
    """

    def __init__(
        self,
        size: int,
        # Per-neuron parameters
        V_th: Union[float, np.ndarray],
        g: Union[float, np.ndarray],
        E_L: Union[float, np.ndarray],
        C_m: Union[float, np.ndarray],
        V_reset: Union[float, np.ndarray],
        t_ref: Union[float, np.ndarray],
        # ASC parameters (can be per-type or per-neuron)
        asc_decay: np.ndarray,  # (n_types, 2) or (n_neurons, 2)
        asc_amps: np.ndarray,   # (n_types, 2) or (n_neurons, 2)
        # Synaptic parameters
        tau_syn: np.ndarray,    # (n_types, n_receptors) or (n_neurons, n_receptors)
        # Optional type mapping
        node_type_ids: Optional[np.ndarray] = None,
        # Spike functions
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),
        spk_fun_training: Callable = brainstate.surrogate.sigmoid,
        mode: str = 'training',
        # Numerical precision
        precision: int = 32,
        name: Optional[str] = None,
    ) -> None:
        """Initialize GLIF3 neuron model."""
        super().__init__(size, name=name, spk_fun=spk_fun)

        # Set precision
        if precision == 32:
            self.dtype = jnp.float32
        elif precision == 64:
            self.dtype = jnp.float64
        else:
            raise ValueError(f"precision must be 32 or 64, got {precision}")

        self.num_neurons = size
        self.node_type_ids = node_type_ids

        # Gather function for per-type to per-neuron expansion
        def gather(param, type_ids):
            if type_ids is None or param.shape[0] == size:
                return param
            # Ensure type_ids is a proper array (avoid NumPy 2.x compatibility issues)
            type_ids_arr = np.asarray(type_ids, dtype=np.int32)
            return jnp.take(param, type_ids_arr, axis=0)

        # Normalize voltage parameters (V_th - E_L is the scale)
        V_th_arr = jnp.asarray(V_th, dtype=self.dtype)
        E_L_arr = jnp.asarray(E_L, dtype=self.dtype)

        self.voltage_scale = V_th_arr - E_L_arr
        self.voltage_offset = E_L_arr

        # Membrane parameters (already per-neuron)
        self.V_th = V_th_arr
        self.E_L = E_L_arr
        self.g = jnp.asarray(g, dtype=self.dtype)
        self.C_m = jnp.asarray(C_m, dtype=self.dtype)
        self.V_reset = jnp.asarray(V_reset, dtype=self.dtype)
        self.t_ref = jnp.asarray(t_ref, dtype=self.dtype)
        self.tau_m = self.C_m / self.g

        # Normalized membrane parameters
        self.v_th_norm = (self.V_th - self.voltage_offset) / self.voltage_scale
        self.v_reset_norm = (self.V_reset - self.voltage_offset) / self.voltage_scale
        self.e_l_norm = (self.E_L - self.voltage_offset) / self.voltage_scale

        # ASC parameters - gather if needed
        asc_decay_arr = jnp.asarray(asc_decay, dtype=self.dtype)
        asc_amps_arr = jnp.asarray(asc_amps, dtype=self.dtype)

        self.asc_decay = gather(asc_decay_arr, node_type_ids)  # (n_neurons, 2)
        # Scale ASC amplitudes by voltage_scale
        asc_amps_gathered = gather(asc_amps_arr, node_type_ids)  # (n_neurons, 2)
        self.asc_amps = asc_amps_gathered / self.voltage_scale[:, None]  # (n_neurons, 2)
        self.num_asc = self.asc_decay.shape[1]

        # Synaptic parameters - gather if needed
        tau_syn_arr = jnp.asarray(tau_syn, dtype=self.dtype)
        self.tau_syn = gather(tau_syn_arr, node_type_ids)  # (n_neurons, n_receptors)
        self.num_receptors = self.tau_syn.shape[1]

        # Spike function for training
        self.spk_fun_training = spk_fun_training
        self.mode = mode

        # Pre-compute constants
        self._precompute_constants()

    def _precompute_constants(self) -> None:
        """Pre-compute constants for JIT optimization."""
        dt = brainstate.environ.get('dt')
        self._dt = dt

        # Membrane decay factor
        self._decay = jnp.exp(-dt / self.tau_m)

        # Current to voltage factor: converts pA to mV
        # (1/g) * (1 - exp(-dt/tau_m)) where tau_m = C_m / g
        # NOTE: Do NOT divide by voltage_scale here!
        # In original TF/JAX implementation, weights are pre-normalized by voltage_scale
        # in prepare_recurrent_connectivity. Since connectivity_brainstate.py doesn't
        # do that normalization, we keep current_factor in physical units (mV/pA).
        # The membrane equation operates in normalized voltage space, but the
        # synaptic weights provide the correct scaling.
        self._current_factor = (1.0 / self.C_m) * (1.0 - jnp.exp(-dt / self.tau_m)) * self.tau_m

        # Synaptic decay (per neuron, per receptor)
        self._syn_decay = jnp.exp(-dt / self.tau_syn)  # (n_neurons, n_receptors)
        # PSC initial amplitude - also normalize for consistent units
        self._psc_initial = jnp.e / self.tau_syn  # (n_neurons, n_receptors)

        # ASC decay factors
        self._asc_decay_factor = jnp.exp(-dt * self.asc_decay)  # (n_neurons, 2)

        # Refractory period in steps
        ref_ratio = self.t_ref / dt
        nearest = jnp.round(ref_ratio)
        close_to_nearest = (jnp.abs(ref_ratio - nearest) < 1e-9).astype(ref_ratio.dtype)
        ref_ratio = close_to_nearest * nearest + (1.0 - close_to_nearest) * ref_ratio
        self._ref_steps = jnp.maximum(1.0, jnp.ceil(ref_ratio))

    def reset_state(self, batch_size: int = 1, **kwargs) -> None:
        """Reset neuron state variables to initial conditions.

        Note: All HiddenState variables must have the same shape for braintrace
        compatibility. We split PSC states into separate HiddenState per receptor.
        """
        shape = (batch_size, self.num_neurons)

        # Membrane potential (normalized)
        self.V = brainstate.HiddenState(
            jnp.broadcast_to(self.v_reset_norm, shape).astype(self.dtype)
        )

        # Refractory counter
        self.ref_count = brainstate.ShortTermState(
            jnp.zeros(shape, dtype=self.dtype)
        )

        # Adaptive spike currents - separate HiddenState for each ASC type
        self.asc_currents = [
            brainstate.HiddenState(jnp.zeros(shape, dtype=self.dtype))
            for _ in range(self.num_asc)
        ]

        # PSC states - separate HiddenState per receptor for braintrace compatibility
        # Each has shape (batch, n_neurons)
        self.psc_rise = [
            brainstate.HiddenState(jnp.zeros(shape, dtype=self.dtype))
            for _ in range(self.num_receptors)
        ]
        self.psc = [
            brainstate.HiddenState(jnp.zeros(shape, dtype=self.dtype))
            for _ in range(self.num_receptors)
        ]

        # Current spike output
        self._current_spike = brainstate.ShortTermState(jnp.zeros(shape, dtype=self.dtype))

    def update(
        self,
        syn_inputs: jnp.ndarray,
        x: Union[float, jnp.ndarray] = 0.,
    ) -> jnp.ndarray:
        """Update neuron state for one simulation time step.

        Parameters
        ----------
        syn_inputs : jnp.ndarray
            Synaptic inputs, shape (batch, n_neurons, n_receptors) or (batch, n_neurons * n_receptors)
        x : float or jnp.ndarray
            External input current in normalized units (default: 0)

        Returns
        -------
        jnp.ndarray
            Normalized neuron output for gradient-based learning
        """
        batch_size = self.V.value.shape[0]

        # Reshape syn_inputs if needed
        if syn_inputs.ndim == 2 and syn_inputs.shape[1] == self.num_neurons * self.num_receptors:
            syn_inputs = syn_inputs.reshape(batch_size, self.num_neurons, self.num_receptors)

        # Refractory handling - use arithmetic for braintrace compatibility
        in_ref = self.ref_count.value > 0
        in_ref_mask = in_ref.astype(self.dtype)
        self.ref_count.value = in_ref_mask * (self.ref_count.value - 1) + (1.0 - in_ref_mask) * 0.0

        # Stack ASC currents for computation
        asc_array = jnp.stack([s.value for s in self.asc_currents], axis=-1)  # (batch, n_neurons, 2)

        # Stack PSC states for computation (each is (batch, n_neurons))
        psc_rise_array = jnp.stack([s.value for s in self.psc_rise], axis=-1)  # (batch, n_neurons, n_receptors)
        psc_array = jnp.stack([s.value for s in self.psc], axis=-1)  # (batch, n_neurons, n_receptors)

        # PSC dynamics (dual exponential)
        new_psc_rise = self._syn_decay * psc_rise_array + syn_inputs * self._psc_initial
        new_psc = psc_array * self._syn_decay + self._dt * self._syn_decay * psc_rise_array

        # Total synaptic current (sum over receptors)
        I_syn = jnp.sum(psc_array, axis=-1)  # (batch, n_neurons)

        # ASC contribution - only when not in refractory
        asc_contribution = jnp.sum(asc_array, axis=-1)  # (batch, n_neurons)
        total_asc = (1.0 - in_ref_mask) * asc_contribution

        # ASC decay - only when not in refractory
        decayed_asc = asc_array * self._asc_decay_factor  # (batch, n_neurons, 2)
        in_ref_mask_3d = in_ref_mask[:, :, None]
        new_asc = in_ref_mask_3d * asc_array + (1.0 - in_ref_mask_3d) * decayed_asc

        # Membrane potential dynamics
        # V_new = decay * V + current_factor * (I_syn + ASC + g*E_L) + reset_current
        gathered_g_el = self.g * self.e_l_norm
        total_current = I_syn + total_asc + gathered_g_el + x

        V_new = self._decay * self.V.value + self._current_factor * total_current

        # Keep voltage at v_reset when in refractory
        V_candidate = in_ref_mask * self.v_reset_norm + (1.0 - in_ref_mask) * V_new

        # Spike detection and generation
        above_threshold = V_candidate > self.v_th_norm
        can_spike = above_threshold & (~in_ref)

        # Different output for simulation vs training mode
        normalizer = self.v_th_norm - self.e_l_norm

        if self.mode == 'simulation':
            can_spike_mask = can_spike.astype(self.dtype)
            new_V = can_spike_mask * self.v_reset_norm + (1.0 - can_spike_mask) * V_candidate
            spike_output = can_spike_mask
            spike_gate = can_spike

            v_scaled = (new_V - self.v_th_norm) / normalizer
            output = jax.nn.standardize(v_scaled)
        else:
            # Training mode with surrogate gradient
            membrane_normalized = (V_candidate - self.v_th_norm) / normalizer
            spike_output = self.spk_fun_training(membrane_normalized)
            spike_gate = spike_output > 0.5

            reset_amount = spike_output * (self.v_reset_norm - V_candidate)
            new_V = V_candidate + reset_amount
            output = spike_output

        # Update refractory counter
        spike_gate_mask = spike_gate.astype(self.dtype)
        new_ref_count = spike_gate_mask * self._ref_steps + (1.0 - spike_gate_mask) * self.ref_count.value

        # ASC spike response
        spike_gate_mask_3d = spike_gate_mask[:, :, None]
        spike_asc_value = self.asc_amps + new_asc  # Add ASC amplitude on spike
        final_asc = spike_gate_mask_3d * spike_asc_value + (1.0 - spike_gate_mask_3d) * new_asc

        # Update all states
        self.V.value = new_V
        self.ref_count.value = new_ref_count

        # Update PSC states (each receptor separately)
        for i in range(self.num_receptors):
            self.psc_rise[i].value = new_psc_rise[:, :, i]
            self.psc[i].value = new_psc[:, :, i]

        for i in range(self.num_asc):
            self.asc_currents[i].value = final_asc[:, :, i]

        self._current_spike.value = spike_output

        return output

    def get_spike(self) -> jnp.ndarray:
        """Get current spike state."""
        return self._current_spike.value

    def get_voltage(self) -> jnp.ndarray:
        """Get current membrane voltage (physical units)."""
        return self.V.value * self.voltage_scale + self.voltage_offset

    @classmethod
    def from_billeh_network(
        cls,
        network_data: Dict[str, Any],
        dt: float = 1.0,
        mode: str = 'training',
        precision: int = 32,
        name: Optional[str] = None,
    ) -> 'GLIF3Brainstate':
        """Create GLIF3 neurons from Billeh network data.

        Parameters
        ----------
        network_data : dict
            Network data from load_network() containing:
            - n_nodes: Number of neurons
            - node_params: Dict with V_th, g, E_L, C_m, V_reset, t_ref, k, asc_amps, tau_syn
            - node_type_ids: Type index per neuron
        dt : float
            Time step in ms
        mode : str
            'simulation' or 'training'
        precision : int
            32 or 64 bit precision
        name : str, optional
            Module name

        Returns
        -------
        GLIF3Brainstate
            Initialized neuron population
        """
        # Set brainstate dt
        brainstate.environ.set(dt=dt)

        n_neurons = int(network_data['n_nodes'])
        node_params = network_data['node_params']
        node_type_ids = np.asarray(network_data['node_type_ids'])

        # Per-type parameters
        V_th_types = np.asarray(node_params['V_th'])
        g_types = np.asarray(node_params['g'])
        E_L_types = np.asarray(node_params['E_L'])
        C_m_types = np.asarray(node_params['C_m'])
        V_reset_types = np.asarray(node_params['V_reset'])
        t_ref_types = np.asarray(node_params['t_ref'])

        # Gather per-neuron from per-type
        V_th = V_th_types[node_type_ids]
        g = g_types[node_type_ids]
        E_L = E_L_types[node_type_ids]
        C_m = C_m_types[node_type_ids]
        V_reset = V_reset_types[node_type_ids]
        t_ref = t_ref_types[node_type_ids]

        # ASC and synaptic params (keep as per-type, will be gathered in __init__)
        asc_decay = np.asarray(node_params['k'])  # (n_types, 2)
        asc_amps = np.asarray(node_params['asc_amps'])  # (n_types, 2)
        tau_syn = np.asarray(node_params['tau_syn'])  # (n_types, n_receptors)

        return cls(
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
            mode=mode,
            precision=precision,
            name=name,
        )
