"""GLIF3 (Generalized Leaky Integrate-and-Fire 3) neuron cell for JAX.

Pure JAX implementation of the GLIF3 neuron model following the BillehColumn
implementation from Chen et al., Science Advances 2022.

Key features:
- NamedTuple state for JIT compatibility
- Custom surrogate gradient (spike_gauss)
- 4-receptor PSC dynamics (AMPA, NMDA, GABA_A, GABA_B)
- Synaptic delay handling via ring buffer
- Adaptive spike currents (ASC)

Reference: models.py:137-356
"""

from __future__ import annotations

from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
from jax import Array

from .spike_functions import spike_gauss


class GLIF3State(NamedTuple):
    """State variables for GLIF3 neuron population.

    All tensors have shape (batch, n_neurons) unless otherwise noted.

    Attributes:
        z_buf: Spike history buffer for delays, shape (batch, n_neurons * max_delay)
        v: Membrane potential (normalized)
        r: Refractory counter (ms remaining)
        asc_1: Adaptive spike current channel 1
        asc_2: Adaptive spike current channel 2
        psc_rise: PSC rising phase, shape (batch, n_neurons * n_receptors)
        psc: Post-synaptic current, shape (batch, n_neurons * n_receptors)
    """
    z_buf: Array      # (batch, n_neurons * max_delay)
    v: Array          # (batch, n_neurons)
    r: Array          # (batch, n_neurons)
    asc_1: Array      # (batch, n_neurons)
    asc_2: Array      # (batch, n_neurons)
    psc_rise: Array   # (batch, n_neurons * n_receptors)
    psc: Array        # (batch, n_neurons * n_receptors)


class GLIF3Params(NamedTuple):
    """Static parameters for GLIF3 neurons.

    These parameters are derived from node type properties and remain
    constant during simulation.

    Attributes:
        v_reset: Reset potential (normalized), shape (n_neurons,)
        v_th: Threshold potential (normalized), shape (n_neurons,)
        e_l: Leak potential (normalized), shape (n_neurons,)
        t_ref: Refractory period (ms), shape (n_neurons,)
        decay: Membrane decay factor, shape (n_neurons,)
        current_factor: Current-to-voltage factor, shape (n_neurons,)
        syn_decay: Synaptic decay per receptor, shape (n_neurons, n_receptors)
        psc_initial: PSC initial amplitude, shape (n_neurons, n_receptors)
        param_k: ASC decay rates, shape (n_neurons, 2)
        asc_amps: ASC amplitudes, shape (n_neurons, 2)
        param_g: Membrane conductance, shape (n_neurons,)
        voltage_scale: Denormalization scale, shape (n_neurons,)
        voltage_offset: Denormalization offset, shape (n_neurons,)
    """
    v_reset: Array
    v_th: Array
    e_l: Array
    t_ref: Array
    decay: Array
    current_factor: Array
    syn_decay: Array
    psc_initial: Array
    param_k: Array
    asc_amps: Array
    param_g: Array
    voltage_scale: Array
    voltage_offset: Array


class GLIF3Cell:
    """GLIF3 neuron cell implementation for JAX.

    This class provides a JAX-compatible implementation of the GLIF3 neuron
    model with sparse recurrent and input connectivity.

    The implementation follows BillehColumn from the TensorFlow source:
    - Normalized membrane potential
    - 4-receptor PSC dynamics
    - Spike delay buffer
    - Adaptive spike currents

    Example:
        >>> params = GLIF3Cell.from_network(network)
        >>> state = GLIF3Cell.init_state(params, batch_size=32)
        >>> new_state, output = GLIF3Cell.step(params, state, inputs)
    """

    @staticmethod
    def from_network(
        network: Dict[str, Any],
        dt: float = 1.0,
        gauss_std: float = 0.5,
        dampening_factor: float = 0.3,
        max_delay: int = 5,
    ) -> Tuple[GLIF3Params, Dict[str, Any]]:
        """Create GLIF3 parameters from Billeh network data.

        Args:
            network: Network dict from load_network()
            dt: Time step in ms
            gauss_std: Gaussian pseudo-derivative width
            dampening_factor: Surrogate gradient amplitude
            max_delay: Maximum synaptic delay in time steps

        Returns:
            Tuple of (params, metadata) where metadata contains:
            - n_neurons: Number of neurons
            - n_receptors: Number of receptor types
            - max_delay: Effective maximum delay
            - node_type_ids: Neuron type indices
            - gauss_std: Surrogate gradient width
            - dampening_factor: Surrogate gradient amplitude
            - dt: Time step
        """
        node_params = network['node_params']
        node_type_ids = network['node_type_ids']
        n_neurons = int(network['n_nodes'])

        # Normalize voltage parameters
        voltage_scale = node_params['V_th'] - node_params['E_L']
        voltage_offset = node_params['E_L']

        # Per-type normalized parameters
        v_th_type = (node_params['V_th'] - voltage_offset) / voltage_scale
        e_l_type = (node_params['E_L'] - voltage_offset) / voltage_scale
        v_reset_type = (node_params['V_reset'] - voltage_offset) / voltage_scale
        asc_amps_type = node_params['asc_amps'] / voltage_scale[..., None]

        # Time constants
        tau = node_params['C_m'] / node_params['g']
        decay_type = jnp.exp(-dt / tau)
        # Current to voltage factor (matching TF: no voltage_scale normalization)
        # Note: weights are already normalized by voltage_scale in prepare_*_connectivity
        current_factor_type = 1 / node_params['C_m'] * (1 - jnp.exp(-dt / tau)) * tau

        # Synaptic parameters (4 receptors)
        tau_syn = jnp.array(node_params['tau_syn'])
        syn_decay_type = jnp.exp(-dt / tau_syn)
        psc_initial_type = jnp.e / tau_syn

        n_receptors = tau_syn.shape[1]

        # Gather per-neuron parameters
        def gather(param_type):
            return jnp.take(param_type, node_type_ids, axis=0)

        params = GLIF3Params(
            v_reset=gather(v_reset_type),
            v_th=gather(v_th_type),
            e_l=gather(e_l_type),
            t_ref=gather(jnp.asarray(node_params['t_ref'])),
            decay=gather(decay_type),
            current_factor=gather(current_factor_type),
            syn_decay=gather(syn_decay_type),
            psc_initial=gather(psc_initial_type),
            param_k=gather(jnp.asarray(node_params['k'])),
            asc_amps=gather(asc_amps_type),
            param_g=gather(jnp.asarray(node_params['g'])),
            voltage_scale=gather(voltage_scale),
            voltage_offset=gather(voltage_offset),
        )

        metadata = {
            'n_neurons': n_neurons,
            'n_receptors': n_receptors,
            'max_delay': max_delay,
            'node_type_ids': node_type_ids,
            'gauss_std': gauss_std,
            'dampening_factor': dampening_factor,
            'dt': dt,
        }

        return params, metadata

    @staticmethod
    def init_state(
        n_neurons: int,
        n_receptors: int,
        max_delay: int,
        batch_size: int,
        params: Optional[GLIF3Params] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> GLIF3State:
        """Initialize GLIF3 state to zero/rest.

        Args:
            n_neurons: Number of neurons
            n_receptors: Number of receptor types (typically 4)
            max_delay: Maximum synaptic delay
            batch_size: Batch dimension
            params: Optional parameters for custom initialization
            dtype: Data type for state arrays

        Returns:
            Initialized GLIF3State
        """
        z_buf = jnp.zeros((batch_size, n_neurons * max_delay), dtype=dtype)

        if params is not None:
            # Initialize voltage near reset
            v_init = params.v_reset * 1.0 + params.v_th * 0.0
            v = jnp.broadcast_to(v_init, (batch_size, n_neurons)).astype(dtype)
        else:
            v = jnp.zeros((batch_size, n_neurons), dtype=dtype)

        r = jnp.zeros((batch_size, n_neurons), dtype=dtype)
        asc_1 = jnp.zeros((batch_size, n_neurons), dtype=dtype)
        asc_2 = jnp.zeros((batch_size, n_neurons), dtype=dtype)
        psc_rise = jnp.zeros((batch_size, n_neurons * n_receptors), dtype=dtype)
        psc = jnp.zeros((batch_size, n_neurons * n_receptors), dtype=dtype)

        return GLIF3State(z_buf, v, r, asc_1, asc_2, psc_rise, psc)

    @staticmethod
    def random_state(
        n_neurons: int,
        n_receptors: int,
        max_delay: int,
        batch_size: int,
        params: GLIF3Params,
        key: Array,
        dtype: jnp.dtype = jnp.float32,
    ) -> GLIF3State:
        """Initialize GLIF3 state with random values.

        Follows the random_state method from TF BillehColumn.

        Args:
            n_neurons: Number of neurons
            n_receptors: Number of receptor types
            max_delay: Maximum synaptic delay
            batch_size: Batch dimension
            params: GLIF3 parameters
            key: JAX random key
            dtype: Data type

        Returns:
            Randomly initialized GLIF3State
        """
        keys = jax.random.split(key, 7)

        z_buf = jax.random.randint(
            keys[0], (batch_size, n_neurons * max_delay), 0, 2
        ).astype(dtype)

        v = jax.random.uniform(
            keys[1], (batch_size, n_neurons),
            minval=params.v_reset, maxval=params.v_th, dtype=dtype
        )

        r = jnp.zeros((batch_size, n_neurons), dtype=dtype)

        asc_1 = jax.random.normal(keys[2], (batch_size, n_neurons), dtype=dtype)
        asc_1 = asc_1 * 1.75 - 0.28

        asc_2 = jax.random.normal(keys[3], (batch_size, n_neurons), dtype=dtype)
        asc_2 = asc_2 * 1.75 - 0.28

        psc_rise = jax.random.normal(
            keys[4], (batch_size, n_neurons * n_receptors), dtype=dtype
        ) * 0.77 + 0.29

        psc = jax.random.normal(
            keys[5], (batch_size, n_neurons * n_receptors), dtype=dtype
        ) * 3.19 + 1.17

        return GLIF3State(z_buf, v, r, asc_1, asc_2, psc_rise, psc)


def glif3_step(
    params: GLIF3Params,
    state: GLIF3State,
    inputs: Array,
    recurrent_current: Array,
    n_neurons: int,
    n_receptors: int,
    max_delay: int,
    dt: float,
    gauss_std: float,
    dampening_factor: float,
) -> Tuple[GLIF3State, Array, Array]:
    """Single timestep update for GLIF3 neuron population.

    Note: This function is NOT JIT-compiled directly. Use make_glif3_step_fn()
    to create a JIT-compiled version with fixed dimensions.

    Args:
        params: Static neuron parameters
        state: Current state
        inputs: External input current (batch, n_neurons * n_receptors)
        recurrent_current: Recurrent input from sparse matmul (batch, n_neurons * n_receptors)
        n_neurons: Number of neurons
        n_receptors: Number of receptor types
        max_delay: Maximum delay
        dt: Time step
        gauss_std: Surrogate gradient width
        dampening_factor: Surrogate gradient amplitude

    Returns:
        Tuple of (new_state, spikes, scaled_voltage)
    """
    batch_size = state.v.shape[0]
    z_buf, v, r, asc_1, asc_2, psc_rise, psc = state

    # Get previous spikes from buffer (oldest entry)
    shaped_z_buf = z_buf.reshape(batch_size, max_delay, n_neurons)
    prev_z = shaped_z_buf[:, 0]

    # Reshape PSC for receptor dimension
    psc_rise_3d = psc_rise.reshape(batch_size, n_neurons, n_receptors)
    psc_3d = psc.reshape(batch_size, n_neurons, n_receptors)

    # Total synaptic input (recurrent + external)
    rec_inputs = recurrent_current + inputs
    rec_inputs = rec_inputs.reshape(batch_size, n_neurons, n_receptors)

    # PSC dynamics (dual exponential)
    syn_decay = params.syn_decay  # (n_neurons, n_receptors)
    psc_initial = params.psc_initial  # (n_neurons, n_receptors)

    new_psc_rise_3d = syn_decay * psc_rise_3d + rec_inputs * psc_initial
    new_psc_3d = psc_3d * syn_decay + dt * syn_decay * psc_rise_3d

    # Refractory period update
    new_r = jnp.maximum(r + prev_z * params.t_ref - dt, 0.0)

    # Adaptive spike current dynamics
    k = params.param_k  # (n_neurons, 2)
    asc_amps = params.asc_amps  # (n_neurons, 2)

    new_asc_1 = jnp.exp(-dt * k[:, 0]) * asc_1 + prev_z * asc_amps[:, 0]
    new_asc_2 = jnp.exp(-dt * k[:, 1]) * asc_2 + prev_z * asc_amps[:, 1]

    # Membrane potential dynamics
    reset_current = prev_z * (params.v_reset - params.v_th)
    input_current = jnp.sum(psc_3d, axis=-1)  # (batch, n_neurons)
    decayed_v = params.decay * v

    # Total current: synaptic + ASC + leak
    gathered_g_el = params.param_g * params.e_l
    c1 = input_current + asc_1 + asc_2 + gathered_g_el
    new_v = decayed_v + params.current_factor * c1 + reset_current

    # Spike generation with surrogate gradient
    normalizer = params.v_th - params.e_l
    v_scaled = (new_v - params.v_th) / normalizer

    new_z = spike_gauss(v_scaled, gauss_std, dampening_factor)

    # Suppress spikes during refractory period
    new_z = jnp.where(new_r > 0.0, jnp.zeros_like(new_z), new_z)

    # Update spike buffer (shift and insert new spikes)
    new_shaped_z_buf = jnp.concatenate(
        [new_z[:, None, :], shaped_z_buf[:, :-1, :]], axis=1
    )
    new_z_buf = new_shaped_z_buf.reshape(batch_size, n_neurons * max_delay)

    # Flatten PSC for state
    new_psc_rise = new_psc_rise_3d.reshape(batch_size, n_neurons * n_receptors)
    new_psc = new_psc_3d.reshape(batch_size, n_neurons * n_receptors)

    # Scale voltage back to physical units
    scaled_v = new_v * params.voltage_scale + params.voltage_offset

    new_state = GLIF3State(
        new_z_buf, new_v, new_r, new_asc_1, new_asc_2, new_psc_rise, new_psc
    )

    return new_state, new_z, scaled_v


def make_glif3_step_fn(
    params: GLIF3Params,
    n_neurons: int,
    n_receptors: int,
    max_delay: int,
    dt: float = 1.0,
    gauss_std: float = 0.5,
    dampening_factor: float = 0.3,
):
    """Create a specialized GLIF3 step function with fixed parameters.

    This returns a closure that captures the static parameters for
    efficient JIT compilation.

    Args:
        params: Neuron parameters
        n_neurons: Number of neurons
        n_receptors: Number of receptor types
        max_delay: Maximum delay
        dt: Time step
        gauss_std: Surrogate gradient width
        dampening_factor: Surrogate gradient amplitude

    Returns:
        step_fn(state, inputs, recurrent_current) -> (new_state, spikes, voltage)
    """
    @jax.jit
    def step_fn(
        state: GLIF3State,
        inputs: Array,
        recurrent_current: Array,
    ) -> Tuple[GLIF3State, Array, Array]:
        return glif3_step(
            params, state, inputs, recurrent_current,
            n_neurons, n_receptors, max_delay, dt, gauss_std, dampening_factor
        )
    return step_fn


def glif3_unroll(
    params: GLIF3Params,
    initial_state: GLIF3State,
    inputs: Array,
    recurrent_fn,
    n_neurons: int,
    n_receptors: int,
    max_delay: int,
    dt: float = 1.0,
    gauss_std: float = 0.5,
    dampening_factor: float = 0.3,
) -> Tuple[GLIF3State, Array, Array]:
    """Unroll GLIF3 dynamics over time using lax.scan.

    Args:
        params: Neuron parameters
        initial_state: Initial state
        inputs: External inputs, shape (time, batch, n_neurons * n_receptors)
        recurrent_fn: Function (state, z_buf) -> recurrent_current
        n_neurons: Number of neurons
        n_receptors: Number of receptor types
        max_delay: Maximum delay
        dt: Time step
        gauss_std: Surrogate gradient width
        dampening_factor: Surrogate gradient amplitude

    Returns:
        Tuple of (final_state, all_spikes, all_voltages)
        where all_spikes has shape (time, batch, n_neurons)
        and all_voltages has shape (time, batch, n_neurons)
    """
    def scan_fn(state, inp):
        # Get recurrent current from connectivity
        rec_current = recurrent_fn(state.z_buf)

        new_state, spikes, voltage = glif3_step(
            params, state, inp, rec_current,
            n_neurons, n_receptors, max_delay, dt, gauss_std, dampening_factor
        )

        return new_state, (spikes, voltage)

    final_state, (all_spikes, all_voltages) = jax.lax.scan(
        scan_fn, initial_state, inputs
    )

    return final_state, all_spikes, all_voltages


def glif3_unroll_checkpointed(
    params: GLIF3Params,
    initial_state: GLIF3State,
    inputs: Array,
    recurrent_fn,
    n_neurons: int,
    n_receptors: int,
    max_delay: int,
    dt: float = 1.0,
    gauss_std: float = 0.5,
    dampening_factor: float = 0.3,
    checkpoint_every_n_steps: int = 50,
) -> Tuple[GLIF3State, Array, Array]:
    """Unroll GLIF3 dynamics with gradient checkpointing to reduce memory.

    This function trades compute for memory by not storing intermediate
    activations during forward pass. Instead, it recomputes them during
    backward pass. The sequence is divided into segments, and each segment
    is checkpointed.

    Args:
        params: Neuron parameters
        initial_state: Initial state
        inputs: External inputs, shape (time, batch, n_neurons * n_receptors)
        recurrent_fn: Function (z_buf) -> recurrent_current
        n_neurons: Number of neurons
        n_receptors: Number of receptor types
        max_delay: Maximum delay
        dt: Time step
        gauss_std: Surrogate gradient width
        dampening_factor: Surrogate gradient amplitude
        checkpoint_every_n_steps: Number of steps per checkpoint segment.
            Smaller values save more memory but increase recomputation.
            Recommended: 50-100 for ~51K neurons.

    Returns:
        Tuple of (final_state, all_spikes, all_voltages)
        where all_spikes has shape (time, batch, n_neurons)
        and all_voltages has shape (time, batch, n_neurons)
    """
    seq_len = inputs.shape[0]

    # Define the step function (same as in glif3_unroll)
    def scan_fn(state, inp):
        rec_current = recurrent_fn(state.z_buf)
        new_state, spikes, voltage = glif3_step(
            params, state, inp, rec_current,
            n_neurons, n_receptors, max_delay, dt, gauss_std, dampening_factor
        )
        return new_state, (spikes, voltage)

    # Define a checkpointed segment function
    @jax.checkpoint
    def process_segment(state, segment_inputs):
        """Process a segment of timesteps with checkpointing.

        The @jax.checkpoint decorator causes JAX to:
        - Not save intermediate activations during forward pass
        - Recompute them during backward pass
        """
        final_state, (spikes, voltages) = jax.lax.scan(
            scan_fn, state, segment_inputs
        )
        return final_state, (spikes, voltages)

    # If sequence is short enough, just use one checkpoint
    if seq_len <= checkpoint_every_n_steps:
        return process_segment(initial_state, inputs)[0], *process_segment(initial_state, inputs)[1]

    # Split inputs into segments
    n_full_segments = seq_len // checkpoint_every_n_steps
    remainder = seq_len % checkpoint_every_n_steps

    # Process full segments
    def segment_scan_fn(state, segment_idx):
        """Process one segment."""
        start = segment_idx * checkpoint_every_n_steps
        segment_inputs = jax.lax.dynamic_slice(
            inputs,
            (start, 0, 0),
            (checkpoint_every_n_steps, inputs.shape[1], inputs.shape[2])
        )
        new_state, (spikes, voltages) = process_segment(state, segment_inputs)
        return new_state, (spikes, voltages)

    # Use scan over segments for efficiency
    state = initial_state
    all_spikes_list = []
    all_voltages_list = []

    # Process full segments using a Python loop (JAX will trace it)
    # We use a for loop here because the number of segments is known at trace time
    for i in range(n_full_segments):
        start = i * checkpoint_every_n_steps
        segment_inputs = inputs[start:start + checkpoint_every_n_steps]
        state, (seg_spikes, seg_voltages) = process_segment(state, segment_inputs)
        all_spikes_list.append(seg_spikes)
        all_voltages_list.append(seg_voltages)

    # Process remainder if any
    if remainder > 0:
        remainder_inputs = inputs[n_full_segments * checkpoint_every_n_steps:]
        # For remainder, we still checkpoint but with a smaller segment
        @jax.checkpoint
        def process_remainder(state, rem_inputs):
            return jax.lax.scan(scan_fn, state, rem_inputs)

        state, (rem_spikes, rem_voltages) = process_remainder(state, remainder_inputs)
        all_spikes_list.append(rem_spikes)
        all_voltages_list.append(rem_voltages)

    # Concatenate all outputs
    all_spikes = jnp.concatenate(all_spikes_list, axis=0)
    all_voltages = jnp.concatenate(all_voltages_list, axis=0)

    return state, all_spikes, all_voltages
