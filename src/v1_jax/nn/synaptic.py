"""Synaptic dynamics and post-synaptic current functions.

Implements exponential convolution, PSC dynamics, and synaptic filters.

Reference: Chen et al., Science Advances 2022
Source: /nvmessd/yinzi/Training-data-driven-V1-model/models.py:32-47
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional, Tuple


def exp_convolve(
    tensor: Array,
    decay: float = 0.8,
    reverse: bool = False,
    initializer: Optional[Array] = None,
    axis: int = 0,
) -> Array:
    """Exponential convolution (causal filter) along an axis.

    Implements: y[t] = decay * y[t-1] + x[t]

    This is used for synaptic current dynamics, implementing exponential
    decay filtering of spike trains.

    Args:
        tensor: Input tensor to filter
        decay: Exponential decay factor (0 < decay < 1)
        reverse: If True, filter in reverse direction
        initializer: Initial state (defaults to zeros)
        axis: Axis along which to apply convolution

    Returns:
        Filtered tensor with same shape as input

    Reference:
        Source: models.py:32-47

    Example:
        >>> spikes = jnp.array([[1, 0, 0, 1, 0]])
        >>> filtered = exp_convolve(spikes, decay=0.8, axis=1)
        >>> # filtered ≈ [[1, 0.8, 0.64, 1.51, 1.21]]
    """
    # Move target axis to position 0
    n_dims = tensor.ndim
    perm = list(range(n_dims))
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = jnp.transpose(tensor, perm)

    if initializer is None:
        initializer = jnp.zeros_like(tensor[0])

    def scan_fn(acc, t):
        new_acc = acc * decay + t
        return new_acc, new_acc

    if reverse:
        tensor = jnp.flip(tensor, axis=0)

    _, filtered = jax.lax.scan(scan_fn, initializer, tensor)

    if reverse:
        filtered = jnp.flip(filtered, axis=0)

    # Transpose back
    filtered = jnp.transpose(filtered, perm)
    return filtered


def alpha_synapse(
    spikes: Array,
    tau_syn: float,
    dt: float = 1.0,
) -> Tuple[Array, Array]:
    """Alpha synapse model: double-exponential PSC.

    Implements the alpha function synapse:
        dy1/dt = -y1/tau_syn + spikes
        dy2/dt = (y1 - y2)/tau_syn
        I_syn = y2

    Args:
        spikes: Spike train (batch, time, neurons)
        tau_syn: Synaptic time constant (ms)
        dt: Time step (ms)

    Returns:
        Tuple of (y1, y2) synaptic state variables
    """
    decay = jnp.exp(-dt / tau_syn)

    # First integrator
    y1 = exp_convolve(spikes.astype(jnp.float32), decay=decay, axis=1)

    # Second integrator (alpha function)
    y2 = exp_convolve(y1 * (dt / tau_syn) * decay, decay=decay, axis=1)

    return y1, y2


def psc_dynamics(
    spikes: Array,
    psc_rise: Array,
    psc: Array,
    syn_decay: Array,
    psc_initial: Array,
    dt: float = 1.0,
) -> Tuple[Array, Array]:
    """Update post-synaptic current with dual-exponential dynamics.

    Implements the PSC update from BillehColumn:
        psc_rise_new = syn_decay * psc_rise + input * psc_initial
        psc_new = psc * syn_decay + dt * syn_decay * psc_rise

    Args:
        spikes: Input spikes or synaptic input
        psc_rise: Rising phase of PSC (state)
        psc: Current PSC value (state)
        syn_decay: Synaptic decay factor (per receptor)
        psc_initial: Initial PSC amplitude (per receptor)
        dt: Time step

    Returns:
        Tuple of (new_psc_rise, new_psc)
    """
    new_psc_rise = syn_decay * psc_rise + spikes * psc_initial
    new_psc = psc * syn_decay + dt * syn_decay * psc_rise

    return new_psc_rise, new_psc


def exponential_synapse(
    spikes: Array, tau_syn: float, dt: float = 1.0
) -> Array:
    """Single exponential synapse model.

    Implements: dI/dt = -I/tau_syn + spikes

    Args:
        spikes: Spike train
        tau_syn: Synaptic time constant
        dt: Time step

    Returns:
        Synaptic current
    """
    decay = jnp.exp(-dt / tau_syn)
    return exp_convolve(spikes.astype(jnp.float32), decay=decay, axis=1)


def compute_synaptic_current(
    psc: Array, receptor_weights: Optional[Array] = None
) -> Array:
    """Compute total synaptic current from PSC across receptors.

    Args:
        psc: Post-synaptic currents (batch, neurons, receptors) or (batch, neurons*receptors)
        receptor_weights: Optional weights per receptor type

    Returns:
        Total synaptic current (batch, neurons)
    """
    if psc.ndim == 3:
        # Shape: (batch, neurons, receptors)
        if receptor_weights is not None:
            psc = psc * receptor_weights
        return jnp.sum(psc, axis=-1)
    else:
        # Assume flat: (batch, neurons*receptors), sum is taken elsewhere
        return psc


class SynapticFilter:
    """Configurable synaptic filter for spike processing.

    Supports multiple receptor types with different time constants.

    Attributes:
        tau_syn: Time constants per receptor (ms)
        dt: Simulation time step (ms)
        n_receptors: Number of receptor types
    """

    def __init__(
        self,
        tau_syn: Array,
        dt: float = 1.0,
    ):
        """Initialize synaptic filter.

        Args:
            tau_syn: Synaptic time constants (n_receptors,) or (n_neurons, n_receptors)
            dt: Time step in ms
        """
        self.tau_syn = jnp.asarray(tau_syn)
        self.dt = dt
        self.decay = jnp.exp(-dt / self.tau_syn)

        if self.tau_syn.ndim == 1:
            self.n_receptors = self.tau_syn.shape[0]
        else:
            self.n_receptors = self.tau_syn.shape[-1]

    def init_state(self, batch_size: int, n_neurons: int) -> Tuple[Array, Array]:
        """Initialize synaptic state.

        Args:
            batch_size: Batch dimension
            n_neurons: Number of neurons

        Returns:
            Tuple of (psc_rise, psc) initialized to zeros
        """
        shape = (batch_size, n_neurons, self.n_receptors)
        return jnp.zeros(shape), jnp.zeros(shape)

    def __call__(
        self,
        inputs: Array,
        psc_rise: Array,
        psc: Array,
        psc_initial: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array]:
        """Update synaptic state and compute current.

        Args:
            inputs: Synaptic inputs (batch, neurons, receptors)
            psc_rise: Rising phase state
            psc: Current PSC state
            psc_initial: Initial amplitude (defaults to 1.0)

        Returns:
            Tuple of (synaptic_current, new_psc_rise, new_psc)
        """
        if psc_initial is None:
            psc_initial = jnp.ones_like(self.decay)

        new_psc_rise, new_psc = psc_dynamics(
            inputs, psc_rise, psc, self.decay, psc_initial, self.dt
        )

        current = compute_synaptic_current(new_psc)

        return current, new_psc_rise, new_psc


# Default receptor types for V1 model (AMPA, NMDA, GABA_A, GABA_B)
DEFAULT_TAU_SYN = jnp.array([2.0, 100.0, 6.0, 150.0])  # ms


def create_v1_synaptic_filter(dt: float = 1.0) -> SynapticFilter:
    """Create synaptic filter with V1 model default parameters.

    Uses 4 receptor types: AMPA, NMDA, GABA_A, GABA_B

    Args:
        dt: Time step in ms

    Returns:
        Configured SynapticFilter
    """
    return SynapticFilter(tau_syn=DEFAULT_TAU_SYN, dt=dt)
