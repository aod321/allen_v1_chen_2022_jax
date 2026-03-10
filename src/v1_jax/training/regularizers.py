"""Regularizers for V1 model training.

Implements voltage regularization, stiff regularization, and spike rate regularization.

Reference: Chen et al., Science Advances 2022
Source: /nvmessd/yinzi/Training-data-driven-V1-model/models.py:389-434
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional

from .loss_functions import spike_rate_distribution_loss


def voltage_regularization(
    voltages: Array,
    v_th: Array,
    v_reset: Array,
    voltage_cost: float = 1e-5,
    voltage_scale: float = 1.0,
    voltage_offset: float = 0.0,
) -> Array:
    """Penalize membrane voltages outside physiological range.

    Encourages voltages to stay between v_reset and v_th by adding
    quadratic penalties for violations.

    Args:
        voltages: Membrane voltage tensor (batch, time, neurons)
        v_th: Spike threshold voltage
        v_reset: Reset voltage after spike
        voltage_cost: Regularization strength
        voltage_scale: Voltage scaling factor (for denormalization)
        voltage_offset: Voltage offset (for denormalization)

    Returns:
        Voltage regularization loss (scalar)

    Reference:
        Source: models.py:389-399
    """
    # Denormalize if needed
    voltage_32 = (voltages - voltage_offset) / voltage_scale

    # Penalty for voltages above threshold
    v_pos = jnp.square(jax.nn.relu(voltage_32 - 1.0))

    # Penalty for voltages below reset (using -1.0 as normalized reset)
    v_neg = jnp.square(jax.nn.relu(-voltage_32 + 1.0))

    # Sum over neurons, mean over batch and time - matches TF implementation
    voltage_loss = jnp.mean(jnp.sum(v_pos + v_neg, axis=-1))

    return voltage_cost * voltage_loss


def voltage_regularization_v2(
    voltages: Array,
    v_th: Array,
    v_reset: Array,
    voltage_cost: float = 0.01,
) -> Array:
    """Alternative voltage regularization using actual voltage values.

    More direct penalty based on actual threshold and reset voltages.

    Args:
        voltages: Membrane voltage tensor (batch, time, neurons)
        v_th: Spike threshold voltage (per neuron)
        v_reset: Reset voltage after spike (per neuron)
        voltage_cost: Regularization strength

    Returns:
        Voltage regularization loss (scalar)

    Reference:
        Source: models.py:425-431
    """
    diff = v_th - v_reset

    # Penalty for voltages above threshold
    v_pos = jnp.square(jnp.clip(jax.nn.relu(voltages - v_th), 0.0, 1.0))

    # Penalty for voltages too far below reset
    v_neg = jnp.square(jnp.clip(jax.nn.relu(-voltages + v_reset - diff), 0.0, 1.0))

    # Sum over neurons, mean over batch and time - matches TF implementation
    voltage_loss = jnp.mean(jnp.sum(v_pos + v_neg, axis=-1))

    return voltage_cost * voltage_loss


def stiff_regularization(
    weights: Array, initial_weights: Array, strength: float = 0.01
) -> Array:
    """Stiff regularization: Penalize deviation from initial weights.

    Encourages weights to stay close to their initial values,
    promoting stability during training.

    Args:
        weights: Current weight values
        initial_weights: Reference initial weights
        strength: Regularization strength

    Returns:
        Stiff regularization loss (scalar)
    """
    return strength * jnp.sum(jnp.square(weights - initial_weights))


def l2_regularization(weights: Array, strength: float = 0.01) -> Array:
    """Standard L2 (weight decay) regularization.

    Args:
        weights: Weight tensor
        strength: Regularization strength

    Returns:
        L2 penalty (scalar)
    """
    return strength * jnp.sum(jnp.square(weights))


def l1_regularization(weights: Array, strength: float = 0.01) -> Array:
    """L1 (sparsity) regularization.

    Args:
        weights: Weight tensor
        strength: Regularization strength

    Returns:
        L1 penalty (scalar)
    """
    return strength * jnp.sum(jnp.abs(weights))


class SpikeRateDistributionRegularizer:
    """Regularizer for matching spike rate distribution to target.

    Maintains target firing rate distribution and applies Huber quantile loss.

    Attributes:
        target_rates: Target firing rate distribution (sorted)
        rate_cost: Regularization strength
    """

    def __init__(self, target_rates: Array, rate_cost: float = 0.5):
        """Initialize spike rate regularizer.

        Args:
            target_rates: Target firing rates per neuron (sorted ascending)
            rate_cost: Regularization strength
        """
        self.target_rates = target_rates
        self.rate_cost = rate_cost

    def __call__(self, spikes: Array, key: jax.random.PRNGKey) -> Array:
        """Compute spike rate distribution loss.

        Args:
            spikes: Spike tensor (batch, time, neurons)
            key: JAX random key for shuffling

        Returns:
            Regularization loss (scalar)
        """
        loss = spike_rate_distribution_loss(spikes, self.target_rates, key)
        return self.rate_cost * loss


class SpikeVoltageRegularizer:
    """Combined spike rate and voltage regularizer.

    Enforces both target firing rates and physiological voltage bounds.

    Attributes:
        rate_cost: Spike rate regularization strength
        voltage_cost: Voltage regularization strength
        target_rate: Target mean firing rate
        v_th: Spike threshold voltage (per neuron)
        v_reset: Reset voltage (per neuron)
    """

    def __init__(
        self,
        v_th: Array,
        v_reset: Array,
        rate_cost: float = 0.1,
        voltage_cost: float = 0.01,
        target_rate: float = 0.02,
    ):
        """Initialize combined regularizer.

        Args:
            v_th: Spike threshold voltage
            v_reset: Reset voltage after spike
            rate_cost: Spike rate regularization strength
            voltage_cost: Voltage regularization strength
            target_rate: Target mean firing rate
        """
        self.v_th = v_th
        self.v_reset = v_reset
        self.rate_cost = rate_cost
        self.voltage_cost = voltage_cost
        self.target_rate = target_rate

    def __call__(
        self, spikes: Array, voltages: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Compute combined regularization loss.

        Args:
            spikes: Spike tensor (batch, time, neurons)
            voltages: Voltage tensor (batch, time, neurons)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Spike rate loss
        rate = jnp.mean(spikes.astype(jnp.float32), axis=(0, 1))
        rate_loss = jnp.sum(jnp.square(rate - self.target_rate)) * self.rate_cost

        # Voltage loss
        voltage_loss = voltage_regularization_v2(
            voltages, self.v_th, self.v_reset, self.voltage_cost
        )

        total_loss = rate_loss + voltage_loss

        metrics = {
            "rate": jnp.mean(rate),
            "rate_loss": rate_loss,
            "voltage_loss": voltage_loss,
        }

        return total_loss, metrics


def activity_regularization(
    activations: Array, l1: float = 0.0, l2: float = 0.0
) -> Array:
    """Activity regularization on neuron activations.

    Args:
        activations: Neuron activation tensor
        l1: L1 regularization strength (sparsity)
        l2: L2 regularization strength (activity magnitude)

    Returns:
        Activity regularization loss
    """
    loss = 0.0
    if l1 > 0:
        loss = loss + l1 * jnp.mean(jnp.abs(activations))
    if l2 > 0:
        loss = loss + l2 * jnp.mean(jnp.square(activations))
    return loss
