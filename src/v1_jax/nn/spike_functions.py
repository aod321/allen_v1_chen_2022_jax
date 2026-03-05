"""Spike functions with custom gradients for surrogate gradient learning.

This module implements the spike generation function with Gaussian pseudo-derivative
for training spiking neural networks via backpropagation.

Reference: Chen et al., Science Advances 2022
Source: /nvmessd/yinzi/Training-data-driven-V1-model/models.py:5-29
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array


def gauss_pseudo(v_scaled: Array, sigma: float, amplitude: float) -> Array:
    """Gaussian pseudo-derivative for surrogate gradient.

    Args:
        v_scaled: Scaled membrane voltage (v - v_th) / normalizer
        sigma: Width of the Gaussian
        amplitude: Peak amplitude (dampening factor)

    Returns:
        Pseudo-derivative value: exp(-v_scaled^2/sigma^2) * amplitude
    """
    return jnp.exp(-jnp.square(v_scaled) / jnp.square(sigma)) * amplitude


def pseudo_derivative(v_scaled: Array, dampening_factor: float) -> Array:
    """Piecewise linear pseudo-derivative (alternative to Gaussian).

    Args:
        v_scaled: Scaled membrane voltage
        dampening_factor: Maximum gradient value at v=0

    Returns:
        Pseudo-derivative: dampening_factor * max(1 - |v_scaled|, 0)
    """
    return dampening_factor * jnp.maximum(1 - jnp.abs(v_scaled), 0)


@jax.custom_vjp
def spike_gauss(v_scaled: Array, sigma: float, amplitude: float) -> Array:
    """Heaviside spike function with Gaussian surrogate gradient.

    Forward pass: Binary spike (v > 0 -> 1, else 0)
    Backward pass: Gaussian pseudo-derivative for gradient approximation

    Args:
        v_scaled: Scaled membrane voltage (v - v_th) / normalizer
        sigma: Width of the Gaussian pseudo-derivative (default ~0.28)
        amplitude: Peak amplitude / dampening factor (default ~0.5)

    Returns:
        Binary spike tensor (0.0 or 1.0)

    Example:
        >>> v = jnp.array([-1.0, 0.0, 0.5, 1.0])
        >>> spikes = spike_gauss(v, 0.28, 0.5)
        >>> # spikes = [0., 0., 1., 1.]
    """
    return (v_scaled > 0.0).astype(jnp.float32)


def _spike_gauss_fwd(
    v_scaled: Array, sigma: float, amplitude: float
) -> Tuple[Array, Tuple[Array, float, float]]:
    """Forward pass for spike_gauss custom VJP.

    Returns:
        Tuple of (output, residuals for backward pass)
    """
    z = spike_gauss(v_scaled, sigma, amplitude)
    return z, (v_scaled, sigma, amplitude)


def _spike_gauss_bwd(
    res: Tuple[Array, float, float], g: Array
) -> Tuple[Array, None, None]:
    """Backward pass for spike_gauss custom VJP.

    Uses Gaussian pseudo-derivative for surrogate gradient.

    Args:
        res: Residuals from forward pass (v_scaled, sigma, amplitude)
        g: Upstream gradient (dy)

    Returns:
        Tuple of gradients: (grad_v_scaled, None, None)
        - sigma and amplitude have no gradients (treated as constants)
    """
    v_scaled, sigma, amplitude = res
    grad_v_scaled = g * gauss_pseudo(v_scaled, sigma, amplitude)
    return (grad_v_scaled, None, None)


# Register the custom VJP
spike_gauss.defvjp(_spike_gauss_fwd, _spike_gauss_bwd)


# Alternative spike functions for experimentation


@jax.custom_vjp
def spike_piecewise(v_scaled: Array, dampening_factor: float) -> Array:
    """Spike function with piecewise linear surrogate gradient.

    Args:
        v_scaled: Scaled membrane voltage
        dampening_factor: Maximum gradient value at v=0

    Returns:
        Binary spike tensor
    """
    return (v_scaled > 0.0).astype(jnp.float32)


def _spike_piecewise_fwd(
    v_scaled: Array, dampening_factor: float
) -> Tuple[Array, Tuple[Array, float]]:
    z = spike_piecewise(v_scaled, dampening_factor)
    return z, (v_scaled, dampening_factor)


def _spike_piecewise_bwd(
    res: Tuple[Array, float], g: Array
) -> Tuple[Array, None]:
    v_scaled, dampening_factor = res
    grad_v_scaled = g * pseudo_derivative(v_scaled, dampening_factor)
    return (grad_v_scaled, None)


spike_piecewise.defvjp(_spike_piecewise_fwd, _spike_piecewise_bwd)


@jax.custom_vjp
def spike_sigmoid(v_scaled: Array, beta: float = 10.0) -> Array:
    """Spike function with sigmoid surrogate gradient.

    Uses sigmoid derivative as surrogate: beta * sigmoid(beta*v) * (1 - sigmoid(beta*v))

    Args:
        v_scaled: Scaled membrane voltage
        beta: Steepness of sigmoid (higher = sharper)

    Returns:
        Binary spike tensor
    """
    return (v_scaled > 0.0).astype(jnp.float32)


def _spike_sigmoid_fwd(
    v_scaled: Array, beta: float
) -> Tuple[Array, Tuple[Array, float]]:
    z = spike_sigmoid(v_scaled, beta)
    return z, (v_scaled, beta)


def _spike_sigmoid_bwd(
    res: Tuple[Array, float], g: Array
) -> Tuple[Array, None]:
    v_scaled, beta = res
    sig = jax.nn.sigmoid(beta * v_scaled)
    grad_v_scaled = g * beta * sig * (1 - sig)
    return (grad_v_scaled, None)


spike_sigmoid.defvjp(_spike_sigmoid_fwd, _spike_sigmoid_bwd)
