"""Constraints for neural network weights.

Implements Dale's law and other biological constraints for the V1 model.

Reference: Chen et al., Science Advances 2022
"""

import jax
import jax.numpy as jnp
from jax import Array


def apply_dale_constraint(weights: Array, is_excitatory: Array) -> Array:
    """Enforce Dale's law on weights.

    Dale's law: A neuron is either excitatory (positive weights)
    or inhibitory (negative weights), never both.

    Args:
        weights: Weight matrix to constrain
        is_excitatory: Boolean mask, True for excitatory neurons

    Returns:
        Constrained weights with correct signs

    Example:
        >>> weights = jnp.array([-1., 2., -3., 4.])
        >>> is_exc = jnp.array([True, True, False, False])
        >>> constrained = apply_dale_constraint(weights, is_exc)
        >>> # constrained = [0., 2., -3., 0.]
    """
    positive = jax.nn.relu(weights)
    negative = -jax.nn.relu(-weights)
    return jnp.where(is_excitatory, positive, negative)


def dale_law_projection(
    weights: Array, is_excitatory: Array, eps: float = 0.0
) -> Array:
    """Project weights to satisfy Dale's law constraint.

    Similar to apply_dale_constraint but with optional epsilon threshold.

    Args:
        weights: Weight matrix to project
        is_excitatory: Boolean mask for excitatory neurons
        eps: Minimum absolute weight value (optional)

    Returns:
        Projected weights satisfying Dale's law
    """
    constrained = apply_dale_constraint(weights, is_excitatory)
    if eps > 0:
        # Ensure minimum weight magnitude
        constrained = jnp.where(
            jnp.abs(constrained) < eps,
            jnp.zeros_like(constrained),
            constrained,
        )
    return constrained


class SignedConstraint:
    """Constraint that enforces sign based on initial weight sign.

    For use with optimizer weight updates. Maintains the sign of weights
    as they were initialized.

    Attributes:
        sign: The sign of initial weights (+1, -1, or 0)
    """

    def __init__(self, initial_weights: Array):
        """Initialize with reference weights.

        Args:
            initial_weights: Initial weight values to determine signs
        """
        self.sign = jnp.sign(initial_weights)

    def __call__(self, weights: Array) -> Array:
        """Apply signed constraint.

        Args:
            weights: Current weights to constrain

        Returns:
            Weights with enforced sign from initialization
        """
        # For positive initial: keep only positive part
        # For negative initial: keep only negative part
        # For zero initial: keep as-is
        positive = jax.nn.relu(weights)
        negative = -jax.nn.relu(-weights)

        return jnp.where(
            self.sign > 0,
            positive,
            jnp.where(self.sign < 0, negative, weights),
        )


class SparseSignedConstraint:
    """Signed constraint for sparse weights with explicit indices.

    Used for sparse connectivity matrices where weights should
    maintain their initial signs.

    Attributes:
        sign: Sign of each weight in the sparse matrix
        indices: Sparse matrix indices (for reference)
    """

    def __init__(self, initial_weights: Array, indices: Array):
        """Initialize with sparse weight reference.

        Args:
            initial_weights: Initial sparse weight values
            indices: Sparse matrix indices (2D: [n_connections, 2])
        """
        self.sign = jnp.sign(initial_weights)
        self.indices = indices

    def __call__(self, weights: Array) -> Array:
        """Apply signed constraint to sparse weights.

        Args:
            weights: Current sparse weights

        Returns:
            Constrained sparse weights
        """
        positive = jax.nn.relu(weights)
        negative = -jax.nn.relu(-weights)

        return jnp.where(
            self.sign > 0,
            positive,
            jnp.where(self.sign < 0, negative, weights),
        )


def apply_weight_bounds(
    weights: Array, min_val: float = None, max_val: float = None
) -> Array:
    """Clip weights to specified bounds.

    Args:
        weights: Weights to bound
        min_val: Minimum allowed value (None = no lower bound)
        max_val: Maximum allowed value (None = no upper bound)

    Returns:
        Bounded weights
    """
    if min_val is not None:
        weights = jnp.maximum(weights, min_val)
    if max_val is not None:
        weights = jnp.minimum(weights, max_val)
    return weights


def soft_sign_constraint(weights: Array, is_excitatory: Array, alpha: float = 10.0) -> Array:
    """Soft version of Dale's law using sigmoid.

    Instead of hard clipping, uses smooth functions to encourage
    correct signs. Useful during early training.

    Args:
        weights: Weights to constrain
        is_excitatory: Boolean mask for excitatory neurons
        alpha: Steepness of transition (higher = sharper)

    Returns:
        Softly constrained weights
    """
    # Sigmoid-based soft constraint
    # For excitatory: weights * sigmoid(alpha * weights)
    # For inhibitory: weights * (1 - sigmoid(alpha * weights))
    sig = jax.nn.sigmoid(alpha * weights)

    exc_weights = weights * sig
    inh_weights = weights * (1 - sig)

    return jnp.where(is_excitatory, exc_weights, -jnp.abs(inh_weights))
