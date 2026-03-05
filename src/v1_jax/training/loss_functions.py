"""Loss functions for V1 model training.

Implements classification losses and spike rate distribution matching.

Reference: Chen et al., Science Advances 2022
Source: /nvmessd/yinzi/Training-data-driven-V1-model/models.py:357-374
"""

import jax
import jax.numpy as jnp
from jax import Array


def huber_quantile_loss(u: Array, tau: Array, kappa: float = 0.002) -> Array:
    """Huber quantile loss for distribution matching.

    Combines quantile regression with Huber loss for robust distribution fitting.
    Used for matching spike rate distributions to target.

    Args:
        u: Residuals (predicted - target)
        tau: Quantile levels (0 to 1)
        kappa: Huber threshold, transitions from quadratic to linear

    Returns:
        Loss values for each element

    Reference:
        Source: models.py:357-360
    """
    abs_u = jnp.abs(u)
    indicator = (u <= 0).astype(jnp.float32)
    tau_weight = jnp.abs(tau - indicator)

    # Quadratic region (|u| <= kappa)
    quadratic = tau_weight / (2 * kappa) * jnp.square(u)

    # Linear region (|u| > kappa)
    linear = tau_weight * (abs_u - 0.5 * kappa)

    return jnp.where(abs_u <= kappa, quadratic, linear)


def spike_rate_distribution_loss(
    spikes: Array, target_rate: Array, key: jax.random.PRNGKey
) -> Array:
    """Match spike rate distribution to target distribution.

    Computes quantile-based loss between observed and target firing rates.
    Uses random shuffling for stochastic gradient estimation.

    Args:
        spikes: Spike tensor (batch, time, neurons) with values 0/1
        target_rate: Target firing rate distribution (sorted, ascending)
        key: JAX random key for shuffling

    Returns:
        Total distribution matching loss (scalar)

    Reference:
        Source: models.py:363-374
    """
    # Compute mean firing rate per neuron (average over batch and time)
    rate = jnp.mean(spikes, axis=(0, 1))

    # Random shuffle to break correlations
    n_neurons = rate.shape[0]
    rand_ind = jax.random.permutation(key, n_neurons)
    rate_shuffled = rate[rand_ind]

    # Sort rates
    sorted_rate = jnp.sort(rate_shuffled)

    # Compute quantile residuals
    u = sorted_rate - target_rate

    # Quantile levels
    tau = (jnp.arange(n_neurons, dtype=jnp.float32) + 1) / n_neurons

    # Compute Huber quantile loss
    loss = huber_quantile_loss(u, tau, kappa=0.002)

    return jnp.sum(loss)


def sparse_categorical_crossentropy(
    logits: Array, labels: Array, from_logits: bool = True
) -> Array:
    """Sparse categorical cross-entropy loss.

    Args:
        logits: Prediction logits or probabilities (batch, num_classes)
        labels: Integer class labels (batch,)
        from_logits: If True, apply softmax to logits

    Returns:
        Cross-entropy loss per sample
    """
    if from_logits:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
    else:
        log_probs = jnp.log(logits + 1e-7)

    # Gather log probabilities at label indices
    batch_size = labels.shape[0]
    batch_indices = jnp.arange(batch_size)
    selected_log_probs = log_probs[batch_indices, labels]

    return -selected_log_probs


def weighted_crossentropy(
    logits: Array, labels: Array, weights: Array, from_logits: bool = True
) -> Array:
    """Weighted cross-entropy loss.

    Args:
        logits: Prediction logits (batch, num_classes)
        labels: Integer class labels (batch,)
        weights: Sample weights (batch,)
        from_logits: If True, apply softmax to logits

    Returns:
        Weighted average cross-entropy loss (scalar)
    """
    ce_loss = sparse_categorical_crossentropy(logits, labels, from_logits)
    weighted_loss = ce_loss * weights
    return jnp.sum(weighted_loss) / jnp.sum(weights)


def binary_crossentropy(
    logits: Array, labels: Array, from_logits: bool = True
) -> Array:
    """Binary cross-entropy loss.

    Args:
        logits: Prediction logits or probabilities
        labels: Binary labels (0 or 1)
        from_logits: If True, apply sigmoid to logits

    Returns:
        BCE loss per sample
    """
    if from_logits:
        # Numerically stable sigmoid cross-entropy:
        # -y*log(sigmoid(x)) - (1-y)*log(1-sigmoid(x))
        # = -y*log(1/(1+exp(-x))) - (1-y)*log(exp(-x)/(1+exp(-x)))
        # = y*log(1+exp(-x)) + (1-y)*(x + log(1+exp(-x)))
        # = (1-y)*x + log(1+exp(-x))   [for numerical stability with large x]
        # But we need to handle both positive and negative logits:
        # For x >= 0: log(1+exp(-x))
        # For x < 0: x + log(1+exp(x)) - x = log(1+exp(x))
        # Combined: max(x,0) - x*y + log(1 + exp(-|x|))
        loss = (
            jax.nn.relu(logits)
            - logits * labels
            + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        )
    else:
        probs = logits
        loss = -labels * jnp.log(probs + 1e-7) - (1 - labels) * jnp.log(
            1 - probs + 1e-7
        )

    return loss


def mean_squared_error(predictions: Array, targets: Array) -> Array:
    """Mean squared error loss.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        MSE per sample
    """
    return jnp.mean(jnp.square(predictions - targets), axis=-1)


def cosine_similarity_loss(
    predictions: Array, targets: Array, eps: float = 1e-8
) -> Array:
    """Cosine similarity loss (1 - cosine_similarity).

    Args:
        predictions: Model predictions (normalized or unnormalized)
        targets: Target vectors
        eps: Small constant for numerical stability

    Returns:
        Cosine distance (1 - cos_sim) per sample
    """
    # Normalize
    pred_norm = predictions / (jnp.linalg.norm(predictions, axis=-1, keepdims=True) + eps)
    target_norm = targets / (jnp.linalg.norm(targets, axis=-1, keepdims=True) + eps)

    # Cosine similarity
    cos_sim = jnp.sum(pred_norm * target_norm, axis=-1)

    return 1 - cos_sim
