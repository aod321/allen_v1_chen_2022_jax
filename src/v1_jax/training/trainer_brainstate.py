"""Trainer using braintrace IODim for memory-efficient online gradient training.

This module provides a trainer that uses braintrace's IODim algorithm
to compute gradients online, dramatically reducing memory usage compared
to standard BPTT.

Memory comparison:
- BPTT: O(T * N) where T=sequence length, N=neurons
- IODim: O(I + O) where I=input dim, O=output dim

For V1 network with T=600, N=52K:
- BPTT: ~8 GB for activations (NOT SUPPORTED)
- IODim: ~1 MB for eligibility traces (DEFAULT)

Based on AlphaBrain/glif3_network.py training implementation.

IMPORTANT: BPTT is intentionally NOT implemented. IODim is the ONLY supported
training method. This follows AlphaBrain's design:
- Forward: brainevent.EventArray @ CSR (event-driven, no VJP needed)
- Gradient: braintrace.IODim (online eligibility traces)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import brainstate
import braintrace

from ..models.v1_network_brainstate import V1NetworkBrainstate

__all__ = ['IODimTrainer', 'IODimConfig', 'train_epoch_iodim']


@dataclass
class IODimConfig:
    """Configuration for IODim training.

    IODim (Input-Output Dimension) uses online eligibility traces for
    memory-efficient gradient computation. This is the ONLY supported
    training method - BPTT is intentionally NOT implemented.

    Attributes
    ----------
    learning_rate : float
        Learning rate for optimizer. Due to small gradients in SNN training,
        higher learning rates (0.1-10.0) may be needed.
    grad_clip_norm : float
        Maximum gradient norm for clipping
    etrace_decay : float
        Decay factor for eligibility traces (0.9-0.999).
        Higher values = more accurate but slower.
        Recommended: 0.99 for most tasks.
    loss_scale : float
        Loss scaling factor to amplify gradients. SNN gradients are typically
        very small (~1e-5) due to:
        - Current-to-voltage conversion factor (~0.001)
        - Surrogate gradient magnitude
        - Sparse connectivity effects
        Recommended: 1000-10000 for effective training.
    optimizer : str
        Optimizer name ('Adam', 'SGD', etc.)
    adam_betas : Tuple[float, float]
        Adam beta parameters
    adam_eps : float
        Adam epsilon for numerical stability
    rate_cost : float
        Spike rate regularization coefficient
    weight_regularization : float
        L2 weight regularization coefficient
    """
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0
    etrace_decay: float = 0.99
    loss_scale: float = 1000.0  # Scale up loss to get reasonable gradients
    optimizer: str = 'Adam'
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    rate_cost: float = 0.1
    weight_regularization: float = 0.0


class IODimTrainer:
    """Trainer using braintrace IODim algorithm.

    This trainer uses online eligibility traces to compute gradients,
    avoiding the need to store all intermediate activations.

    Based on AlphaBrain/glif3_network.py training implementation.

    IMPORTANT: Uses braintrace.IODimVjpAlgorithm to wrap the network.
    Forward propagation uses brainevent.EventArray @ CSR (event-driven),
    and gradients are computed via eligibility traces, NOT VJP.

    Parameters
    ----------
    network : V1NetworkBrainstate
        Network to train
    config : IODimConfig
        Training configuration
    loss_fn : Callable
        Loss function (output, target) -> loss
    """

    def __init__(
        self,
        network: V1NetworkBrainstate,
        config: IODimConfig,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """Initialize IODim trainer."""
        self.network = network
        self.config = config
        self.loss_fn = loss_fn or self._default_loss_fn

        # Get trainable weights
        self._trainable_weights = network.get_trainable_weights()
        if self._trainable_weights is None:
            raise ValueError("Network has no trainable weights")

        # Save initial weights for regularization
        self._initial_weights = self._trainable_weights.value.copy()

        # Build train_states dict for brainstate.augment.grad
        self._train_states = {'weights': self._trainable_weights}

        # Initialize optimizer
        self._init_optimizer()

        # IODim model (compiled on first use)
        self._etrace_model = None
        self._compiled = False
        self._train_step_fn = None

    def _init_optimizer(self) -> None:
        """Initialize optimizer."""
        optimizer_class = getattr(brainstate.optim, self.config.optimizer, None)
        if optimizer_class is None:
            raise ValueError(f"Optimizer '{self.config.optimizer}' not found")

        if self.config.optimizer == 'Adam':
            self._optimizer = optimizer_class(
                lr=self.config.learning_rate,
                beta1=self.config.adam_betas[0],
                beta2=self.config.adam_betas[1],
                eps=self.config.adam_eps,
            )
        else:
            self._optimizer = optimizer_class(lr=self.config.learning_rate)

        self._optimizer.register_trainable_weights(self._train_states)

    def _default_loss_fn(self, outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """Default MSE loss function."""
        return jnp.mean((outputs - targets) ** 2)

    def _compile_etrace_model(self, sample_input: jnp.ndarray) -> None:
        """Compile IODim eligibility trace model.

        Parameters
        ----------
        sample_input : jnp.ndarray
            Sample input for shape inference, shape (batch, n_neurons)
        """
        if self._compiled:
            return

        # Reset network state before compiling (required for braintrace)
        if sample_input.ndim == 2:
            batch_size = sample_input.shape[0]
        else:
            batch_size = 1
            sample_input = sample_input[None, :]

        self.network.reset(batch_size=batch_size)

        # Create IODim model wrapper - this wraps the network's update() method
        # and computes eligibility traces internally
        self._etrace_model = braintrace.IODimVjpAlgorithm(
            self.network,
            decay_or_rank=self.config.etrace_decay,
        )

        # Compile graph with sample input
        self._etrace_model.compile_graph(sample_input)
        self._compiled = True

    def _clip_gradients(self, grads: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Clip gradients by global norm."""
        if self.config.grad_clip_norm is None or self.config.grad_clip_norm <= 0:
            return grads

        # Flatten all gradients
        flat = []
        for v in grads.values():
            if v is not None:
                flat.append(jnp.reshape(v, (-1,)))

        if len(flat) == 0:
            return grads

        all_grads = jnp.concatenate(flat)
        total_norm = jnp.linalg.norm(all_grads)
        clip_coeff = jnp.minimum(self.config.grad_clip_norm / (total_norm + 1e-6), 1.0)
        return {k: clip_coeff * v for k, v in grads.items()}

    def train_step(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        reset_before: bool = True,
    ) -> float:
        """Execute one training step using IODim online gradient.

        Based on AlphaBrain/glif3_network.py training implementation.

        Parameters
        ----------
        inputs : jnp.ndarray
            Input sequence, shape (T, batch, n_neurons) or (T, n_neurons)
        targets : jnp.ndarray
            Target outputs, shape (T, batch, n_output) or (T, n_output)
        reset_before : bool
            Whether to reset network state before forward pass

        Returns
        -------
        float
            Loss value
        """
        T = inputs.shape[0]

        # Ensure batch dimension
        if inputs.ndim == 2:
            inputs = inputs[:, None, :]
            targets = targets[:, None, :]
        batch_size = inputs.shape[1]

        # Compile IODim model if needed
        if not self._compiled:
            self._compile_etrace_model(inputs[0])

        # Get references for closure
        network = self.network
        etrace_model = self._etrace_model
        loss_fn = self.loss_fn
        train_states = self._train_states
        initial_weights = self._initial_weights
        weight_reg = self.config.weight_regularization
        loss_scale = self.config.loss_scale

        # Time indices for brainstate.compile.for_loop
        time_idx = jnp.arange(T)

        # Define loss function following AlphaBrain pattern
        def loss_fn_wrapper():
            # Reset network state
            if reset_before:
                network.reset(batch_size=batch_size)

            # Reset eligibility traces each eval to avoid cross-epoch accumulation
            etrace_model.reset_state()

            # Single step function for for_loop
            def step_fn(i):
                # Use etrace_model instead of network.update()
                # This computes forward pass AND updates eligibility traces
                return etrace_model(inputs[i])

            # Run sequence using brainstate's for_loop
            y_pred = brainstate.compile.for_loop(step_fn, time_idx, pbar=None)

            # Compute loss
            mse_loss = loss_fn(y_pred, targets)

            # Add weight regularization if specified
            regularization_loss = 0.0
            if weight_reg > 0.0:
                current_weights = train_states['weights'].value
                weight_diff = current_weights - initial_weights
                regularization_loss = weight_reg * jnp.mean(jnp.square(weight_diff))

            # Scale loss to get reasonable gradients
            # SNN gradients are typically ~1e-5 due to current-to-voltage conversion
            # Scaling by 1000-10000 brings them to reasonable magnitude
            return (mse_loss + regularization_loss) * loss_scale

        # Compute gradients using brainstate.augment.grad
        # This works because IODimVjpAlgorithm has prepared the eligibility traces
        grad_fn = brainstate.augment.grad(loss_fn_wrapper, train_states, return_value=True)
        grads, scaled_loss_val = grad_fn()

        # Clip gradients
        clipped_grads = self._clip_gradients(grads)

        # Update weights
        self._optimizer.update(clipped_grads)

        # Return unscaled loss for reporting
        return float(scaled_loss_val / loss_scale)

    def train_epoch(
        self,
        data_loader,
        verbose: bool = True,
        log_interval: int = 10,
    ) -> List[float]:
        """Train for one epoch.

        Parameters
        ----------
        data_loader : iterable
            Yields (inputs, targets) batches
        verbose : bool
            Print progress
        log_interval : int
            Steps between log messages

        Returns
        -------
        List[float]
            Loss values for each step
        """
        losses = []

        for step, (inputs, targets) in enumerate(data_loader):
            loss = self.train_step(inputs, targets)
            losses.append(loss)

            if verbose and step % log_interval == 0:
                print(f"[IODim] Step {step}: loss={loss:.6f}")

        return losses


def train_epoch_iodim(
    network: V1NetworkBrainstate,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    config: IODimConfig,
    num_epochs: int = 1,
    verbose: bool = True,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """Convenience function for IODim training.

    Parameters
    ----------
    network : V1NetworkBrainstate
        Network to train
    inputs : jnp.ndarray
        Input sequence, shape (T, batch, n_neurons)
    targets : jnp.ndarray
        Target outputs, shape (T, batch, n_output)
    config : IODimConfig
        Training configuration
    num_epochs : int
        Number of training epochs
    verbose : bool
        Print progress
    log_interval : int
        Epochs between log messages

    Returns
    -------
    Dict[str, Any]
        Training results with 'loss_curve' and 'final_weights'
    """
    trainer = IODimTrainer(network, config)
    loss_curve = []

    for epoch in range(num_epochs):
        loss = trainer.train_step(inputs, targets)
        loss_curve.append(loss)

        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(f"[IODim] Epoch {epoch + 1}/{num_epochs}: loss={loss:.6f}")

    return {
        'loss_curve': loss_curve,
        'final_weights': network.get_trainable_weights().value,
    }
