"""Training infrastructure for V1 model.

Implements TrainState, loss computation, and JIT-compiled training steps
for the V1 cortical network model.

Reference: Chen et al., Science Advances 2022
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import optax

from ..models.v1_network import V1Network, V1NetworkState, V1NetworkOutput
from .loss_functions import (
    sparse_categorical_crossentropy,
    weighted_crossentropy,
    spike_rate_distribution_loss,
)
from .regularizers import (
    voltage_regularization,
    stiff_regularization,
    SpikeRateDistributionRegularizer,
)


# =============================================================================
# Training State
# =============================================================================

class TrainState(NamedTuple):
    """Training state for V1 network.

    Attributes:
        step: Current training step
        params: Trainable network parameters
        opt_state: Optimizer state
        initial_params: Initial parameter values (for stiff regularization)
        rng_key: Random key for training
        sign_masks: Boolean masks for Dale's law (True = excitatory, False = inhibitory).
                   Keys: 'input_weights', 'recurrent_weights'. Based on initial weight signs.
    """
    step: int
    params: Dict[str, Array]
    opt_state: Any
    initial_params: Dict[str, Array]
    rng_key: Array
    sign_masks: Dict[str, Array] = {}  # Default empty for backward compatibility


class TrainMetrics(NamedTuple):
    """Metrics from a training step.

    Attributes:
        loss: Total loss
        classification_loss: Classification/task loss
        rate_loss: Spike rate distribution loss
        voltage_loss: Voltage regularization loss
        weight_loss: Weight regularization loss
        accuracy: Classification accuracy
        mean_rate: Mean firing rate
    """
    loss: Array
    classification_loss: Array
    rate_loss: Array
    voltage_loss: Array
    weight_loss: Array
    accuracy: Array
    mean_rate: Array


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Configuration for training.

    Attributes:
        learning_rate: Optimizer learning rate
        rate_cost: Spike rate distribution regularization strength
        voltage_cost: Voltage regularization strength
        weight_cost: Recurrent weight stiff regularization strength
        use_rate_regularization: Whether to apply rate regularization
        use_voltage_regularization: Whether to apply voltage regularization
        use_weight_regularization: Whether to apply weight stiff regularization
        use_dale_law: Whether to enforce Dale's law during training
        gradient_clip_norm: Max gradient norm (0 to disable)
        warmup_steps: Learning rate warmup steps
        weight_decay: AdamW weight decay
        voltage_scale: Voltage scale factor for regularization
        voltage_offset: Voltage offset for regularization
    """
    learning_rate: float = 1e-3
    rate_cost: float = 0.1
    voltage_cost: float = 1e-5
    weight_cost: float = 0.0
    use_rate_regularization: bool = True
    use_voltage_regularization: bool = True
    use_weight_regularization: bool = False
    use_dale_law: bool = True
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 0
    weight_decay: float = 0.0
    voltage_scale: float = 1.0
    voltage_offset: float = 0.0


# =============================================================================
# Trainer Class
# =============================================================================

class V1Trainer:
    """Trainer for V1 cortical network.

    Handles training loop, loss computation, and parameter updates.

    Example:
        >>> network = V1Network.from_billeh(network_path)
        >>> trainer = V1Trainer(network, config=TrainConfig())
        >>> state = trainer.init_train_state(rng_key, readout_params={'weights': w, 'bias': b})
        >>> for batch in dataloader:
        ...     state, metrics = trainer.train_step(state, batch, network_state)
    """

    def __init__(
        self,
        network: V1Network,
        config: Optional[TrainConfig] = None,
        target_firing_rates: Optional[Array] = None,
        readout_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize trainer.

        Args:
            network: V1Network instance
            config: Training configuration
            target_firing_rates: Target firing rate distribution for rate regularization
            readout_config: Readout layer configuration (temporal_pooling, chunk_size, apply_softmax)
        """
        self.network = network
        self.config = config or TrainConfig()
        self.target_firing_rates = target_firing_rates
        self.readout_config = readout_config or {
            'temporal_pooling': 'chunks',
            'chunk_size': 50,
            'apply_softmax': True,
        }

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create rate regularizer if needed
        if self.config.use_rate_regularization and target_firing_rates is not None:
            self.rate_regularizer = SpikeRateDistributionRegularizer(
                target_rates=target_firing_rates,
                rate_cost=self.config.rate_cost,
            )
        else:
            self.rate_regularizer = None

    def _apply_readout(
        self,
        spikes: Array,
        weights: Array,
        bias: Array,
    ) -> Array:
        """Apply readout layer with trainable parameters.

        Args:
            spikes: Spike trains (time, batch, n_neurons)
            weights: Readout weights (n_neurons, n_classes)
            bias: Readout bias (n_classes,)

        Returns:
            Predictions (batch, n_chunks, n_classes) or (batch, n_classes)
        """
        temporal_pooling = self.readout_config.get('temporal_pooling', 'chunks')
        chunk_size = self.readout_config.get('chunk_size', 50)
        apply_softmax = self.readout_config.get('apply_softmax', True)

        # Temporal pooling
        if temporal_pooling == 'mean':
            pooled = jnp.mean(spikes, axis=0)  # (batch, n_neurons)
        elif temporal_pooling == 'sum':
            pooled = jnp.sum(spikes, axis=0)
        elif temporal_pooling == 'last':
            pooled = spikes[-1]
        elif temporal_pooling == 'chunks':
            seq_len = spikes.shape[0]
            n_chunks = seq_len // chunk_size
            # Reshape to (n_chunks, chunk_size, batch, n_neurons)
            spikes_chunked = spikes.reshape(n_chunks, chunk_size, *spikes.shape[1:])
            pooled = jnp.mean(spikes_chunked, axis=1)  # (n_chunks, batch, n_neurons)
            # Transpose to (batch, n_chunks, n_neurons)
            pooled = jnp.transpose(pooled, (1, 0, 2))
        else:
            raise ValueError(f"Unknown temporal_pooling: {temporal_pooling}")

        # Linear projection
        logits = pooled @ weights + bias

        # Apply softmax if requested
        if apply_softmax:
            return jax.nn.softmax(logits, axis=-1)
        return logits

    def _apply_dale_constraints(
        self,
        params: Dict[str, Array],
        sign_masks: Dict[str, Array],
    ) -> Dict[str, Array]:
        """Apply Dale's law constraints to parameters after gradient update.

        Ensures that:
        - Excitatory weights (sign_mask=True) are clipped to >= 0
        - Inhibitory weights (sign_mask=False) are clipped to <= 0

        This enforces the biological constraint that neurons are either
        excitatory or inhibitory, never both (Dale's Law).

        Args:
            params: Current parameters (may violate Dale's law after gradient update)
            sign_masks: Boolean masks from initial weights (True = excitatory)

        Returns:
            Constrained parameters satisfying Dale's law
        """
        if not self.config.use_dale_law:
            return params

        constrained_params = dict(params)

        # Constrain input weights
        if 'input_weights' in params and 'input_weights' in sign_masks:
            constrained_params['input_weights'] = jnp.where(
                sign_masks['input_weights'],
                jnp.maximum(params['input_weights'], 0.0),  # Excitatory: >= 0
                jnp.minimum(params['input_weights'], 0.0),  # Inhibitory: <= 0
            )

        # Constrain recurrent weights
        if 'recurrent_weights' in params and 'recurrent_weights' in sign_masks:
            constrained_params['recurrent_weights'] = jnp.where(
                sign_masks['recurrent_weights'],
                jnp.maximum(params['recurrent_weights'], 0.0),
                jnp.minimum(params['recurrent_weights'], 0.0),
            )

        return constrained_params

    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optax optimizer with optional gradient clipping and warmup."""
        transforms = []

        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            transforms.append(optax.clip_by_global_norm(self.config.gradient_clip_norm))

        # Learning rate schedule with optional warmup
        if self.config.warmup_steps > 0:
            lr_schedule = optax.warmup_constant_schedule(
                init_value=0.0,
                peak_value=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
            )
        else:
            lr_schedule = self.config.learning_rate

        # Adam or AdamW
        if self.config.weight_decay > 0:
            transforms.append(optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=self.config.weight_decay,
                eps=1e-11,  # Match TF epsilon
            ))
        else:
            transforms.append(optax.adam(
                learning_rate=lr_schedule,
                eps=1e-11,
            ))

        return optax.chain(*transforms)

    def init_train_state(
        self,
        rng_key: Array,
        readout_params: Optional[Dict[str, Array]] = None,
    ) -> TrainState:
        """Initialize training state.

        Args:
            rng_key: Random key for initialization
            readout_params: Optional readout layer parameters (weights, bias).
                           If provided, these will be included in trainable params.

        Returns:
            Initial TrainState
        """
        # Get trainable parameters from network
        params = self.network.get_trainable_params()

        # Compute sign masks from initial weights for Dale's law enforcement
        # True = excitatory (weight >= 0), False = inhibitory (weight < 0)
        sign_masks = {
            'input_weights': params['input_weights'] >= 0,
            'recurrent_weights': params['recurrent_weights'] >= 0,
        }

        # Add readout parameters if provided
        if readout_params is not None:
            params['readout_weights'] = readout_params['weights']
            params['readout_bias'] = readout_params['bias']

        # Initialize optimizer state
        opt_state = self.optimizer.init(params)

        # Store initial params for stiff regularization
        initial_params = jax.tree.map(lambda x: x.copy(), params)

        return TrainState(
            step=0,
            params=params,
            opt_state=opt_state,
            initial_params=initial_params,
            rng_key=rng_key,
            sign_masks=sign_masks,
        )

    def _compute_loss(
        self,
        params: Dict[str, Array],
        initial_params: Dict[str, Array],
        inputs: Array,
        labels: Array,
        sample_weights: Array,
        network_state: V1NetworkState,
        readout_fn: Callable,
        rng_key: Array,
    ) -> Tuple[Array, Tuple[V1NetworkOutput, TrainMetrics]]:
        """Compute total loss including all regularization terms.

        Args:
            params: Current trainable parameters (includes readout_weights, readout_bias)
            initial_params: Initial parameters (for stiff regularization)
            inputs: Input tensor (seq_len, batch, n_inputs)
            labels: Target labels (batch,)
            sample_weights: Sample weights (batch,)
            network_state: Initial network state
            readout_fn: Function to compute predictions from spikes (used if no readout params)
            rng_key: Random key

        Returns:
            Tuple of (total_loss, (network_output, metrics))
        """
        # Apply parameters to network
        updated_network = self.network.apply_trainable_params(
            params, use_dale_law=self.config.use_dale_law
        )

        # Split random key
        rng_key, noise_key, rate_key = jax.random.split(rng_key, 3)

        # Forward pass
        output = updated_network(inputs, network_state, noise_key)

        # Compute predictions via readout
        # Use trainable readout params if available, otherwise use the provided function
        if 'readout_weights' in params and 'readout_bias' in params:
            predictions = self._apply_readout(
                output.spikes,
                params['readout_weights'],
                params['readout_bias'],
            )
        else:
            predictions = readout_fn(output.spikes)

        # Classification loss
        # Note: When readout uses apply_softmax=True, predictions are probabilities
        # and we should use from_logits=False. When apply_softmax=False, use from_logits=True.
        # For TF alignment, readout should have apply_softmax=True.
        classification_loss = weighted_crossentropy(
            predictions, labels, sample_weights, from_logits=False
        )

        # Voltage regularization
        if self.config.use_voltage_regularization:
            # Use voltage_scale and voltage_offset from the network's GLIF3 params
            # This normalizes voltages to 0-1 range before computing penalty
            voltage_loss = voltage_regularization(
                output.voltages,
                v_th=jnp.ones((self.network.n_neurons,)),  # Normalized threshold
                v_reset=jnp.zeros((self.network.n_neurons,)),  # Normalized reset
                voltage_cost=self.config.voltage_cost,
                voltage_scale=self.network.glif3_params.voltage_scale,
                voltage_offset=self.network.glif3_params.voltage_offset,
            )
        else:
            voltage_loss = jnp.array(0.0)

        # Spike rate regularization
        if self.rate_regularizer is not None and self.config.use_rate_regularization:
            # Transpose spikes to (batch, time, neurons) for regularizer
            spikes_bth = jnp.transpose(output.spikes, (1, 0, 2))
            rate_loss = self.rate_regularizer(spikes_bth, rate_key)
        else:
            rate_loss = jnp.array(0.0)

        # Weight stiff regularization
        if self.config.use_weight_regularization and self.config.weight_cost > 0:
            weight_loss = stiff_regularization(
                params['recurrent_weights'],
                initial_params['recurrent_weights'],
                strength=self.config.weight_cost,
            )
        else:
            weight_loss = jnp.array(0.0)

        # Total loss
        total_loss = classification_loss + voltage_loss + rate_loss + weight_loss

        # Compute accuracy
        # Handle 3D predictions (batch, n_chunks, n_classes)
        if predictions.ndim == 3:
            # Use the last chunk for accuracy (matching TF behavior)
            predictions_last = predictions[:, -1, :]
            pred_classes = jnp.argmax(predictions_last, axis=-1)
        else:
            pred_classes = jnp.argmax(predictions, axis=-1)
        correct = (pred_classes == labels).astype(jnp.float32)
        accuracy = jnp.sum(correct * sample_weights) / jnp.sum(sample_weights)

        # Mean firing rate
        mean_rate = jnp.mean(output.spikes)

        metrics = TrainMetrics(
            loss=total_loss,
            classification_loss=classification_loss,
            rate_loss=rate_loss,
            voltage_loss=voltage_loss,
            weight_loss=weight_loss,
            accuracy=accuracy,
            mean_rate=mean_rate,
        )

        return total_loss, (output, metrics)

    def train_step(
        self,
        state: TrainState,
        inputs: Array,
        labels: Array,
        sample_weights: Array,
        network_state: V1NetworkState,
        readout_fn: Callable,
    ) -> Tuple[TrainState, V1NetworkOutput, TrainMetrics]:
        """Execute single training step.

        Args:
            state: Current training state
            inputs: Input tensor (seq_len, batch, n_inputs)
            labels: Target labels (batch,)
            sample_weights: Sample weights (batch,)
            network_state: Initial network state
            readout_fn: Function to compute predictions from spikes

        Returns:
            Tuple of (new_state, network_output, metrics)
        """
        # Split key for this step
        rng_key, new_key = jax.random.split(state.rng_key)

        # Compute gradients
        grad_fn = jax.value_and_grad(self._compute_loss, has_aux=True)
        (loss, (output, metrics)), grads = grad_fn(
            state.params,
            state.initial_params,
            inputs,
            labels,
            sample_weights,
            network_state,
            readout_fn,
            rng_key,
        )

        # Apply optimizer updates
        updates, new_opt_state = self.optimizer.update(
            grads, state.opt_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)

        # Apply Dale's law constraints immediately after parameter update
        # This ensures stored params always satisfy biological constraints
        new_params = self._apply_dale_constraints(new_params, state.sign_masks)

        # Create new state
        new_state = TrainState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            initial_params=state.initial_params,
            rng_key=new_key,
            sign_masks=state.sign_masks,
        )

        return new_state, output, metrics

    def eval_step(
        self,
        state: TrainState,
        inputs: Array,
        labels: Array,
        sample_weights: Array,
        network_state: V1NetworkState,
        readout_fn: Callable,
    ) -> Tuple[V1NetworkOutput, TrainMetrics]:
        """Execute single evaluation step (no gradients).

        Args:
            state: Current training state
            inputs: Input tensor (seq_len, batch, n_inputs)
            labels: Target labels (batch,)
            sample_weights: Sample weights (batch,)
            network_state: Initial network state
            readout_fn: Function to compute predictions from spikes

        Returns:
            Tuple of (network_output, metrics)
        """
        # Split key for this step
        rng_key, _ = jax.random.split(state.rng_key)

        # Forward pass only (no gradients)
        _, (output, metrics) = self._compute_loss(
            state.params,
            state.initial_params,
            inputs,
            labels,
            sample_weights,
            network_state,
            readout_fn,
            rng_key,
        )

        return output, metrics


# =============================================================================
# JIT-compiled Training Functions
# =============================================================================

def create_train_step_fn(
    trainer: V1Trainer,
    readout_fn: Callable,
) -> Callable:
    """Create JIT-compiled training step function.

    Args:
        trainer: V1Trainer instance
        readout_fn: Readout function (spikes -> predictions)

    Returns:
        JIT-compiled train step function
    """
    @jax.jit
    def train_step_jit(
        state: TrainState,
        inputs: Array,
        labels: Array,
        sample_weights: Array,
        network_state: V1NetworkState,
    ) -> Tuple[TrainState, V1NetworkOutput, TrainMetrics]:
        return trainer.train_step(
            state, inputs, labels, sample_weights, network_state, readout_fn
        )

    return train_step_jit


def create_eval_step_fn(
    trainer: V1Trainer,
    readout_fn: Callable,
) -> Callable:
    """Create JIT-compiled evaluation step function.

    Args:
        trainer: V1Trainer instance
        readout_fn: Readout function (spikes -> predictions)

    Returns:
        JIT-compiled eval step function
    """
    @jax.jit
    def eval_step_jit(
        state: TrainState,
        inputs: Array,
        labels: Array,
        sample_weights: Array,
        network_state: V1NetworkState,
    ) -> Tuple[V1NetworkOutput, TrainMetrics]:
        return trainer.eval_step(
            state, inputs, labels, sample_weights, network_state, readout_fn
        )

    return eval_step_jit


# =============================================================================
# Metrics Aggregation
# =============================================================================

class MetricsAccumulator:
    """Accumulator for training metrics.

    Tracks running averages of metrics over multiple steps.
    """

    def __init__(self):
        """Initialize accumulator."""
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self._sum_loss = 0.0
        self._sum_cls_loss = 0.0
        self._sum_rate_loss = 0.0
        self._sum_voltage_loss = 0.0
        self._sum_weight_loss = 0.0
        self._sum_accuracy = 0.0
        self._sum_rate = 0.0
        self._count = 0

    def update(self, metrics: TrainMetrics):
        """Update accumulator with new metrics.

        Args:
            metrics: TrainMetrics from a step
        """
        self._sum_loss += float(metrics.loss)
        self._sum_cls_loss += float(metrics.classification_loss)
        self._sum_rate_loss += float(metrics.rate_loss)
        self._sum_voltage_loss += float(metrics.voltage_loss)
        self._sum_weight_loss += float(metrics.weight_loss)
        self._sum_accuracy += float(metrics.accuracy)
        self._sum_rate += float(metrics.mean_rate)
        self._count += 1

    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics.

        Returns:
            Dictionary of averaged metrics
        """
        if self._count == 0:
            return {}

        return {
            'loss': self._sum_loss / self._count,
            'classification_loss': self._sum_cls_loss / self._count,
            'rate_loss': self._sum_rate_loss / self._count,
            'voltage_loss': self._sum_voltage_loss / self._count,
            'weight_loss': self._sum_weight_loss / self._count,
            'accuracy': self._sum_accuracy / self._count,
            'mean_rate': self._sum_rate / self._count,
        }

    def format_string(self, prefix: str = '') -> str:
        """Format metrics as a string for logging.

        Args:
            prefix: Prefix for metric names

        Returns:
            Formatted string
        """
        metrics = self.compute()
        if not metrics:
            return f'{prefix}No metrics'

        parts = [
            f"Loss {metrics['loss']:.4f}",
            f"Acc {metrics['accuracy']:.4f}",
            f"Rate {metrics['mean_rate']:.4f}",
        ]

        if metrics['rate_loss'] > 0:
            parts.append(f"RLoss {metrics['rate_loss']:.4f}")
        if metrics['voltage_loss'] > 0:
            parts.append(f"VLoss {metrics['voltage_loss']:.4f}")
        if metrics['weight_loss'] > 0:
            parts.append(f"WLoss {metrics['weight_loss']:.4f}")

        return f'{prefix}{", ".join(parts)}'


# =============================================================================
# Gradient Checkpointing Support
# =============================================================================

def create_checkpointed_forward_fn(
    network: V1Network,
    checkpoint_every_n_steps: int = 100,
) -> Callable:
    """Create a gradient-checkpointed forward function.

    For very long sequences, gradient checkpointing reduces memory usage
    by recomputing activations during backward pass.

    Args:
        network: V1Network instance
        checkpoint_every_n_steps: Recompute gradients every N steps

    Returns:
        Checkpointed forward function
    """
    @jax.checkpoint
    def checkpointed_segment(
        state: V1NetworkState,
        inputs_segment: Array,
        key: Optional[Array],
    ) -> Tuple[V1NetworkState, Array, Array]:
        """Process one segment with gradient checkpointing."""
        output = network(inputs_segment, state, key)
        return output.final_state, output.spikes, output.voltages

    def forward_with_checkpointing(
        inputs: Array,
        initial_state: V1NetworkState,
        key: Optional[Array] = None,
    ) -> V1NetworkOutput:
        """Forward pass with gradient checkpointing.

        Args:
            inputs: Full input sequence (seq_len, batch, n_inputs)
            initial_state: Initial network state
            key: Random key for noise

        Returns:
            V1NetworkOutput
        """
        seq_len = inputs.shape[0]
        n_segments = (seq_len + checkpoint_every_n_steps - 1) // checkpoint_every_n_steps

        # Split inputs into segments
        def scan_fn(carry, segment_idx):
            state, current_key = carry

            # Get segment bounds
            start = segment_idx * checkpoint_every_n_steps
            end = jnp.minimum(start + checkpoint_every_n_steps, seq_len)

            # Extract segment
            segment = jax.lax.dynamic_slice_in_dim(
                inputs, start, checkpoint_every_n_steps, axis=0
            )

            # Split key
            if current_key is not None:
                current_key, segment_key = jax.random.split(current_key)
            else:
                segment_key = None

            # Process segment with checkpointing
            new_state, spikes, voltages = checkpointed_segment(
                state, segment, segment_key
            )

            return (new_state, current_key), (spikes, voltages)

        # Run scan over segments
        (final_state, _), (all_spikes, all_voltages) = jax.lax.scan(
            scan_fn,
            (initial_state, key),
            jnp.arange(n_segments),
        )

        # Concatenate results
        all_spikes = all_spikes.reshape(-1, *all_spikes.shape[2:])[:seq_len]
        all_voltages = all_voltages.reshape(-1, *all_voltages.shape[2:])[:seq_len]

        return V1NetworkOutput(
            spikes=all_spikes,
            voltages=all_voltages,
            final_state=V1NetworkState(
                glif3_state=final_state.glif3_state,
                step=initial_state.step + seq_len,
            ),
        )

    return forward_with_checkpointing


# =============================================================================
# Learning Rate Schedules
# =============================================================================

def create_lr_schedule(
    base_lr: float,
    warmup_steps: int = 0,
    decay_steps: int = 0,
    decay_rate: float = 0.1,
    schedule_type: str = 'constant',
) -> optax.Schedule:
    """Create learning rate schedule.

    Args:
        base_lr: Base learning rate
        warmup_steps: Number of warmup steps
        decay_steps: Steps between decays (for step decay)
        decay_rate: Decay multiplier
        schedule_type: 'constant', 'cosine', 'exponential', or 'step'

    Returns:
        optax Schedule
    """
    if schedule_type == 'constant':
        schedule = optax.constant_schedule(base_lr)
    elif schedule_type == 'cosine':
        schedule = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=decay_steps,
        )
    elif schedule_type == 'exponential':
        schedule = optax.exponential_decay(
            init_value=base_lr,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
        )
    elif schedule_type == 'step':
        boundaries = [decay_steps * (i + 1) for i in range(10)]
        values = [base_lr * (decay_rate ** i) for i in range(11)]
        schedule = optax.piecewise_constant_schedule(
            init_value=base_lr,
            boundaries_and_scales={b: decay_rate for b in boundaries},
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    # Add warmup if requested
    if warmup_steps > 0:
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=base_lr,
                    transition_steps=warmup_steps,
                ),
                schedule,
            ],
            [warmup_steps],
        )

    return schedule
