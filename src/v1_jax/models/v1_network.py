"""Complete V1 network model integrating LGN, GLIF3, and sparse connectivity.

This module provides the high-level V1 network model that combines:
- LGN preprocessing (spatial/temporal filtering)
- Sparse input connectivity (LGN -> V1)
- GLIF3 neuron dynamics with recurrent connectivity
- Optional readout layer

The network follows the architecture from Chen et al., Science Advances 2022.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Callable, NamedTuple
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from ..nn.glif3_cell import (
    GLIF3Cell,
    GLIF3State,
    GLIF3Params,
    glif3_step,
    glif3_unroll,
    glif3_unroll_checkpointed,
)
from ..nn.sparse_layer import (
    InputLayer,
    RecurrentLayer,
    SparseConnectivity,
    SparseFormat,
    BCSRStructure,
    prepare_input_connectivity,
    prepare_recurrent_connectivity,
)
from ..lgn import LGN, LGNParams
from ..data import load_billeh


@dataclass
class V1NetworkConfig:
    """Configuration for V1 network.

    Attributes:
        dt: Time step in ms
        gauss_std: Gaussian surrogate gradient width
        dampening_factor: Surrogate gradient amplitude
        max_delay: Maximum synaptic delay in time steps
        input_weight_scale: Scaling for input weights
        recurrent_weight_scale: Scaling for recurrent weights
        use_dale_law: Whether to enforce Dale's law
        use_decoded_noise: Whether to use decoded background noise
        noise_scale: (quick_noise_scale, slow_noise_scale)
        use_gradient_checkpointing: Whether to use gradient checkpointing
            to reduce memory at the cost of recomputation
        checkpoint_every_n_steps: Number of timesteps per checkpoint segment.
            Only used when use_gradient_checkpointing=True.
            Smaller values save more memory but increase recomputation.
        sparse_format: Sparse matrix format ("bcsr" or "bcoo").
            BCSR provides ~1.8x faster matmul via cuSPARSE.
    """
    dt: float = 1.0
    gauss_std: float = 0.5
    dampening_factor: float = 0.3
    max_delay: int = 5
    input_weight_scale: float = 1.0
    recurrent_weight_scale: float = 1.0
    use_dale_law: bool = True
    use_decoded_noise: bool = False
    noise_scale: Tuple[float, float] = (1.0, 1.0)
    use_gradient_checkpointing: bool = False
    checkpoint_every_n_steps: int = 50
    sparse_format: SparseFormat = "bcsr"


class V1NetworkState(NamedTuple):
    """State of V1 network during simulation.

    Attributes:
        glif3_state: State of GLIF3 neurons
        step: Current timestep (for tracking)
    """
    glif3_state: GLIF3State
    step: int = 0


class V1NetworkOutput(NamedTuple):
    """Output from V1 network forward pass.

    Attributes:
        spikes: Spike trains (time, batch, n_neurons)
        voltages: Membrane voltages (time, batch, n_neurons)
        final_state: Final network state
    """
    spikes: Array
    voltages: Array
    final_state: V1NetworkState


class V1Network:
    """Complete V1 cortical network model.

    This class integrates all components of the V1 model:
    - LGN preprocessing (optional, can use pre-computed LGN outputs)
    - Sparse input layer (LGN -> V1)
    - GLIF3 recurrent network
    - Trainable synaptic weights

    Example:
        >>> network = V1Network.from_billeh(network_path)
        >>> state = network.init_state(batch_size=32)
        >>> lgn_input = ...  # (seq_len, batch, n_lgn)
        >>> output = network(lgn_input, state)
        >>> spikes = output.spikes  # (seq_len, batch, n_neurons)
    """

    def __init__(
        self,
        glif3_params: GLIF3Params,
        input_layer: InputLayer,
        recurrent_layer: RecurrentLayer,
        metadata: Dict[str, Any],
        config: V1NetworkConfig,
        lgn_model: Optional[LGN] = None,
    ):
        """Initialize V1 network.

        Args:
            glif3_params: Parameters for GLIF3 neurons
            input_layer: Sparse input connectivity
            recurrent_layer: Sparse recurrent connectivity
            metadata: Network metadata (n_neurons, n_receptors, etc.)
            config: Network configuration
            lgn_model: Optional LGN model for preprocessing
        """
        self.glif3_params = glif3_params
        self.input_layer = input_layer
        self.recurrent_layer = recurrent_layer
        self.metadata = metadata
        self.config = config
        self.lgn_model = lgn_model

        # Cache dimensions
        self.n_neurons = metadata['n_neurons']
        self.n_receptors = metadata['n_receptors']
        self.max_delay = metadata['max_delay']
        self.n_inputs = input_layer.n_inputs

    @classmethod
    def from_billeh(
        cls,
        network_path: str,
        lgn_model: Optional[LGN] = None,
        config: Optional[V1NetworkConfig] = None,
        bkg_weights: Optional[np.ndarray] = None,
        noise_data: Optional[np.ndarray] = None,
        network_data: Optional[Dict[str, Any]] = None,
        input_pop: Optional[Dict[str, Any]] = None,
    ) -> 'V1Network':
        """Create V1 network from Billeh network data files.

        Args:
            network_path: Path to network data directory (used if network_data/input_pop not provided)
            lgn_model: Optional LGN model
            config: Network configuration
            bkg_weights: Background weights (n_neurons * n_receptors,)
            noise_data: Pre-loaded noise data for decoded noise mode
            network_data: Pre-loaded network data dict (optional, avoids reloading)
            input_pop: Pre-loaded input population dict (optional)

        Returns:
            Initialized V1Network
        """
        if config is None:
            config = V1NetworkConfig()

        # Use pre-loaded data if provided, otherwise load from path
        if network_data is not None and input_pop is not None:
            network = network_data
        else:
            # Load network data from path
            input_pop, network, bkg_weights = load_billeh(
                n_input=17400,  # Default value
                n_neurons=51978,  # Default: all neurons
                core_only=False,
                data_dir=network_path,
            )

        # Create GLIF3 parameters
        glif3_params, metadata = GLIF3Cell.from_network(
            network,
            dt=config.dt,
            gauss_std=config.gauss_std,
            dampening_factor=config.dampening_factor,
            max_delay=config.max_delay,
        )

        n_neurons = metadata['n_neurons']
        n_receptors = metadata['n_receptors']
        node_type_ids = metadata['node_type_ids']

        # Prepare voltage scale for weight normalization
        voltage_scale = np.array(glif3_params.voltage_scale)
        voltage_scale_types = network['node_params']['V_th'] - network['node_params']['E_L']

        # Prepare input connectivity
        input_indices, input_weights, input_shape = prepare_input_connectivity(
            indices=input_pop['indices'],
            weights=input_pop['weights'] * config.input_weight_scale,
            n_neurons=n_neurons,
            n_receptors=n_receptors,
            n_inputs=input_pop['n_inputs'],
            voltage_scale=voltage_scale_types,
            node_type_ids=node_type_ids,
        )

        # Default background weights if not provided
        if bkg_weights is None:
            bkg_weights = np.ones(n_neurons * n_receptors, dtype=np.float32)
        # Scale by voltage (matching TF: divide by voltage_scale, then multiply by 10)
        # TF: bkg_weights = bkg_weights / np.repeat(voltage_scale[self._node_type_ids], self._n_receptors)
        bkg_weights = bkg_weights / np.repeat(
            voltage_scale_types[node_type_ids], n_receptors
        )
        # Multiply by 10 to match TF implementation
        # TF: self.bkg_weights = tf.Variable(bkg_weights * 10., ...)
        bkg_weights = bkg_weights * 10.0

        input_layer = InputLayer(
            indices=input_indices,
            weights=input_weights,
            dense_shape=input_shape,
            bkg_weights=bkg_weights,
            use_decoded_noise=config.use_decoded_noise,
            noise_data=noise_data,
            noise_scale=config.noise_scale,
            sparse_format=config.sparse_format,
        )

        # Prepare recurrent connectivity
        rec_indices, rec_weights, rec_shape = prepare_recurrent_connectivity(
            indices=network['synapses']['indices'],
            weights=network['synapses']['weights'] * config.recurrent_weight_scale,
            delays=network['synapses']['delays'],
            n_neurons=n_neurons,
            n_receptors=n_receptors,
            max_delay=config.max_delay,
            dt=config.dt,
            voltage_scale=voltage_scale_types,
            node_type_ids=node_type_ids,
        )

        recurrent_layer = RecurrentLayer(
            indices=rec_indices,
            weights=rec_weights,
            dense_shape=rec_shape,
            n_neurons=n_neurons,
            n_receptors=n_receptors,
            max_delay=config.max_delay,
            sparse_format=config.sparse_format,
        )

        return cls(
            glif3_params=glif3_params,
            input_layer=input_layer,
            recurrent_layer=recurrent_layer,
            metadata=metadata,
            config=config,
            lgn_model=lgn_model,
        )

    def init_state(
        self,
        batch_size: int,
        key: Optional[Array] = None,
        random: bool = False,
    ) -> V1NetworkState:
        """Initialize network state.

        Args:
            batch_size: Batch dimension
            key: Random key for random initialization
            random: If True, use random initialization

        Returns:
            Initial V1NetworkState
        """
        if random and key is not None:
            glif3_state = GLIF3Cell.random_state(
                n_neurons=self.n_neurons,
                n_receptors=self.n_receptors,
                max_delay=self.max_delay,
                batch_size=batch_size,
                params=self.glif3_params,
                key=key,
            )
        else:
            glif3_state = GLIF3Cell.init_state(
                n_neurons=self.n_neurons,
                n_receptors=self.n_receptors,
                max_delay=self.max_delay,
                batch_size=batch_size,
                params=self.glif3_params,
            )

        return V1NetworkState(glif3_state=glif3_state, step=0)

    def __call__(
        self,
        inputs: Array,
        state: V1NetworkState,
        key: Optional[Array] = None,
    ) -> V1NetworkOutput:
        """Run forward pass through V1 network.

        Args:
            inputs: LGN inputs (seq_len, batch, n_lgn) or movie (seq_len, H, W)
            state: Initial network state
            key: Random key for noise generation

        Returns:
            V1NetworkOutput containing spikes, voltages, and final state
        """
        # Check if inputs need LGN processing
        if self.lgn_model is not None and inputs.ndim == 3 and inputs.shape[-1] > self.n_inputs:
            # Input is a movie, process through LGN
            inputs = self._process_through_lgn(inputs)

        seq_len = inputs.shape[0]
        batch_size = inputs.shape[1]

        # Process inputs through input layer
        # Transpose to (batch, seq_len, n_inputs) for input layer
        inputs_batched = jnp.transpose(inputs, (1, 0, 2))
        input_current = self.input_layer(inputs_batched, key)
        # Transpose back to (seq_len, batch, n_neurons * n_receptors)
        input_current = jnp.transpose(input_current, (1, 0, 2))

        # Create recurrent function
        def recurrent_fn(z_buf):
            return self.recurrent_layer(z_buf)

        # Run GLIF3 unroll (with or without gradient checkpointing)
        if self.config.use_gradient_checkpointing:
            final_glif3_state, all_spikes, all_voltages = glif3_unroll_checkpointed(
                params=self.glif3_params,
                initial_state=state.glif3_state,
                inputs=input_current,
                recurrent_fn=recurrent_fn,
                n_neurons=self.n_neurons,
                n_receptors=self.n_receptors,
                max_delay=self.max_delay,
                dt=self.config.dt,
                gauss_std=self.config.gauss_std,
                dampening_factor=self.config.dampening_factor,
                checkpoint_every_n_steps=self.config.checkpoint_every_n_steps,
            )
        else:
            final_glif3_state, all_spikes, all_voltages = glif3_unroll(
                params=self.glif3_params,
                initial_state=state.glif3_state,
                inputs=input_current,
                recurrent_fn=recurrent_fn,
                n_neurons=self.n_neurons,
                n_receptors=self.n_receptors,
                max_delay=self.max_delay,
                dt=self.config.dt,
                gauss_std=self.config.gauss_std,
                dampening_factor=self.config.dampening_factor,
            )

        final_state = V1NetworkState(
            glif3_state=final_glif3_state,
            step=state.step + seq_len,
        )

        return V1NetworkOutput(
            spikes=all_spikes,
            voltages=all_voltages,
            final_state=final_state,
        )

    def _process_through_lgn(self, movie: Array) -> Array:
        """Process movie through LGN model.

        Args:
            movie: Input movie (seq_len, H, W)

        Returns:
            LGN firing rates (seq_len, 1, n_lgn)
        """
        if self.lgn_model is None:
            raise ValueError("LGN model not provided")

        firing_rates = self.lgn_model(movie)
        return firing_rates[:, None, :]  # Add batch dimension

    def get_trainable_params(self) -> Dict[str, Array]:
        """Get trainable parameters for optimization.

        Returns:
            Dictionary of trainable parameters
        """
        return {
            'input_weights': self.input_layer.connectivity.weights,
            'recurrent_weights': self.recurrent_layer.connectivity.weights,
        }

    def apply_trainable_params(
        self,
        params: Dict[str, Array],
        use_dale_law: bool = True,
    ) -> 'V1Network':
        """Apply updated trainable parameters.

        Args:
            params: Dictionary of updated parameters
            use_dale_law: Whether to enforce Dale's law

        Returns:
            New V1Network with updated parameters
        """
        # Apply Dale's law if needed
        if use_dale_law:
            input_weights = self._apply_dale_constraint(
                params['input_weights'],
                self.input_layer.connectivity.weights >= 0,
            )
            recurrent_weights = self._apply_dale_constraint(
                params['recurrent_weights'],
                self.recurrent_layer.connectivity.weights >= 0,
            )
        else:
            input_weights = params['input_weights']
            recurrent_weights = params['recurrent_weights']

        # Create new layers with updated weights
        # Reuse BCSR structures for efficiency (only weights change)
        new_input_layer = InputLayer(
            indices=self.input_layer.connectivity.indices,
            weights=input_weights,
            dense_shape=self.input_layer.connectivity.shape,
            bkg_weights=self.input_layer.bkg_weights,
            use_decoded_noise=self.input_layer.use_decoded_noise,
            noise_data=self.input_layer.noise_data,
            noise_scale=self.input_layer.noise_scale,
            sparse_format=self.input_layer.sparse_format,
            bcsr_structure=getattr(self.input_layer, '_bcsr_structure', None),
        )

        new_recurrent_layer = RecurrentLayer(
            indices=self.recurrent_layer.connectivity.indices,
            weights=recurrent_weights,
            dense_shape=self.recurrent_layer.connectivity.shape,
            n_neurons=self.n_neurons,
            n_receptors=self.n_receptors,
            max_delay=self.max_delay,
            sparse_format=self.recurrent_layer.sparse_format,
            bcsr_structure=getattr(self.recurrent_layer, '_bcsr_structure', None),
        )

        return V1Network(
            glif3_params=self.glif3_params,
            input_layer=new_input_layer,
            recurrent_layer=new_recurrent_layer,
            metadata=self.metadata,
            config=self.config,
            lgn_model=self.lgn_model,
        )

    @staticmethod
    def _apply_dale_constraint(weights: Array, is_positive: Array) -> Array:
        """Apply Dale's law constraint to weights.

        Args:
            weights: Connection weights
            is_positive: Boolean mask for excitatory connections

        Returns:
            Constrained weights
        """
        return jnp.where(
            is_positive,
            jnp.maximum(weights, 0.0),
            jnp.minimum(weights, 0.0),
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_v1_network(
    network_path: str,
    lgn_data_path: Optional[str] = None,
    config: Optional[V1NetworkConfig] = None,
) -> V1Network:
    """Create V1 network with LGN preprocessing.

    Convenience function for creating a complete V1 model.

    Args:
        network_path: Path to Billeh network data
        lgn_data_path: Path to LGN data (optional)
        config: Network configuration

    Returns:
        Initialized V1Network
    """
    lgn_model = None
    if lgn_data_path is not None:
        lgn_model = LGN(lgn_data_path=lgn_data_path)

    return V1Network.from_billeh(
        network_path=network_path,
        lgn_model=lgn_model,
        config=config,
    )


def make_v1_forward_fn(
    network: V1Network,
) -> Callable:
    """Create a JIT-compilable forward function for V1 network.

    Args:
        network: V1Network instance

    Returns:
        A pure function: (inputs, state, key) -> V1NetworkOutput
    """
    @jax.jit
    def forward_fn(
        inputs: Array,
        state: V1NetworkState,
        key: Optional[Array] = None,
    ) -> V1NetworkOutput:
        return network(inputs, state, key)

    return forward_fn


# =============================================================================
# Single Step Interface (for custom loops)
# =============================================================================

def v1_network_step(
    network: V1Network,
    state: V1NetworkState,
    inputs: Array,
    key: Optional[Array] = None,
) -> Tuple[V1NetworkState, Array, Array]:
    """Single timestep update for V1 network.

    This function is useful for custom simulation loops where
    you need to access intermediate states.

    Args:
        network: V1Network instance
        state: Current network state
        inputs: Input for single timestep (batch, n_lgn)
        key: Random key for noise

    Returns:
        Tuple of (new_state, spikes, voltages)
    """
    # Add time dimension for processing
    inputs_seq = inputs[None, :, :]  # (1, batch, n_lgn)

    output = network(inputs_seq, state, key)

    return (
        output.final_state,
        output.spikes[0],  # Remove time dimension
        output.voltages[0],
    )


def make_v1_step_fn(
    network: V1Network,
) -> Callable:
    """Create a JIT-compiled single step function.

    Args:
        network: V1Network instance

    Returns:
        A pure function: (state, inputs, key) -> (new_state, spikes, voltages)
    """
    @jax.jit
    def step_fn(
        state: V1NetworkState,
        inputs: Array,
        key: Optional[Array] = None,
    ) -> Tuple[V1NetworkState, Array, Array]:
        return v1_network_step(network, state, inputs, key)

    return step_fn
