"""V1 Network using brainstate + brainevent for IODim training.

This module provides a brainstate-based implementation of the V1 visual cortex
network, using brainevent for event-driven spike propagation and braintrace's
IODim algorithm for memory-efficient gradient computation.

Key design:
- Forward: brainevent.EventArray @ CSR (event-driven, 5-20x faster)
- Gradient: braintrace.IODim (online eligibility traces, 500x less memory)

Based on AlphaBrain/glif3_network.py.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

import brainstate

# Apply JAX compatibility patches BEFORE importing brainevent
from ..compat import apply_jax_compat_patches
apply_jax_compat_patches()

import brainevent
from brainevent import CSR

from ..nn.glif3_brainstate import GLIF3Brainstate
from ..nn.connectivity_brainstate import (
    Connection, SynapticDelayBuffer, build_connection_from_billeh
)

__all__ = ['V1NetworkBrainstate']


class V1NetworkBrainstate(brainstate.nn.Module):
    """V1 visual cortex network using brainstate + brainevent for IODim training.

    This network integrates GLIF3 neurons with sparse recurrent connectivity
    and synaptic delays, designed for memory-efficient training with braintrace.

    Features:
    - Event-driven spike propagation using brainevent.EventArray @ CSR
    - Ring buffer for synaptic delays
    - Multi-receptor synaptic inputs (AMPA, NMDA, GABA_A, GABA_B)
    - Compatible with braintrace IODim algorithm (no BPTT needed)
    - **Input layer support** for LGN → V1 connectivity (Garrett task)

    Parameters
    ----------
    network_data : dict
        Network data from load_network() containing neuron params and connectivity
    input_data : dict, optional
        Input connectivity data from load_billeh() containing LGN → V1 connections
    dt : float
        Time step in ms
    mode : str
        'simulation' or 'training'
    precision : int
        32 or 64 bit precision
    """

    def __init__(
        self,
        network_data: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None,
        bkg_weights: Optional[np.ndarray] = None,
        dt: float = 1.0,
        mode: str = 'training',
        precision: int = 32,
        name: Optional[str] = None,
    ) -> None:
        """Initialize V1 network from Billeh data."""
        super().__init__(name=name)

        self.dt = dt
        self.mode = mode
        self.precision = precision

        # Set dtype
        self.dtype = jnp.float32 if precision == 32 else jnp.float64

        # Set brainstate environment
        brainstate.environ.set(dt=dt, precision=precision)

        # Extract network info
        self.n_neurons = int(network_data['n_nodes'])
        self.n_receptors = network_data['node_params']['tau_syn'].shape[1]
        self.n_inputs = input_data['n_inputs'] if input_data is not None else 0
        self.has_input_layer = input_data is not None

        # Initialize GLIF3 population
        self.population = GLIF3Brainstate.from_billeh_network(
            network_data,
            dt=dt,
            mode=mode,
            precision=precision,
        )

        # Build connectivity (uses brainevent.CSR)
        # Include input connectivity if input_data is provided
        self.connectivity = build_connection_from_billeh(
            synapses=network_data['synapses'],
            n_neurons=self.n_neurons,
            n_receptors=self.n_receptors,
            dt=dt,
            include_input=(input_data is not None),
            input_data=input_data,
            n_inputs=self.n_inputs,
            dtype=self.dtype,
        )

        # Initialize delay buffer
        self.delay_buffer = SynapticDelayBuffer(
            max_delay=self.connectivity.max_delay,
            num_neurons=self.n_neurons,
            num_receptors=self.n_receptors,
            dtype=self.dtype,
        )

        # Background weights for input layer
        if bkg_weights is not None:
            # Scale by voltage (matching TF implementation)
            voltage_scale_types = network_data['node_params']['V_th'] - network_data['node_params']['E_L']
            node_type_ids = network_data.get('node_type_ids', np.zeros(self.n_neurons, dtype=np.int32))
            bkg_weights_scaled = bkg_weights / np.repeat(
                voltage_scale_types[node_type_ids], self.n_receptors
            )
            bkg_weights_scaled = bkg_weights_scaled * 10.0
            self.bkg_weights = jnp.asarray(bkg_weights_scaled, dtype=self.dtype)
        else:
            self.bkg_weights = jnp.ones(self.n_neurons * self.n_receptors, dtype=self.dtype)

        # Create trainable weight state for recurrent connections
        self._init_trainable_weights()

        # Step counter
        self._step_count = brainstate.ShortTermState(jnp.array(0, dtype=jnp.int32))

        # Input mode: if True, expect LGN inputs and use input layer
        # If False, expect direct V1 current injection
        self._use_lgn_input = False

    def _init_trainable_weights(self) -> None:
        """Initialize trainable weight parameters.

        We flatten all CSR weights into a single trainable array for optimization.
        Store (key, start_idx, end_idx) for reconstruction during forward pass.
        """
        # Collect all recurrent weights
        all_weights = []
        self._weight_info = []  # Track (key, start_idx, end_idx) for reconstruction

        current_idx = 0
        for key in sorted(self.connectivity.csr_matrices.keys()):
            csr = self.connectivity.csr_matrices[key]
            weights = csr.data
            n_weights = len(weights)
            all_weights.append(weights)
            self._weight_info.append((key, current_idx, current_idx + n_weights))
            current_idx += n_weights

        if all_weights:
            self._trainable_weights = brainstate.ParamState(
                jnp.concatenate(all_weights)
            )
        else:
            self._trainable_weights = None

    def reset(self, batch_size: int = 1) -> None:
        """Reset all network state.

        Parameters
        ----------
        batch_size : int
            Batch size for state initialization
        """
        self.population.reset_state(batch_size=batch_size)
        self.delay_buffer.reset()
        self._step_count.value = jnp.array(0, dtype=jnp.int32)

    def update(self, external_input: jnp.ndarray) -> jnp.ndarray:
        """Single step network update with direct current injection.

        This method is designed for use with braintrace IODim algorithm.
        Use update_with_lgn() for LGN firing rate inputs.

        Parameters
        ----------
        external_input : jnp.ndarray
            External input current, shape (batch, n_neurons) or (n_neurons,)

        Returns
        -------
        jnp.ndarray
            Network output (normalized membrane potential or spikes)
        """
        # Ensure batch dimension
        if external_input.ndim == 1:
            external_input = external_input[None, :]

        batch_size = external_input.shape[0]

        # Get delayed synaptic input from buffer
        # Shape: (n_receptors, n_neurons) -> need to expand for batch
        delayed_input = self.delay_buffer.get_current_synaptic_input()
        # Transpose to (n_neurons, n_receptors) for GLIF3
        syn_input = delayed_input.T  # (n_neurons, n_receptors)
        # Add batch dimension
        syn_input = jnp.broadcast_to(syn_input, (batch_size, self.n_neurons, self.n_receptors))

        # Update neurons
        output = self.population.update(syn_input, x=external_input)

        # Get spikes for propagation
        spikes = self.population.get_spike()  # (batch, n_neurons)

        # Propagate spikes through recurrent connections (event-driven)
        self._propagate_spikes(spikes)

        # Advance delay buffer
        self.delay_buffer.advance_and_clear_current()

        # Increment step counter
        self._step_count.value = self._step_count.value + 1

        return output

    def update_with_lgn(self, lgn_input: jnp.ndarray) -> jnp.ndarray:
        """Single step network update with LGN firing rate input.

        This method processes LGN inputs through the input layer (sparse CSR)
        to compute V1 input currents, then updates the network.

        Parameters
        ----------
        lgn_input : jnp.ndarray
            LGN firing rates, shape (batch, n_inputs) or (n_inputs,)
            Values typically in range [0, 1.3] (rate-coded input)

        Returns
        -------
        jnp.ndarray
            Network output (normalized membrane potential or spikes)
        """
        if not self.has_input_layer:
            raise ValueError(
                "Network was not initialized with input_data. "
                "Use update() with direct current input instead."
            )

        # Ensure batch dimension
        if lgn_input.ndim == 1:
            lgn_input = lgn_input[None, :]

        batch_size = lgn_input.shape[0]

        # Convert LGN firing rates to V1 input current using input CSR matrices
        # CSR has shape (n_neurons, n_inputs), we need CSR @ lgn_input
        # Process each (delay, receptor) group
        if self.connectivity.input_csr_matrices is not None:
            for key, csr in self.connectivity.input_csr_matrices.items():
                delay, receptor = key
                # CSR @ input = (n_neurons,) for each batch element
                for b in range(batch_size):
                    contrib = csr @ lgn_input[b]
                    # Add to appropriate delay slot in buffer
                    self.delay_buffer.add_delayed_synaptic_input(
                        delay_steps=delay,
                        receptor=receptor,
                        synaptic_input=contrib,
                    )

        # Background current from bkg_weights
        # Sum over receptors to get per-neuron background current
        bkg_current = self.bkg_weights.reshape(self.n_neurons, self.n_receptors)
        bkg_current = jnp.sum(bkg_current, axis=1)  # (n_neurons,)

        # Create external input current with background only
        # The LGN contributions are now in the delay buffer
        input_current = jnp.broadcast_to(bkg_current[None, :], (batch_size, self.n_neurons))

        # Get delayed synaptic input from buffer (now includes LGN contributions)
        delayed_input = self.delay_buffer.get_current_synaptic_input()
        # Transpose to (n_neurons, n_receptors) for GLIF3
        syn_input = delayed_input.T  # (n_neurons, n_receptors)
        # Add batch dimension
        syn_input = jnp.broadcast_to(syn_input, (batch_size, self.n_neurons, self.n_receptors))

        # Update neurons with synaptic input and background current
        output = self.population.update(syn_input, x=input_current)

        # Get spikes for propagation
        spikes = self.population.get_spike()  # (batch, n_neurons)

        # Propagate spikes through recurrent connections (event-driven)
        self._propagate_spikes(spikes)

        # Advance delay buffer
        self.delay_buffer.advance_and_clear_current()

        # Increment step counter
        self._step_count.value = self._step_count.value + 1

        return output

    def _propagate_spikes(self, spikes: jnp.ndarray) -> None:
        """Propagate spikes through recurrent connectivity using event-driven method.

        Uses brainevent.EventArray @ CSR for efficient event-driven propagation.
        Only computes contributions from neurons that actually fired.

        Parameters
        ----------
        spikes : jnp.ndarray
            Spike outputs, shape (batch, n_neurons)
        """
        # Use mean over batch for spike propagation (simplified)
        # In full implementation, would handle batch properly
        if spikes.ndim > 1:
            spikes_flat = jnp.mean(spikes, axis=0)
        else:
            spikes_flat = spikes

        # Get current trainable weights
        if self._trainable_weights is not None:
            trainable_weights = self._trainable_weights.value
        else:
            trainable_weights = None

        # Propagate through each (delay, receptor) group using event-driven method
        for key, start_idx, end_idx in self._weight_info:
            delay, receptor = key
            csr = self.connectivity.csr_matrices[key]

            # Get current weights for this CSR
            if trainable_weights is not None:
                current_weights = trainable_weights[start_idx:end_idx]
                # Create temporary CSR with updated weights
                temp_csr = CSR(
                    (current_weights, csr.indices, csr.indptr),
                    shape=csr.shape
                )
            else:
                temp_csr = csr

            # Event-driven sparse matrix multiplication
            # EventArray only processes non-zero (spiking) elements
            target_input = brainevent.EventArray(spikes_flat) @ temp_csr

            # Add to delay buffer
            self.delay_buffer.add_delayed_synaptic_input(
                delay_steps=delay,
                receptor=receptor,
                synaptic_input=target_input,
            )

    def simulate(
        self,
        external_inputs: jnp.ndarray,
        reset_before: bool = True,
        use_lgn_input: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run simulation over multiple time steps.

        Parameters
        ----------
        external_inputs : jnp.ndarray
            External inputs:
            - If use_lgn_input=False: shape (T, batch, n_neurons) - direct current
            - If use_lgn_input=True: shape (T, batch, n_inputs) - LGN firing rates
        reset_before : bool
            Whether to reset state before simulation
        use_lgn_input : bool
            If True, treat inputs as LGN firing rates and process through input layer

        Returns
        -------
        outputs : jnp.ndarray
            Network outputs, shape (T, batch, n_neurons)
        spikes : jnp.ndarray
            Spike outputs, shape (T, batch, n_neurons)
        """
        T = external_inputs.shape[0]

        # Determine batch size
        if external_inputs.ndim == 3:
            batch_size = external_inputs.shape[1]
        else:
            batch_size = 1
            external_inputs = external_inputs[:, None, :]

        if reset_before:
            self.reset(batch_size=batch_size)

        outputs = []
        spikes = []

        # Choose update method based on input type
        update_fn = self.update_with_lgn if use_lgn_input else self.update

        for t in range(T):
            out = update_fn(external_inputs[t])
            outputs.append(out)
            spikes.append(self.population.get_spike())

        return jnp.stack(outputs), jnp.stack(spikes)

    def get_trainable_weights(self) -> Optional[brainstate.ParamState]:
        """Get trainable weight parameters."""
        return self._trainable_weights

    @classmethod
    def from_billeh(
        cls,
        network_data: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None,
        bkg_weights: Optional[np.ndarray] = None,
        dt: float = 1.0,
        mode: str = 'training',
        precision: int = 32,
    ) -> 'V1NetworkBrainstate':
        """Create V1 network from Billeh network data.

        Parameters
        ----------
        network_data : dict
            Network data from load_billeh() or load_network()
        input_data : dict, optional
            Input connectivity from load_billeh() for LGN → V1 connections.
            Contains 'indices', 'weights', 'n_inputs'.
            If not provided, use update() with direct current injection.
        bkg_weights : np.ndarray, optional
            Background weights for noise injection
        dt : float
            Time step in ms
        mode : str
            'simulation' or 'training'
        precision : int
            32 or 64 bit precision

        Returns
        -------
        V1NetworkBrainstate
            Initialized network
        """
        return cls(
            network_data=network_data,
            input_data=input_data,
            bkg_weights=bkg_weights,
            dt=dt,
            mode=mode,
            precision=precision,
        )
