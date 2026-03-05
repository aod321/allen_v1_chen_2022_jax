"""High-level model composition modules."""

from .v1_network import (
    V1NetworkConfig,
    V1NetworkState,
    V1NetworkOutput,
    V1Network,
    create_v1_network,
    make_v1_forward_fn,
    v1_network_step,
    make_v1_step_fn,
)

from .readout import (
    ReadoutParams,
    dense_readout,
    select_readout_neurons,
    sparse_readout,
    chunk_readout,
    DenseReadout,
    BinaryReadout,
    MultiClassReadout,
    create_readout,
    apply_readout_jit,
    make_readout_fn,
)

__all__ = [
    # V1 Network
    "V1NetworkConfig",
    "V1NetworkState",
    "V1NetworkOutput",
    "V1Network",
    "create_v1_network",
    "make_v1_forward_fn",
    "v1_network_step",
    "make_v1_step_fn",
    # Readout
    "ReadoutParams",
    "dense_readout",
    "select_readout_neurons",
    "sparse_readout",
    "chunk_readout",
    "DenseReadout",
    "BinaryReadout",
    "MultiClassReadout",
    "create_readout",
    "apply_readout_jit",
    "make_readout_fn",
]
