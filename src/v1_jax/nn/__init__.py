"""Neural network modules for V1 model."""

from .spike_functions import (
    spike_gauss,
    gauss_pseudo,
    pseudo_derivative,
    spike_piecewise,
    spike_sigmoid,
)
from .constraints import (
    apply_dale_constraint,
    dale_law_projection,
    SignedConstraint,
    SparseSignedConstraint,
)
from .synaptic import (
    exp_convolve,
    psc_dynamics,
    alpha_synapse,
    exponential_synapse,
    SynapticFilter,
)
from .glif3_cell import (
    GLIF3State,
    GLIF3Params,
    GLIF3Cell,
    glif3_step,
    make_glif3_step_fn,
    glif3_unroll,
)
from .sparse_layer import (
    SparseConnectivity,
    sparse_matmul_bcoo,
    sparse_input_layer,
    create_recurrent_matmul_fn,
    InputLayer,
    RecurrentLayer,
    prepare_recurrent_connectivity,
    prepare_input_connectivity,
)

__all__ = [
    # Spike functions
    "spike_gauss",
    "gauss_pseudo",
    "pseudo_derivative",
    "spike_piecewise",
    "spike_sigmoid",
    # Constraints
    "apply_dale_constraint",
    "dale_law_projection",
    "SignedConstraint",
    "SparseSignedConstraint",
    # Synaptic
    "exp_convolve",
    "psc_dynamics",
    "alpha_synapse",
    "exponential_synapse",
    "SynapticFilter",
    # GLIF3 neuron
    "GLIF3State",
    "GLIF3Params",
    "GLIF3Cell",
    "glif3_step",
    "make_glif3_step_fn",
    "glif3_unroll",
    # Sparse layers
    "SparseConnectivity",
    "sparse_matmul_bcoo",
    "sparse_input_layer",
    "create_recurrent_matmul_fn",
    "InputLayer",
    "RecurrentLayer",
    "prepare_recurrent_connectivity",
    "prepare_input_connectivity",
]
