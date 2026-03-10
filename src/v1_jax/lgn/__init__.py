"""LGN preprocessing modules.

This package implements the LGN (Lateral Geniculate Nucleus) preprocessing
model for converting visual stimuli into firing rates.

Modules:
    params_loader: LGN parameter loading utilities
    spatial_filter: Gaussian spatial filtering and bilinear interpolation
    temporal_filter: Temporal convolution and firing rate computation
    lgn_model: Main LGN class combining spatial and temporal filtering
    bmtk_lgn: BMTK-style LGN with cell type-specific parameters
"""

from .params_loader import (
    LGNParams,
    find_lgn_files,
    load_lgn_params,
    load_lgn_params_from_dir,
    get_neuron_groups,
)

from .spatial_filter import (
    create_gaussian_kernel,
    create_gaussian_kernel_trimmed,
    gaussian_conv2d,
    bilinear_select,
    batch_spatial_filter,
    spatial_filter_single_sigma,
    precompute_gaussian_kernels,
    SpatialFilter,
)

from .temporal_filter import (
    temporal_filter,
    temporal_filter_scan,
    temporal_filter_fft,
    transfer_function,
    compute_firing_rates,
    TemporalFilter,
)

from .lgn_model import (
    LGN,
    make_lgn_forward_fn,
    create_lgn_model,
)

from .bmtk_lgn import (
    BMTKLGN,
    BMTKLGNParams,
    CellTypeParams,
    CELL_TYPE_PARAMS,
    cosine_bump_kernel,
    create_temporal_kernel_from_params,
    load_bmtk_lgn_params,
    bmtk_transfer_function,
    create_bmtk_lgn_model,
)

__all__ = [
    # Params loading
    "LGNParams",
    "find_lgn_files",
    "load_lgn_params",
    "load_lgn_params_from_dir",
    "get_neuron_groups",
    # Spatial filtering
    "create_gaussian_kernel",
    "create_gaussian_kernel_trimmed",
    "gaussian_conv2d",
    "bilinear_select",
    "batch_spatial_filter",
    "spatial_filter_single_sigma",
    "precompute_gaussian_kernels",
    "SpatialFilter",
    # Temporal filtering
    "temporal_filter",
    "temporal_filter_scan",
    "temporal_filter_fft",
    "transfer_function",
    "compute_firing_rates",
    "TemporalFilter",
    # LGN model
    "LGN",
    "make_lgn_forward_fn",
    "create_lgn_model",
    # BMTK LGN model
    "BMTKLGN",
    "BMTKLGNParams",
    "CellTypeParams",
    "CELL_TYPE_PARAMS",
    "cosine_bump_kernel",
    "create_temporal_kernel_from_params",
    "load_bmtk_lgn_params",
    "bmtk_transfer_function",
    "create_bmtk_lgn_model",
]
