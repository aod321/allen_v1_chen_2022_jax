"""LGN preprocessing modules.

This package implements the LGN (Lateral Geniculate Nucleus) preprocessing
model for converting visual stimuli into firing rates.

Modules:
    spatial_filter: Gaussian spatial filtering and bilinear interpolation
    temporal_filter: Temporal convolution and firing rate computation
    lgn_model: Main LGN class combining spatial and temporal filtering
"""

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
    LGNParams,
    load_lgn_params,
    LGN,
    make_lgn_forward_fn,
    create_lgn_model,
)

__all__ = [
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
    "LGNParams",
    "load_lgn_params",
    "LGN",
    "make_lgn_forward_fn",
    "create_lgn_model",
]
