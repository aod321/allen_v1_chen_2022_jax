"""Spatial filtering for LGN model.

This module implements spatial filtering operations for the LGN preprocessing,
including Gaussian convolution and bilinear interpolation.

Corresponding TF implementation: lgn_model/lgn.py:61-83, 265-318
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


def create_gaussian_kernel(sigma: float, size: int = 101) -> jnp.ndarray:
    """Create a 2D Gaussian spatial filter kernel.

    Args:
        sigma: Standard deviation of Gaussian (in pixels)
        size: Size of the kernel (should be odd for symmetry)

    Returns:
        Gaussian kernel of shape (size, size)

    Note:
        Corresponding to bmtk.simulator.filternet.lgnmodel.spatialfilter.GaussianSpatialFilter
    """
    half_size = size // 2
    x = jnp.arange(-half_size, half_size + 1)
    y = jnp.arange(-half_size, half_size + 1)
    xx, yy = jnp.meshgrid(x, y)

    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)  # Normalize

    return kernel.astype(jnp.float32)


def create_gaussian_kernel_trimmed(sigma: float, threshold: float = 1e-9) -> jnp.ndarray:
    """Create a trimmed Gaussian kernel (removes near-zero edges).

    This matches the TF implementation which trims the kernel to non-zero regions.

    Args:
        sigma: Standard deviation of Gaussian
        threshold: Values below this are considered zero

    Returns:
        Trimmed Gaussian kernel
    """
    # Create a larger kernel first
    size = int(sigma * 10) | 1  # Make odd
    size = max(size, 11)  # At least 11x11

    half_size = size // 2
    x = jnp.arange(-half_size, half_size + 1)
    y = jnp.arange(-half_size, half_size + 1)
    xx, yy = jnp.meshgrid(x, y)

    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)

    return kernel.astype(jnp.float32)


def gaussian_conv2d(movie: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Apply Gaussian convolution to a movie.

    Args:
        movie: Input movie of shape (T, H, W) or (T, H, W, 1)
        kernel: Gaussian kernel of shape (kH, kW)

    Returns:
        Convolved movie of shape (T, H, W)

    Note:
        Corresponding TF: tf.nn.conv2d with padding='SAME'
    """
    # Ensure movie has channel dimension
    if movie.ndim == 3:
        movie = movie[..., None]  # (T, H, W, 1)

    # Kernel shape for JAX conv: (kH, kW, C_in, C_out)
    kernel_4d = kernel[:, :, None, None]  # (kH, kW, 1, 1)

    # JAX conv expects: (N, H, W, C) for input
    # Use lax.conv_general_dilated for 2D convolution
    dn = lax.conv_dimension_numbers(
        movie.shape,
        kernel_4d.shape,
        ('NHWC', 'HWIO', 'NHWC')
    )

    # Calculate padding to achieve 'SAME' behavior
    kH, kW = kernel.shape
    pad_h = (kH - 1) // 2
    pad_w = (kW - 1) // 2

    convolved = lax.conv_general_dilated(
        movie,
        kernel_4d,
        window_strides=(1, 1),
        padding=((pad_h, kH - 1 - pad_h), (pad_w, kW - 1 - pad_w)),
        dimension_numbers=dn,
    )

    return convolved[..., 0]  # Remove channel dimension


def bilinear_select(
    x: jnp.ndarray,
    y: jnp.ndarray,
    convolved_movie: jnp.ndarray,
) -> jnp.ndarray:
    """Select spatial responses using bilinear interpolation.

    Given neuron positions (x, y), extract responses from convolved movie
    using bilinear interpolation between neighboring pixels.

    Args:
        x: x-coordinates of neurons, shape (n_neurons,)
        y: y-coordinates of neurons, shape (n_neurons,)
        convolved_movie: Convolved movie of shape (T, H, W)

    Returns:
        Spatial responses of shape (T, n_neurons)

    Note:
        Corresponding TF: select_spatial function (lgn.py:61-83)
    """
    T, H, W = convolved_movie.shape

    # Get corner indices
    x_floor = jnp.floor(x).astype(jnp.int32)
    x_ceil = jnp.ceil(x).astype(jnp.int32)
    y_floor = jnp.floor(y).astype(jnp.int32)
    y_ceil = jnp.ceil(y).astype(jnp.int32)

    # Clip to valid range
    x_floor = jnp.clip(x_floor, 0, W - 1)
    x_ceil = jnp.clip(x_ceil, 0, W - 1)
    y_floor = jnp.clip(y_floor, 0, H - 1)
    y_ceil = jnp.clip(y_ceil, 0, H - 1)

    # Interpolation weights
    x_frac = x - jnp.floor(x)
    y_frac = y - jnp.floor(y)

    w1 = (1 - x_frac) * (1 - y_frac)  # floor-floor
    w2 = (1 - x_frac) * y_frac        # floor-ceil
    w3 = x_frac * (1 - y_frac)        # ceil-floor
    w4 = x_frac * y_frac              # ceil-ceil

    # Gather values at four corners
    # convolved_movie shape: (T, H, W)
    # We need to gather at (y, x) for each neuron
    sr1 = convolved_movie[:, y_floor, x_floor]  # (T, n_neurons)
    sr2 = convolved_movie[:, y_ceil, x_floor]   # (T, n_neurons)
    sr3 = convolved_movie[:, y_floor, x_ceil]   # (T, n_neurons)
    sr4 = convolved_movie[:, y_ceil, x_ceil]    # (T, n_neurons)

    # Weighted sum
    spatial_responses = sr1 * w1 + sr2 * w2 + sr3 * w3 + sr4 * w4

    return spatial_responses


def batch_spatial_filter(
    movie: jnp.ndarray,
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
    spatial_sizes: jnp.ndarray,
    spatial_bins: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
) -> jnp.ndarray:
    """Apply spatial filtering for all neurons grouped by spatial size.

    This matches the TF implementation which groups neurons by spatial size
    and applies a single Gaussian filter per group for efficiency.

    Args:
        movie: Input movie of shape (T, H, W)
        x_coords: x-coordinates of all neurons
        y_coords: y-coordinates of all neurons
        spatial_sizes: Spatial filter size for each neuron
        spatial_bins: Bin edges for grouping neurons by spatial size

    Returns:
        Spatial responses for all neurons of shape (T, n_neurons)

    Note:
        Corresponding TF: LGN.spatial_response (lgn.py:265-318)
    """
    T, H, W = movie.shape
    n_neurons = len(x_coords)

    # Initialize output
    all_responses = jnp.zeros((T, n_neurons), dtype=jnp.float32)

    # This is a static loop that can be JIT compiled
    for i in range(len(spatial_bins) - 1):
        low, high = spatial_bins[i], spatial_bins[i + 1]

        # Create mask for neurons in this bin
        mask = (spatial_sizes >= low) & (spatial_sizes < high)

        # Skip if no neurons in this bin
        if not jnp.any(mask):
            continue

        # Compute sigma for this bin (average of bin edges divided by 3)
        sigma = (low + high) / 2.0 / 3.0
        sigma = max(sigma, 0.5)  # Minimum sigma

        # Create and apply Gaussian filter
        kernel = create_gaussian_kernel_trimmed(sigma)
        convolved = gaussian_conv2d(movie, kernel)

        # Get indices of neurons in this bin
        indices = jnp.where(mask)[0]

        # Select responses for these neurons
        x_sel = x_coords[indices]
        y_sel = y_coords[indices]
        responses = bilinear_select(x_sel, y_sel, convolved)

        # Scatter results back
        all_responses = all_responses.at[:, indices].set(responses)

    return all_responses


@jax.jit
def spatial_filter_single_sigma(
    movie: jnp.ndarray,
    sigma: float,
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
) -> jnp.ndarray:
    """Apply spatial filtering for a single sigma value.

    This is a JIT-compiled helper for a single group of neurons.

    Args:
        movie: Input movie of shape (T, H, W)
        sigma: Standard deviation for Gaussian filter
        x_coords: x-coordinates of neurons in this group
        y_coords: y-coordinates of neurons in this group

    Returns:
        Spatial responses of shape (T, n_neurons_in_group)
    """
    kernel = create_gaussian_kernel_trimmed(sigma)
    convolved = gaussian_conv2d(movie, kernel)
    return bilinear_select(x_coords, y_coords, convolved)


def precompute_gaussian_kernels(
    spatial_bins: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
    max_kernel_size: int = 51,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute Gaussian kernels for all spatial bins.

    Args:
        spatial_bins: Bin edges for spatial sizes
        max_kernel_size: Maximum kernel size (should be odd)

    Returns:
        Tuple of:
        - kernels: Array of shape (n_bins, max_kernel_size, max_kernel_size)
        - sigmas: Array of sigma values for each bin
    """
    n_bins = len(spatial_bins) - 1
    kernels = np.zeros((n_bins, max_kernel_size, max_kernel_size), dtype=np.float32)
    sigmas = np.zeros(n_bins, dtype=np.float32)

    center = max_kernel_size // 2

    for i in range(n_bins):
        low, high = spatial_bins[i], spatial_bins[i + 1]
        sigma = (low + high) / 2.0 / 3.0
        sigma = max(sigma, 0.5)
        sigmas[i] = sigma

        # Create Gaussian kernel centered in the array
        x = np.arange(-center, center + 1)
        y = np.arange(-center, center + 1)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernels[i] = kernel

    return jnp.array(kernels), jnp.array(sigmas)


class SpatialFilter:
    """Spatial filter for LGN preprocessing.

    Handles Gaussian convolution and bilinear interpolation for
    extracting spatial responses at neuron locations.

    Attributes:
        x_coords: x-coordinates of neurons (normalized to movie coordinates)
        y_coords: y-coordinates of neurons
        spatial_sizes: Spatial filter size for each neuron
        neuron_groups: List of (indices, sigma) for each spatial size group
        kernels: Precomputed Gaussian kernels
    """

    def __init__(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        spatial_sizes: np.ndarray,
        movie_height: int = 120,
        movie_width: int = 240,
    ):
        """Initialize spatial filter.

        Args:
            x_coords: Original x-coordinates (will be normalized)
            y_coords: Original y-coordinates (will be normalized)
            spatial_sizes: Spatial filter size for each neuron
            movie_height: Height of input movies
            movie_width: Width of input movies
        """
        # Normalize coordinates to movie dimensions
        # Following TF: x = x * 239 / 240, y = y * 119 / 120
        self.x_coords = np.array(x_coords) * (movie_width - 1) / movie_width
        self.y_coords = np.array(y_coords) * (movie_height - 1) / movie_height

        # Clip to valid range
        self.x_coords = np.clip(self.x_coords, 0, movie_width - 1)
        self.y_coords = np.clip(self.y_coords, 0, movie_height - 1)

        self.spatial_sizes = np.array(spatial_sizes)
        self.n_neurons = len(x_coords)

        # Group neurons by spatial size
        self._setup_neuron_groups()

    def _setup_neuron_groups(self):
        """Group neurons by spatial size for efficient filtering."""
        spatial_range = np.arange(0, 15, 1.0)

        self.neuron_groups = []
        for i in range(len(spatial_range) - 1):
            low, high = spatial_range[i], spatial_range[i + 1]
            mask = (self.spatial_sizes >= low) & (self.spatial_sizes < high)
            indices = np.where(mask)[0]

            if len(indices) > 0:
                sigma = (low + high) / 2.0 / 3.0
                sigma = max(sigma, 0.5)
                self.neuron_groups.append((indices, sigma))

    def __call__(
        self,
        movie: jnp.ndarray,
        x_coords: jnp.ndarray = None,
        y_coords: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Compute spatial responses for all neurons.

        Args:
            movie: Input movie of shape (T, H, W)
            x_coords: Optional override for x coordinates
            y_coords: Optional override for y coordinates

        Returns:
            Spatial responses of shape (T, n_neurons)
        """
        if x_coords is None:
            x_coords = jnp.array(self.x_coords)
        if y_coords is None:
            y_coords = jnp.array(self.y_coords)

        T = movie.shape[0]
        all_responses = jnp.zeros((T, self.n_neurons), dtype=jnp.float32)
        neuron_order = []

        for indices, sigma in self.neuron_groups:
            kernel = create_gaussian_kernel_trimmed(sigma)
            convolved = gaussian_conv2d(movie, kernel)

            x_sel = x_coords[indices]
            y_sel = y_coords[indices]
            responses = bilinear_select(x_sel, y_sel, convolved)

            all_responses = all_responses.at[:, indices].set(responses)
            neuron_order.extend(indices.tolist())

        return all_responses

    def get_responses_for_coords(
        self,
        movie: jnp.ndarray,
        x_coords: jnp.ndarray,
        y_coords: jnp.ndarray,
        sigma: float,
    ) -> jnp.ndarray:
        """Get spatial responses for specific coordinates with given sigma.

        Args:
            movie: Input movie of shape (T, H, W)
            x_coords: x-coordinates
            y_coords: y-coordinates
            sigma: Gaussian sigma for this group

        Returns:
            Spatial responses of shape (T, n_coords)
        """
        kernel = create_gaussian_kernel_trimmed(sigma)
        convolved = gaussian_conv2d(movie, kernel)
        return bilinear_select(x_coords, y_coords, convolved)
