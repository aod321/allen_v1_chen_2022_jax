#!/usr/bin/env python3
"""Convert TensorFlow checkpoints to JAX format.

This script converts pre-trained TensorFlow models from the original
Training-data-driven-V1-model implementation to the JAX format.

Usage:
    python scripts/convert_checkpoint.py \\
        --tf_checkpoint /path/to/tf_checkpoint \\
        --output /path/to/output.npz \\
        --verbose

Reference: Chen et al., Science Advances 2022
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import jax.numpy as jnp
    from jax import Array
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# =============================================================================
# Variable Mapping
# =============================================================================

# Default TensorFlow variable name patterns and their JAX equivalents
DEFAULT_VAR_MAPPING = {
    # GLIF3 neuron parameters
    'billeh_column/threshold_adaptation': 'threshold_adaptation',
    'billeh_column/asc_1': 'asc_amp_1',
    'billeh_column/asc_2': 'asc_amp_2',
    'billeh_column/k_1': 'asc_k_1',
    'billeh_column/k_2': 'asc_k_2',
    'billeh_column/t_ref': 't_ref',
    'billeh_column/g': 'g_leak',
    'billeh_column/E_L': 'E_L',
    'billeh_column/delta_V': 'delta_V',

    # Synaptic weights
    'billeh_column/w_rec': 'recurrent_weights',
    'billeh_column/w_in': 'input_weights',
    'billeh_column/w_out': 'output_weights',

    # Readout weights
    'dense/kernel': 'readout_weights',
    'dense/bias': 'readout_bias',
    'output_layer/kernel': 'readout_weights',
    'output_layer/bias': 'readout_bias',

    # LGN weights
    'lgn/spatial_weights': 'lgn_spatial_weights',
    'lgn/temporal_weights': 'lgn_temporal_weights',
}

# Variables to skip (optimizer state, etc.)
SKIP_PATTERNS = [
    'Adam',
    'adam',
    'optimizer',
    'Momentum',
    'global_step',
    'learning_rate',
    'beta1_power',
    'beta2_power',
    '_slot_',
    'save_counter',
]


# =============================================================================
# Checkpoint Analysis
# =============================================================================

@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: str
    format: str  # 'tf1', 'tf2', 'keras', 'h5'
    variables: Dict[str, Tuple[tuple, str]]  # name -> (shape, dtype)
    total_params: int

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Checkpoint: {self.path}",
            f"Format: {self.format}",
            f"Total parameters: {self.total_params:,}",
            f"Variables ({len(self.variables)}):",
        ]
        for name, (shape, dtype) in sorted(self.variables.items())[:20]:
            lines.append(f"  {name}: {shape} ({dtype})")
        if len(self.variables) > 20:
            lines.append(f"  ... and {len(self.variables) - 20} more")
        return '\n'.join(lines)


def analyze_tf_checkpoint(checkpoint_path: str) -> CheckpointInfo:
    """Analyze TensorFlow checkpoint to get variable info.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        CheckpointInfo with variable details
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for checkpoint analysis")

    path = Path(checkpoint_path)

    # Determine checkpoint format
    if path.suffix == '.h5':
        return _analyze_h5_checkpoint(str(path))
    elif path.suffix == '.keras':
        return _analyze_keras_checkpoint(str(path))
    elif (path.parent / 'checkpoint').exists() or path.suffix == '.index':
        return _analyze_tf_checkpoint_v2(str(path))
    else:
        return _analyze_tf_checkpoint_v1(str(path))


def _analyze_tf_checkpoint_v1(checkpoint_path: str) -> CheckpointInfo:
    """Analyze TensorFlow 1.x checkpoint."""
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_to_shape = reader.get_variable_to_shape_map()
    var_to_dtype = reader.get_variable_to_dtype_map()

    variables = {}
    total_params = 0

    for name, shape in var_to_shape.items():
        # Skip optimizer variables
        if any(p in name for p in SKIP_PATTERNS):
            continue

        dtype = str(var_to_dtype.get(name, 'float32'))
        variables[name] = (tuple(shape), dtype)
        total_params += int(np.prod(shape))

    return CheckpointInfo(
        path=checkpoint_path,
        format='tf1',
        variables=variables,
        total_params=total_params,
    )


def _analyze_tf_checkpoint_v2(checkpoint_path: str) -> CheckpointInfo:
    """Analyze TensorFlow 2.x SavedModel checkpoint."""
    # Remove .index extension if present
    if checkpoint_path.endswith('.index'):
        checkpoint_path = checkpoint_path[:-6]

    reader = tf.train.load_checkpoint(checkpoint_path)
    var_to_shape = reader.get_variable_to_shape_map()
    var_to_dtype = reader.get_variable_to_dtype_map()

    variables = {}
    total_params = 0

    for name, shape in var_to_shape.items():
        if any(p in name for p in SKIP_PATTERNS):
            continue

        dtype = str(var_to_dtype.get(name, 'float32'))
        variables[name] = (tuple(shape), dtype)
        total_params += int(np.prod(shape))

    return CheckpointInfo(
        path=checkpoint_path,
        format='tf2',
        variables=variables,
        total_params=total_params,
    )


def _analyze_h5_checkpoint(checkpoint_path: str) -> CheckpointInfo:
    """Analyze HDF5/Keras checkpoint."""
    import h5py

    variables = {}
    total_params = 0

    def visit_fn(name, obj):
        nonlocal total_params
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = str(obj.dtype)
            variables[name] = (tuple(shape), dtype)
            total_params += int(np.prod(shape))

    with h5py.File(checkpoint_path, 'r') as f:
        f.visititems(visit_fn)

    return CheckpointInfo(
        path=checkpoint_path,
        format='h5',
        variables=variables,
        total_params=total_params,
    )


def _analyze_keras_checkpoint(checkpoint_path: str) -> CheckpointInfo:
    """Analyze Keras .keras format checkpoint."""
    # Keras format is a zip containing model config and weights
    import zipfile
    import tempfile

    variables = {}
    total_params = 0

    with zipfile.ZipFile(checkpoint_path, 'r') as zf:
        # Look for weights file
        weight_files = [f for f in zf.namelist() if 'weights' in f.lower()]
        if weight_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                for wf in weight_files:
                    zf.extract(wf, tmpdir)
                    if wf.endswith('.h5'):
                        sub_info = _analyze_h5_checkpoint(os.path.join(tmpdir, wf))
                        variables.update(sub_info.variables)
                        total_params += sub_info.total_params

    return CheckpointInfo(
        path=checkpoint_path,
        format='keras',
        variables=variables,
        total_params=total_params,
    )


# =============================================================================
# Checkpoint Conversion
# =============================================================================

def convert_tf_to_jax(
    checkpoint_path: str,
    output_path: str,
    variable_mapping: Optional[Dict[str, str]] = None,
    skip_patterns: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Convert TensorFlow checkpoint to JAX format.

    Args:
        checkpoint_path: Path to TF checkpoint
        output_path: Path for output .npz file
        variable_mapping: Optional custom variable name mapping
        skip_patterns: Additional patterns to skip
        verbose: Print detailed conversion info

    Returns:
        Dictionary of converted parameters
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for checkpoint conversion")

    # Analyze checkpoint first
    info = analyze_tf_checkpoint(checkpoint_path)
    if verbose:
        print(info.summary())
        print()

    # Combine mappings
    var_map = DEFAULT_VAR_MAPPING.copy()
    if variable_mapping:
        var_map.update(variable_mapping)

    # Combine skip patterns
    all_skip = SKIP_PATTERNS.copy()
    if skip_patterns:
        all_skip.extend(skip_patterns)

    # Load and convert variables
    params = {}
    converted_count = 0
    skipped_count = 0

    if info.format == 'h5':
        params = _convert_h5_checkpoint(checkpoint_path, var_map, all_skip, verbose)
    else:
        params = _convert_tf_checkpoint(checkpoint_path, var_map, all_skip, verbose)

    # Save to output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with metadata
    metadata = {
        'source_path': checkpoint_path,
        'source_format': info.format,
        'conversion_date': str(np.datetime64('now')),
        'variable_count': len(params),
    }

    np.savez(
        output_path,
        **params,
        _metadata=json.dumps(metadata),
    )

    if verbose:
        print(f"\nConversion complete:")
        print(f"  Converted: {len(params)} variables")
        print(f"  Output: {output_path}")
        total_params = sum(np.prod(v.shape) for v in params.values())
        print(f"  Total parameters: {total_params:,}")

    return params


def _convert_tf_checkpoint(
    checkpoint_path: str,
    var_map: Dict[str, str],
    skip_patterns: List[str],
    verbose: bool,
) -> Dict[str, np.ndarray]:
    """Convert TF 1.x/2.x checkpoint."""
    # Remove .index suffix if present
    if checkpoint_path.endswith('.index'):
        checkpoint_path = checkpoint_path[:-6]

    reader = tf.train.load_checkpoint(checkpoint_path)
    var_names = list(reader.get_variable_to_shape_map().keys())

    params = {}

    for tf_name in var_names:
        # Skip optimizer variables
        if any(p in tf_name for p in skip_patterns):
            if verbose:
                print(f"  Skip: {tf_name}")
            continue

        # Load tensor
        value = reader.get_tensor(tf_name)

        # Map variable name
        jax_name = None
        for pattern, mapped_name in var_map.items():
            if pattern in tf_name:
                jax_name = mapped_name
                break

        if jax_name is None:
            # Default: clean up TF name
            jax_name = (
                tf_name
                .replace('/', '_')
                .replace(':0', '')
                .replace('.', '_')
            )

        # Handle potential duplicates
        if jax_name in params:
            # Append suffix for duplicates
            i = 2
            while f"{jax_name}_{i}" in params:
                i += 1
            jax_name = f"{jax_name}_{i}"

        params[jax_name] = value.astype(np.float32)

        if verbose:
            print(f"  {tf_name} -> {jax_name}: {value.shape}")

    return params


def _convert_h5_checkpoint(
    checkpoint_path: str,
    var_map: Dict[str, str],
    skip_patterns: List[str],
    verbose: bool,
) -> Dict[str, np.ndarray]:
    """Convert HDF5/Keras checkpoint."""
    import h5py

    params = {}

    def extract_fn(name: str, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        # Skip optimizer variables
        if any(p in name for p in skip_patterns):
            if verbose:
                print(f"  Skip: {name}")
            return

        # Load data
        value = np.array(obj)

        # Map variable name
        jax_name = None
        for pattern, mapped_name in var_map.items():
            if pattern in name:
                jax_name = mapped_name
                break

        if jax_name is None:
            # Default: clean up H5 path
            jax_name = (
                name
                .replace('/', '_')
                .replace('model_weights_', '')
                .replace('_weights', '')
            )

        # Handle potential duplicates
        if jax_name in params:
            i = 2
            while f"{jax_name}_{i}" in params:
                i += 1
            jax_name = f"{jax_name}_{i}"

        params[jax_name] = value.astype(np.float32)

        if verbose:
            print(f"  {name} -> {jax_name}: {value.shape}")

    with h5py.File(checkpoint_path, 'r') as f:
        f.visititems(extract_fn)

    return params


# =============================================================================
# JAX Checkpoint Loading
# =============================================================================

def load_converted_checkpoint(
    checkpoint_path: str,
    as_jax: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load converted JAX checkpoint.

    Args:
        checkpoint_path: Path to .npz file
        as_jax: Convert to JAX arrays (requires JAX)

    Returns:
        Tuple of (params_dict, metadata_dict)
    """
    data = np.load(checkpoint_path, allow_pickle=True)

    # Extract metadata if present
    metadata = {}
    if '_metadata' in data:
        metadata = json.loads(str(data['_metadata']))

    # Extract params
    params = {}
    for key in data.files:
        if key.startswith('_'):
            continue
        value = data[key]
        if as_jax and JAX_AVAILABLE:
            params[key] = jnp.array(value)
        else:
            params[key] = value

    return params, metadata


def validate_converted_checkpoint(
    checkpoint_path: str,
    expected_shapes: Optional[Dict[str, tuple]] = None,
) -> Dict[str, Any]:
    """Validate converted checkpoint.

    Args:
        checkpoint_path: Path to .npz file
        expected_shapes: Optional dict of expected shapes

    Returns:
        Validation report dictionary
    """
    params, metadata = load_converted_checkpoint(checkpoint_path, as_jax=False)

    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'variable_count': len(params),
        'total_params': sum(np.prod(v.shape) for v in params.values()),
    }

    # Check NaN/Inf
    for name, value in params.items():
        if np.any(np.isnan(value)):
            report['errors'].append(f"{name}: contains NaN values")
            report['valid'] = False
        if np.any(np.isinf(value)):
            report['warnings'].append(f"{name}: contains Inf values")

    # Check expected shapes
    if expected_shapes:
        for name, expected in expected_shapes.items():
            if name not in params:
                report['errors'].append(f"{name}: missing from checkpoint")
                report['valid'] = False
            elif params[name].shape != expected:
                report['errors'].append(
                    f"{name}: shape mismatch (got {params[name].shape}, expected {expected})"
                )
                report['valid'] = False

    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert TensorFlow checkpoints to JAX format'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze checkpoint')
    analyze_parser.add_argument('checkpoint', type=str, help='Checkpoint path')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert checkpoint')
    convert_parser.add_argument('--tf_checkpoint', type=str, required=True,
                                help='TensorFlow checkpoint path')
    convert_parser.add_argument('--output', type=str, required=True,
                                help='Output .npz path')
    convert_parser.add_argument('--mapping', type=str, default=None,
                                help='Custom variable mapping JSON file')
    convert_parser.add_argument('--verbose', '-v', action='store_true',
                                help='Verbose output')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate converted checkpoint')
    validate_parser.add_argument('checkpoint', type=str, help='JAX checkpoint path (.npz)')

    args = parser.parse_args()

    if args.command == 'analyze':
        info = analyze_tf_checkpoint(args.checkpoint)
        print(info.summary())

    elif args.command == 'convert':
        # Load custom mapping if provided
        custom_mapping = None
        if args.mapping:
            with open(args.mapping, 'r') as f:
                custom_mapping = json.load(f)

        convert_tf_to_jax(
            args.tf_checkpoint,
            args.output,
            variable_mapping=custom_mapping,
            verbose=args.verbose,
        )

    elif args.command == 'validate':
        report = validate_converted_checkpoint(args.checkpoint)
        print(f"Validation: {'PASSED' if report['valid'] else 'FAILED'}")
        print(f"Variables: {report['variable_count']}")
        print(f"Parameters: {report['total_params']:,}")

        if report['errors']:
            print("\nErrors:")
            for err in report['errors']:
                print(f"  - {err}")

        if report['warnings']:
            print("\nWarnings:")
            for warn in report['warnings']:
                print(f"  - {warn}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
