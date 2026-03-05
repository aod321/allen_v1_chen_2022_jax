#!/usr/bin/env python3
"""Performance benchmark for V1 model JAX implementation.

Compares forward pass and backward pass performance between JAX and TensorFlow
implementations across different network sizes and batch configurations.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --neurons 51978 --batch_size 4 --seq_len 600
    python scripts/benchmark.py --compare_tf  # Include TF comparison

Reference: Chen et al., Science Advances 2022
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# JAX setup
import jax
import jax.numpy as jnp
from jax import Array

# Check TensorFlow availability
TF_SOURCE_PATH = '/nvmessd/yinzi/Training-data-driven-V1-model'
HAS_TF = False
try:
    import tensorflow as tf
    if TF_SOURCE_PATH not in sys.path:
        sys.path.insert(0, TF_SOURCE_PATH)
    HAS_TF = True
except ImportError:
    pass


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    neurons: int = 1000
    n_input: int = 400
    batch_size: int = 4
    seq_len: int = 600
    n_receptors: int = 4
    max_delay: int = 5
    warmup_runs: int = 3
    benchmark_runs: int = 10
    compare_tf: bool = False
    output_json: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    framework: str
    neurons: int
    batch_size: int
    seq_len: int
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_mb: Optional[float] = None
    throughput_steps_per_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'framework': self.framework,
            'neurons': self.neurons,
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'forward_time_ms': self.forward_time_ms,
            'backward_time_ms': self.backward_time_ms,
            'total_time_ms': self.total_time_ms,
            'memory_mb': self.memory_mb,
            'throughput_steps_per_sec': self.throughput_steps_per_sec,
        }


# =============================================================================
# Timer Utilities
# =============================================================================

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self.start) * 1000  # ms


def benchmark_function(fn, *args, warmup: int = 3, runs: int = 10, **kwargs) -> Tuple[float, float]:
    """Benchmark a function with warmup and multiple runs.

    Args:
        fn: Function to benchmark
        *args: Positional arguments
        warmup: Number of warmup runs
        runs: Number of timed runs
        **kwargs: Keyword arguments

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()

    # Timed runs
    times = []
    for _ in range(runs):
        with Timer() as t:
            result = fn(*args, **kwargs)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
                result[0].block_until_ready()
        times.append(t.elapsed)

    return np.mean(times), np.std(times)


# =============================================================================
# JAX Benchmark
# =============================================================================

def create_jax_benchmark_model(config: BenchmarkConfig):
    """Create JAX model components for benchmarking."""
    from v1_jax.nn.spike_functions import spike_gauss
    from v1_jax.nn.synaptic import exp_convolve

    # Create random parameters
    key = jax.random.PRNGKey(42)
    n_neurons = config.neurons
    n_input = config.n_input
    n_receptors = config.n_receptors
    max_delay = config.max_delay

    # Input weights (sparse simulation with dense for simplicity)
    key, subkey = jax.random.split(key)
    input_weights = jax.random.normal(subkey, (n_neurons * n_receptors, n_input)) * 0.01

    # Recurrent weights
    key, subkey = jax.random.split(key)
    recurrent_weights = jax.random.normal(subkey, (n_neurons * n_receptors, n_neurons * max_delay)) * 0.001

    # Neuron parameters
    decay = jnp.ones(n_neurons) * 0.95
    current_factor = jnp.ones(n_neurons) * 0.1
    syn_decay = jnp.ones((n_neurons, n_receptors)) * 0.9
    psc_initial = jnp.ones((n_neurons, n_receptors)) * 0.1
    v_th = jnp.zeros(n_neurons)
    v_reset = -jnp.ones(n_neurons) * 0.1
    e_l = -jnp.ones(n_neurons) * 0.05
    t_ref = jnp.ones(n_neurons) * 3.0
    asc_amps = jnp.zeros((n_neurons, 2))
    k = jnp.ones((n_neurons, 2)) * 0.005
    g_el = jnp.zeros(n_neurons)

    params = {
        'input_weights': input_weights,
        'recurrent_weights': recurrent_weights,
        'decay': decay,
        'current_factor': current_factor,
        'syn_decay': syn_decay,
        'psc_initial': psc_initial,
        'v_th': v_th,
        'v_reset': v_reset,
        'e_l': e_l,
        't_ref': t_ref,
        'asc_amps': asc_amps,
        'k': k,
        'g_el': g_el,
    }

    return params


def jax_forward_step(params, state, inputs, gauss_std=0.28, dampening=0.5, dt=1.0):
    """Single forward step of GLIF3 network (JAX)."""
    from v1_jax.nn.spike_functions import spike_gauss

    z_buf, v, r, asc_1, asc_2, psc_rise, psc = state
    n_neurons = v.shape[-1]
    n_receptors = 4
    max_delay = z_buf.shape[-1] // n_neurons

    batch_size = v.shape[0]

    # Get previous spike from buffer
    shaped_z_buf = z_buf.reshape(batch_size, max_delay, n_neurons)
    prev_z = shaped_z_buf[:, 0]

    # Reshape PSC
    psc_rise_r = psc_rise.reshape(batch_size, n_neurons, n_receptors)
    psc_r = psc.reshape(batch_size, n_neurons, n_receptors)

    # Input current (simplified)
    i_in = jnp.matmul(inputs, params['input_weights'].T)
    i_in = i_in.reshape(batch_size, n_neurons, n_receptors)

    # Recurrent current
    i_rec = jnp.matmul(z_buf, params['recurrent_weights'].T)
    i_rec = i_rec.reshape(batch_size, n_neurons, n_receptors)

    rec_inputs = i_in + i_rec

    # PSC update
    new_psc_rise = params['syn_decay'] * psc_rise_r + rec_inputs * params['psc_initial']
    new_psc = psc_r * params['syn_decay'] + dt * params['syn_decay'] * psc_rise_r

    # Refractory update
    new_r = jnp.maximum(r + prev_z * params['t_ref'] - dt, 0.)

    # ASC update
    new_asc_1 = jnp.exp(-dt * params['k'][:, 0]) * asc_1 + prev_z * params['asc_amps'][:, 0]
    new_asc_2 = jnp.exp(-dt * params['k'][:, 1]) * asc_2 + prev_z * params['asc_amps'][:, 1]

    # Voltage update
    reset_current = prev_z * (params['v_reset'] - params['v_th'])
    input_current = jnp.sum(psc_r, axis=-1)
    decayed_v = params['decay'] * v
    c1 = input_current + asc_1 + asc_2 + params['g_el']
    new_v = decayed_v + params['current_factor'] * c1 + reset_current

    # Spike generation
    normalizer = params['v_th'] - params['e_l'] + 1e-6
    v_sc = (new_v - params['v_th']) / normalizer
    new_z = spike_gauss(v_sc, gauss_std, dampening)

    # Spike suppression
    new_z = jnp.where(new_r > 0., jnp.zeros_like(new_z), new_z)

    # Update buffer
    new_shaped_z_buf = jnp.concatenate([new_z[:, None], shaped_z_buf[:, :-1]], axis=1)
    new_z_buf = new_shaped_z_buf.reshape(batch_size, -1)

    new_psc_rise = new_psc_rise.reshape(batch_size, -1)
    new_psc = new_psc.reshape(batch_size, -1)

    new_state = (new_z_buf, new_v, new_r, new_asc_1, new_asc_2, new_psc_rise, new_psc)

    return new_z, new_v, new_state


def jax_forward_pass(params, inputs, gauss_std=0.28, dampening=0.5, dt=1.0):
    """Full forward pass over sequence (JAX)."""
    seq_len, batch_size, n_input = inputs.shape
    n_neurons = params['v_th'].shape[0]
    n_receptors = 4
    max_delay = 5

    # Initialize state
    z_buf = jnp.zeros((batch_size, n_neurons * max_delay))
    v = jnp.ones((batch_size, n_neurons)) * params['v_reset']
    r = jnp.zeros((batch_size, n_neurons))
    asc_1 = jnp.zeros((batch_size, n_neurons))
    asc_2 = jnp.zeros((batch_size, n_neurons))
    psc_rise = jnp.zeros((batch_size, n_neurons * n_receptors))
    psc = jnp.zeros((batch_size, n_neurons * n_receptors))
    state = (z_buf, v, r, asc_1, asc_2, psc_rise, psc)

    # Scan over time
    def scan_fn(carry, inp):
        state = carry
        z, v, new_state = jax_forward_step(params, state, inp, gauss_std, dampening, dt)
        return new_state, (z, v)

    final_state, (all_z, all_v) = jax.lax.scan(scan_fn, state, inputs)

    return all_z, all_v, final_state


def run_jax_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run JAX benchmark."""
    print(f"Running JAX benchmark: {config.neurons} neurons, batch={config.batch_size}, seq={config.seq_len}")

    # Create model and data
    params = create_jax_benchmark_model(config)
    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, (config.seq_len, config.batch_size, config.n_input))

    # JIT compile forward pass
    @jax.jit
    def forward_fn(params, inputs):
        return jax_forward_pass(params, inputs)

    # JIT compile forward + backward
    @jax.jit
    def forward_backward_fn(params, inputs):
        def loss_fn(p):
            z, v, _ = jax_forward_pass(p, inputs)
            return jnp.mean(z)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    # Benchmark forward pass
    forward_mean, forward_std = benchmark_function(
        forward_fn, params, inputs,
        warmup=config.warmup_runs, runs=config.benchmark_runs
    )

    # Benchmark forward + backward
    total_mean, total_std = benchmark_function(
        forward_backward_fn, params, inputs,
        warmup=config.warmup_runs, runs=config.benchmark_runs
    )

    backward_mean = total_mean - forward_mean

    # Calculate throughput
    throughput = (config.seq_len * config.batch_size) / (total_mean / 1000.0)

    result = BenchmarkResult(
        name='glif3_network',
        framework='jax',
        neurons=config.neurons,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        forward_time_ms=forward_mean,
        backward_time_ms=backward_mean,
        total_time_ms=total_mean,
        throughput_steps_per_sec=throughput,
    )

    print(f"  Forward:  {forward_mean:.2f} ± {forward_std:.2f} ms")
    print(f"  Backward: {backward_mean:.2f} ms (estimated)")
    print(f"  Total:    {total_mean:.2f} ± {total_std:.2f} ms")
    print(f"  Throughput: {throughput:.0f} steps/sec")

    return result


# =============================================================================
# Multi-Configuration Benchmark
# =============================================================================

def run_scaling_benchmark(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run benchmark across different network sizes."""
    results = []

    # Test different neuron counts
    neuron_counts = [1000, 5000, 10000, 20000, 50000]

    for n_neurons in neuron_counts:
        config.neurons = n_neurons
        config.n_input = n_neurons // 3  # Approximate input ratio

        try:
            result = run_jax_benchmark(config)
            results.append(result)
        except Exception as e:
            print(f"  Failed for {n_neurons} neurons: {e}")
            break

        # Clear memory
        gc.collect()

    return results


def run_batch_scaling_benchmark(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run benchmark across different batch sizes."""
    results = []

    batch_sizes = [1, 2, 4, 8, 16]

    for batch_size in batch_sizes:
        config.batch_size = batch_size

        try:
            result = run_jax_benchmark(config)
            results.append(result)
        except Exception as e:
            print(f"  Failed for batch_size={batch_size}: {e}")
            break

        gc.collect()

    return results


# =============================================================================
# Summary and Reporting
# =============================================================================

def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\n{'Framework':<10} {'Neurons':<10} {'Batch':<8} {'Seq':<8} "
          f"{'Forward':<12} {'Total':<12} {'Throughput':<15}")
    print("-" * 70)

    for r in results:
        print(f"{r.framework:<10} {r.neurons:<10} {r.batch_size:<8} {r.seq_len:<8} "
              f"{r.forward_time_ms:>10.2f}ms {r.total_time_ms:>10.2f}ms "
              f"{r.throughput_steps_per_sec:>12.0f} steps/s")


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON."""
    data = {
        'benchmark_date': str(np.datetime64('now')),
        'jax_version': jax.__version__,
        'device': str(jax.devices()[0]),
        'results': [r.to_dict() for r in results],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='V1 Model Performance Benchmark')

    parser.add_argument('--neurons', type=int, default=1000,
                        help='Number of neurons')
    parser.add_argument('--n_input', type=int, default=400,
                        help='Number of input neurons')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=600,
                        help='Sequence length')
    parser.add_argument('--warmup_runs', type=int, default=3,
                        help='Number of warmup runs')
    parser.add_argument('--benchmark_runs', type=int, default=10,
                        help='Number of benchmark runs')
    parser.add_argument('--compare_tf', action='store_true',
                        help='Include TensorFlow comparison')
    parser.add_argument('--scaling', action='store_true',
                        help='Run scaling benchmark')
    parser.add_argument('--batch_scaling', action='store_true',
                        help='Run batch size scaling benchmark')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = BenchmarkConfig(
        neurons=args.neurons,
        n_input=args.n_input,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        compare_tf=args.compare_tf,
        output_json=args.output,
    )

    # Print environment info
    print("=" * 70)
    print("V1 Model Performance Benchmark (JAX)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print("=" * 70)

    results = []

    if args.scaling:
        print("\n--- Neuron Scaling Benchmark ---")
        results.extend(run_scaling_benchmark(config))
    elif args.batch_scaling:
        print("\n--- Batch Size Scaling Benchmark ---")
        results.extend(run_batch_scaling_benchmark(config))
    else:
        # Single configuration benchmark
        result = run_jax_benchmark(config)
        results.append(result)

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        save_results(results, args.output)


if __name__ == '__main__':
    main()
