#!/usr/bin/env python3
"""Benchmark script for comparing optimization strategies.

Tests:
1. Baseline (no optimizations)
2. Gradient Checkpointing only
3. ZeRO-2 only
4. Gradient Checkpointing + ZeRO-2

Measures:
- Memory usage per device
- Step time (forward + backward)
- Maximum batch size
"""

import os
import sys
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

# Set memory preallocation to allow measuring actual usage
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.95')


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    data_dir: str
    batch_size: int = 4
    seq_len: int = 600
    n_warmup_steps: int = 2
    n_benchmark_steps: int = 5
    use_gradient_checkpointing: bool = False
    checkpoint_every_n_steps: int = 50
    use_zero2: bool = False
    use_pmap: bool = True


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage info."""
    try:
        # Try nvidia-smi for detailed info
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        total_used = 0
        total_total = 0
        per_gpu = []
        for line in lines:
            used, total = map(int, line.split(','))
            total_used += used
            total_total += total
            per_gpu.append({'used_mb': used, 'total_mb': total})
        return {
            'total_used_gb': total_used / 1024,
            'total_total_gb': total_total / 1024,
            'per_gpu': per_gpu,
            'num_gpus': len(per_gpu),
        }
    except Exception as e:
        return {'error': str(e)}


def clear_memory():
    """Clear JAX caches and run garbage collection."""
    jax.clear_caches()
    gc.collect()


def run_single_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run benchmark with specific configuration.

    Returns:
        Dict with timing and memory results
    """
    from v1_jax.models import V1Network, V1NetworkConfig
    from v1_jax.models.readout import MultiClassReadout
    from v1_jax.data.network_loader import load_billeh
    from v1_jax.training.trainer import V1Trainer, TrainConfig, create_train_step_fn
    from v1_jax.training.distributed import (
        DistributedConfig, create_distributed_trainer, get_device_count
    )

    results = {
        'config': {
            'batch_size': config.batch_size,
            'seq_len': config.seq_len,
            'use_gradient_checkpointing': config.use_gradient_checkpointing,
            'checkpoint_every_n_steps': config.checkpoint_every_n_steps,
            'use_zero2': config.use_zero2,
        }
    }

    num_devices = get_device_count()
    results['num_devices'] = num_devices

    print(f"\n{'='*60}")
    print(f"Config: batch_size={config.batch_size}, "
          f"checkpoint={config.use_gradient_checkpointing}, "
          f"zero2={config.use_zero2}")
    print(f"{'='*60}")

    # Clear memory before test
    clear_memory()
    mem_before = get_gpu_memory_info()
    results['memory_before'] = mem_before

    try:
        # Create network config
        network_config = V1NetworkConfig(
            dt=1.0,
            gauss_std=0.28,
            dampening_factor=0.5,
            max_delay=5,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            checkpoint_every_n_steps=config.checkpoint_every_n_steps,
        )

        # Load network
        print("Loading network...")
        input_pop, network_data, bkg_weights = load_billeh(
            n_input=17400,
            n_neurons=51978,
            core_only=False,
            data_dir=config.data_dir,
            seed=42,
        )

        network = V1Network.from_billeh(
            network_path=config.data_dir,
            config=network_config,
            bkg_weights=bkg_weights,
            network_data=network_data,
            input_pop=input_pop,
        )
        print(f"Network: {network.n_neurons} neurons, {network.n_inputs} inputs")

        # Create readout
        readout = MultiClassReadout(
            n_neurons=network.n_neurons,
            n_classes=2,
            temporal_pooling='chunks',
            chunk_size=50,
            apply_softmax=True,
        )

        def readout_fn(spikes):
            return readout(spikes)

        # Create trainer
        train_config = TrainConfig(
            learning_rate=1e-3,
            rate_cost=0.1,
            voltage_cost=1e-5,
            gradient_clip_norm=1.0,
        )

        # Load target rates (simplified)
        target_rates = jnp.full(network.n_neurons, 0.02, dtype=jnp.float32)

        trainer = V1Trainer(
            network=network,
            config=train_config,
            target_firing_rates=target_rates,
        )

        # Create distributed trainer if needed
        key = jax.random.PRNGKey(42)

        if num_devices > 1 and (config.use_pmap or config.use_zero2):
            dist_config = DistributedConfig(
                num_devices=num_devices,
                use_pmap=config.use_pmap,
                use_zero2=config.use_zero2,
            )
            dist_trainer = create_distributed_trainer(trainer, dist_config)
            train_state = dist_trainer.init_state(key)
            train_step_fn = dist_trainer.create_train_step_fn(readout_fn)

            if config.use_zero2:
                train_state = dist_trainer.replicate_state(train_state)

            devices = dist_trainer.devices
            use_distributed = True
        else:
            train_state = trainer.init_train_state(key)
            train_step_fn = create_train_step_fn(trainer, readout_fn)
            devices = None
            use_distributed = False

        # Prepare dummy data
        batch_per_device = config.batch_size // num_devices if use_distributed else config.batch_size

        def create_batch():
            if use_distributed:
                inputs = np.random.randn(num_devices, config.seq_len, batch_per_device, network.n_inputs).astype(np.float32)
                labels = np.random.randint(0, 2, (num_devices, batch_per_device)).astype(np.int32)
                weights = np.ones((num_devices, batch_per_device), dtype=np.float32)

                inputs = jax.device_put_sharded(list(inputs), devices)
                labels = jax.device_put_sharded(list(labels), devices)
                weights = jax.device_put_sharded(list(weights), devices)
                network_state = network.init_state(batch_per_device)
                network_state = jax.device_put_replicated(network_state, devices)
            else:
                inputs = jnp.array(np.random.randn(config.seq_len, config.batch_size, network.n_inputs).astype(np.float32))
                labels = jnp.array(np.random.randint(0, 2, config.batch_size).astype(np.int32))
                weights = jnp.ones(config.batch_size, dtype=jnp.float32)
                network_state = network.init_state(config.batch_size)

            return inputs, labels, weights, network_state

        # Warmup
        print(f"Warmup ({config.n_warmup_steps} steps)...")
        for i in range(config.n_warmup_steps):
            inputs, labels, weights, network_state = create_batch()
            train_state, output, metrics = train_step_fn(
                train_state, inputs, labels, weights, network_state
            )
            # Block until complete
            if hasattr(metrics, 'loss'):
                jax.block_until_ready(metrics.loss)
            print(f"  Warmup step {i+1}/{config.n_warmup_steps}")

        # Memory after model loaded and warmed up
        mem_after_warmup = get_gpu_memory_info()
        results['memory_after_warmup'] = mem_after_warmup

        # Benchmark
        print(f"Benchmarking ({config.n_benchmark_steps} steps)...")
        step_times = []

        for i in range(config.n_benchmark_steps):
            inputs, labels, weights, network_state = create_batch()

            # Time the step
            jax.block_until_ready(inputs)
            t_start = time.perf_counter()

            train_state, output, metrics = train_step_fn(
                train_state, inputs, labels, weights, network_state
            )

            # Block until complete
            if hasattr(metrics, 'loss'):
                jax.block_until_ready(metrics.loss)

            t_end = time.perf_counter()
            step_time = t_end - t_start
            step_times.append(step_time)
            print(f"  Step {i+1}/{config.n_benchmark_steps}: {step_time*1000:.1f}ms")

        # Memory after benchmark
        mem_after_benchmark = get_gpu_memory_info()
        results['memory_after_benchmark'] = mem_after_benchmark

        # Compute statistics
        results['step_times_ms'] = [t * 1000 for t in step_times]
        results['mean_step_time_ms'] = np.mean(step_times) * 1000
        results['std_step_time_ms'] = np.std(step_times) * 1000
        results['throughput_samples_per_sec'] = config.batch_size / np.mean(step_times)

        # Memory usage
        if 'per_gpu' in mem_after_benchmark:
            results['memory_per_gpu_gb'] = [g['used_mb'] / 1024 for g in mem_after_benchmark['per_gpu']]
            results['max_memory_per_gpu_gb'] = max(results['memory_per_gpu_gb'])

        results['success'] = True

    except Exception as e:
        import traceback
        results['success'] = False
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()
        print(f"ERROR: {e}")

    finally:
        # Clean up
        clear_memory()

    return results


def print_comparison_table(all_results: Dict[str, Dict]):
    """Print comparison table of all results."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*80)

    # Header
    headers = ["Configuration", "Batch", "Step Time", "Throughput", "GPU Memory", "Status"]
    col_widths = [30, 8, 12, 15, 12, 8]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for name, result in all_results.items():
        if result['success']:
            batch = str(result['config']['batch_size'])
            step_time = f"{result['mean_step_time_ms']:.1f}ms"
            throughput = f"{result['throughput_samples_per_sec']:.1f} samp/s"
            memory = f"{result.get('max_memory_per_gpu_gb', 0):.1f} GB"
            status = "OK"
        else:
            batch = str(result['config']['batch_size'])
            step_time = "N/A"
            throughput = "N/A"
            memory = "N/A"
            status = "FAIL"

        row = [name, batch, step_time, throughput, memory, status]
        row_line = " | ".join(str(r).ljust(w) for r, w in zip(row, col_widths))
        print(row_line)

    print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark optimization strategies')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GLIF_network data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for benchmarking')
    parser.add_argument('--seq_len', type=int, default=600,
                       help='Sequence length')
    parser.add_argument('--warmup_steps', type=int, default=2,
                       help='Number of warmup steps')
    parser.add_argument('--benchmark_steps', type=int, default=5,
                       help='Number of benchmark steps')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'baseline', 'checkpoint', 'zero2', 'combined'],
                       help='Which tests to run')

    args = parser.parse_args()

    print("="*80)
    print("V1 SNN Training Optimization Benchmark")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"JAX devices: {jax.device_count()}")
    print(f"Device type: {jax.devices()[0].device_kind}")

    all_results = {}

    # Test configurations
    configs = {
        'baseline': BenchmarkConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_warmup_steps=args.warmup_steps,
            n_benchmark_steps=args.benchmark_steps,
            use_gradient_checkpointing=False,
            use_zero2=False,
        ),
        'checkpoint': BenchmarkConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_warmup_steps=args.warmup_steps,
            n_benchmark_steps=args.benchmark_steps,
            use_gradient_checkpointing=True,
            checkpoint_every_n_steps=50,
            use_zero2=False,
        ),
        'zero2': BenchmarkConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_warmup_steps=args.warmup_steps,
            n_benchmark_steps=args.benchmark_steps,
            use_gradient_checkpointing=False,
            use_zero2=True,
        ),
        'combined': BenchmarkConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_warmup_steps=args.warmup_steps,
            n_benchmark_steps=args.benchmark_steps,
            use_gradient_checkpointing=True,
            checkpoint_every_n_steps=50,
            use_zero2=True,
        ),
    }

    # Run selected tests
    if args.test == 'all':
        tests_to_run = ['baseline', 'checkpoint', 'zero2', 'combined']
    else:
        tests_to_run = [args.test]

    for test_name in tests_to_run:
        config = configs[test_name]
        result = run_single_benchmark(config)
        all_results[test_name] = result

        # Clear between tests
        clear_memory()
        time.sleep(2)

    # Print comparison
    print_comparison_table(all_results)

    # Save results
    import json
    results_file = Path(args.data_dir).parent / 'benchmark_results.json'

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
