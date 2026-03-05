# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX implementation of the Allen Institute V1 visual cortex model from Chen et al., Science Advances 2022. This is a spiking neural network model using GLIF3 neurons with sparse connectivity, refactored from TensorFlow/Keras.

## Commands

```bash
# Install dependencies
uv sync

# Run training with Hydra config
uv run python scripts/train.py data_dir=/path/to/GLIF_network

# Override training params
uv run python scripts/train.py data_dir=/path/to/data training.learning_rate=1e-4 training.batch_size=4

# Switch task
uv run python scripts/train.py data_dir=/path/to/data task=evidence

# Run all tests
uv run pytest tests/ -v

# Run specific module tests
uv run pytest tests/test_glif3_cell.py -v

# Run TensorFlow comparison tests
uv run pytest tests/tf_comparison/ -v

# Convert TF checkpoint to JAX
uv run python scripts/convert_checkpoint.py --tf_checkpoint /path/to/tf_model --output /path/to/jax_checkpoint --verify

# Run benchmarks
uv run python scripts/benchmark.py --neurons 1000 --batch_size 4 --seq_len 600
```

## Architecture

### Core Neural Network Modules (`src/v1_jax/nn/`)
- **spike_functions.py**: `spike_gauss` with `jax.custom_vjp` for surrogate gradient (Gaussian pseudo-derivative)
- **glif3_cell.py**: GLIF3 neuron model with `GLIF3State` (NamedTuple), `GLIF3Params`, and `glif3_unroll` using `jax.lax.scan`
- **sparse_layer.py**: BCOO sparse connectivity for input/recurrent layers
- **synaptic.py**: PSC dynamics with 4 receptor types (AMPA, NMDA, GABA_A, GABA_B)
- **constraints.py**: Dale's law enforcement

### LGN Preprocessing (`src/v1_jax/lgn/`)
- Spatial filtering (Gaussian convolution)
- Temporal filtering (transfer function via `lax.scan`)

### Models (`src/v1_jax/models/`)
- **v1_network.py**: `V1Network` integrates LGN, sparse input layer, and GLIF3 recurrent network
- **readout.py**: Classification readout layers

### Training (`src/v1_jax/training/`)
- **trainer.py**: `V1Trainer` with `TrainState`, loss computation, gradient clipping
- **distributed.py**: Multi-GPU via `jax.pmap`
- **loss_functions.py**, **regularizers.py**: Spike rate, voltage, and weight regularization

### Data (`src/v1_jax/data/`)
- **network_loader.py**: `load_billeh()` for Billeh network data
- **stim_generator.py**: Drifting grating and classification stimuli

## Key Patterns

### State Management
All neuron states use `NamedTuple` for JIT compatibility:
```python
class GLIF3State(NamedTuple):
    z_buf: Array      # Spike buffer (batch, n_neurons * max_delay)
    v: Array          # Membrane potential
    r: Array          # Refractory counter
    asc_1: Array      # Adaptive spike current 1
    asc_2: Array      # Adaptive spike current 2
    psc_rise: Array   # PSC rising phase
    psc: Array        # Post-synaptic current
```

### RNN Unrolling
Time dynamics use `jax.lax.scan` for efficient compilation:
```python
final_state, (all_spikes, all_voltages) = jax.lax.scan(scan_fn, initial_state, inputs)
```

### Sparse Operations
BCOO format for large sparse connectivity matrices (~51K neurons):
```python
from jax.experimental.sparse import BCOO
connectivity = BCOO((weights, indices), shape=(n_post, n_pre))
output = connectivity @ input_spikes
```

### Configuration
Hydra configs in `configs/` with hierarchical structure:
- `config.yaml` - Main entry point
- `network/default.yaml` - Architecture params
- `training/default.yaml` - Optimizer/regularization
- `task/{garrett,evidence,10class}.yaml` - Task-specific
- `wandb/default.yaml` - Logging

## Test Structure

- `tests/test_*.py` - Unit tests for each module
- `tests/tf_comparison/` - Numerical equivalence tests against TensorFlow (rtol=1e-4 to 1e-5)
- Test markers: `unit`, `integration`, `benchmark`, `slow`, `gpu`
