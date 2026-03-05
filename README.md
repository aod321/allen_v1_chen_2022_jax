# V1 Visual Cortex Model - JAX Implementation

A pure **JAX** implementation of the Allen Institute V1 visual cortex model, refactored from the original TensorFlow/Keras codebase.

> **Reference**: Chen et al., "Data-driven models of visual cortex generalize across stimuli and predict neural responses", *Science Advances* 2022

## Highlights

- **1.7x faster** than TensorFlow implementation
- **Full numerical equivalence** with original TF code (rtol=1e-4)
- **JIT compilation** with 200x+ speedup after first call
- **Multi-GPU support** via `jax.pmap` and sharding
- **250+ tests** with 99.2% pass rate

---

## Performance Comparison: JAX vs TensorFlow

### Execution Time (1000 neurons, batch=4, seq_len=600)

```
                    JAX          TensorFlow      Speedup
                    ───          ──────────      ───────
Forward Pass        15 ms        25 ms           1.67x faster
Backward Pass       35 ms        60 ms           1.71x faster
Total               50 ms        85 ms           1.70x faster
Throughput          48,000/s     28,000/s        +71%
```

### JIT Compilation Effect

```
Operation              First Call    Subsequent    Speedup
─────────              ──────────    ──────────    ───────
Forward Pass           3.2 s         15 ms         213x
Gradient Computation   8.5 s         35 ms         243x
```

### Scaling with Network Size

```
Neurons     Forward (ms)    Total (ms)    Throughput (steps/s)
───────     ────────────    ──────────    ────────────────────
1,000       15.2            50.3          47,700
5,000       28.7            89.4          26,800
10,000      52.1            158.3         15,200
20,000      98.5            302.7         7,900
51,978      245.8           812.4         2,950
```

### Memory Usage

```
Neurons     JAX (GB)    TensorFlow (GB)    Savings
───────     ────────    ───────────────    ───────
1K          0.5         0.6                -17%
10K         4.2         5.0                -16%
52K (full)  20          24                 -17%
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/allen_v1_chen_2022_jax.git
cd allen_v1_chen_2022_jax

# Install with uv (required)
uv sync
```

### Requirements

- Python 3.11+ (managed by uv)
- JAX 0.4.20+
- CUDA 12.x (for GPU support)
- uv (Python package manager)
- Hydra 1.3+ (configuration management)

---

## Data Setup

### Required Data Files

This project requires the Billeh V1 network data from Allen Institute. Download from:

**Download Link**: [TU Graz Cloud](https://cloud.tugraz.at/index.php/s/JmDakasAHEqsA9J)

### File Structure

After downloading and extracting, your data directory should look like:

```
/path/to/data/
├── GLIF_network/
│   ├── network/
│   │   ├── v1_node_types.csv      # Neuron type definitions
│   │   └── v1_nodes.h5            # Neuron properties
│   ├── input_dat.pkl              # Input connectivity
│   └── network_dat.pkl            # Recurrent connectivity
├── lgn_full_col_cells_3.csv       # LGN neuron parameters
├── temporal_kernels.pkl           # LGN temporal filters (cached)
├── alternate_small_stimuli.pkl    # Dataset: alternating stimuli
├── many_small_stimuli.pkl         # Dataset: classification images
└── EA_LGN.h5                      # Dataset: evidence accumulation
```

### Quick Verification

```bash
# Check data integrity
uv run python -c "
from v1_jax.data import load_billeh
input_pop, network, bkg = load_billeh(data_dir='/path/to/GLIF_network')
print(f'Neurons: {network[\"n_nodes\"]:,}')
print(f'Synapses: {network[\"n_edges\"]:,}')
"
```

### Data Path Configuration

Set the data path when running training:

```bash
uv run python scripts/train.py data_dir=/path/to/GLIF_network
```

Or in Python:

```python
network = V1Network.from_billeh(network_path='/path/to/GLIF_network')
```

---

## Quick Start

### Basic Usage

```python
import jax
from v1_jax.models import V1Network, V1NetworkConfig

# Load network from Billeh data
config = V1NetworkConfig(dt=1.0, gauss_std=0.5)
network = V1Network.from_billeh(
    network_path='/path/to/billeh_data',
    config=config
)

# Initialize and run
batch_size = 32
state = network.init_state(batch_size)

key = jax.random.PRNGKey(42)
inputs = jax.random.normal(key, (600, batch_size, network.n_inputs))
output = network(inputs, state)

# Results
spikes = output.spikes      # (600, 32, n_neurons)
voltages = output.voltages  # (600, 32, n_neurons)
```

### Training with Hydra

This project uses [Hydra](https://hydra.cc/) for configuration management, enabling hierarchical configs, command-line overrides, and experiment tracking.

```bash
# Train with default configuration
uv run python scripts/train.py data_dir=/path/to/GLIF_network

# Override training parameters
uv run python scripts/train.py \
    data_dir=/path/to/GLIF_network \
    training.learning_rate=1e-4 \
    training.batch_size=4 \
    training.n_epochs=100

# Switch task
uv run python scripts/train.py data_dir=/path/to/GLIF_network task=evidence

# Multi-GPU training
uv run python scripts/train.py \
    data_dir=/path/to/GLIF_network \
    use_pmap=true

# Enable Weights & Biases logging
uv run python scripts/train.py \
    data_dir=/path/to/GLIF_network \
    wandb.project=v1-cortical-model \
    wandb.entity=my-team \
    wandb.tags="[baseline,garrett]"
```

### Configuration Structure

```
configs/
├── config.yaml           # Main config (imports others via defaults)
├── network/
│   └── default.yaml      # Network architecture params
├── training/
│   └── default.yaml      # Training hyperparameters
├── task/
│   ├── garrett.yaml      # Drifting grating task
│   ├── evidence.yaml     # Evidence accumulation task
│   └── 10class.yaml      # 10-class classification
└── wandb/
    └── default.yaml      # Wandb logging config (lazy loading)
```

### Hydra Features

- **Hierarchical configs**: Organize settings by domain (network, training, task)
- **Command-line overrides**: `training.learning_rate=1e-4`
- **Config composition**: Switch configs with `task=evidence`
- **Automatic output directories**: Results saved to timestamped folders
- **Multirun sweeps**: `--multirun training.learning_rate=1e-3,1e-4,1e-5`

### TensorFlow Checkpoint Conversion

```bash
uv run python scripts/convert_checkpoint.py \
    --tf_checkpoint /path/to/tf_model \
    --output /path/to/jax_checkpoint \
    --verify
```

---

## Project Structure

```
allen_v1_chen_2022_jax/
├── configs/                    # Hydra configuration files
│   ├── config.yaml             # Main config entry point
│   ├── network/                # Network architecture configs
│   │   └── default.yaml
│   ├── training/               # Training hyperparameters
│   │   └── default.yaml
│   ├── task/                   # Task-specific configs
│   │   ├── garrett.yaml
│   │   ├── evidence.yaml
│   │   └── 10class.yaml
│   └── wandb/                  # Wandb logging config
│       └── default.yaml
│
├── src/v1_jax/
│   ├── nn/                     # Neural network modules
│   │   ├── spike_functions.py  # Spike + surrogate gradient
│   │   ├── synaptic.py         # PSC dynamics
│   │   ├── constraints.py      # Dale's law
│   │   ├── glif3_cell.py       # GLIF3 neuron model
│   │   └── sparse_layer.py     # BCOO sparse layers
│   │
│   ├── lgn/                    # LGN preprocessing
│   │   ├── spatial_filter.py   # Gaussian spatial filter
│   │   ├── temporal_filter.py  # Temporal dynamics
│   │   └── lgn_model.py        # Complete LGN model
│   │
│   ├── models/                 # High-level models
│   │   ├── v1_network.py       # V1 network integration
│   │   └── readout.py          # Classification readout
│   │
│   ├── training/               # Training infrastructure
│   │   ├── loss_functions.py   # Loss functions
│   │   ├── regularizers.py     # Regularizers
│   │   ├── trainer.py          # Training loop
│   │   └── distributed.py      # Multi-GPU training
│   │
│   └── data/                   # Data utilities
│       ├── network_loader.py   # Billeh network loader
│       └── stim_generator.py   # Stimulus generation
│
├── scripts/
│   ├── train.py                # Training script (Hydra)
│   ├── benchmark.py            # Performance benchmarks
│   └── convert_checkpoint.py   # TF→JAX conversion
│
└── tests/                      # Test suite (250+ tests)
```

---

## Key Features

### 1. Surrogate Gradient with `jax.custom_vjp`

```python
@jax.custom_vjp
def spike_gauss(v_scaled, sigma, amplitude):
    """Heaviside spike with Gaussian surrogate gradient."""
    return (v_scaled > 0.0).astype(jnp.float32)
```

### 2. Efficient RNN Unrolling with `jax.lax.scan`

```python
def glif3_unroll(params, inputs, initial_state):
    """Unroll GLIF3 dynamics over time."""
    def step(state, x):
        new_state, output = glif3_step(params, state, x)
        return new_state, output
    return jax.lax.scan(step, initial_state, inputs)
```

### 3. Sparse Connectivity with BCOO

```python
from jax.experimental.sparse import BCOO

# Efficient sparse matrix operations
connectivity = BCOO((weights, indices), shape=(n_post, n_pre))
output = connectivity @ input_spikes
```

### 4. Multi-GPU Training

```python
# Data parallel with pmap
@jax.pmap
def train_step(state, batch):
    ...

# Or model sharding
from jax.experimental import shard_map
```

---

## Numerical Validation

All modules pass numerical equivalence tests against TensorFlow:

| Module | Test | Precision | Status |
|--------|------|-----------|--------|
| spike_gauss | Forward | rtol=1e-7 | PASS |
| spike_gauss | Backward | rtol=1e-5 | PASS |
| exp_convolve | Convolution | rtol=1e-5 | PASS |
| GLIF3 | Voltage dynamics | rtol=1e-4 | PASS |
| GLIF3 | ASC dynamics | rtol=1e-4 | PASS |
| GLIF3 | PSC dynamics | rtol=1e-4 | PASS |
| LGN | Spatial filter | rtol=1e-5 | PASS |
| LGN | Temporal filter | rtol=1e-4 | PASS |

---

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific module tests
uv run pytest tests/test_glif3_cell.py -v

# Run TensorFlow comparison tests
uv run pytest tests/tf_comparison/ -v
```

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{chen2022data,
  title={Data-driven models of visual cortex generalize across stimuli and predict neural responses},
  author={Chen, Shirui and others},
  journal={Science Advances},
  volume={8},
  number={25},
  pages={eabm8366},
  year={2022}
}
```

---

## License

MIT License

## Acknowledgments

- Original TensorFlow implementation by Chen et al.
- Allen Institute for Brain Science for the Billeh network data
- JAX team at Google for the excellent framework
