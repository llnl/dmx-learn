# dmx-learn

[![Tests](https://github.com/LLNL/dmx-learn/workflows/Tests/badge.svg)](https://github.com/LLNL/dmx-learn/actions/workflows/test.yml)
[![Code Quality](https://github.com/LLNL/dmx-learn/workflows/Code%20Quality/badge.svg)](https://github.com/LLNL/dmx-learn/actions/workflows/quality.yml)
[![Documentation](https://readthedocs.org/projects/dmx-learn/badge/?version=latest)](https://dmx-learn.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-BSD-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Distributed Mixture Learning** - A package for distributed heterogeneous density estimation. With only a few lines of code you can specify and fit complex models on variable-length heterogeneous data.

---

## 📚 Documentation

View the full documentation on **Read the Docs**:

👉 [https://dmx-learn.readthedocs.io/en/latest/](https://dmx-learn.readthedocs.io/en/latest/)

---

## 🚀 Installation

### Using Poetry (Recommended)

dmx-learn uses Poetry for dependency management. Install Poetry first if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then install dmx-learn:

```bash
# Clone or navigate to the package directory
cd /path/to/dmx-learn

# Install core dependencies only
poetry install

# Or install with optional features using extras
poetry install -E torch            # Include PyTorch support
poetry install -E optional         # Include MPI and UMAP support
poetry install -E all              # Include all optional features

# Or using dependency groups (for dev/docs)
poetry install --with dev          # Include dev tools (pytest)
poetry install --with docs         # Include documentation tools
```

### Using pip

You can also install with pip using extras:

```bash
# Basic installation
pip install /path/to/dmx-learn

# With PyTorch support
pip install /path/to/dmx-learn[torch]

# With MPI and UMAP support
pip install /path/to/dmx-learn[optional]

# With all optional features
pip install /path/to/dmx-learn[all]
```

---

## 📦 Optional Dependencies

dmx-learn supports several optional dependency extras and groups:

### Extras (for pip and Poetry)

| Extra | Install Command | Includes | Use Case |
|-------|----------------|----------|----------|
| **torch** | `pip install .[torch]` or `poetry install -E torch` | PyTorch | GPU-accelerated distributions |
| **optional** | `pip install .[optional]` or `poetry install -E optional` | mpi4py, umap-learn | Distributed computing |
| **all** | `pip install .[all]` or `poetry install -E all` | All above | Everything |

### Groups (Poetry only - for development)

| Group | Install Command | Includes | Use Case |
|-------|----------------|----------|----------|
| **dev** | `poetry install --with dev` | pytest, pytest-dependency | Testing |
| **docs** | `poetry install --with docs` | sphinx, sphinx-rtd-theme | Documentation building |

**Install multiple extras and groups:**
```bash
# Poetry: extras with -E, groups with --with
poetry install -E all --with dev,docs

# pip: multiple extras with comma
pip install .[torch,optional]
```

**Check what's installed:**
```bash
poetry show
```

---

## 📖 Usage Examples

### Stats Examples (scipy-based)

Examples using `stats` distributions (always available) are located in `./examples/`:

```bash
# Using Poetry
poetry run python ./examples/stats_examples/mixture_example.py

# Or set PYTHONPATH
export PYTHONPATH=/path/to/dmx-learn:$PYTHONPATH
python ./examples/stats_examples/mixture_example.py
```

### Torch Examples (PyTorch-based)

Examples using `torch_stats` (requires PyTorch) are located in `./examples_torch/`:

```bash
poetry run python ./examples_torch/stats_examples/mixture_example.py
```

---

## 🌐 Running with Spark

Examples that run with Apache Spark are located in `./examples_spark/`.

First, build a wheel:

```bash
cd /path/to/dmx-learn
pip install setuptools wheel
python setup.py bdist_wheel
```

Then run with Spark:

```bash
/path/to/spark/bin/spark-submit \
  --master local[*] \
  --py-files /path/to/dmx-learn/dist/dmx-learn-1.0.0-py3-none-any.whl \
  ./examples_spark/mixture_example.py
```

---

## 🚄 Running with MPI4PY

Examples using mpi4py are located in `./examples_mpi4py/`.

**Install MPI support:**
```bash
# Using Poetry
poetry install -E optional

# Using pip
pip install /path/to/dmx-learn[optional]
```

**Run with MPI:**
```bash
# Run with 4 processes
mpiexec -n 4 poetry run python ./examples_mpi4py/estimation_example.py
```

---

## 🔥 PyTorch Support (Optional)

dmx-learn includes `torch_stats`, a PyTorch-based implementation of statistical distributions for GPU-accelerated computation.

### Installing with PyTorch

```bash
# Using Poetry (recommended)
poetry install -E torch

# Using pip
pip install /path/to/dmx-learn[torch]
```

### Using torch_stats

```python
from dmx.torch_stats import GaussianDistribution, MixtureDistribution

# Create distributions (runs on GPU if available)
dist = GaussianDistribution(mu=0.0, sigma2=1.0)

# Automatic device detection
from dmx.torch_utils import detect_device
device = detect_device()  # Returns 'cuda', 'mps', or 'cpu'
```

### Running Torch Examples

Examples using `torch_stats` are located in `./examples_torch/`:

```bash
poetry run python ./examples_torch/stats_examples/mixture_example.py
```

### Testing with PyTorch

Run torch_stats tests on different devices:

```bash
# Default (CPU)
poetry run pytest tests/torch_stats

# On Apple Silicon MPS
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps poetry run pytest tests/torch_stats

# On CUDA
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=cuda poetry run pytest tests/torch_stats
```

See [tests/torch_stats/README.md](tests/torch_stats/README.md) for detailed testing information.

---

## 🧪 Testing

### Run All Tests

```bash
# Using Poetry
poetry run pytest tests/

# Using the test runner script
./test_runner.sh all
```

### Run Specific Test Suites

```bash
# Stats tests (always available)
poetry run pytest tests/stats

# Torch tests (requires PyTorch)
export NUMBA_DISABLE_JIT=1
poetry run pytest tests/torch_stats

# Quick sanity check
./test_runner.sh quick
```

### Test Runner Script

The `test_runner.sh` helper script makes testing easier:

```bash
./test_runner.sh stats        # Run stats tests
./test_runner.sh torch        # Run torch tests (CPU)
./test_runner.sh torch-mps    # Run torch tests (MPS)
./test_runner.sh all          # Run all tests
./test_runner.sh quick        # Quick sanity check
```

---

## 🔧 Development Setup

For development with all features:

```bash
# Clone the repository
git clone https://github.com/llnl/dmx-learn.git
cd dmx-learn

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies including dev tools
poetry install -E all --with dev,docs

# Run tests
poetry run pytest tests/

# Build documentation
cd docs
poetry run make html
```

---

## 📋 Requirements

- **Python**: >= 3.10
- **Core Dependencies**: numpy, scipy, numba, pandas, pyspark, mpmath
- **Optional**: torch (for GPU acceleration), mpi4py (for distributed computing)

---

## 🆘 Troubleshooting

### PyTorch Not Found

If you see errors about PyTorch not being installed:

```bash
# Install torch extra with Poetry
poetry install -E torch

# Or with pip
pip install /path/to/dmx-learn[torch]
```

### Tests Failing

Make sure you have the test dependencies:

```bash
poetry install --with dev
```

For torch tests, set the environment variable:

```bash
export NUMBA_DISABLE_JIT=1
```

### Import Errors

Make sure the package is installed:

```bash
poetry install
```

Or add to PYTHONPATH:

```bash
export PYTHONPATH=/path/to/dmx-learn:$PYTHONPATH
```

---

## 📄 License

BSD License

---
