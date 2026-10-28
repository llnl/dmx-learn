# dmx-learn

[![Tests](https://github.com/LLNL/dmx-learn/workflows/Tests/badge.svg)](https://github.com/LLNL/dmx-learn/actions/workflows/test.yml)
[![Code Quality](https://github.com/LLNL/dmx-learn/workflows/Code%20Quality/badge.svg)](https://github.com/LLNL/dmx-learn/actions/workflows/quality.yml)
[![Documentation](https://readthedocs.org/projects/dmx-learn/badge/?version=latest)](https://dmx-learn.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **The expert for mixture models on messy tables. Every model it fits is one
> you can read, rerun, and scale to a cluster.**

dmx-learn is a dedicated tool for **latent-variable graphical models on
heterogeneous tabular data**. It fits mixtures, hierarchical mixtures, HMMs,
and tree-structured models over records that mix continuous values, counts,
categories, sets, and variable-length sequences. Each field keeps its own
type, so there is no one-hot encoding and no rescaling. Every fit uses exact
EM or variational updates that you can inspect, and the same model runs
locally, with MPI, or on Spark.

dmx-learn stays narrow on purpose. It does not try to be a general
probabilistic or neural modeling platform. It aims to be exact, transparent,
and easy to reproduce. See [Scope and Non-Goals](docs/scope.rst) for what is
in scope, what is deliberately left out, and the rule for adding new models.

### Workflow: from table to agent

The primary user surface is a conversation with an expert agent backed by a
small, deterministic tool API:

```text
your table (CSV / DataFrame)
  -> profile the columns and agree on a typed schema   ("treat zip as an ID")
  -> fit a baseline and a latent-variable model
  -> compare, diagnose, and refine the structure        (leaderboard + lineage)
  -> explain it: clusters, dependencies, anomalies, search depth, embeddings
  -> export plain dmx Python code that reproduces the fit exactly
  -> scale the same model to MPI or Spark when the data grows
```

> **Status:** the agent (`dmx-mcp`, an MCP server installed through a future
> `dmx-learn[agent]` extra) is in development for the 1.2 release. The
> modeling engine underneath it is available today.

### Today: the same models in plain Python

```python
import numpy as np
import pandas as pd

from dmx.stats import (
    CategoricalEstimator,
    CompositeEstimator,
    GaussianEstimator,
    MixtureEstimator,
    PoissonEstimator,
    SequenceEstimator,
)
from dmx.utils.estimation import optimize

# Columns: age (float), visits (int), plan (str), tags (list of str)
df = pd.DataFrame(...)
data = list(df.itertuples(index=False, name=None))

# A two-class latent model over typed fields: no one-hot encoding, no scaling.
row = CompositeEstimator(
    (
        GaussianEstimator(),
        PoissonEstimator(),
        CategoricalEstimator(),
        SequenceEstimator(CategoricalEstimator()),
    )
)
model = optimize(data, MixtureEstimator([row] * 2), max_its=100, rng=np.random.RandomState(1))

print(model)                    # readable, re-creatable model
print(model.posterior(data[0]))  # latent-class membership for one row
```

`dmx.utils.automatic.get_estimator(data)` can also infer an estimator
directly from example records. See
[`examples/utils_examples/auto_example.py`](examples/utils_examples/auto_example.py).

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
poetry install -E spark            # Include PySpark support
poetry install -E optional         # Include MPI and UMAP support
poetry install -E distributed      # Include PySpark and MPI support
poetry install -E viz              # Include matplotlib plotting support
poetry install -E agent            # Include the MCP agent SDK and plotting
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

# With PySpark support
pip install /path/to/dmx-learn[spark]

# With MPI and UMAP support
pip install /path/to/dmx-learn[optional]

# With plotting support (matplotlib)
pip install /path/to/dmx-learn[viz]

# With the expert agent (MCP SDK, includes plotting)
pip install /path/to/dmx-learn[agent]

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
| **spark** | `pip install .[spark]` or `poetry install -E spark` | pyspark | Spark RDD estimation and sampling |
| **optional** | `pip install .[optional]` or `poetry install -E optional` | mpi4py, umap-learn | MPI estimation and UMAP embeddings |
| **distributed** | `pip install .[distributed]` or `poetry install -E distributed` | pyspark, mpi4py | Spark and MPI scale-out |
| **viz** | `pip install .[viz]` or `poetry install -E viz` | matplotlib | Plotting helpers for explore results |
| **agent** | `pip install .[agent]` or `poetry install -E agent` | mcp (official MCP Python SDK), matplotlib | The `dmx-mcp` expert agent server (in development) |
| **all** | `pip install .[all]` or `poetry install -E all` | All above | Everything |

> **Changed in 1.2:** pyspark is no longer a core dependency. Install the
> `spark` (or `distributed`/`all`) extra to use Spark RDDs or
> `dmx.stats.rdd_sampler`.

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

Examples using `stats` distributions (always available) are located in
`./examples/stats_examples/`:

```bash
# Using Poetry
poetry run python ./examples/stats_examples/mixture_example.py

# Or set PYTHONPATH
export PYTHONPATH=/path/to/dmx-learn:$PYTHONPATH
python ./examples/stats_examples/mixture_example.py
```

### Bayesian Stats Examples

Examples using `bstats` are located in `./examples/bstats_examples/`:

```bash
poetry run python ./examples/bstats_examples/dpm_auto_example1.py
```

### Utility Examples

Examples for automatic model selection, detailed estimation, embeddings, and
random graphs are located in `./examples/utils_examples/`:

```bash
poetry run python ./examples/utils_examples/auto_example.py
```

### Torch Examples (PyTorch-based)

Examples using `torch_stats` (requires PyTorch) are located in
`./examples/torch_examples/`:

```bash
poetry run python ./examples/torch_examples/stats_examples/mixture_example.py
```

---

## 🌐 Running with Spark

Examples that run with Apache Spark are located in `./examples/spark_examples/`.
Spark support requires the `spark` extra:

```bash
# Using Poetry
poetry install -E spark

# Using pip
pip install /path/to/dmx-learn[spark]
```

To submit to a Spark installation, first build a wheel:

```bash
cd /path/to/dmx-learn
poetry build -f wheel
```

Then run with Spark:

```bash
/path/to/spark/bin/spark-submit \
  --master local[*] \
  --py-files /path/to/dmx-learn/dist/dmx_learn-1.1.1-py3-none-any.whl \
  ./examples/spark_examples/mixture_example.py
```

---

## 🚄 Running with MPI4PY

Examples using mpi4py are located in `./examples/mpi4py_examples/`.

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
mpiexec -n 4 poetry run python ./examples/mpi4py_examples/estimation_example.py
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

Examples using `torch_stats` are located in `./examples/torch_examples/`:

```bash
poetry run python ./examples/torch_examples/stats_examples/mixture_example.py
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
- **Core Dependencies**: numpy, scipy, numba, pandas, mpmath
- **Optional**: torch (for GPU acceleration), pyspark (for Spark), mpi4py (for MPI), umap-learn (for UMAP embeddings), matplotlib (for plotting), mcp (for the expert agent)

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

License
-------

dmx-learn is distributed under the terms of the BSD 3-Clause license.  See
[LICENSE](./LICENSE) and [NOTICE](./NOTICE) for details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-844837
