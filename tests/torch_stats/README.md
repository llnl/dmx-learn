# Torch Stats Tests

This directory contains the torch-backed distribution test suite.

## Device Selection

Torch tests select their device from the `TEST_TORCH_DEVICE` environment
variable.

- Default: `cpu`
- Supported values: `cpu`, `mps`, `cuda`, `cuda:<index>`

Examples:

```bash
python -m pytest tests/torch_stats
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats
TEST_TORCH_DEVICE=cuda python -m pytest tests/torch_stats
TEST_TORCH_DEVICE=cuda:0 python -m pytest tests/torch_stats
```

If a requested backend is not available, the tests fail fast with a clear
runtime error.

## Notes on Precision

The shared torch test base in `tests/torch_stats/torch_stats_tests.py` uses
device-aware tolerances:

- CPU/CUDA: float64-oriented tolerance
- MPS: relaxed tolerance for float32 behavior

This is mainly relevant for log-density comparisons and EM-related checks.

## Recommended Validation Flow

When validating a non-CPU backend, start with a smaller subset before running
the full suite.

Examples:

```bash
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/exponential_test.py
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/hidden_markov_test.py
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/heterogeneous_mixture_test.py
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/int_plsi_test.py
```

Then run the full suite:

```bash
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats
```

## Current Status

The harness is now backend-selectable via `TEST_TORCH_DEVICE`. Remaining device
validation work is to execute the suite on actual MPS/CUDA hardware and review
any backend-specific numerical or operator issues.
