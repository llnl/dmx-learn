# MPS Torch Test Status

`tests/torch_stats` now passes on `TEST_TORCH_DEVICE=mps`.

## Current State

- MPS is available locally.
- The torch test harness is device-selectable through `TEST_TORCH_DEVICE`.
- `NUMBA_DISABLE_JIT=1` should be exported before running these tests to avoid
  unnecessary numba compilation.
- Full suite status on MPS:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats -q
```

- Latest result: `293 passed`.
- The remaining output is limited to `pytest.mark.dependency` warnings when the
  optional plugin is not installed.

## What Was Fixed

- `src/dmx/torch_utils/vector.py`
  - float tensor creation already defaulted to `float32` on MPS
  - `sample_dirichlet()` now returns samples on the intended target device even
    when the draw itself falls back to CPU
- `tests/torch_stats/torch_stats_tests.py`
  - test harness sets default float dtype from the selected test device
- `src/dmx/torch_stats/binomial.py`
  - constant creation path was aligned to encoded input device
- `src/dmx/torch_stats/poisson.py`
  - `PoissonAccumulator.seq_update()` aligns encoded observations to the
    weights device/dtype before `torch.dot`
- Torch test files under `tests/torch_stats`
  - distributions are moved onto `self.device` during setup
- `src/dmx/torch_stats/dmvn.py`
  - cached coefficients are aligned to encoded tensor device/dtype for matmul
    paths
- `src/dmx/torch_stats/intmultinomial.py`
  - `.to(device)` now updates cached probability tensors instead of leaving
    stale CPU-backed state behind
  - encoded length tensors are now created on the selected device
- `src/dmx/torch_stats/int_plsi.py`
  - `.to(device)` now migrates cached probability/document tensors correctly
  - MPS indexing and Dirichlet initialization paths now stay device-consistent
- `src/dmx/torch_stats/mixture.py`
  - `.to(device)` now rebuilds cached `zw` and `log_w` tensors on the target
    device
- `src/dmx/torch_stats/heterogenous_mixture.py`
  - `.to(device)` now rebuilds cached `zw` and `log_w` tensors on the target
    device
- `src/dmx/torch_stats/hmm.py`
  - `.to(device)` now actually moves transition tensors
  - sampler inputs are renormalized on CPU before passing probabilities to the
    numpy/Markov-chain sampling path
- `src/dmx/torch_stats/mvn.py`
  - scalar multivariate Gaussian log-density now uses a CPU fallback for
    `cholesky_solve` on MPS

## Resolved Failure Patterns

### 1. Device-mismatched index and mask tensors

Previously failing paths indexed MPS tensors with CPU tensors or wrote through
CPU masks.

Resolution pattern:

- move index tensors onto the parameter tensor device before indexed reads or
  writes
- move boolean masks onto the target tensor device before masked assignment
- ensure nested encoded data uses the selected device all the way down

### 2. Cached tensors not updated by `.to(device)`

Several distributions moved some tensors but kept cached derived tensors on CPU.

Resolution pattern:

- reassign moved tensors instead of calling `.to(device)` without storing the
  result
- recompute cached logs and zero-weight masks after moving the base tensors

### 3. Unsupported MPS operators

The main unsupported operators encountered were:

```text
aten::_sample_dirichlet
aten::_cholesky_solve_helper
```

Resolution pattern:

- perform the unsupported computation on CPU
- move the result back to the model device
- preserve the MPS-facing public API for callers and tests

### 4. HMM float32 normalization instability

After the MPS float32 switch, some sampler probabilities no longer summed to 1
closely enough for downstream numpy sampling.

Resolution pattern:

- normalize initial-state probabilities on CPU before sampler construction
- normalize each transition row on CPU before constructing the Markov-chain
  sampler

## Useful Commands

Full suite:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats -q
```

Targeted validation:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/intmultinomial_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/int_plsi_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/mixture_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/heterogeneous_mixture_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/hidden_markov_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/mvn_test.py -q
```

## Practical Notes

- If you want to remove the `pytest.mark.dependency` warnings, reinstall test
  dependencies with:

```bash
pip install -e ".[test]"
```

- Prefer small-file validation while iterating on future MPS regressions, then
  rerun the full suite as a final check.

## Exit Criteria

This work is complete for the current target because:

- `tests/torch_stats` passes on `TEST_TORCH_DEVICE=mps`
- no remaining test failures are due to MPS dtype/device mismatches
- unsupported MPS ops in the exercised paths are handled through explicit
  fallbacks
