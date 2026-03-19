# MPS Device Plan

This document captures the remaining work to get `tests/torch_stats` passing on
`TEST_TORCH_DEVICE=mps`.

## Current State

- MPS is available locally.
- The torch test harness is now device-selectable through
  `TEST_TORCH_DEVICE`.
- `NUMBA_DISABLE_JIT=1` should be exported before running these tests to
  avoid unnecessary numba compilation.
- A stable subset already passes on MPS, including:
  - `tests/torch_stats/gaussian_test.py`
  - `tests/torch_stats/exponential_test.py`
  - `tests/torch_stats/binomial_test.py`
- The full `tests/torch_stats` suite still has MPS-specific failures.

## Useful Command

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats
```

For smaller iterations:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/<file> -q
```

## What Has Already Been Fixed

- `src/dmx/torch_utils/vector.py`
  - float tensor creation now defaults to `float32` on MPS instead of
    `float64`
  - `sample_dirichlet()` now has an MPS fallback path that samples on CPU and
    moves the result back
- `tests/torch_stats/torch_stats_tests.py`
  - test harness now sets default float dtype from the selected test device
- `src/dmx/torch_stats/binomial.py`
  - fixed a constant creation path so tensors are created on the same device as
    encoded inputs
- `src/dmx/torch_stats/poisson.py`
  - `PoissonAccumulator.seq_update()` now aligns encoded observations to the
    weights device/dtype before `torch.dot`
- Torch test files under `tests/torch_stats`
  - distributions are now moved onto `self.device` in test setup
- `src/dmx/torch_stats/dmvn.py`
  - one MPS matmul/device mismatch was fixed by aligning cached coefficients to
    encoded tensor device/dtype

## Main Remaining Failure Categories

### 1. Index Tensors on One Device, Parameters on Another

These failures look like:

```text
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor
```

This still shows up in:

- `src/dmx/torch_stats/conditional.py`
- `src/dmx/torch_stats/hmm.py`
- `src/dmx/torch_stats/int_plsi.py`
- `src/dmx/torch_stats/intmultinomial.py`
- `src/dmx/torch_stats/intrange.py`
- `src/dmx/torch_stats/intsetdist.py`

Resolution pattern:

- Move index tensors onto the parameter tensor device before indexed reads or
  writes
- Move boolean masks onto the target tensor device before masked assignment
- Keep the output tensor on the model device and only convert index tensors as
  needed

### 2. Unsupported MPS Operators

These failures look like:

```text
NotImplementedError: The operator 'aten::_sample_dirichlet' is not currently implemented for the MPS device
```

and:

```text
NotImplementedError: The operator 'aten::_cholesky_solve_helper' is not currently implemented for the MPS device
```

Affected areas:

- `src/dmx/torch_stats/int_plsi.py` initialization path
- `src/dmx/torch_stats/mixture.py` / `src/dmx/torch_stats/heterogenous_mixture.py`
  initialization paths
- `src/dmx/torch_stats/mvn.py` for multivariate Gaussian log-density

Resolution pattern:

- For Dirichlet initialization, use CPU fallback and move results back to MPS
- For unsupported linear algebra like `cholesky_solve`, compute on CPU for the
  affected path and move the result back to the model device

### 3. Nested Encoded Structures Keep Mixed Devices Internally

Some top-level encoded sequences are on MPS, but nested tensors or cached model
parameters are still on CPU.

This is especially relevant in:

- `src/dmx/torch_stats/conditional.py`
- `src/dmx/torch_stats/hmm.py`
- `src/dmx/torch_stats/int_plsi.py`
- `src/dmx/torch_stats/sequence.py`

Resolution pattern:

- Ensure `to(device)` actually migrates all cached tensors
- Ensure nested distributions used inside composite/conditional/HMM structures
  are moved to the selected device during test setup and model construction
- When this is too fragile, explicitly coerce intermediate tensors inside the
  hot path

### 4. HMM Sampling / Normalization Instability After MPS Float32 Switch

Current HMM failures are not only device mismatches. Some now fail during
sampling with:

```text
ValueError: probabilities do not sum to 1
```

Likely cause:

- probability maps derived from float32 tensors are not renormalized before
  they are handed to numpy sampling code

Resolution pattern:

- inspect `HiddenMarkovSampler` and `MarkovChainDistribution` handoff
- normalize `w` and each transition row on CPU before constructing the numpy
  sampler inputs

## Recommended Order of Work

### Step 1. Fix discrete/indexed distributions first

Start here because the fixes are localized and unblock several higher-level
 models.

Files:

- `src/dmx/torch_stats/intrange.py`
- `src/dmx/torch_stats/intsetdist.py`
- `src/dmx/torch_stats/intmultinomial.py`
- `src/dmx/torch_stats/int_plsi.py`

Goal:

- all direct indexing uses tensors on the same device

Suggested validation:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/intrange_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/intsetdist_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/intmultinomial_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/int_plsi_test.py -q
```

### Step 2. Fix conditional and sequence-style nested models

Files:

- `src/dmx/torch_stats/conditional.py`
- `src/dmx/torch_stats/sequence.py`

Goal:

- nested encodings and index tensors operate consistently on MPS

Suggested validation:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/conditional_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/sequence_test.py -q
```

### Step 3. Fix HMM device and normalization issues

Files:

- `src/dmx/torch_stats/hmm.py`

Focus areas:

- device alignment for masks / indexing
- transition tensor construction on MPS
- sampler probability normalization before numpy sampling

Suggested validation:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/hidden_markov_test.py -q
```

### Step 4. Fix unsupported-op fallbacks for remaining structured models

Files:

- `src/dmx/torch_stats/int_plsi.py`
- `src/dmx/torch_stats/mixture.py`
- `src/dmx/torch_stats/heterogenous_mixture.py`
- `src/dmx/torch_stats/mvn.py`

Goal:

- CPU fallback for MPS-missing ops, while preserving MPS-facing API behavior

Suggested validation:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/mixture_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/heterogeneous_mixture_test.py -q
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats/mvn_test.py -q
```

## Practical Notes

- Reinstall test dependencies if the `pytest.mark.dependency` warnings persist:

```bash
pip install -e ".[test]"
```

- Prefer small-file validation as you go rather than rerunning the full suite
  after every edit
- Once the targeted files pass, rerun the full suite:

```bash
export NUMBA_DISABLE_JIT=1
TEST_TORCH_DEVICE=mps python -m pytest tests/torch_stats
```

## Exit Criteria

This task is complete when:

- `tests/torch_stats` passes on `TEST_TORCH_DEVICE=mps`
- no remaining failures are due to MPS dtype/device mismatches
- unsupported MPS ops are either handled or explicitly avoided
