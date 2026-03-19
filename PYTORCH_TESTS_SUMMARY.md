# PyTorch Stats Test Suite — Summary

## What Was Accomplished

### Example Files (`examples_torch/stats_examples/`)
Created torch-adapted example scripts for all remaining distributions:
- `mvn_example.py` — MultivariateGaussianDistribution
- `mixture_example.py` — MixtureDistribution (with `seq_posterior` demo)
- `heterogeneous_mixture_example.py` — HeterogeneousMixtureDistribution
- `conditional_example.py` — ConditionalDistribution (integer-keyed dmap)
- `composite_example.py` — CompositeDistribution
- `sequence_example.py` — SequenceDistribution
- `hidden_markov_example.py` — HiddenMarkovModelDistribution
- `jmixture_example.py` — JointMixtureDistribution

---

### Base Test Class (`tests/torch_stats/torch_stats_tests.py`)
Created `TorchStatsTestClass`, a torch-adapted equivalent of `tests/stats/stats_tests.py`. Key differences from the stats version:
- No `__eq__` checks — replaced with isinstance/type-correctness assertions
- `encoder.seq_encode(data, device=device)` — explicit device argument
- `seq_initialize` uses `int` seed instead of `numpy.random.RandomState`
- Float tolerance: `1e-10` for float64 (CPU/CUDA), `1e-4` for float32 (MPS)
- Added `test_09_seq_initialize` and `test_10_device` (not in stats base class)

**10 inherited tests per distribution:**

| Test | Description |
|---|---|
| `test_01_sampler` | Same seed yields identical samples |
| `test_02_log_density` | `log_density(x)` matches element of `seq_log_density` |
| `test_03_dist_to_encoder` | `dist_to_encoder()` returns `TorchSequenceEncoder` |
| `test_04_estimator` | `dist.estimator()` returns `TorchParameterEstimator` |
| `test_05_estimator_factory` | `est.accumulator_factory()` returns `TorchStatisticAccumulatorFactory` |
| `test_06_factory_make` | `factory.make()` returns `TorchStatisticAccumulator` |
| `test_07_acc_to_encoder` | `acc.acc_to_encoder()` returns `TorchSequenceEncoder` |
| `test_08_seq_update` | One EM step does not decrease log-likelihood |
| `test_09_seq_initialize` | `seq_initialize` produces a model with finite log-likelihood |
| `test_10_device` | Fitted model can be moved to CPU via `model.to(torch.device('cpu'))` |

---

### Test Files (`tests/torch_stats/`)
Created 18 test files covering all torch_stats distributions. Each inherits the 10 base tests and adds distribution-specific tests:

| File | Extra Tests |
|---|---|
| `gaussian_test.py` | encoder/accumulator type checks |
| `exponential_test.py` | sampler positivity, beta effect on mean |
| `gamma_test.py` | sampler positivity, mean approximation (`k * theta`) |
| `geometric_test.py` | sampler non-negativity, mean approximation (`1/p`) |
| `poisson_test.py` | sampler non-negativity, lambda mean approximation |
| `binomial_test.py` | sample range `[0, n]`, mean approximation (`n * p`) |
| `dmvn_test.py` | sample shape matches dimension, column-wise mean |
| `mvn_test.py` | sample shape matches dimension, column-wise mean |
| `intrange_test.py` | sample range `[min_val, max_val]`, empirical frequency |
| `intsetdist_test.py` | set element range, empirical inclusion frequency |
| `intmultinomial_test.py` | `(value, count)` pair element range |
| `mixture_test.py` | `seq_posterior` rows sum to 1, `seq_component_log_density` consistency with `seq_log_density` |
| `heterogeneous_mixture_test.py` | `seq_posterior` rows sum to 1 |
| `composite_test.py` | tuple length matches component count, log-density additivity |
| `conditional_test.py` | sample pair structure `(given, obs)`, finite log-density |
| `sequence_test.py` | sample is list, length distribution, finite log-density |
| `hidden_markov_test.py` | `viterbi` state range, `viterbi` length matches sequence |
| `jmixture_test.py` | sample is `(x1, x2)` pair, finite log-density |
| `int_plsi_test.py` | `(doc_id, [(word, count)])` structure, word index range, finite log-density, `component_log_density` shape |

**Result: 286 passing, 3 failing**

---

## Bugs Discovered in `torch_stats`

The failing tests expose real implementation bugs:

### 1. `ExponentialEstimator` — Wrong MLE formula [COMPLETED]
**File:** `src/dmx/torch_stats/exponential.py` (~line 231)

`ExponentialEstimator.estimate()` computes `p = count / sum` (rate = 1/mean) but `ExponentialDistribution` treats `beta` as the **mean**. The formula should be `p = sum / count`.

**Effect:** For `beta != 1.0`, each EM step moves the estimate away from the true mean, causing the log-likelihood to decrease after `seq_estimate`.

**Implemented:** Updated `ExponentialEstimator.estimate()` to interpret the torch sufficient statistics as `(sum, count)` and compute `beta = sum / count` in all branches, and added a targeted regression test in `tests/torch_stats/exponential_test.py`.

**Fix:** Swap the numerator and denominator in the `estimate` method:
```python
# Current (wrong):
p = suff_stat[1] / suff_stat[0]   # count / sum = rate

# Correct:
p = suff_stat[0] / suff_stat[1]   # sum / count = mean
```

---

### 2. `HeterogeneousMixtureDistribution` — wrong encoder-group indexing in EM [COMPLETED]
**File:** `src/dmx/torch_stats/heterogenous_mixture.py`

The original dtype diagnosis turned out to be stale. `PoissonDataEncoder.seq_encode()` already uses `vec.tensor(...)`, so Poisson observations are encoded as floating-point tensors. The real bug was in `HeterogeneousMixtureAccumulator.seq_update()`: it indexed grouped encoded data as `enc_data[i]` by component index, even though `enc_data` is grouped by encoder type and must be accessed as `enc_data[tag]`.

**Effect:** During EM updates, components could receive encoded data from the wrong encoder group. This breaks heterogeneous mixtures with shared encoder families and can surface as misleading runtime errors during `seq_estimate`.

**Implemented:** Updated `HeterogeneousMixtureAccumulator.seq_update()` to mirror the numpy implementation and use `enc_data[tag]` for component updates. Added regression coverage in `tests/torch_stats/heterogeneous_mixture_test.py` for a mixture with two Poisson components sharing one encoder group plus a Binomial component.

---

### 3. `HiddenMarkovAccumulator.seq_initialize` — sequence-weight indexing bug [COMPLETED]
**File:** `src/dmx/torch_stats/hmm.py` (~line 492)

The original issue was not in the shared `seq_initialize()` helper. The helper correctly passes one weight per top-level sequence. The real bug was in `HiddenMarkovAccumulator.seq_initialize()`: it compressed weights to the non-empty subsequence set and then indexed them with `idx_vec`, even though `idx_vec` stores original sequence indices from the encoded batch. Causes:
```
IndexError: index 496 is out of bounds for dimension 0 with size 496
```

**Effect:** Batches containing empty sequences could fail during HMM initialization because per-sequence weights no longer aligned with the original sequence ids used by `idx_vec`.

**Implemented:** Updated `HiddenMarkovAccumulator.seq_initialize()` to expand sequence weights with `weights[idx_vec]`, matching the encoder semantics and the existing `seq_update()` path. Re-enabled HMM EM/device coverage in `tests/torch_stats/hidden_markov_test.py` and added a regression test with explicit empty sequences.

---

### 4. `viterbi()` API inconsistency
**File:** `src/dmx/torch_stats/hmm.py` (~line 266)

`viterbi(x)` accepts a **single raw sequence** (`List[T]`), unlike all other `seq_*` methods which accept a batch-encoded `TorchEncodedSequence`. This makes it impossible to use `viterbi` in the same pipeline as `seq_log_density` and `seq_encode`.

Tests in `hidden_markov_test.py` are written to call `viterbi` with individual raw sequences to match the current API.

---

## Next Steps

1. [x] **Fix `ExponentialEstimator`** — completed in `src/dmx/torch_stats/exponential.py`; added regression coverage in `tests/torch_stats/exponential_test.py`.
2. [x] **Fix `HeterogeneousMixtureDistribution` EM update bug** — corrected encoder-group indexing in `src/dmx/torch_stats/heterogenous_mixture.py`; added regression coverage in `tests/torch_stats/heterogeneous_mixture_test.py`.
3. [x] **Fix `HiddenMarkovAccumulator.seq_initialize`** — corrected sequence-weight indexing in `src/dmx/torch_stats/hmm.py`; re-enabled HMM EM coverage and added empty-sequence regression coverage in `tests/torch_stats/hidden_markov_test.py`.
4. **Fix `IntegerPLSIDistribution.component_log_density`** — verify return shape matches `num_states`.
5. **Standardize `viterbi` API** — consider making it accept a `HiddenMarkovTorchSequence` (encoded batch) for consistency with other `seq_*` methods.
6. **Run on MPS/CUDA** — all tests currently run on CPU only; validate float32 tolerances on MPS devices.
7. **Add `pytest-dependency`** — the `@pytest.mark.dependency` markers in the base class generate warnings because the plugin is not installed. Add it to dev dependencies.
