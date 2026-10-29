# Fisher-Geometry humap: GitHub Epic and Issues

Source plan: `HUMAP_FISHER_ROADMAP.md`. Decision references (F1–F12) point to
section 5 of that file; milestones (M1–M6) to section 6.

This file is formatted for `scripts/create_github_issue_plan.py`, the same
tool used for `DMX_EXPERT_AGENT_ISSUES.md`. The whole file is one epic: one
run creates the epic as a parent issue, creates source issues 1–16 as
children, and fills the parent with an ordered checklist. Issue numbers below
are **source numbers**. GitHub assigns its own numbers. Dependencies are
written as source numbers plus titles so they stay resolvable.

Branching: open one branch per child issue with GitHub's **Create a branch**
button (`<issue-number>-<title-slug>`). Each issue lists a suggested short
slug. Target branch for PRs: `dmx-agent`.

Labels: `enhancement,tests`.

## Epic

| Epic | Source issues | Labels |
|---|---|---|
| Fisher-geometry humap for heterogeneous embeddings | 1–16 | `enhancement,tests` |

Dry-run first by adding `--dry-run`.

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file HUMAP_FISHER_ISSUES.md --link-sub-issues --labels "enhancement,tests" --start-issue 1 --end-issue 16 --parent-title "Epic: Fisher-geometry humap for heterogeneous embeddings" --parent-summary "Replace the lossy posterior-only input to humap with a Fisher-geometry feature map built from the fitted mixture: the Hellinger distance between per-record locally tilted models, computed as a sparse Euclidean feature vector [sqrt(z_i), sqrt(c) sqrt(z_ik) s_ik] where s_ik is the whitened score of record i under component k. Adds a seq_fisher_score primitive to dmx.stats and dmx.bstats distributions (leaves first, then combinators and latent-variable components), a mixture-level embedding builder with local and MPI paths, and an opt-in humap(representation='fisher'). Reduces exactly to current humap at c = 0, recovers within-component structure, keeps per-record embarrassingly parallel scaling, and supports out-of-sample transform. See HUMAP_FISHER_ROADMAP.md."
```

Order of work (critical path to a usable feature in bold):

| Phase | Source issues |
|---|---|
| A. API and harness | **1**, **2** |
| B. `dmx.stats` leaves | **3**, **4**, **5**, 6 |
| C. Observed combinators | **7**, 8 |
| D. Latent-variable components | 9, 10 |
| E. `dmx.bstats` parity | **11** |
| F. Mixture embedding and scale-out | **12**, 13 |
| G. humap | **14** |
| H. Evidence and docs | 15, 16 |

---

## Issue 1

**Title:** Add the `seq_fisher_score` distribution API, options, and shared score kernels

**Body:**

Epic: Fisher-geometry humap. Phase A. Milestone M1. Suggested branch slug:
`fisher-score-api`.

Introduce the per-distribution primitive that the whole epic builds on
(roadmap sections 2 and 4.1, decisions F1, F10, F12). No distribution
implements it yet in this issue beyond the base class.

Context: `humap` currently embeds only the posterior matrix, which loses each
record's position inside its component. The fix needs every distribution to
return a **whitened score**: `s(x) = I^{-1/2} (T(x) - E[T])`, with
`E[s] = 0` and `Cov[s] = I` under the distribution.

Scope:

- in `src/dmx/stats/pdist.py`, add to `SequenceEncodableProbabilityDistribution`:
  - `seq_fisher_score(self, x, options=None) -> np.ndarray` of shape
    `(N, self.fisher_dim(options))`, float64; the base implementation raises
    `NotImplementedError` with the class name and a pointer to the docs page
  - `fisher_dim(self, options=None) -> int`; base raises
    `NotImplementedError`
  - `fisher_field_slices(self, options=None) -> Dict[str, slice]`; base
    returns `{self.name or type(self).__name__: slice(0, fisher_dim)}`
  - `has_fisher_score(dist) -> bool` helper (module-level) that checks
    recursively whether a distribution tree is fully supported
- mirror the same three methods on the `dmx.bstats` base class in
  `src/dmx/bstats/pdist.py`
- add `FisherScoreOptions` (frozen dataclass) in a new module
  `src/dmx/utils/fisher.py` with fields and defaults:
  - `prob_floor: float = 1e-6`
  - `location_only: bool = False` (MVN / DiagonalGaussian)
  - `sequence_scaling: Literal["conditional", "marginal"] = "conditional"`
  - `balance_fields: bool = False`
  - `on_missing: Literal["raise", "posterior_only"] = "raise"`
  - `dense_category_limit: int = 256`
- in the same module, add the place for pure, parameter-only score kernels
  (for example `gaussian_score(x, mu, sigma2)`), so `dmx.stats` and
  `dmx.bstats` share implementations; this issue adds the module skeleton
  and docstring conventions, leaf issues fill it in
- add `docs/utils/fisher_embedding.rst` (linked from `docs/utils/index.rst`
  and from the humap section of `docs/utils/embeddings.rst`) with
  the math from roadmap sections 2–3: definition of `Q_i`, second-order form,
  feature map, scale and truncation, UMAP compatibility, and the score
  contract
- document F12: keyed (shared) parameters do not change the score definition

Acceptance criteria:

- calling `seq_fisher_score` on any distribution raises a clear
  `NotImplementedError` naming the class
- `FisherScoreOptions` is importable from `dmx.utils.fisher` and documented
- docs build passes and the new page renders the formulas
- no behavior change for existing APIs; full test suite passes

Depends on: none.

---

## Issue 2

**Title:** Add a generic contract test for Fisher scores across all distributions

**Body:**

Epic: Fisher-geometry humap. Phase A. Milestone M1. Suggested branch slug:
`fisher-score-contract-test`.

One parametrised test validates every implementation, using the samplers that
dmx already provides (pattern: `tests/stats/num_params_test.py`).

Scope:

- add `tests/stats/fisher_score_test.py` with a registry of `FisherCase`
  entries (distribution factory, implementing issue, options, expected
  covariance `rank_deficit`) covering every family named in roadmap section
  4.3 phases B, C, D; unimplemented families are `pytest.skip`ped with the
  implementing issue so the file tracks progress
- shared checks live in `tests/fisher_contract.py`
- for each case:
  - draw `n = 1000` samples with a seed derived from the case id via
    `dist.sampler(seed)`, encode with the package's encoder
    (`dist.dist_to_encoder().seq_encode` for `dmx.stats`, `dist.seq_encode`
    for `dmx.bstats`), call `seq_fisher_score`
  - assert shape `(n, fisher_dim())`, all finite, dtype float64
  - assert sample mean ≈ 0 and sample covariance ≈ the identity, or a
    projector of rank `fisher_dim() - rank_deficit` for one-hot style
    encodings, within 5 standard errors estimated from the sample
  - assert determinism: same input gives identical output
  - assert `fisher_field_slices` partitions `range(fisher_dim())`
- add a hand-computed value test per family (small input, closed-form
  expected rows) in each leaf issue, not here
- add `tests/bstats/fisher_score_test.py` reusing the same helpers for
  `dmx.bstats` (cases implemented in issue 11)
- self-test the contract on correct and deliberately broken toy scores
- keep the contract suite well under a second, excluding the one-time
  `dmx.stats` import

Acceptance criteria:

- test file runs green with every case skipped (nothing implemented yet)
- helpers are reusable by later issues without modification
- documented in `tests/stats/README.md`

Depends on: 1 (Add the `seq_fisher_score` distribution API, options, and
shared score kernels).

---

## Issue 3

**Title:** Implement Fisher scores for continuous scalar leaves (Gaussian, LogGaussian, Exponential, Gamma)

**Body:**

Epic: Fisher-geometry humap. Phase B. Milestone M2. Suggested branch slug:
`fisher-score-continuous-leaves`.

Closed-form whitened scores. Add the kernel to `src/dmx/utils/fisher.py` and
a thin `seq_fisher_score` / `fisher_dim` on each `dmx.stats` class.

Scores (with `u = (x - mu) / sigma`):

| Class | Parameters | Score | dim |
|---|---|---|---|
| `GaussianDistribution` | `mu`, `sigma2` | `[u, (u^2 - 1) / sqrt(2)]` | 2 |
| `LogGaussianDistribution` | `mu`, `sigma2` on `log x` | Gaussian score of `log x` | 2 |
| `ExponentialDistribution` | mean `beta` | `x / beta - 1` | 1 |
| `GammaDistribution` | shape `k`, scale `theta` | `L^{-1} ([x, log x] - [k theta, digamma(k) + log theta])` with `L L^T = [[k theta^2, theta], [theta, trigamma(k)]]` | 2 |

Scope:

- implement the four kernels and class methods, vectorised over the encoded
  array (same idiom as `seq_log_density`)
- `location_only` does not apply to scalar leaves (ignored)
- register the four families in the contract test (issue 2)
- hand-computed value tests (for example Gaussian `mu=1, sigma2=4` at
  `x = [1, 3, -1]` gives `[[0, -0.7071], [1, 0], [-1, 0]]`)

Acceptance criteria:

- contract test passes for all four families across a few parameter
  settings (including small and large Gamma shape)
- value tests pass
- docstrings follow `docs/docstring_style.rst`

Depends on: 2 (Add a generic contract test for Fisher scores across all
distributions).

---

## Issue 4

**Title:** Implement Fisher scores for count leaves (Poisson, Geometric, Binomial)

**Body:**

Epic: Fisher-geometry humap. Phase B. Milestone M2. Suggested branch slug:
`fisher-score-count-leaves`.

Closed-form whitened scores; kernels in `src/dmx/utils/fisher.py`.

| Class | Parameters | Score | dim |
|---|---|---|---|
| `PoissonDistribution` | `lam` | `(x - lam) / sqrt(lam)` | 1 |
| `GeometricDistribution` | `p`, support 1, 2, ... | `(p x - 1) / sqrt(1 - p)` | 1 |
| `BinomialDistribution` | `p`, `n`, `min_val` | `y = x - min_val`; `(y - n p) / sqrt(n p (1 - p))` | 1 |

Scope:

- implement kernels and class methods
- guard degenerate parameters (`lam -> 0`, `p -> 0` or `1`): clamp with
  `options.prob_floor` and document the behavior
- add a Bernoulli kernel `(x - p) / sqrt(p (1 - p))` to the shared module
  now (used by `dmx.bstats` in issue 11 and by Optional in issue 7)
- register families in the contract test; add value tests

Acceptance criteria:

- contract test passes, including near-degenerate parameters without
  non-finite output
- value tests pass

Depends on: 2 (Add a generic contract test for Fisher scores across all
distributions).

---

## Issue 5

**Title:** Implement Fisher scores for categorical leaves (Categorical, IntegerCategorical)

**Body:**

Epic: Fisher-geometry humap. Phase B. Milestone M2. Suggested branch slug:
`fisher-score-categorical-leaves`.

Categorical fields are the most common heterogeneous type and the most
delicate (decision F7).

Math: for probabilities `p` over `C` levels the whitened score is
`s(x) = e_x / sqrt(p_x) - sqrt(p)`, with covariance the projector
`I - sqrt(p) sqrt(p)^T` and `||s(a) - s(b)||^2 = 1/p_a + 1/p_b` for `a != b`.

Scope:

- `CategoricalDistribution` (`src/dmx/stats/categorical.py`): data is encoded
  as `(xs, val_map_inv)`; compute the score row per unique value, then index
  by `xs`
  - level order: the distribution's `pmap` keys in a stable order, plus one
    trailing "other" coordinate when `default_value > 0` so unseen levels
    have a defined position
  - unseen levels use probability `default_value` (or `prob_floor` if
    zero); all probabilities are floored at `prob_floor`
  - when `C <= options.dense_category_limit`: dense form above
  - when `C > dense_category_limit`: offset-free form `e_x / sqrt(p_x)`
    returned as a `scipy.sparse.csr_matrix`; document that it is exact for
    pairs in the same component and approximate across components
- `IntegerCategoricalDistribution` (`src/dmx/stats/intrange.py`): same
  kernel over `p_vec`, with `x - min_val` as the index; out-of-range values
  get the floor
- extend the return type contract of `seq_fisher_score` to allow
  `np.ndarray | scipy.sparse.csr_matrix`, and update issue 2's helpers to
  densify for the covariance check
- register both families (small `C` dense, large `C` sparse) in the contract
  test; value test: `p = [0.5, 0.25, 0.25]`, check pairwise distances equal
  `1/p_a + 1/p_b`

Acceptance criteria:

- contract test passes for dense and sparse regimes
- unseen and rare levels produce finite rows
- pairwise-distance value test passes

Depends on: 2 (Add a generic contract test for Fisher scores across all
distributions).

---

## Issue 6

**Title:** Implement Fisher scores for multivariate leaves (MVN, DiagonalGaussian, Dirichlet)

**Body:**

Epic: Fisher-geometry humap. Phase B. Milestone M2. Suggested branch slug:
`fisher-score-multivariate-leaves`.

Scores (with `u = L^{-1} (x - mu)`, `L L^T = Sigma`):

| Class | Score | dim |
|---|---|---|
| `MultivariateGaussianDistribution` | location `u`; scale block: `u_i u_j` for `i < j` and `(u_i^2 - 1) / sqrt(2)` on the diagonal | `d + d(d+1)/2`, or `d` if `location_only` |
| `DiagonalGaussianDistribution` | per-dimension Gaussian score from issue 3 | `2d`, or `d` if `location_only` |
| `DirichletDistribution` | `L^{-1} (log x - (digamma(alpha) - digamma(alpha_0)))` with `L L^T = diag(trigamma(alpha)) - trigamma(alpha_0)` | `d` |

Scope:

- implement kernels and class methods; cache the Cholesky factor on the
  distribution (computed lazily, invalidated if parameters change)
- honor `options.location_only` for MVN and DiagonalGaussian; document that
  it drops scale information and is the recommended default when `d` is
  large
- register in the contract test (including `location_only=True`); value
  tests for `d = 2`

Acceptance criteria:

- contract test passes for `d` in {1, 2, 5}
- `location_only` changes `fisher_dim` and output accordingly

Depends on: 3 (Implement Fisher scores for continuous scalar leaves).

---

## Issue 7

**Title:** Implement Fisher scores for observed combinators (Composite, Optional, Ignored, NullDistribution)

**Body:**

Epic: Fisher-geometry humap. Phase C. Milestone M3. Suggested branch slug:
`fisher-score-combinators`.

This issue is what makes heterogeneous records work: independent fields have
block-diagonal Fisher information, so the composite score is a concatenation.

Scope:

- `CompositeDistribution`: `hstack` of each child's
  `seq_fisher_score(x.data[i], options)` (sparse-aware: return CSR if any
  child is sparse); `fisher_dim` is the sum; `fisher_field_slices` returns
  one slice per field named by the child's `name` or `field_<i>`, recursing
  so nested composites produce `parent.child` names
- `options.balance_fields=True` (decision F9): divide each field block by
  `sqrt(fisher_dim(field))`; default off
- `OptionalDistribution` (missing probability `q`): score is
  `[ (m - q) / sqrt(q (1 - q)),  (1 - m) * s_child / sqrt(1 - q) ]` where
  `m` is the missing indicator and missing rows have a zero child block;
  when `q` is 0 or not estimated, drop the indicator coordinate. Uses the
  Bernoulli kernel from issue 4.
- `IgnoredDistribution`, `NullDistribution`: `fisher_dim = 0`, return an
  `(N, 0)` array
- `has_fisher_score` recurses through these
- contract test cases: composite of Gaussian + Categorical + Poisson +
  Optional(Gaussian) + Ignored; value test checking field slices and that
  the composite equals the concatenation of child scores

Acceptance criteria:

- contract test passes for the mixed composite
- `fisher_field_slices` covers every coordinate exactly once with readable
  names
- `balance_fields` changes block norms as specified

Depends on: 3, 4, 5 (continuous, count, and categorical leaves).

---

## Issue 8

**Title:** Implement Fisher scores for variable-length observed data (Sequence, BernoulliSet, multinomials)

**Body:**

Epic: Fisher-geometry humap. Phase C. Milestone M3. Suggested branch slug:
`fisher-score-variable-length`.

Variable-length records need a scaling decision (F8). The summed element
score of an iid sequence of length `n` has covariance `n I`.

Scope:

- `SequenceDistribution`: score is
  `[ sum_t s_elem(x_t) / scale(n),  s_len(n) ]` where
  - `sequence_scaling="conditional"` (default): `scale(n) = sqrt(n)`
    (whitened conditional on the observed length; zero block for `n = 0`)
  - `sequence_scaling="marginal"`: `scale(n) = sqrt(E[n])` from the length
    distribution (longer sequences end up farther apart)
  - the length block is the length distribution's own score
    (`NullDistribution` gives no block)
  - aggregate per-element scores with the encoded index arrays
    (`np.add.at` or `np.bincount` per column), as `seq_log_density` does
- `IntegerMultinomialDistribution` and `MultinomialDistribution` (count
  vectors with total `N_i`): `(n_c - N_i p_c) / sqrt(N_i p_c)` per level,
  with the same `sequence_scaling` rule and the length block
- `BernoulliSetDistribution` (`pmap`, `min_prob`): per-element Bernoulli
  scores `(1[e in set] - p_e) / sqrt(p_e (1 - p_e))`; dense below
  `dense_category_limit` elements, otherwise sparse offset-free form as in
  issue 5
- stretch (may be split to a follow-up): fully observed
  `MarkovChainDistribution` via per-row categorical transition scores
- contract test: sequences with Poisson length, both scaling modes (for
  `"conditional"` check whitening conditional on `n`); multinomial; set

Acceptance criteria:

- contract test passes for each family and scaling mode
- `n = 0` rows are finite (zero element block)
- docs page explains the two scaling modes

Depends on: 7 (Implement Fisher scores for observed combinators).

---

## Issue 9

**Title:** Add empirical whitening and Fisher scores for nested mixtures

**Body:**

Epic: Fisher-geometry humap. Phase D. Milestone M5. Suggested branch slug:
`fisher-score-latent-mixtures`.

Latent-variable components do not have closed-form Fisher information. Their
score is `E[T | x] - E[T]`, which the E-step already computes, but only summed
over observations in `seq_update`.

Scope:

- `EmpiricalFisherWhitener` in `src/dmx/utils/fisher.py`: fit on raw
  centered per-record score vectors (optionally posterior-weighted) and apply
  `diag`, `full`, or `pca(rank)` whitening; deterministic given data and
  seed; serialisable with the embedding
- nested `MixtureDistribution` as a component (and
  `HeterogeneousMixtureDistribution`): complete-data score
  `[ (z_j - w_j) / sqrt(w_j),  sqrt(z_j / w_j) * s_j(x) ]` over inner
  components `j`, followed by empirical whitening of the result (the
  complete-data information over-states the observed information);
  `fisher_dim = J + sum_j dim_j`
- the top-level mixture in issue 12 is **not** scored this way (it uses the
  `sqrt(z)` block, decision F2); this issue covers mixtures nested inside a
  component
- `options.on_missing="posterior_only"` path: a component whose tree is not
  fully supported contributes no score block, and a single warning lists
  the unsupported classes
- contract test: nested mixture of Gaussians, checked after empirical
  whitening (mean ≈ 0, covariance ≈ I on the fitted sample)

Acceptance criteria:

- nested-mixture components score and pass the post-whitening check
- unsupported components raise by default and degrade with a warning under
  `posterior_only`

Depends on: 7 (Implement Fisher scores for observed combinators).

---

## Issue 10

**Title:** Implement Fisher scores for the HMM family via per-record expected sufficient statistics

**Body:**

Epic: Fisher-geometry humap. Phase D. Milestone M5. Suggested branch slug:
`fisher-score-hmm`. Stretch within the epic: may move to a follow-up if
M4 ships first.

Scope:

- add a per-record variant of the forward-backward statistics to
  `HiddenMarkovModelDistribution` (`src/dmx/stats/hidden_markov.py`):
  expected initial-state counts, expected transition counts, and
  posterior-weighted emission scores, each per sequence instead of summed
  (reuse the existing numba kernels where possible)
- score = concatenation of the centered blocks, then empirical whitening
  (issue 9), with `sequence_scaling` applied to the transition and emission
  blocks as in issue 8
- extend to `IntegerHiddenMarkovModelDistribution` if the same kernel
  applies; list other HMM variants (`look_back_hmm`, `tree_hmm`) as
  follow-ups in the PR description
- contract test after whitening; value test on a 2-state HMM with a
  hand-checkable sequence

Acceptance criteria:

- HMM components score and pass the post-whitening check
- runtime of per-record statistics within 2x of a `seq_log_density` pass
  on the same data

Depends on: 8 (variable-length data), 9 (empirical whitening and nested
mixtures).

---

## Issue 11

**Title:** Add Fisher scores to `dmx.bstats` distributions used by the automatic DPM

**Body:**

Epic: Fisher-geometry humap. Phase E. Milestone M4. Suggested branch slug:
`fisher-score-bstats`.

The default `humap` path fits a **bstats** Dirichlet-process mixture via
`get_dpm_mixture` / `prepare_mixture_model` (`src/dmx/utils/automatic.py`),
whose leaves are `dmx.bstats` Gaussian, Categorical, Poisson, Optional,
Sequence, Composite, and Ignored. Without this issue `representation="fisher"`
would only work for user-supplied `dmx.stats` mixtures.

Scope:

- implement `seq_fisher_score`, `fisher_dim`, `fisher_field_slices` on
  `dmx.bstats` Gaussian, Categorical, Poisson, Bernoulli, Exponential,
  Geometric, Gamma, Composite, Optional, Sequence, Ignored, and NullDist,
  calling the shared kernels in `src/dmx/utils/fisher.py` (no duplicated
  math)
- use plug-in parameters (`get_parameters()`, posterior means), decision F6;
  note in docstrings that expected scores under the parameter posterior are
  future work
- bstats Categorical: map its parameter layout onto the issue-5 kernel,
  including unseen-level handling
- register cases in `tests/bstats/fisher_score_test.py`; add a parity test
  that a bstats distribution and the `dmx.stats` distribution with the same
  parameters return identical scores

Acceptance criteria:

- bstats contract tests pass
- stats/bstats parity test passes for every shared family
- `has_fisher_score` is true for a DPM produced by `get_dpm_mixture` on the
  repo's heterogeneous example data

Depends on: 7 (observed combinators), 8 (variable-length data) for
`Sequence`.

---

## Issue 12

**Title:** Build the mixture-level Fisher embedding (`seq_fisher_embedding` and `FisherEmbedding`)

**Body:**

Epic: Fisher-geometry humap. Phase F. Milestone M4. Suggested branch slug:
`mixture-fisher-embedding`.

Assemble the feature map from roadmap section 2.3:
`phi_i = [sqrt(z_i), sqrt(c) sqrt(z_ik) s_ik for k in top_m(i)]`.

Scope:

- `FisherEmbedding` dataclass in `src/dmx/utils/fisher.py`: `matrix`
  (CSR float32), `scale` (the `c` used), `top_m`, `posterior_slice`,
  `component_slices` (per component k), `field_slices` (per component and
  field, from `fisher_field_slices`), `options`, and an optional fitted
  `EmpiricalFisherWhitener` per component
- `MixtureDistribution.seq_fisher_embedding(x, scale=None, top_m=3,
  options=None, dtype=np.float32) -> FisherEmbedding` on both
  `dmx.stats.mixture.MixtureDistribution` and
  `dmx.bstats.mixture.MixtureDistribution`:
  - `Z = seq_posterior(x)`; keep each row's top `top_m` components (all if
    `top_m >= K` or `None`)
  - for each component, compute `seq_fisher_score` on all rows (first
    version; row subsetting of encoded data is a follow-up), multiply by
    `sqrt(c * z_ik)`, and keep only rows where k is in the top `m`
  - default `scale`: `c = 1 / p_eff`, `p_eff = mean_i sum_k z_ik dim_k`
    (decision F4)
  - zero-weight components (`zw`) are skipped
- `fisher_embedding(model, data, *, scale=None, top_m=3, options=None,
  chunk_size=None) -> FisherEmbedding` convenience function that encodes the
  data and processes it in chunks to bound peak memory, producing identical
  rows to the unchunked call
- `transform` support: given a stored `FisherEmbedding` (scale, options,
  whiteners), `fisher_embedding(model, new_data, like=embedding)` reuses
  them so new records land in the same feature space
- tests:
  - `scale=0` gives exactly `[sqrt(Z)]` (current humap features) padded
    with zeros
  - two records with identical posteriors: `||phi_i - phi_j||^2 = c *
    ||s_i - s_j||^2`
  - chunked equals unchunked
  - `top_m = K` vs `top_m = 3` differ by less than a small tolerance on a
    well-separated Gaussian mixture

Acceptance criteria:

- works for stats and bstats mixtures whose components pass
  `has_fisher_score`
- output is CSR float32 with the documented slices
- tests above pass

Depends on: 7 (observed combinators), 11 (bstats parity) for the bstats
mixture.

---

## Issue 13

**Title:** Compute Fisher embeddings in parallel with MPI and Spark

**Body:**

Epic: Fisher-geometry humap. Phase F. Milestone M5. Suggested branch slug:
`fisher-embedding-distributed`.

The embedding is per record, so it distributes like `seq_posterior`. Only the
final UMAP fit is centralised.

Scope:

- `fisher_embedding_mpi(model, local_data, comm, ...)` in
  `src/dmx/mpi4py/utils/` next to `humap_mpi`: each rank computes its rows;
  `p_eff` (for the default scale) is reduced across ranks with one
  `allreduce`; empirical whiteners (issue 9) are fitted from reduced
  moments; rows are gathered to root (or returned per rank with
  `gather=False`)
- Spark: a `mapPartitions` helper in `src/dmx/utils/spark.py` that returns
  an RDD of `(index, sparse row)` pairs, using the same two-pass scale and
  whitener reduction
- tests in `tests/mpi4py/` (run with `NUMBA_DISABLE_JIT=1 mpiexec -n 2`)
  and `tests/utils/test_spark.py`: distributed rows equal the local
  `fisher_embedding` rows bit-for-bit for float64 and within float32
  tolerance

Acceptance criteria:

- MPI and Spark paths reproduce local rows
- one collective for the default scale; no N x N or N x K gather beyond the
  final rows

Depends on: 12 (mixture-level Fisher embedding).

---

## Issue 14

**Title:** Add `representation="fisher"` to `humap` and `humap_mpi`

**Body:**

Epic: Fisher-geometry humap. Phase G. Milestone M4. Suggested branch slug:
`humap-fisher-representation`.

Wire the embedding into the public entry points without changing defaults
(decision F11).

Scope:

- `src/dmx/utils/humap.py`, new keyword-only arguments:
  - `representation: Literal["posterior", "fisher"] = "posterior"`
  - `fisher_scale: Optional[float] = None` (default `1 / p_eff`)
  - `top_m: int = 3`
  - `fisher_options: Optional[FisherScoreOptions] = None`
- when `representation="fisher"`:
  - build features with `fisher_embedding` (issue 12) from the fitted or
    supplied mixture
  - default UMAP metric becomes `"euclidean"` unless the caller set
    `metric`; pass the CSR matrix directly to UMAP
  - the 4th return value is the feature matrix actually fed to UMAP
    (posteriors in `"posterior"` mode, as today); also expose the
    `FisherEmbedding` via a new `return_embedding: bool = False` flag
    (appends it as a 5th element) so the existing 4-tuple is unchanged by
    default
  - emit progress events (`dmx.utils.progress`) for the embedding stage and
    log the scale, `p_eff`, `top_m` and feature dimension at DEBUG
- `umap_kwargs=None` currently binds the shared module-level `DEFAULT_UMAP`
  dict. Fisher mode changes the default `metric`, so copy the defaults
  (`dict(DEFAULT_UMAP)`) before any per-call change, or the first fisher
  call would alter later posterior calls. Remove the unused module-level
  `umap_model`.
- `humap_transform(fit, mix_model, new_data, embedding=None)`: compute
  features for new records (posterior or Fisher, matching the fit) and call
  `fit.transform`
- `humap_mpi` (`src/dmx/mpi4py/utils/humap.py`): same arguments, using
  `fisher_embedding_mpi` when issue 13 has landed, otherwise compute on root
  with a documented limitation
- tests: default call unchanged (same output for fixed seeds);
  `representation="fisher"` runs on the default bstats DPM path and on a
  supplied `dmx.stats` mixture; `fisher_scale=0` reproduces the posterior
  embedding's kNN graph; `humap_transform` places training records near
  their fitted coordinates

Acceptance criteria:

- `humap(data, representation="fisher")` works end to end on repo example
  data
- existing callers and tests unaffected
- docstrings describe the new arguments and link the docs page

Depends on: 12 (mixture-level Fisher embedding), 11 (bstats parity) for the
default path; 13 (distributed) for the full `humap_mpi` path.

---

## Issue 15

**Title:** Validate and benchmark Fisher humap against posterior humap and raw-feature baselines

**Body:**

Epic: Fisher-geometry humap. Phase H. Milestone M6. Suggested branch slug:
`fisher-humap-benchmark`.

Show that the method recovers within-component structure, quantify the
approximation, and measure scaling.

Scope:

- synthetic generator in `tests/utils/` (and reused by the benchmark):
  heterogeneous records whose clusters have known internal structure, for
  example a ring or line in two Gaussian fields inside each cluster, a
  categorical field correlated with position on the ring, and a Poisson
  count; ground-truth within-cluster coordinates retained
- reference metric `fisher_hellinger_reference(model, data)` for small N:
  exact Hellinger between local models `Q_i` using closed-form
  Bhattacharyya coefficients for Gaussian, Poisson and Categorical leaves
  (roadmap section 2.1); used only in tests and the benchmark
- metrics: trustworthiness and continuity, kNN recall against ground truth,
  within-cluster rank correlation with the true coordinates, and agreement
  of kNN graphs between the Euclidean `phi` and the exact reference
- compare: posterior humap, Fisher humap (default, `balance_fields=True`,
  several `fisher_scale` values), and UMAP on standardised raw features
  with one-hot categoricals
- scaling: wall time and peak memory of the embedding stage and of UMAP for
  N in {1e4, 1e5, 1e6} on one node, plus the MPI path at 1e6 if issue 13
  has landed
- write results to `_benchmarks/fisher-humap.md` following
  `_benchmarks/evaluation-rubric.md`
- a fast regression test (small N) asserting Fisher humap beats posterior
  humap on within-cluster rank correlation by a set margin

Acceptance criteria:

- benchmark page with tables and one figure per comparison
- regression test in CI
- recommended defaults (`fisher_scale`, `top_m`, `balance_fields`)
  confirmed or updated in the roadmap

Depends on: 14 (`representation="fisher"` in humap).

---

## Issue 16

**Title:** Document Fisher humap with a user guide page, example, and changelog entry

**Body:**

Epic: Fisher-geometry humap. Phase H. Milestone M6. Suggested branch slug:
`fisher-humap-docs`.

Scope:

- add a short "Fisher representation" subsection with a usage snippet to
  `docs/utils/embeddings.rst`
- extend `docs/utils/fisher_embedding.rst` (from issue 1) with a user
  section: when to use `representation="fisher"`, choosing `fisher_scale`,
  `top_m`, `balance_fields`, `sequence_scaling`, variable-length data,
  handling unsupported components, and a table of supported distributions
  (generated from the contract-test registry if practical)
- add `examples/humap_fisher_example.py` using the synthetic data from
  issue 15 and one repo dataset, plotting posterior vs Fisher embeddings
  side by side
- add `fisher_embedding` and `humap_transform` to the API reference
- `CHANGELOG.md` entry under the next release
- update `HUMAP_FISHER_ROADMAP.md` with milestone status and any decisions
  changed by the benchmark; add a pointer from the embeddings row in
  `DMX_EXPERT_AGENT_ROADMAP.md` (section 7) so agent tools can expose the
  new option
- example passes the repo's pylint gate

Acceptance criteria:

- docs build passes; example runs in under a minute on a laptop
- changelog and roadmap updated

Depends on: 14 (`representation="fisher"` in humap), 15 (validation and
benchmarks).
