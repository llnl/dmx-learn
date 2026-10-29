# Fisher-Geometry humap Roadmap

**Goal:** make `humap` embed heterogeneous records using the full local
geometry of a fitted dmx mixture, not only the component posteriors. The fix
must keep today's scaling story: fit the model (locally, MPI, or Spark),
compute a per-record feature vector in one embarrassingly parallel pass, and
hand it to UMAP. No N x N matrix is ever formed.

Issue plan: `HUMAP_FISHER_ISSUES.md` (one epic, source issues 1–16).

---

## 1. Problem

`humap` (`src/dmx/utils/humap.py`) fits UMAP to the posterior matrix
`Z = (z_ik)` with the Hellinger metric. That representation is lossy:

- With a reasonably fitted DPM most posteriors are close to one-hot. Every
  record in a component sits at distance ~0 from every other record in the
  same component, so UMAP's local-connectivity step (`rho_i`, `sigma_i`) is
  degenerate and the shape of each cluster in the embedding is arbitrary.
- `htsne` does not fix this. Its affinity
  `P_ab ∝ sum_k z_ak f_k(x_b) = p(x_b) sum_k z_ak z_bk / pi_k` has rank K and
  depends on each record only through `(z_i, p(x_i))`. It carries the same
  information as `humap` plus a density weighting, at O(N^2) cost.

What both discard is **where a record sits inside its component**.

## 2. The metric

### 2.1 Definition

Fitted model: `p(x) = sum_k pi_k f_k(x | eta_k)`. For each record `x_i`
define a local model over (component, observation):

$$Q_i(k, x) = z_{ik}\, f_k(x \mid \tilde\eta_{ik}),\qquad
\tilde\eta_{ik} = \eta_k + \varepsilon\, I_k^{-1}\big(T_k(x_i) - \mu_k\big)$$

`T_k` are the sufficient statistics of component k, `mu_k = E_k[T]`,
`I_k = Cov_k[T]` is the Fisher information. In mean parameters the tilt is
`mu~ = (1 - eps) mu_k + eps T(x_i)`: a conjugate update that adds `x_i` with
weight `eps`, i.e. one observation's worth of evidence.

The distance is Hellinger between local models:

$$H^2(Q_i,Q_j) = 1 - \sum_k \sqrt{z_{ik}z_{jk}}\;
\mathrm{BC}\big(f_k(\cdot\mid\tilde\eta_{ik}), f_k(\cdot\mid\tilde\eta_{jk})\big)$$

For exponential families
`BC(eta1, eta2) = exp(A((eta1+eta2)/2) - A(eta1)/2 - A(eta2)/2)`. For a
Composite, fields are independent, so BC is a product over fields.

### 2.2 Second-order form

With the whitened score `s_ik = I_k^{-1/2} (T_k(x_i) - mu_k)`,
`BC ≈ 1 - (eps^2 / 8) ||s_ik - s_jk||^2`, and

$$2H^2(Q_i,Q_j) \approx \|\sqrt{z_i}-\sqrt{z_j}\|^2
+ c \sum_k \sqrt{z_{ik}z_{jk}}\,\|s_{ik}-s_{jk}\|^2,\qquad c = \varepsilon^2/4$$

The first term is exactly the current `humap` metric.

### 2.3 Euclidean feature map (what we compute)

$$\phi_i = \big[\sqrt{z_{i1}},\dots,\sqrt{z_{iK}},\;
\sqrt{c}\sqrt{z_{i1}}\,s_{i1},\dots,\sqrt{c}\sqrt{z_{iK}}\,s_{iK}\big]$$

`||phi_i - phi_j||^2` equals the second-order form whenever
`z_ik = z_jk`, and differs only by a term proportional to
`sqrt(z_ik) - sqrt(z_jk)` otherwise, i.e. for records that are already far
apart. UMAP only consumes each record's k nearest neighbours, which almost
always share components, so the approximation is exact where it matters.

### 2.4 Scale and truncation

- Inside one component the neighbour ranking does not depend on `c`. `c`
  only sets how many within-component standard deviations equal a change of
  component. Default `c = 1 / p_eff`, with `p_eff` the posterior-weighted
  mean score dimension, so a typical within-component pair (`E||s_i-s_j||^2
  = 2 p_eff`) is as far apart as two records in different components.
- For exponential families `z_ik` decays like `exp(-||s_ik||^2 / 2)`, faster
  than `||s_ik||` grows, so keeping each record's top `m` components
  (default 3) loses essentially nothing. `phi` is stored sparse, float32.

### 2.5 Why it is principled

1. **Generalises humap:** `c = 0` gives today's metric exactly.
2. **Keeps the lost information:** `T` is sufficient for an
   exponential-family component, so `s_ik` carries everything component k
   can say about `x_i`. What is dropped is what the model declares
   irrelevant.
3. **Heterogeneity without hand weights:** every field is tilted by the same
   evidence `eps`. The Fisher metric is the unique metric invariant under
   sufficient-statistic maps (Chentsov), so units of a Gaussian field,
   category labels and parametrisation choices do not change distances.
4. **Local Mahalanobis, generalised:** all-Gaussian data gives
   posterior-blended per-component Mahalanobis distance; a Categorical
   mismatch between levels `a` and `b` costs `1/p_a + 1/p_b`.
5. **Local version of htsne's kernel:** tilting `sum_k z_ik f_k(x_j)` toward
   `x_i` adds, to first order, the Fisher inner product that htsne lacks.
6. **Out-of-sample:** `phi` is defined for new records through the fitted
   model, so `UMAP.transform` works.

## 3. UMAP compatibility

| UMAP assumption / requirement | Status |
|---|---|
| Locally (approximately) constant Riemannian metric | Met by construction: the pull-back of the Fisher–Rao metric. Constant (`I_k^{-1}`) inside a component, smoothly blended by `sqrt(z)` across components. |
| Data uniform on the manifold | Handled by UMAP's per-point `sigma_i`, as for any input. Whitening already partly equalises density. |
| Local connectivity (`rho_i > 0`) | Currently violated (ties at distance ~0); fixed by `phi`. Exact duplicate records remain ties: deduplicate with counts. |
| Searchable dissimilarity for NN-descent | `phi` is Euclidean (a true metric); sparse input, PCA or random projection all work. |
| `transform()` on new data | Supported. |

Caveats: discrete-only data has no true manifold (UMAP acts as a graph
method, as on any categorical data); the second-order form is unbounded for
rare categories (mitigated by a probability floor); the metric is only as
good as the fitted model.

## 4. Design

### 4.1 The per-distribution primitive

Each distribution returns its own **whitened score** rather than `(T, mu,
I)` separately (decision F1):

```python
def seq_fisher_score(self, x: EncodedDataSequence,
                     options: Optional[FisherScoreOptions] = None) -> np.ndarray:
    """Whitened per-observation score, shape (N, self.fisher_dim(options)).

    Contract: under this distribution, E[s] = 0 and Cov[s] = I on the
    score's support (a projector for over-complete encodings such as
    Categorical).
    """

def fisher_dim(self, options: Optional[FisherScoreOptions] = None) -> int: ...

def fisher_field_slices(self, options=None) -> Dict[str, slice]: ...  # optional, for attribution
```

`FisherScoreOptions` (frozen dataclass, passed down recursively):
`prob_floor`, `location_only` (MVN/DMVN), `sequence_scaling`
(`"conditional"` | `"marginal"`), `balance_fields` (opt-in, see F9),
`on_missing` (`"raise"` | `"posterior_only"`).

Closed-form kernels live as pure functions of parameters in the private
module `src/dmx/utils/_fisher_kernels.py` (for example
`gaussian_score(x, mu, sigma2) -> (N, 2)`), so `dmx.stats` and `dmx.bstats`,
which never import each other, share one implementation and one set of
tests. The kernels are not public API: callers use `seq_fisher_score`, and
each distribution's docstring states its formula. `dmx.utils.fisher` holds
only the public pieces (options, support check, score-array helper, and the
shared error).

### 4.2 Data flow

```
encoded data ──► MixtureDistribution.seq_posterior ──► Z (N, K)
            └──► component_k.seq_fisher_score ─────► S_k (N, p_k)   for each k
                                  │
                 MixtureDistribution.seq_fisher_embedding
                                  ▼
                 FisherEmbedding(matrix=phi (CSR float32), scale=c,
                                 component_slices, field_slices, top_m)
                                  ▼
                 humap(representation="fisher") ──► UMAP(metric="euclidean")
```

Every step is per record, so local, chunked, MPI and Spark paths compute the
same rows; only the final UMAP fit is centralised.

### 4.3 Coverage by phase

| Phase | Distributions | Score |
|---|---|---|
| B. Leaves (`dmx.stats`) | Gaussian, LogGaussian, Exponential, Gamma, Poisson, Geometric, Binomial, Categorical, IntRange, MVN, DiagonalGaussian, Dirichlet | Closed form |
| C. Observed combinators | Composite, Optional, Ignored, NullDist, Sequence, SetDist, IntMultinomial, CatMultinomial | Closed form, recursive |
| D. Latent-variable | nested Mixture, HeterogeneousMixture; HMM family (stretch) | Expected sufficient statistics + empirical whitening |
| E. `dmx.bstats` parity | Gaussian, Categorical, Poisson, Bernoulli, Exponential, Geometric, Gamma, Composite, Optional, Sequence, Ignored, NullDist | Shared kernels, plug-in parameters |
| Later / out of epic | VMF, MarkovChain, IntSetDist, edit set distributions, Spearman rho, LDA, PLSI, ICL tree, tree HMM, hidden association, Conditional, Weighted | Use `on_missing="posterior_only"` until implemented |

## 5. Decisions

| ID | Decision | Rationale |
|---|---|---|
| F1 | Whitened score is the per-distribution primitive | Each family knows a convenient orthonormal basis (Gaussian `(x, x^2)` is strongly correlated, so diagonal whitening of `T` is wrong); Euclidean distance is rotation invariant |
| F2 | `sqrt(z)` (Hellinger) block, not the Fisher-vector weight block `(z - pi)/sqrt(pi)` | Bounded, independent of `pi`, reduces exactly to current humap, and follows from the `Q_i` construction |
| F3 | Euclidean `phi` is the default; exact Hellinger is a validation reference only | Works with standard UMAP/pynndescent, PCA and sparse input; exact in the regime UMAP uses |
| F4 | Default `c = 1 / p_eff`; user-overridable `fisher_scale` | Balances within- vs between-component distances; ranking inside a component is independent of `c` |
| F5 | Top-`m` truncation (default 3), CSR float32 output | Memory: dense float64 `phi` at N = 1e6, K = 30, p = 50 is ~12 GB |
| F6 | `dmx.bstats` uses plug-in (posterior mean) parameters | The default `humap` path fits a bstats DPM; expected scores under the parameter posterior are a later refinement |
| F7 | Categorical: probability floor from `default_value` / `prob_floor`; dense `e_x/sqrt(p_x) - sqrt(p)` up to 256 levels, sparse offset-free `e_x/sqrt(p_x)` above (exact for same-component pairs) | Avoids `1/p` blow-ups for unseen or rare levels and dense blow-up for high cardinality |
| F8 | Sequences: `sequence_scaling="conditional"` divides the summed element score by `sqrt(n)`; length information comes from the length distribution's own score block | Whitened conditional on the observed length; consistent with `Sequence(len_normalized=...)` and htsne's `vlen` |
| F9 | Pure Fisher weighting by default; `balance_fields=True` (each field block divided by `sqrt(dim_field)`) is opt-in and benchmarked | Under the Fisher metric a field contributes in proportion to its parameter count; high-cardinality categoricals can dominate |
| F10 | Components without a score raise by default; `on_missing="posterior_only"` contributes only the `sqrt(z)` block for them, with a warning | Never silently mix geometries |
| F11 | `representation="posterior"` stays the `humap` default in 1.x; `"fisher"` is opt-in | Additive change policy (D3 in `DMX_EXPERT_AGENT_ROADMAP.md`) |
| F12 | Shared (keyed) parameters do not change the score definition; scores are computed per occurrence with the occurrence's parameters | Keeps the primitive local; documented in the API page |

## 6. Milestones

| Milestone | Issues | Exit criteria |
|---|---|---|
| M1. API and harness | 1, 2 | Base API, options, `dmx/utils/fisher.py`, design page; generic contract test runs (and skips unimplemented families) |
| M2. `dmx.stats` leaves | 3–6 | Every Phase B leaf passes the contract test |
| M3. Observed combinators | 7, 8 | Composite of mixed leaves passes the contract test; sequences whitened conditional on length |
| M4. Usable humap (MVP) | 11, 12, 14 | `humap(data, representation="fisher")` works end to end on the default bstats DPM path and on a supplied `dmx.stats` mixture |
| M5. Latent and scale-out | 9, 10, 13 | Nested mixtures score via empirical whitening; MPI path produces identical rows to the local path |
| M6. Evidence and docs | 15, 16 | Benchmark shows within-cluster structure recovered where posterior humap collapses it; docs, example, changelog |

Critical path to a usable feature: 1 → 2 → 3, 4, 5 → 7 → 11 → 12 → 14.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Second-order distances unbounded (rare categories, heavy tails) | `prob_floor`; exact per-field BC for Categorical available in the reference metric; benchmark clipping |
| High-cardinality fields dominate | F7 sparse form; F9 opt-in balancing; report per-field contribution via `fisher_field_slices` |
| Model misspecification yields misleading geometry | Document; validation issue compares against raw-feature baselines |
| Latent-variable components lack closed-form Fisher information | Empirical whitening fallback (issue 9) |
| Memory of `phi` for large N | CSR, float32, top-`m`; optional PCA / random projection before UMAP |
| bstats and stats implementations drift | Shared private kernels in `dmx/utils/_fisher_kernels.py`; one contract test parametrised over both packages |

## 8. Out of scope for this epic

- Changing the default `humap` representation (revisit for 2.0).
- Scaling `htsne` itself (edge sampling from the rank-K `P`); tracked
  separately if wanted.
- Expected (posterior-integrated) scores for bstats.
- Families listed under "Later" in section 4.3.
