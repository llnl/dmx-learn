# Changelog

All notable changes to dmx-learn are recorded here.

## Unreleased (1.2)

### Changed

- `optimize` and `iterate` (`dmx.utils.estimation`) and the MPI
  `optimize_mpi` helpers no longer share one import-time
  `RandomState()` default across calls; `rng` now defaults to `None`, and each
  unseeded call gets its own reported seed. Calls that pass `rng` or `seed` are
  unchanged.
- `torch_utils.estimation.optimize` and `optimize_mp`, and
  `dmx.utils.vector.row_choice`, no longer draw from NumPy's global random
  state when no seed or `rng` is given.
- **Packaging: pyspark is no longer a core dependency.** A plain
  `pip install dmx-learn` no longer pulls in pyspark (or its Java runtime
  requirement), and `import dmx`, `dmx.stats`, `dmx.bstats`, and
  `dmx.utils.estimation` work without it. Spark support moved to new extras:
  - `spark`: pyspark
  - `distributed`: pyspark and mpi4py
  - `all`: now also includes pyspark

  If you use Spark RDDs, `dmx.stats.rdd_sampler`, or the Spark examples,
  install the extra with `pip install dmx-learn[spark]` (or
  `poetry install -E spark`). Importing `dmx.stats.rdd_sampler` without
  pyspark raises an `ImportError` with this install hint. The estimation
  helpers still accept RDDs unchanged when pyspark is installed.

### Added

- **Fisher-geometry `humap`.** `humap(..., representation="fisher")` embeds
  each record by its posteriors plus its whitened Fisher score under its most
  probable components, so records are spread out by where they sit inside a
  cluster, not only by which cluster they belong to. The default stays
  `representation="posterior"`, and existing calls are unchanged.
  - New keyword-only options on `humap` and `humap_mpi`: `fisher_scale`
    (default `1 / p_eff`), `top_m` (default 3), `fisher_options`, and
    `return_embedding`. In Fisher mode the default UMAP metric is
    `"euclidean"`, fields are balanced by their dimension
    (`balance_fields=True`), and the fourth return value is the sparse feature
    matrix.
  - `humap_transform(fit, mix_model, new_data, embedding=None)` places new
    records in a fitted embedding.
  - `dmx.utils.fisher`: `FisherScoreOptions`, `has_fisher_score`,
    `FisherEmbedding`, `fisher_embedding` (chunked, with optional
    posterior-weighted whitening and `like=` for new records), and
    `EmpiricalFisherWhitener`.
  - Distributions gain `seq_fisher_score`, `fisher_dim`, and
    `fisher_field_slices`. They are implemented for the `dmx.stats` leaves,
    combinators, variable-length families, nested mixtures and HMMs, and for
    the `dmx.bstats` families used by the automatic DPM. Mixtures gain
    `seq_fisher_embedding`.
  - Distributed computation: `dmx.mpi4py.utils.fisher.fisher_embedding_mpi`
    and `dmx.utils.spark.fisher_embedding_rdd` reproduce the local rows bit
    for bit (without `whiten`).
  - Docs: user guide in `docs/utils/fisher_embedding.rst`, example
    `examples/utils_examples/humap_fisher_example.py`, and the benchmark
    `_benchmarks/fisher-humap.md`.
- New optional extras for upcoming features:
  - `viz`: matplotlib, for plotting helpers
  - `agent`: the official MCP Python SDK (`mcp`) plus everything in `viz`,
    for the `dmx-mcp` expert agent server
  - `all`: now also includes matplotlib and mcp

  The core installation still has no matplotlib or mcp dependency.
- `dmx.utils.optional`, a shared helper for optional dependencies.
  `require_optional()` (with `require_matplotlib()` and `require_mcp()`
  wrappers) raises an `ImportError` naming the extra to install, and
  `is_available()` checks for a package without importing it. The Spark and
  PyTorch guards now use it, so their messages share one format; the PyTorch
  message previously suggested `poetry install --with torch`, which is not a
  valid command for an extra.
- `dmx.utils.spark` with `is_rdd()`, which detects RDDs without importing
  pyspark, and `require_pyspark()`, which raises an actionable install hint.
- `num_params()` on every `dmx.stats` distribution. It returns the number of
  free parameters the estimator fits, for use in BIC, AIC, and ICL. A
  probability vector over `K` values counts `K - 1`; support bounds,
  dimensions, and other fixed structure count 0. Combinators (composite, the
  mixture family, optional, sequence, conditional, the HMM variants, the ICL
  tree, and others) add up their children. Parameters tied through estimator
  `keys` are counted once. `IgnoredDistribution` and `NullDistribution` return
  0. Custom distributions implement the `_num_params(seen_keys)` hook; the base
  class raises `NotImplementedError` until they do. Existing behavior is
  unchanged.
- `num_params()` on every `dmx.bstats` distribution, using the same rules and
  `_num_params(seen_keys)` hook as `dmx.stats`. A likelihood counts the
  parameters its estimator fits; its attached prior is fixed metadata and is
  not counted. Distributions used as priors (Beta, Gamma, Dirichlet,
  NormalGamma, and others) count their own hyperparameters. Keys on Bernoulli,
  geometric, Poisson, optional, and composite distributions are counted once.
- `DirichletProcessMixtureDistribution.num_params()` counts the full
  truncation (`K - 1` weights plus all `K` components; the concentration is not
  counted). New `effective_num_components(min_weight)` and
  `effective_num_params(min_weight)` count only the components with
  `w >= min_weight`. With `min_weight = mix_threshold_count / len(data)` they
  match the components `get_dpm_mixture` keeps and the `num_params()` of the
  mixture it returns. `get_dpm_mixture` itself is unchanged.
- `dmx.utils.scores`, model scoring helpers for `dmx.stats` and `dmx.bstats`
  models. `heldout_log_likelihood(model, data)` scores held-out data;
  `kfold_log_likelihood(data, estimator, k, seed)` refits on each training
  fold with `optimize` and scores the held-out fold, reproducibly from `seed`;
  `bic`, `aic`, and `icl` return information criteria on the deviance scale
  (lower is better: `-2 * LL + p * ln(n)`, `-2 * LL + 2 * p`, and BIC plus twice
  the posterior assignment entropy from `seq_posterior`). Results are frozen
  dataclasses and nothing is printed. The criteria take an optional
  `num_params` override, for example a DPM's `effective_num_params()`.
- Quiet fitting and embedding. `optimize`, `best_of`, `iterate`, and
  `hill_climb` (in `dmx.utils.estimation` and `dmx.bstats.bestimation`), the
  MPI `optimize_mpi` and `best_of_mpi`, `get_dpm_mixture`,
  `get_dpm_mixture_mpi`, `prepare_mixture_model`, `htsne`, `dpmsne`, `humap`,
  and `humap_mpi` take `verbose: bool = True` and an optional
  `progress_callback`. The default output is unchanged; `verbose=False` writes
  nothing to standard output or error. Every progress message is also logged
  at `DEBUG` on the module's logger (silent unless logging is configured), and
  the callback receives a `dmx.utils.progress.ProgressEvent` for every
  iteration (independent of `print_iter`), with `to_dict()` for JSON. Existing
  `out=` arguments work as before.
- Seed discipline on every stochastic entry point, with an audit table on the
  new "Seeds and reproducibility" docs page. Fitting helpers (`optimize`,
  `best_of`, `iterate` in `dmx.utils.estimation`, `dmx.bstats.bestimation`, and
  `dmx.mpi4py.utils`, plus `get_dpm_mixture` and `prepare_mixture_model`) gain a
  `seed` argument next to `rng`; passing both raises `ValueError`. With
  neither, a fresh seed is drawn from OS entropy instead of NumPy's global
  state, and the seed used is reported as a `"start"` progress event and a
  DEBUG log record (`htsne`, `dpmsne`, and `humap` report their `seed` the same
  way; MPI helpers broadcast a drawn seed from rank 0). Samplers record their
  seed as `sampler.seed`, so `dist.sampler(seed=sampler.seed)` reproduces an
  unseeded sampler. New `dmx.utils.seeding` helpers: `new_seed`,
  `resolve_seed`, `resolve_sampler_seed`, and `resolve_rng`.

### Fixed

- `humap` no longer shares the module-level `DEFAULT_UMAP` dictionary with
  UMAP when `umap_kwargs` is None, so per-call changes do not leak into later
  calls. The unused module-level `umap_model` was removed.

- The pylint quality gate passes again. A late import in `dmx.bstats` (added
  when pyspark moved to the `spark` extra) failed `pylint src/dmx/bstats`, and
  `pylint examples` linted only `examples/__init__.py` because the example
  folders were not packages. Each example folder now has an `__init__.py`, so
  all example files are linted; they still run as scripts.
