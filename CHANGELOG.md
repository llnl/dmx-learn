# Changelog

All notable changes to dmx-learn are recorded here.

## Unreleased (1.2)

### Changed

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

### Fixed

- The pylint quality gate passes again. A late import in `dmx.bstats` (added
  when pyspark moved to the `spark` extra) failed `pylint src/dmx/bstats`, and
  `pylint examples` linted only `examples/__init__.py` because the example
  folders were not packages. Each example folder now has an `__init__.py`, so
  all example files are linted; they still run as scripts.
