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
