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

- `dmx.utils.spark` with `is_rdd()`, which detects RDDs without importing
  pyspark, and `require_pyspark()`, which raises an actionable install hint.
