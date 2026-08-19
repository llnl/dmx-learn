# `dmx.bstats` baseline audit

Baseline: 2026-08-19 at commit `908b3e8`. This is an inventory for later
cleanup MRs, not a request to add `bstats` to CI or to make the diagnostics
pass.

## Public package API

`src/dmx/bstats/__init__.py` declares the following 65 names in `__all__`.
Later work must keep these imports available, or provide an intentional
compatibility alias and migration plan before removing one.

- Distributions: `BernoulliDistribution`, `BernoulliSetDistribution`,
  `BetaDistribution`, `CategoricalDistribution`, `CompositeDistribution`,
  `DiagonalGaussianDistribution`, `DictDirichletDistribution`,
  `DirichletDistribution`, `DirichletProcessMixtureDistribution`,
  `ExponentialDistribution`, `GaussianDistribution`, `GammaDistribution`,
  `GeometricDistribution`, `IgnoredDistribution`,
  `IntegerCategoricalDistribution`, `MixtureDistribution`,
  `MultivariateNormalGammaDistribution`, `NullDistribution`,
  `OptionalDistribution`, `PoissonDistribution`, and `SequenceDistribution`.
- Estimators: `BernoulliEstimator`, `BernoulliSetEstimator`,
  `CategoricalEstimator`, `CompositeEstimator`, `DiagonalGaussianEstimator`,
  `DirichletEstimator`, `DirichletProcessMixtureEstimator`,
  `ExponentialEstimator`, `GaussianEstimator`, `GammaEstimator`,
  `GeometricEstimator`, `IgnoredEstimator`, `IntegerCategoricalEstimator`,
  `MixtureEstimator`, `NullEstimator`, `OptionalEstimator`, `PoissonEstimator`,
  and `SequenceEstimator`.
- Samplers: `BernoulliSampler`, `BernoulliSetSampler`, `BetaSampler`,
  `CategoricalSampler`, `CompositeSampler`, `DiagonalGaussianSampler`,
  `DirichletSampler`, `DirichletProcessMixtureSampler`, `ExponentialSampler`,
  `GaussianSampler`, `GammaSampler`, `GeometricSampler`, `IgnoredSampler`,
  `IntegerCategoricalSampler`, `MixtureSampler`,
  `MultivariateNormalGammaSampler`, `NullSampler`, `OptionalSampler`,
  `PoissonSampler`, and `SequenceSampler`.
- Functions: `estimate`, `seq_estimate`, `initialize`,
  `seq_log_density_sum`, `seq_encode`, and `seq_log_density`.

The module also deliberately binds non-private names that are absent from
`__all__`: `DataSequenceEncoder`, `EncodedDataSequence`, `ParameterEstimator`,
`ProbabilityDistribution`, `load_models`, and `dump_models`. In particular,
`from dmx.bstats import ParameterEstimator` is already used downstream, so it
is part of the minimum compatibility surface despite the `__all__` omission.
The other five names must be reviewed before removal; imported dependencies
and names leaked by `from dmx.arithmetic import *` are not treated as a
supported API by this audit.

## Downstream integration points

The inventory searched Python files below `src/dmx/utils`, `src/dmx/mpi4py`,
`examples`, and `tests` for static and string-based `dmx.bstats` references.

| Area | File | Dependency |
| --- | --- | --- |
| utils | `src/dmx/utils/automatic.py` | `mixture.MixtureDistribution`; dynamically loads `optional.OptionalEstimator`, `sequence.SequenceEstimator`, `ignored.IgnoredEstimator`, `composite.CompositeEstimator`, `categorical.CategoricalEstimator`, `poisson.PoissonEstimator`, `gaussian.GaussianEstimator`, `bestimation.optimize`, and `dpm.DirichletProcessMixtureEstimator` |
| utils | `src/dmx/utils/htsne.py` | package-level `MixtureDistribution`; calls the mixture integration in `prepare_mixture_model` |
| utils | `src/dmx/utils/humap.py` | package-level `MixtureDistribution`; relies on `seq_encode`, `seq_posterior`, `num_components`, `components`, and `w` through the fitted model |
| MPI | `src/dmx/mpi4py/bstats/__init__.py` | `pdist.ParameterEstimator` and `pdist.ProbabilityDistribution`; estimator/accumulator sequence protocol |
| MPI | `src/dmx/mpi4py/utils/automatic.py` | package-level `ParameterEstimator`, `mixture.MixtureDistribution`, and dynamically loaded `dpm.DirichletProcessMixtureEstimator` |
| MPI | `src/dmx/mpi4py/utils/humap.py` | package-level `MixtureDistribution` and `pdist.ParameterEstimator`; direct bstats model encoding/posterior protocol |
| MPI | `src/dmx/mpi4py/utils/bestimation.py` | `pdist.ParameterEstimator`, `pdist.ProbabilityDistribution`, and the MPI bstats sequence helpers |
| tests | `tests/data/generate_data.py` | package-level `CategoricalDistribution`, `CompositeDistribution`, `GaussianDistribution`, and `MixtureDistribution` |
| tests | `tests/mpi4py/test_humap.py` | package-level `MixtureDistribution` and its runtime type identity |
| tests | `tests/mpi4py/test_bestimation.py` | package-level `seq_encode`, `bestimation.empirical_kl_divergence`, and the pickled `tests/data/testInput_mpi_b_optimize.pkl` model |
| examples | `examples/` | No `dmx.bstats` imports or string-based module references found. |

`tests/data/testInput_bstats_estimator.pkl` is also a tracked bstats pickle,
although no current Python reference to it was found. Class module paths and
state names can therefore be serialization-sensitive even when they do not
appear in an import search.

## Compatibility constraints for cleanup

- Preserve both spellings `accumulator_factory` and `accumulatorFactory` until
  callers are migrated through an adapter. The base protocol and almost all
  estimators use `accumulator_factory`; `DirichletEstimator` implements only
  `accumulatorFactory`. Package-level Spark branches in `estimate` and
  `seq_estimate` still call the camel-case spelling, while local, newer Spark,
  and MPI paths call the snake-case spelling.
- Do not normalize `ParameterEstimator.estimate` arity without a compatibility
  plan. Most local/MPI paths call `estimate(suff_stat)`, legacy distributed
  branches call `estimate(nobs, suff_stat)`, and `DirichletEstimator` implements
  the latter form.
- Estimator and accumulator protocol methods are integration points:
  `accumulator_factory`/`accumulatorFactory`, `make`, `initialize`, `update`,
  `seq_initialize`, `seq_update`, `combine`, `value`, `from_value`,
  `key_merge`, `key_replace`, `acc_to_encoder`, and `estimate`.
- Distribution sequence behavior is integration-sensitive: `seq_encode`,
  `seq_log_density`, `seq_posterior`, `sampler`, and `estimator`, plus mixture
  attributes `num_components`, `components`, and `w`.
- Preserve importable legacy module spellings used by imports or dynamic
  loading: `bestimation`, `dpm`, `pdist`, `dmvn`, `intrange`, `nulldist`, and
  `catdirichlet`. Renaming files requires module aliases because dynamic imports
  and pickles store these paths.
- `dump_models`/`load_models` use constructor-like `str(...)` output and
  `eval(...)`; class names, constructor signatures, and string forms may be
  compatibility-sensitive independently of type/lint cleanup.

## Focused quality baseline

These commands mirror the repository tools but target only `src/dmx/bstats`.
All three exited with status 1. Tool versions were mypy 1.20.0, pylint 3.3.9
(astroid 3.3.11), pydocstyle 6.3.0, and Python 3.13.14.

```text
poetry run mypy --explicit-package-bases src/dmx/bstats
poetry run pylint src/dmx/bstats --jobs=1 --fail-under=10
poetry run pydocstyle src/dmx/bstats
```

| Gate | Observed failure shape |
| --- | --- |
| mypy | 907 errors in all 28 source files. Codes: `no-untyped-def` 624, `no-any-return` 58, `attr-defined` 57, `assignment` 31, `has-type` 28, `arg-type` 15, `call-arg` 12, `var-annotated` 12, `abstract` 10, `return-value` 9, `syntax` 8, `union-attr` 8, `return` 7, `override` 6, `valid-type` 5, `call-overload` 4, `name-defined` 4, `operator` 4, `empty-body` 2, `misc` 2, and `index` 1. |
| pylint | 493 messages across all 28 files; score 0.00/10. Dominant codes are `R1705` 71, `W0611` 58, `C0301` 47, `R0205` 38, `C0209` 30, `W0613` 26, `W0231` 25, and `W0612` 23. Error/fatal shape includes `E0611` 18, `E0602` 4, `E1120` 4, `E0606` 1, `E1121` 1, and an `F0002` astroid crash while checking `categorical.py`. |
| pydocstyle | 918 violations in 27 files (all except `__init__.py`): `D102` 622, `D101` 121, `D107` 110, `D105` 45, `D103` 15, `D415` 2, and one each of `D205`, `D212`, and `D403`. |

This baseline records counts and categories only. Later issues can reduce them
incrementally; matching zero is not a condition of this audit.

## Generated files

`src/dmx/bstats/__pycache__/` currently contains 52 ignored `.pyc` files: 26
for CPython 3.11 and 26 for CPython 3.13. Neither the directory nor any bytecode
file is tracked, and `.gitignore` already ignores `__pycache__/`. These are
runtime artifacts, not cleanup targets or source changes, and must not be added
to a later MR.
