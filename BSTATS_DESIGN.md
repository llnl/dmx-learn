# BSTATS GitHub Issue Plan

Use this file as the child issue source for
`scripts/create_github_issue_plan.py`. The script creates the top-level parent
issue first, creates each `## Issue N` block as a child issue, then updates the
parent issue with an ordered checklist linking all created children.

Parent issue title:

`Make dmx.bstats a first-class Bayesian modeling package`

Parent issue summary:

Clean `src/dmx/bstats` as the compatibility-preserving Bayesian counterpart to
`src/dmx/stats`. Preserve current public imports and helper signatures while
adding typed interfaces, readable docstrings, deterministic tests, mypy,
pylint, pydocstyle, and workflow coverage.

Recommended labels:

- `bstats`
- `quality`
- `mypy`
- `pylint`
- `tests`
- `docs`
- `bayesian`

Recommended command:

```bash
python scripts/create_github_issue_plan.py \
  --repo OWNER/REPO \
  --issue-file BSTATS_DESIGN.md \
  --parent-title "Make dmx.bstats a first-class Bayesian modeling package" \
  --parent-summary "Clean src/dmx/bstats as the compatibility-preserving Bayesian counterpart to src/dmx/stats. Preserve current public imports and helper signatures while adding typed interfaces, readable docstrings, deterministic tests, mypy, pylint, pydocstyle, and workflow coverage." \
  --labels "bstats,quality,mypy,pylint,tests,docs,bayesian"
```

Recommended order:

1. Issue 1
2. Issue 2
3. Issue 3
4. Issue 4
5. Issue 5
6. Issue 6
7. Issue 7
8. Issue 8
9. Issue 9
10. Issue 10
11. Issue 11
12. Issue 12
13. Issue 13
14. Issue 14
15. Issue 15
16. Issue 16
17. Issue 17

Design invariants for all child issues:

- Preserve existing `dmx.bstats` public imports unless a child issue explicitly
  documents a compatibility shim or deprecation.
- Prefer typed, documented compatibility refactors over behavior changes.
- Keep one MR scoped to one child issue.
- Add or update `tests/bstats` coverage for every behavior-bearing code change.
- Do not add `src/dmx/bstats` to global CI quality gates until the relevant
  child issue makes that gate pass locally.

---

## Issue 1

**Title:** Audit `dmx.bstats` public API and current quality failures

**Body:**

Create a focused baseline audit for `src/dmx/bstats` before refactoring code.
This issue should make the cleanup measurable and identify compatibility
constraints that later MRs must preserve.

Scope:

- inventory exported names from `dmx.bstats.__init__`
- inventory downstream imports from `src/dmx/utils`, `src/dmx/mpi4py`, examples,
  and tests
- identify legacy spellings and compatibility-sensitive methods, including
  `accumulatorFactory` versus `accumulator_factory`
- record current targeted failures for mypy, pylint, and pydocstyle on
  `src/dmx/bstats`
- identify generated files such as `__pycache__` artifacts that should not be
  part of source cleanup

Acceptance criteria:

- a short audit artifact exists in the repo and names the public API that must
  remain compatible
- the audit records the current quality-gate failure shape without trying to
  fix all failures
- downstream `bstats` integration points are listed explicitly
- later child issues can cite this audit for scope and compatibility decisions

Out of scope:

- broad code refactors
- behavior changes
- adding `bstats` to CI gates

---

## Issue 2

**Title:** Stabilize typed core interfaces in `src/dmx/bstats/pdist.py`

**Body:**

Make `pdist.py` the documented and typed interface layer for Bayesian
distributions, samplers, accumulators, estimator factories, and sequence
encoders. This is the first code cleanup MR because most other modules depend
on these base contracts.

Scope:

- add clear module, class, and method docstrings using the repo's Google-style
  convention
- type the base distribution, sampler, accumulator, accumulator factory,
  estimator, data encoder, and encoded data APIs
- add Bayesian hooks used across `bstats`, including prior access, expected
  log-density, entropy, and cross-entropy where appropriate
- replace silent `None` stubs with `NotImplementedError` or safe default
  implementations
- add compatibility aliases only where current repo code depends on them

Acceptance criteria:

- `poetry run mypy --explicit-package-bases src/dmx/bstats/pdist.py` passes
- `poetry run pylint src/dmx/bstats/pdist.py --jobs=1 --fail-under=10` passes
- `poetry run pydocstyle src/dmx/bstats/pdist.py` passes
- downstream modules can still import the same base names from
  `dmx.bstats.pdist`
- no child distribution is intentionally broken by abstract enforcement

Out of scope:

- cleaning every concrete distribution
- redesigning public `bstats` class names

---

## Issue 3

**Title:** Normalize `dmx.bstats` package exports and top-level fitting helpers

**Body:**

Clean `src/dmx/bstats/__init__.py` so package exports and public fitting helpers
are readable, typed, and compatible. This should preserve the existing
top-level import surface while removing avoidable type and lint failures.

Scope:

- keep the existing `__all__` names available unless Issue 1 identifies a safe
  deprecation path
- replace wildcard imports with explicit imports
- type and document `load_models`, `dump_models`, `estimate`, `seq_estimate`,
  `initialize`, `seq_encode`, `seq_log_density`, and related public helpers
- standardize internal calls on `accumulator_factory` while preserving required
  compatibility aliases
- keep PySpark RDD, pandas DataFrame, and local sequence behavior compatible

Acceptance criteria:

- existing imports such as `from dmx.bstats import MixtureDistribution` still
  work
- targeted mypy and pylint pass for `src/dmx/bstats/__init__.py`
- helper docstrings explain supported data inputs and return values
- existing MPI and utility modules that import from `dmx.bstats` still import
  successfully

Out of scope:

- replacing `eval` serialization with a new format
- changing fitting algorithms

---

## Issue 4

**Title:** Add a shared `tests/bstats` distribution harness

**Body:**

Create a reusable pytest harness for Bayesian distributions, adapted from
`tests/stats/stats_tests.py` but aware of `bstats` prior and variational
interfaces.

Scope:

- create `tests/bstats`
- add shared helpers for string round-tripping where supported
- add shared helpers for sampler repeatability
- add scalar versus sequence log-density checks
- add estimator, accumulator factory, accumulator value, and encoder checks
- add Bayesian checks for `get_prior`, `set_prior`, `expected_log_density`,
  `seq_expected_log_density`, entropy, and cross-entropy when implemented

Acceptance criteria:

- the harness can test one simple `bstats` distribution without bespoke test
  scaffolding
- tests are deterministic under fixed seeds
- unsupported optional Bayesian methods can be skipped explicitly instead of
  silently ignored
- `poetry run pytest tests/bstats -v --tb=short` runs successfully for the
  initial covered modules

Out of scope:

- full coverage for every `bstats` module
- convergence-heavy DPM tests

---

## Issue 5

**Title:** Clean null and ignored `bstats` distributions

**Body:**

Clean the no-op and intentionally ignored distribution implementations first
because they are used as defaults and priors throughout the Bayesian package.

Scope:

- clean `src/dmx/bstats/nulldist.py`
- clean `src/dmx/bstats/ignored.py`
- type constructors, samplers, accumulators, estimators, and encoders
- document the semantics of ignored observations and null priors
- add harness coverage in `tests/bstats`

Acceptance criteria:

- targeted mypy and pylint pass for `nulldist.py` and `ignored.py`
- public names exported from `dmx.bstats` remain compatible
- tests cover scalar scoring, sequence scoring, sampler behavior, estimator
  wiring, and accumulator value round-trips where meaningful
- null and ignored distributions can still be used as priors/defaults by other
  modules

Out of scope:

- changing missing-value policy
- changing automatic estimator routing

---

## Issue 6

**Title:** Clean scalar and continuous conjugate prior distributions

**Body:**

Clean the scalar and continuous prior distributions that support Bayesian
updates for primitive likelihoods.

Scope:

- clean `src/dmx/bstats/beta.py`
- clean `src/dmx/bstats/gamma.py`
- clean `src/dmx/bstats/normgamma.py`
- clean `src/dmx/bstats/mvngamma.py`
- document parameterization, support, prior/posterior semantics, entropy, and
  cross-entropy behavior
- add deterministic tests for density, expected log-density inputs where
  relevant, sampling, string round-tripping, and parameter access

Acceptance criteria:

- targeted mypy and pylint pass for the four modules
- pydocstyle passes for public classes and methods touched in these modules
- tests cover finite values for valid inputs and explicit behavior for invalid
  support where current code defines it
- downstream likelihood modules can still instantiate their default priors

Out of scope:

- changing mathematical parameterization
- optimizing numerical routines

---

## Issue 7

**Title:** Clean categorical and vector prior distributions

**Body:**

Clean the prior distributions used by categorical, integer categorical, and
composite Bayesian estimators.

Scope:

- clean `src/dmx/bstats/dirichlet.py`
- clean `src/dmx/bstats/symdirichlet.py`
- clean `src/dmx/bstats/catdirichlet.py`
- document dense versus dictionary parameter behavior
- type sampler and estimator-related methods
- add tests for finite log densities, sampling shape, parameter access,
  expected log-density where present, and entropy/cross-entropy behavior

Acceptance criteria:

- targeted mypy and pylint pass for the three modules
- tests cover both dense and dictionary categorical-prior paths
- current default prior construction in categorical modules remains compatible
- public imports from `dmx.bstats` remain unchanged

Out of scope:

- replacing dictionary categorical priors with a new representation
- changing smoothing defaults

---

## Issue 8

**Title:** Clean discrete primitive Bayesian likelihood distributions

**Body:**

Clean the discrete primitive likelihood distributions after their prior
dependencies are stable.

Scope:

- clean `src/dmx/bstats/bernoulli.py`
- clean `src/dmx/bstats/categorical.py`
- clean `src/dmx/bstats/intrange.py`
- clean `src/dmx/bstats/poisson.py`
- clean `src/dmx/bstats/geometric.py`
- document observation support, prior defaults, sufficient statistics, and
  expected log-density behavior
- add tests for scalar and sequence log-density agreement, sampler
  repeatability, estimator wiring, accumulator value round-trips, and
  prior/posterior updates

Acceptance criteria:

- targeted mypy and pylint pass for the five modules
- tests cover representative valid observations for each distribution
- invalid-support behavior is explicit and covered where current code defines
  it
- existing automatic estimator routing for categorical, integer categorical,
  count, and binary data remains compatible

Out of scope:

- changing automatic type inference thresholds
- changing distribution names or constructor signatures

---

## Issue 9

**Title:** Clean continuous primitive Bayesian likelihood distributions

**Body:**

Clean the continuous primitive likelihood distributions after scalar and
continuous prior modules are stable.

Scope:

- clean `src/dmx/bstats/exponential.py`
- clean `src/dmx/bstats/gaussian.py`
- clean `src/dmx/bstats/dmvn.py`
- clean `src/dmx/bstats/dirac.py`
- document parameterization, support, default priors, sufficient statistics,
  and expected log-density behavior
- add tests for scalar and sequence log-density agreement, sampler
  repeatability, estimator wiring, accumulator value round-trips, and
  prior/posterior updates

Acceptance criteria:

- targeted mypy and pylint pass for the four modules
- tests cover finite values for representative valid observations
- Gaussian and diagonal Gaussian estimators remain compatible with
  `dmx.utils.automatic`
- no constructor or import path used by current examples is broken

Out of scope:

- replacing Gaussian prior families
- changing numerical estimation algorithms except to fix correctness bugs found
  by tests

---

## Issue 10

**Title:** Clean `bstats` composite product distributions

**Body:**

Clean `CompositeDistribution` as the central product distribution for
heterogeneous Bayesian observations. This is a prerequisite for reliable
automatic modeling and Dirichlet-process mixture work.

Scope:

- clean `src/dmx/bstats/composite.py`
- type component distribution, sampler, accumulator, factory, estimator, and
  encoder interactions
- document tuple observation semantics, component ordering, names, keys, priors,
  and expected log-density behavior
- preserve public constructor and export behavior
- add tests for product log-density, sequence encoding, estimator wiring, key
  merge/replace behavior, and prior propagation

Acceptance criteria:

- targeted mypy and pylint pass for `composite.py`
- tests cover composite distributions over at least one discrete and one
  continuous child
- `CompositeDistribution` remains usable as the base distribution inside
  mixture and DPM workflows
- key-sharing behavior is documented and covered by tests

Out of scope:

- redesigning the composite observation representation
- changing component order semantics

---

## Issue 11

**Title:** Clean optional, sequence, and set `bstats` distributions

**Body:**

Clean structural distributions that wrap or repeat child distributions while
preserving their current observation semantics.

Scope:

- clean `src/dmx/bstats/optional.py`
- clean `src/dmx/bstats/sequence.py`
- clean `src/dmx/bstats/setdist.py`
- document missing-value behavior, variable-length sequence behavior, set
  observation behavior, priors, and sufficient statistics
- add tests for scalar and sequence scoring, encoding, estimator wiring,
  accumulator value round-trips, and child prior propagation

Acceptance criteria:

- targeted mypy and pylint pass for the three modules
- tests cover missing and present optional observations
- tests cover variable-length sequences
- tests cover set-like observations without assuming order unless current
  behavior requires it
- existing automatic estimator routing for optional and sequence-like data
  remains compatible

Out of scope:

- changing missing-value inference policy
- replacing sequence or set observation formats

---

## Issue 12

**Title:** Clean conditional `bstats` distributions

**Body:**

Clean conditional Bayesian distributions and encoders as a focused MR because
they have more complex keyed data flow than the other structural wrappers.

Scope:

- clean `src/dmx/bstats/conditional.py`
- type condition keys, default distributions, encoded conditional data, and
  accumulators
- document default-distribution behavior when a condition is absent
- add tests for known conditions, missing conditions with defaults, sequence
  encoding, sampler behavior, and estimator wiring

Acceptance criteria:

- targeted mypy and pylint pass for `conditional.py`
- tests cover both explicit condition maps and default fallback behavior
- encoded conditional data shape is documented and stable
- no downstream import path is changed

Out of scope:

- adding new conditional modeling features
- changing how condition keys are represented

---

## Issue 13

**Title:** Clean finite Bayesian mixture distributions

**Body:**

Clean finite Bayesian mixture distributions after primitive and structural
component distributions are stable.

Scope:

- clean `src/dmx/bstats/mixture.py`
- type mixture weights, components, sampler, accumulator, factory, estimator,
  encoder, and priors
- document normalized weight expectations, component ordering, posterior
  responsibility calculations, and prior behavior
- add tests for log-sum-exp scoring, sequence scoring, sampler repeatability,
  posterior responsibility shape, estimator wiring, and component prior
  propagation

Acceptance criteria:

- targeted mypy and pylint pass for `mixture.py`
- tests cover mixtures over at least one primitive and one composite component
- mixture weights remain normalized after estimation paths covered by tests
- existing imports and helper usage in `humap`, `htsne`, and MPI utilities
  remain compatible

Out of scope:

- redesigning mixture APIs
- changing initialization strategy for DPM mixtures

---

## Issue 14

**Title:** Clean Dirichlet-process mixture and variational estimation workflows

**Body:**

Clean the DPM implementation and local variational estimation helpers after
finite mixtures and composite distributions are stable. This is the most
important behavior-bearing Bayesian workflow and should remain compatibility
preserving.

Scope:

- clean `src/dmx/bstats/dpm.py`
- clean `src/dmx/bstats/bestimation.py`
- document truncation behavior, stick-breaking weights, component priors,
  variational parameters, expected log-density, initialization, and optimizer
  convergence outputs
- type local optimize, initialize, iterate, likelihood, and empirical KL helper
  functions
- add tests for initialization, one-step update behavior, normalized
  nonnegative weights, component sorting behavior, finite output values, and
  conversion through `get_dpm_mixture`

Acceptance criteria:

- targeted mypy and pylint pass for `dpm.py` and `bestimation.py`
- DPM tests are deterministic and small enough for normal CI
- DPM output remains compatible with `dmx.bstats.MixtureDistribution`
- existing local and MPI automatic DPM helper imports still work

Out of scope:

- changing the public DPM constructor signature
- introducing a new inference algorithm
- making MPI behavior changes beyond preserving compatibility

---

## Issue 15

**Title:** Update automatic, embedding, and MPI integrations for cleaned `bstats`

**Body:**

Update repo integration points only after the cleaned `bstats` APIs are stable.
The goal is to make `bstats` first-class for existing automatic and mixture
workflows without changing user-facing helper signatures.

Scope:

- update `src/dmx/utils/automatic.py` where it routes through `dmx.bstats`
- update `src/dmx/utils/humap.py` and `src/dmx/utils/htsne.py` imports and type
  expectations
- update `src/dmx/mpi4py/utils/automatic.py`
- update `src/dmx/mpi4py/utils/bestimation.py`
- update `src/dmx/mpi4py/bstats/__init__.py` if interface cleanup requires it
- add or update tests for local `get_estimator`, local `get_dpm_mixture`, and
  existing MPI bstats smoke paths where practical

Acceptance criteria:

- local automatic helpers still return `bstats` estimators when
  `use_bstats=True`
- `get_dpm_mixture` and `get_dpm_mixture_mpi` keep their public signatures
- targeted mypy and pylint pass for touched integration files
- existing `tests/mpi4py` bstats tests still pass when MPI test dependencies
  are available

Out of scope:

- expanding automatic model-selection behavior
- changing embedding algorithms

---

## Issue 16

**Title:** Add `bstats` to CI quality gates after local checks pass

**Body:**

Add `bstats` to repository quality workflows only after the package passes the
same local gates expected by CI. This issue makes the cleanup enforceable.

Scope:

- add `src/dmx/bstats` to the mypy workflow command
- add `src/dmx/bstats` to the pylint workflow command
- add `tests/bstats` to the pytest workflow command
- add `src/dmx/bstats` to pydocstyle coverage once public docstrings pass
- update the `dmx-test-and-quality` skill to include the new `bstats` checks

Acceptance criteria:

- `poetry run black --check .` passes
- `poetry run isort --check .` passes
- `poetry run mypy --explicit-package-bases src/dmx/bstats tests/bstats` passes
- `poetry run pylint src/dmx/bstats tests/bstats --jobs=1 --fail-under=10`
  passes
- `poetry run pydocstyle src/dmx/bstats` passes or the issue documents a
  narrower staged pydocstyle target with all touched public modules passing
- `TEST_TORCH_DEVICE=cpu poetry run pytest tests/bstats tests/stats tests/torch_stats tests/utils -v --tb=short`
  passes

Out of scope:

- adding new behavior after quality gates are enabled
- weakening existing repo quality settings

---

## Issue 17

**Title:** Update modeling skills to treat `bstats` as first-class where appropriate

**Body:**

Update repo-local modeling skills after code and tests establish `bstats` as a
quality-gated package surface. The skills should route to `bstats` for Bayesian,
variational, DPM, and automatic mixture workflows while keeping `dmx.stats` as
the default non-Bayesian path.

Scope:

- update `.codex/skills/dmx-expert-orchestrator/SKILL.md`
- update `.codex/skills/dmx-local-modeling/SKILL.md`
- update relevant references under `.codex/skills/dmx-local-modeling/references`
- document when to choose `dmx.bstats` instead of `dmx.stats`
- document `get_estimator(..., use_bstats=True)` and `get_dpm_mixture` as valid
  first-class Bayesian/automatic routes
- document DPM over composite distributions as the main high-value `bstats`
  modeling path

Acceptance criteria:

- skills no longer describe `bstats` as out of scope or second-class for
  Bayesian/DPM use cases
- skills still default to explicit `dmx.stats` construction for ordinary
  non-Bayesian local modeling
- skill guidance points to tests or examples that exercise cleaned `bstats`
  workflows
- guidance clearly separates local `bstats` workflows from MPI-specific
  workflows

Out of scope:

- changing code behavior
- adding new notebooks
