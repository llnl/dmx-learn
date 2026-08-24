---
name: dmx-test-and-quality
description: Use when adding or updating tests, or when validating dmx-learn changes against the repo's CI quality gates. Covers pytest layout, the shared stats test harness, torch test constraints, and the exact Black, isort, mypy, pylint, and pydocstyle checks used in GitHub Actions.
---

# dmx Test And Quality

Use this skill for changes under `tests/` and for final verification of Python changes.

## Test Layout

- Follow the repo's pytest discovery rules:
  - files: `*_test.py` or `test_*.py`
  - functions: `test_*`
  - classes: `Test*` or `*TestCase`
- Put new tests beside the code family they exercise:
  - `tests/stats`
  - `tests/bstats`
  - `tests/torch_stats`
  - `tests/utils`
  - `tests/mpi4py` for MPI-specific behavior

## Existing Test Patterns

- For `dmx.stats` distributions, start by reading `tests/stats/stats_tests.py`.
- Reuse the shared harness when possible instead of building bespoke assertions.
- Preserve the repo's standard distribution checks:
  - string round-tripping with `eval(str(...))` where that pattern already exists
  - estimator and encoder wiring
  - sampler repeatability for fixed seeds
  - agreement between scalar and sequence log-density paths
  - estimation or sequence-estimation improvement checks where relevant
- Add custom assertions only for behavior that the harness does not already cover.

## Torch-Specific Rules

- Default tests to CPU behavior unless the task is explicitly device-specific.
- Respect `TEST_TORCH_DEVICE`; CI sets it to `cpu`.
- Do not require CUDA or MPS for routine validation.
- Be careful with dtype expectations because MPS may force `float32` paths in repo code.

## Quality Gates From CI

Format and import order:

```bash
poetry run black --check .
poetry run isort --check .
```

Type checking:

```bash
poetry run mypy --explicit-package-bases \
  src/dmx/arithmetic.py \
  src/dmx/utils \
  src/dmx/stats \
  src/dmx/bstats \
  src/dmx/torch_utils \
  src/dmx/torch_stats \
  src/dmx/mpi4py \
  examples \
  examples_mpi4py \
  examples_spark \
  examples_torch \
  tests
```

Linting:

```bash
poetry run pylint src/dmx/stats --jobs=1 --fail-under=10
poetry run pylint src/dmx/bstats --jobs=1 --fail-under=10
poetry run pylint src/dmx/torch_stats --jobs=1 --fail-under=10
poetry run pylint src/dmx/mpi4py --jobs=1 --fail-under=10
poetry run pylint src/dmx/utils --jobs=1 --fail-under=10
poetry run pylint src/dmx/torch_utils --jobs=1 --fail-under=10
poetry run pylint examples --jobs=1 --fail-under=10
poetry run pylint examples_mpi4py --jobs=1 --fail-under=10
poetry run pylint examples_spark --jobs=1 --fail-under=10
poetry run pylint examples_torch --jobs=1 --fail-under=10
poetry run pylint tests --jobs=1 --fail-under=10
```

Docstring quality:

```bash
poetry run pydocstyle src/dmx/stats/pdist.py \
  src/dmx/bstats \
  src/dmx/torch_stats/pdist.py \
  src/dmx/utils/optsutil.py \
  src/dmx/utils/vector.py
```

Test job command:

```bash
TEST_TORCH_DEVICE=cpu poetry run pytest tests/bstats/ tests/stats/ tests/torch_stats/ tests/utils/ -v --tb=short
```

## Practical Workflow

1. Run the smallest relevant test target while iterating.
2. Run the matching formatter, import-sort, and type checks on changed paths.
3. Before finishing a substantial Python change, run the CI-equivalent test command plus any affected lint or docstring checks.
4. If optional dependencies are involved, keep CI's install shape in mind:
   - tests use `poetry install --no-interaction --extras ci --with dev`
   - mypy and pylint use `poetry install --no-interaction --extras "ci optional" --with dev`
