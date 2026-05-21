# Documentation & Code Quality Progress Tracker

**Repository:** `dmx-learn`
**Date:** 2026-05-21
**Status:** In Progress

## Current Snapshot

- Completed: Phase 1 tooling setup, Phase 2 baseline cleanup, and most CI/workflow setup.
- Remaining: Phase 3 docs/docstrings work, Phase 4 Sphinx docs completion, repo-wide `mypy` enforcement, and `pylint` coverage for `src/dmx/stats/` and `src/dmx/bstats/`.
- Removed from plan: `PHASE_6`.

## Live Status Verified In Repo

- `pyproject.toml` includes `black`, `isort`, `pylint`, `mypy`, `pydocstyle`, and a dedicated `docs` dependency group.
- `.pre-commit-config.yaml` is active for formatting/basic checks, but `pydocstyle` is still commented out.
- `.github/workflows/quality.yml` exists and runs:
  - `black --check .`
  - `isort --check .`
  - targeted `mypy`
  - targeted `pylint`
  - targeted `pydocstyle`
- `.github/workflows/docs.yml` exists and builds Sphinx HTML docs in CI.
- `docs/conf.py` is configured with `autodoc`, `napoleon`, `sphinx_autodoc_typehints`, and `mathjax`.
- Local docs generation works: `poetry run sphinx-build -W -b html docs/ docs/_build/html` passed on 2026-05-21.
- Repo-wide typing is not ready yet: `poetry run mypy src/dmx` currently reports `1034` errors.
- Current `mypy` blockers include:
  - internal error in `src/dmx/stats/tree_hmm.py`
  - missing typing support for `umap` in `src/dmx/mpi4py/utils/humap.py`
- Current `pylint` enforcement does not yet cover all source packages:
  - covered in CI: `src/dmx/torch_stats`, `src/dmx/mpi4py`, `src/dmx/utils`, `src/dmx/torch_utils`, examples, tests, and `src/dmx/stats/pdist.py`
  - not yet covered repo-wide: `src/dmx/stats/` and `src/dmx/bstats/`

## Phase Tracker

| Phase | Scope | Status | Notes |
| --- | --- | --- | --- |
| Phase 1 | Tooling and standards | Complete | Tooling is configured in `pyproject.toml`, `.pylintrc`, and pre-commit. |
| Phase 2 | Baseline cleanup and initial fixes | Complete | Formatting and targeted quality fixes have already landed. |
| Phase 3 | Documentation and docstring quality | Not Complete | This is still open. `pydocstyle` is configured but not yet enabled in pre-commit. |
| Phase 4 | Sphinx docs structure and generation | Not Complete | Sphinx is configured and HTML build passes, but this phase is still open until docs coverage and generation workflow are fully finalized. |
| Phase 5 | Automation and CI | Mostly Complete | CI workflows exist and run, but `mypy` and `pylint` are not yet enforced across the full repo. |

## Outstanding Work

### Phase 3: Docs and Docstrings

- [ ] Review and improve docstrings in the remaining public modules.
- [ ] Decide the expected docstring coverage standard for older modules versus active modules.
- [ ] Expand `pydocstyle` beyond the current small CI target set.
- [ ] Re-enable `pydocstyle` in `.pre-commit-config.yaml` once the noise level is acceptable.

### Phase 4: Sphinx Documentation

- [ ] Audit `docs/` pages against the public API and examples that should be documented.
- [ ] Fill in or update missing `.rst` pages where module coverage is incomplete or stale.
- [ ] Keep `poetry run sphinx-build -W -b html docs/ docs/_build/html` passing.
- [ ] Decide whether broken-link checking should stay non-blocking or become a required CI gate.

### Repo-wide Type Checking

- [ ] Expand `mypy` CI from the current targeted file list to the full repository.
- [ ] Reduce repo-wide `mypy` failures from the current `1034` errors.
- [ ] Resolve or intentionally isolate the `mypy` internal error in `src/dmx/stats/tree_hmm.py`.
- [ ] Address the `umap` typing issue in `src/dmx/mpi4py/utils/humap.py`.

### Remaining Lint Coverage

- [ ] Add `pylint` coverage for `src/dmx/stats/` beyond `src/dmx/stats/pdist.py`.
- [ ] Add `pylint` coverage for `src/dmx/bstats/`.

## Recommended Completion Order

1. Stabilize repo-wide `mypy` enough to run it on all source files in CI.
2. Finish Phase 3 docstring cleanup so `pydocstyle` can become an active gate.
3. Finish Phase 4 docs coverage and keep Sphinx HTML generation green.
4. Expand `pylint` to `src/dmx/stats/` and `src/dmx/bstats/`.
5. Tighten any remaining CI gates after the codebase is clean enough to support them.

## Definition Of Done For This Tracker

- [ ] Phase 3 marked complete.
- [ ] Phase 4 marked complete.
- [ ] `mypy` runs on all intended repo files in CI.
- [ ] `pylint` runs on `src/dmx/stats/` and `src/dmx/bstats/`.
- [ ] `pydocstyle` is enabled in pre-commit and CI at the desired scope.
