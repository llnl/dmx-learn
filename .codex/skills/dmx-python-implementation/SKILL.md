---
name: dmx-python-implementation
description: Use when modifying Python source, utilities, examples, or distribution code in dmx-learn. Applies the repo's enforced coding standards from CI: Poetry workflows, Black line length 88, isort black profile, typed function signatures for mypy, pylint-friendly structure, and Google-style docstrings for touched public APIs.
---

# dmx Python Implementation

Use this skill for changes under `src/dmx`, `examples*`, and related Python entry points.

## Start Here

1. Read the target module and at least one neighboring module in the same package before editing.
2. Read the matching tests when they exist.
3. Keep changes scoped to the package split already used by the repo:
   - `src/dmx/stats`: NumPy/SciPy statistical distributions
   - `src/dmx/torch_stats`: PyTorch-backed statistical distributions
   - `src/dmx/utils`: non-torch utilities
   - `src/dmx/torch_utils`: torch-specific utilities
   - `examples*`: runnable examples, kept simple and explicit

## Coding Rules Enforced by CI

- Format for Black with line length `88`.
- Sort imports with isort using the Black profile.
- Add type annotations for every new or changed function or method.
- Do not rely on implicit `Optional`; write `Optional[T]` or `T | None` explicitly.
- Return concrete types where possible. Avoid untyped helper functions.
- Keep code compatible with the repo's Python floor of `3.10`.
- Prefer small, direct functions over clever abstractions. This repo already tolerates some repetitive statistical scaffolding; do not invent framework layers just to deduplicate a few lines.
- Preserve device and dtype behavior in torch code. In this repo, MPS and CPU/GPU differences are often handled intentionally.
- Avoid broad exception handling unless the surrounding code already depends on it or the failure mode is genuinely variable.
- If a `pylint` disable is necessary, keep it local and explain why in one short comment.

## Docstrings

- Use Google-style docstrings for public functions, methods, and classes you touch.
- Keep the first line short and descriptive.
- Include `Args`, `Returns`, and `Raises` when they add real value.
- Match the repo's current style: concise summaries, then argument details only where behavior is not obvious.
- Be stricter in modules similar to:
  - `src/dmx/stats/pdist.py`
  - `src/dmx/torch_stats/pdist.py`
  - `src/dmx/utils/optsutil.py`
  - `src/dmx/utils/vector.py`
  Those paths are checked directly by `pydocstyle` in CI.

## Implementation Guidance

- Preserve public names and constructor signatures unless the task explicitly changes API.
- When touching distributions, maintain consistency between scalar and sequence paths such as `log_density`, `seq_log_density`, encoders, samplers, and estimators.
- Prefer existing helper utilities over reimplementing numeric or tensor behavior.
- In torch code, keep generators, devices, and dtype conversions explicit.
- In utility code, favor NumPy arrays and deterministic conversions over loose container handling.

## Validation

Run focused checks on touched files first, then broader checks if the change is substantial.

Common commands:

```bash
poetry run black --check <paths>
poetry run isort --check <paths>
poetry run mypy --explicit-package-bases <paths>
poetry run pylint <paths> --jobs=1 --fail-under=10
```

If the change touches one of the docstring-gated files, also run:

```bash
poetry run pydocstyle <path>
```

If behavior changed, run the matching tests in `tests/` as well.
