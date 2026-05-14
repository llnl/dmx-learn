# Pylint CI/CD Plan

## Goal

Use `pylint` as a strict, incremental quality gate.

Current enforced goal:

- Keep the existing CI scope at `10.00/10` for:
  - `src/dmx/stats/pdist.py`
  - `src/dmx/torch_stats/pdist.py`
  - `src/dmx/utils/optsutil.py`
  - `src/dmx/utils/vector.py`

Next expansion goal:

- Bring every Python file in these directories to `10.00/10`:
  - `src/dmx/utils`
  - `src/dmx/torch_utils`
  - `src/dmx/mpi4py/utils`

Long-term goal:

- Expand the same standard to every Python file in the repository.

This plan keeps the rollout small and controlled so that CI enforcement grows
only after the targeted file set is actually clean and stable.

## Style Standard

The intended `pylint` style for this repository is:

- PEP 8 layout
- Black as the formatting authority
- line length `88`
- Google-style docstrings
- Sphinx-friendly prose and examples
- explicit imports instead of wildcard imports
- limited, documented exceptions for scientific Python code

### Hard Requirements

For any file added to the CI lint scope:

- No `pylint` fatal or error messages
- Final score must be `10.00/10`
- No `line-too-long` violations
- No undocumented local `pylint: disable=...` comments
- No wildcard imports
- No unnecessary `pass` statements
- No config warnings from `.pylintrc`

### Repo-Wide Relaxations That Should Stay

These are reasonable for scientific and numerical code and should remain
relaxed unless there is a concrete reason to tighten them:

- `too-many-arguments`
- `too-many-locals`
- `too-many-branches`
- `too-many-statements`
- `invalid-name` for mathematical variables such as `x`, `y`, `n`, `mu`, `df`
- `no-member` when it is a known dynamic-library inference issue

### Docstring Policy

Docstrings must work well for both human readers and generated Sphinx docs.

Guidelines:

- Prefer concise summary lines.
- Use valid Google-style `Args`, `Returns`, `Raises`, and `Examples` sections.
- Wrap long prose at `88` characters.
- Avoid massive prose blocks when a shorter summary plus argument detail is
  enough.
- Avoid awkward docstring line breaks that would render poorly in Sphinx.
- Let `pydocstyle` remain the authority for docstring convention details.
- Use `pylint` to enforce code quality, not duplicate docstring formatting
  checks already owned by `pydocstyle`.

## Current Enforced Scope

The current GitHub Actions lint job enforces `10.00/10` on:

- `src/dmx/stats/pdist.py`
- `src/dmx/torch_stats/pdist.py`
- `src/dmx/utils/optsutil.py`
- `src/dmx/utils/vector.py`

Current CI behavior:

- Uses `--jobs=1`
- Uses `--fail-under=10`
- Previously used a file-scoped workaround for `src/dmx/utils/vector.py`:
  - `--ignored-modules=numpy,scipy,scipy.linalg,scipy.special`

That workaround is no longer needed after switching the SciPy imports in
`src/dmx/utils/vector.py` to `import_module(...)`.

## Next Scope Inventory

The next expansion includes 20 Python files.

### `src/dmx/utils`

- `src/dmx/utils/__init__.py`
- `src/dmx/utils/automatic.py`
- `src/dmx/utils/builder.py`
- `src/dmx/utils/estimation.py`
- `src/dmx/utils/htsne.py`
- `src/dmx/utils/humap.py`
- `src/dmx/utils/metrics.py`
- `src/dmx/utils/optsutil.py`
- `src/dmx/utils/pvalues.py`
- `src/dmx/utils/special.py`
- `src/dmx/utils/vector.py`

### `src/dmx/torch_utils`

- `src/dmx/torch_utils/__init__.py`
- `src/dmx/torch_utils/estimation.py`
- `src/dmx/torch_utils/optsutil.py`
- `src/dmx/torch_utils/vector.py`

### `src/dmx/mpi4py/utils`

- `src/dmx/mpi4py/utils/automatic.py`
- `src/dmx/mpi4py/utils/bestimation.py`
- `src/dmx/mpi4py/utils/estimation.py`
- `src/dmx/mpi4py/utils/humap.py`
- `src/dmx/mpi4py/utils/optsutil.py`

## Rollout Strategy

The right way to expand this is by directory and by risk level, not by trying
to lint all 20 files in one pass.

### Phase 1: Finish the Low-Risk Utility Surface

Target files:

- `src/dmx/utils/__init__.py`
- `src/dmx/utils/automatic.py`
- `src/dmx/utils/builder.py`
- `src/dmx/utils/estimation.py`
- `src/dmx/utils/metrics.py`
- `src/dmx/utils/pvalues.py`
- `src/dmx/utils/special.py`

Progress so far:

- Done: `src/dmx/utils/__init__.py` at `10.00/10`
- Done: `src/dmx/utils/automatic.py` at `10.00/10`
- Done: `src/dmx/utils/builder.py` at `10.00/10`
- Done: `src/dmx/utils/estimation.py` at `10.00/10`
- Done: `src/dmx/utils/htsne.py` at `10.00/10`
- Done: `src/dmx/utils/humap.py` at `10.00/10`
- Done: `src/dmx/utils/metrics.py` at `10.00/10`
- Done: `src/dmx/utils/pvalues.py` at `10.00/10`
- Done: `src/dmx/utils/special.py` at `10.00/10`
- All of `src/dmx/utils` now lint clean at `10.00/10`

Phase 1 status:

- Complete

Why first:

- Smaller surface area than the numerical/vector-heavy modules
- Likely to need mostly docstring cleanup, import cleanup, and minor style fixes
- Good place to validate the repeatable cleanup process

Expected issues:

- long docstrings
- wildcard imports
- outdated naming or helper patterns
- minor refactor warnings

### Phase 2: Finish the Remaining Core `utils` Files

Target files:

- `src/dmx/utils/humap.py`
- `src/dmx/utils/htsne.py`
- confirm `src/dmx/utils/optsutil.py` remains clean
- confirm `src/dmx/utils/vector.py` remains clean

Why second:

- These are more likely to involve optional dependencies, more complex numeric
  behavior, and docstrings that need extra Sphinx attention
- `vector.py` already demonstrated that some files may need command-scoped
  lint workarounds rather than global config changes

Expected issues:

- import-analysis problems around scientific libraries
- dense numerical docstrings
- examples that need to render cleanly in Sphinx

### Phase 3: Bring `torch_utils` to `10.00/10`

Target files:

- `src/dmx/torch_utils/__init__.py`
- `src/dmx/torch_utils/estimation.py`
- `src/dmx/torch_utils/optsutil.py`
- `src/dmx/torch_utils/vector.py`

Why third:

- This is a small directory with a narrow conceptual surface
- It likely mirrors some of the cleanup patterns already solved in `utils`
- It may need torch-specific inference workarounds, so it is safer to handle
  after the base utility patterns are established

Expected issues:

- dynamic torch attribute inference
- long docstrings
- mirrored code that should be cleaned consistently with `src/dmx/utils`

### Phase 4: Bring `mpi4py/utils` to `10.00/10`

Target files:

- `src/dmx/mpi4py/utils/automatic.py`
- `src/dmx/mpi4py/utils/bestimation.py`
- `src/dmx/mpi4py/utils/estimation.py`
- `src/dmx/mpi4py/utils/humap.py`
- `src/dmx/mpi4py/utils/optsutil.py`

Why fourth:

- MPI-specific code tends to have more environment-specific edge cases
- Some modules may be harder to lint or test locally depending on optional
  dependencies
- By this point, the repo should already have established patterns for utility
  cleanup and mirrored modules

Expected issues:

- optional import patterns
- duplicated utility logic that should be brought into consistent style
- environment-specific code paths that may need careful docstrings and narrow
  suppressions

## CI Expansion Plan

Do not add a directory to CI until every file in that directory set is already
at `10.00/10` locally.

Recommended CI expansion order:

1. Current four-file enforced scope
2. All of `src/dmx/utils`
3. All of `src/dmx/torch_utils`
4. All of `src/dmx/mpi4py/utils`

### Step 1: Enforce `src/dmx/utils`

Every file in `src/dmx/utils` is now at `10.00/10`, so the lint job can be
extended to include all of them.

Current CI behavior for this directory:

- Runs `poetry run pylint src/dmx/utils --jobs=1 --fail-under=10`
- No `vector.py`-specific workaround is needed now

### Step 2: Enforce `src/dmx/torch_utils`

Only after all four `torch_utils` files reach `10.00/10`, add them to the same
workflow.

If torch-specific import-analysis issues appear, prefer command-scoped
workarounds over global `.pylintrc` changes.

### Step 3: Enforce `src/dmx/mpi4py/utils`

Only after all five MPI utility files are stable at `10.00/10`, add them.

If MPI-specific modules need special lint handling, scope it to those commands
only.

## Required Workflow Discipline

For each phase, use the same process.

### Per-File Cleanup Order

1. Run `black` and `isort`.
2. Run `pylint` on the single file.
3. Fix line length, imports, naming, and control-flow issues.
4. Rewrite docstrings so they satisfy both style and Sphinx readability.
5. Re-run `pylint` until the file is `10.00/10`.
6. Move to the next file.

### Per-Directory Completion Gate

Do not update CI for a directory until all files in that directory:

- run without `pylint` crashes
- score `10.00/10`
- use only documented, narrow exceptions
- do not require broad repo-level workarounds that weaken other files

## Sphinx-Specific Guidance

Because documentation generation matters here, `pylint` cleanup should not turn
docstrings into awkward lint-driven text.

Preferred docstring style:

- short summary sentence first
- optional one short explanatory paragraph
- Google-style sections with readable wrapping
- simple examples when they materially help understanding

Avoid:

- giant copied mathematical prose blocks
- one-line summaries that hide important argument behavior
- mechanically wrapped text that reads badly in generated docs
- disabling lint/doc rules when a cleaner docstring would solve the problem

## Command Patterns

### Local Per-File Pattern

```bash
poetry run black <file>
poetry run isort <file>
poetry run pylint <file> --jobs=1 --fail-under=10
```

### Local Pattern for `src/dmx/utils/vector.py`

```bash
poetry run black src/dmx/utils/vector.py
poetry run isort src/dmx/utils/vector.py
poetry run pylint src/dmx/utils/vector.py --jobs=1 --fail-under=10 --ignored-modules=numpy,scipy,scipy.linalg,scipy.special
```

### CI Pattern

```bash
poetry run pylint <file> --jobs=1 --fail-under=10
```

Use command-scoped `--ignored-modules=...` only for the specific files that
actually require it.

## Contributor Guidance

For any file being prepared for inclusion in the CI lint scope:

1. Keep changes minimal and behavior-preserving.
2. Prefer explicit imports.
3. Prefer the smallest correct docstring rewrite.
4. Avoid broad repo-level lint config changes unless multiple files truly need
   the same behavior.
5. If a file needs a special `pylint` workaround, prove it with a direct
   command-line test and scope it to that file if possible.
6. Document the reason for any local disable or command-scoped workaround in
   this file.

## Definition Of Done For The Next Expansion

The next expansion is complete when all of the following are true:

1. Every file in `src/dmx/utils`, `src/dmx/torch_utils`, and
   `src/dmx/mpi4py/utils` runs successfully under `pylint`.
2. Every file in those directories scores `10.00/10`.
3. Any required import-analysis workaround is file-scoped, not global, unless a
   broader rule is clearly justified.
4. `.github/workflows/quality.yml` is updated to enforce the newly completed
   directory set.
5. `CONTRIBUTING.md` stays aligned with the actual enforced commands.

## Recommended Next Session Start

When starting the next session, begin with:

1. audit `src/dmx/utils` file-by-file
2. start with the lowest-risk files in that directory
3. leave `htsne.py`, `humap.py`, and any heavy numerical module for later in
   the phase unless they are already close to clean
4. only expand CI after the whole directory is finished

## Summary Recommendation

The best rollout is:

- strict score
- narrow CI scope
- phased directory expansion
- Sphinx-friendly docstring cleanup
- file-scoped workarounds only when necessary

That gives you a realistic path from 4 enforced files to utility-directory
coverage, and from there to full-repo `pylint` enforcement.
