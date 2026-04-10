# Prompt to Start: Phase 5 - CI/CD Pipeline Setup

## Context

I'm working on improving the dmx-learn Python repository following a comprehensive documentation and code quality improvement plan outlined in `DOC_UPDATE.md`. Phase 5 can be done in parallel with Phase 3 since it focuses on automation infrastructure rather than code changes.

## Current Status

**Branch:** `feature/doc-phase-5`

**Phase 1:** ✅ COMPLETE
- All code quality tools configured (Black, isort, pylint, mypy, pydocstyle)
- Pre-commit hooks installed and active (`.pre-commit-config.yaml`)
- Developer documentation created

**Phase 2:** ✅ COMPLETE
- Code formatted with Black and isort (273 files compliant)
- Type checking passing (0 errors in core modules)
- Pylint scores: 8.10-9.73/10 (all core modules meet quality standards)
- Docstrings fixed (0 pydocstyle errors in core modules)
- Sphinx documentation builds successfully (0 warnings)
- All tests passing (442 tests, zero regressions)

**See `PHASE2_REPORT.md` for comprehensive Phase 2 completion summary.**

**Phase 3:** IN PROGRESS
- Comprehensive docstring enhancement across all modules

**Phase 5:** IN PROGRESS
- CI/CD Pipeline Setup

## Phase 5 - CI/CD Pipeline Setup Progress

### Completed Steps:

#### ✅ Step 5.1: GitHub Actions Test Workflow (COMPLETE)
**File:** `.github/workflows/test.yml`

**Configuration:**
- Tests on Python 3.10, 3.11, 3.12, 3.13 (4 versions)
- Tests on Ubuntu, macOS, Windows (3 OS platforms)
- Total: 12 test job combinations
- Added new `ci` extra in `pyproject.toml` for CI-specific dependencies

**Dependencies Installed:**
- Core dependencies (via poetry.lock)
- torch (CPU-only via `TEST_TORCH_DEVICE=cpu`)
- umap-learn (for all utils tests)
- NO mpi4py (excluded per requirement)

**Tests Run:**
- `tests/stats/` - Core statistics tests
- `tests/torch_stats/` - PyTorch tests (CPU-only, no MPS/CUDA)
- `tests/utils/` - All utility tests (including test_humap.py)
- EXCLUDED: `tests/mpi4py/` (not in test paths)

**Key Features:**
- Dependency caching for faster builds
- Conforms to poetry.lock (exact versions)
- Fail-fast disabled (all combinations run independently)
- Test artifacts uploaded (7-day retention)
- Triggers: push/PR to main/develop/feature/* branches

#### ✅ Step 5.2: GitHub Actions Code Quality Workflow (COMPLETE)
**File:** `.github/workflows/quality.yml`

**Configuration:**
- 4 parallel jobs for fast CI feedback
- Runs on Python 3.11, Ubuntu-only
- Uses `poetry install --with dev`

**Jobs:**
1. **Formatting Job** (Black & isort)
   - `black --check .` - All files
   - `isort --check .` - All imports

2. **Type Checking Job** (mypy)
   - Core modules: pdist.py (stats + torch_stats), optsutil.py, vector.py
   - Enforces 0 type errors

3. **Linting Job** (pylint)
   - Same core modules as mypy
   - Uses `--jobs=1` (prevents crashes)
   - Uses `--exit-zero` (reports but doesn't fail)
   - Enforces ≥ 8.0/10 score

4. **Docstring Quality Job** (pydocstyle)
   - Same core modules
   - Google-style convention
   - Enforces 0 errors

**Performance:** ~60-90 seconds total (parallel execution)

### In Progress:

## Step 5.3: GitHub Actions - Documentation Build (NEXT)

### Objective

Automate code quality checks and documentation deployment to ensure:
- Consistent code quality enforcement
- Automated testing on multiple environments
- Continuous documentation updates
- Easy contribution workflow
- Professional project infrastructure

### From DOC_UPDATE.md - Phase 5 Overview:

**Goal:** Automate quality checks and documentation deployment

**Key Components:**
1. GitHub Actions workflows for testing and linting
2. Read the Docs integration for documentation
3. Code coverage reporting
4. Automated dependency updates
5. Release automation

### Phase 5 Steps

#### Step 5.1: GitHub Actions - Test Workflow

Create `.github/workflows/test.yml` to:
- [ ] Run tests on multiple Python versions (3.9, 3.10, 3.11, 3.12, 3.13)
- [ ] Test on multiple OS (Ubuntu, macOS, Windows)
- [ ] Run with and without optional dependencies
- [ ] Cache dependencies for faster builds
- [ ] Upload test results

**Test matrix considerations:**
- Core tests (no optional deps): All Python versions, all OS
- Torch tests: Subset of versions (torch compatibility)
- MPI tests: Ubuntu only (complex setup)

**From Phase 2 experience:**
- 442 stats tests pass reliably
- Tests complete in ~90 seconds
- Some tests require optional dependencies (torch, mpi4py, umap)
- Use `pytest --ignore=tests/mpi4py/ --ignore=tests/utils/test_humap.py` for core tests

#### Step 5.2: GitHub Actions - Code Quality Workflow

Create `.github/workflows/quality.yml` to:
- [ ] Run Black formatting check
- [ ] Run isort import check
- [ ] Run mypy type checking
- [ ] Run pylint on core modules
- [ ] Run pydocstyle for documentation validation
- [ ] Report quality metrics

**Quality standards from Phase 2:**
- Black: All files must pass `black --check .`
- isort: All imports must pass `isort --check .`
- mypy: Core modules must have 0 errors
- pylint: Core modules should score ≥ 8.0/10
- pydocstyle: Critical modules should have 0 errors

#### Step 5.3: GitHub Actions - Documentation Build

Create `.github/workflows/docs.yml` to:
- [ ] Build Sphinx documentation
- [ ] Treat warnings as errors (`-W` flag)
- [ ] Check for broken links
- [ ] Verify all cross-references
- [ ] Upload documentation artifacts

**From Phase 2 experience:**
- Sphinx build requires `docs` dependency group
- Build command: `sphinx-build -W docs/ docs/_build/html`
- Build time: Fast (< 30 seconds)
- Extensions needed: autodoc, napoleon, sphinx_autodoc_typehints, mathjax

#### Step 5.4: Read the Docs Integration

Configure `.readthedocs.yml` to:
- [ ] Specify Python version
- [ ] Install dependencies (including docs group)
- [ ] Configure Sphinx build
- [ ] Set up version management
- [ ] Enable PDF generation (optional)

**Configuration requirements:**
- Python version: 3.11 or higher
- Install: `poetry install --with docs`
- Build: Standard Sphinx build
- Requirements: All packages in `[tool.poetry.group.docs]`

#### Step 5.5: Code Coverage Reporting

Integrate code coverage:
- [ ] Add pytest-cov configuration
- [ ] Generate coverage reports in CI
- [ ] Upload to Codecov or Coveralls
- [ ] Add coverage badge to README
- [ ] Set coverage thresholds

**Configuration:**
```toml
[tool.pytest.ini_options]
addopts = "--cov=src/dmx --cov-report=xml --cov-report=html"
```

#### Step 5.6: Dependency Management

Set up automated dependency updates:
- [ ] Configure Dependabot for Python dependencies
- [ ] Set up automatic security updates
- [ ] Configure update frequency
- [ ] Define version constraints

**Poetry considerations:**
- Use `pyproject.toml` for dependency management
- Keep `poetry.lock` in version control
- Use version ranges appropriately
- Test updates before merging

#### Step 5.7: Status Badges and Documentation

Add to README.md:
- [ ] Test status badge (GitHub Actions)
- [ ] Code quality badge
- [ ] Documentation build badge (Read the Docs)
- [ ] Coverage badge
- [ ] PyPI version badge (if published)
- [ ] License badge

### Current Repository Structure

**Important Files:**
- `pyproject.toml` - Poetry configuration with all dependencies
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `docs/conf.py` - Sphinx configuration
- `tests/` - Test suite (442 tests)

**Dependency Groups in pyproject.toml:**
```toml
[tool.poetry.dependencies]
# Core dependencies

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
black = "^24.1.0"
isort = "^5.13.0"
pylint = "^3.0.0"
mypy = "^1.8.0"
pydocstyle = {extras = ["toml"], version = "^6.3.0"}
pre-commit = "^3.6.0"
# ... other dev tools

[tool.poetry.group.docs]
optional = true

[tool.poetry.group.docs.dependencies]
sphinx = "^8.0.0"
sphinx-rtd-theme = "^3.0.0"
sphinx-autodoc-typehints = "^3.0.0"
```

**Python Version Support:**
- Primary: Python 3.11+
- Testing: Should cover 3.9-3.13 if possible
- Note: Some features may require newer Python versions

### GitHub Actions Best Practices

**Workflow Triggers:**
```yaml
on:
  push:
    branches: [main, develop, feature/*]
  pull_request:
    branches: [main, develop]
```

**Caching Dependencies:**
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pypoetry
    key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock') }}
```

**Matrix Strategy:**
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12', '3.13']
    os: [ubuntu-latest, macos-latest, windows-latest]
  fail-fast: false
```

### Read the Docs Configuration

**Minimal `.readthedocs.yml`:**
```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs

sphinx:
  configuration: docs/conf.py
  fail_on_warning: true
```

### Testing Strategy for CI

**Test Categories:**

1. **Fast Core Tests** (run on all platforms/versions)
   ```bash
   pytest tests/stats/ --ignore=tests/mpi4py/ --ignore=tests/utils/test_humap.py
   ```
   - Duration: ~90 seconds
   - No optional dependencies required
   - 442 tests

2. **Torch Tests** (subset of platforms)
   ```bash
   pytest tests/ -m torch
   ```
   - Requires torch installation
   - GPU tests skipped in CI (CPU only)

3. **Full Test Suite** (on main branch only)
   ```bash
   pytest tests/
   ```
   - Includes all optional dependencies
   - Longer runtime

### Quality Check Strategy for CI

**Parallel Jobs for Speed:**

1. **Formatting Job** (fastest)
   ```bash
   black --check .
   isort --check .
   ```

2. **Type Checking Job**
   ```bash
   mypy src/dmx/stats/pdist.py src/dmx/torch_stats/pdist.py \
        src/dmx/utils/optsutil.py src/dmx/utils/vector.py
   ```

3. **Linting Job**
   ```bash
   pylint src/dmx/stats/pdist.py --exit-zero
   pylint src/dmx/torch_stats/pdist.py --exit-zero
   pylint src/dmx/utils/optsutil.py --exit-zero
   ```

4. **Documentation Job**
   ```bash
   pydocstyle src/dmx/stats/pdist.py src/dmx/torch_stats/pdist.py
   ```

### Known Issues and Considerations

**From Phase 2 Experience:**

1. **Pylint Multiprocessing:**
   - Use `--jobs=1` flag to avoid crashes
   - Example: `pylint src/dmx/ --jobs=1 --exit-zero`

2. **Test Warnings:**
   - 11 expected numerical warnings (divide by zero, etc.)
   - These are normal for statistical computations
   - Not failures

3. **Optional Dependencies:**
   - torch: Large download, platform-specific
   - mpi4py: Complex setup, Ubuntu only
   - umap: Optional dependency
   - Strategy: Core tests without these, separate jobs for optional features

4. **Sphinx Dependencies:**
   - Must install `docs` group: `poetry install --with docs`
   - Requires sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints

5. **Platform Differences:**
   - Windows: Path separators, line endings
   - macOS: Case-sensitive filesystem
   - Linux: Most compatible for scientific Python

### Security Considerations

**GitHub Actions:**
- [ ] Use minimal permissions for tokens
- [ ] Pin action versions (e.g., `actions/checkout@v4`)
- [ ] Review third-party actions before use
- [ ] Use secrets for sensitive data (API tokens, etc.)

**Dependabot:**
- [ ] Enable security updates
- [ ] Review dependency changes before merging
- [ ] Set up automatic PR creation
- [ ] Configure merge strategy

### Success Criteria for Phase 5

- [ ] GitHub Actions workflows running successfully
- [ ] Tests passing on multiple Python versions and platforms
- [ ] Code quality checks integrated into CI
- [ ] Documentation building automatically
- [ ] Read the Docs integration active and updating
- [ ] Status badges displaying correct information
- [ ] Automated dependency updates configured
- [ ] Clear contribution guidelines with CI expectations

### Resources

- **Phase 2 Report:** `PHASE2_REPORT.md` - Lessons learned, test metrics
- **Repository:** Check `.readthedocs.yml` if it exists
- **GitHub:** Check if Actions are already partially configured
- **Poetry Config:** `pyproject.toml` - All dependencies and tool configs

### Recommended Approach

**Phase 5 can be done in parallel with Phase 3 because:**
1. It focuses on automation infrastructure, not code changes
2. It uses the quality standards already established in Phase 2
3. It doesn't conflict with ongoing documentation work
4. Early CI setup provides continuous feedback during Phase 3

**Start with:**
1. **Step 5.1:** Basic test workflow (highest value, validates core functionality)
2. **Step 5.4:** Read the Docs integration (documents current state)
3. **Step 5.2:** Code quality workflow (enforces standards)
4. **Steps 5.3, 5.5-5.7:** Additional automation and polish

**Test workflows on feature branch first:**
- Create workflows in feature branch
- Test all workflows thoroughly
- Verify badge URLs work
- Document any limitations or known issues
- Merge to main once stable

## Request

Please help me set up **Phase 5: CI/CD Pipeline** following the approach outlined above.

**Begin with:**
1. Creating a basic GitHub Actions test workflow (Step 5.1)
2. Setting up Read the Docs integration (Step 5.4)
3. Adding code quality checks (Step 5.2)
4. Expanding with additional automation as needed

**Work systematically through the steps, testing each workflow before proceeding to the next.**

Let's begin by creating the GitHub Actions test workflow!
