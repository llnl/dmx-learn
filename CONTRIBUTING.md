# Contributing to dmx-learn

Thank you for your interest in contributing to dmx-learn! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Running Quality Checks](#running-quality-checks)
- [Testing](#testing)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dmx-learn.git
   cd dmx-learn
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/llnl/dmx-learn.git
   ```

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Poetry (for dependency management)

### Installation

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install project dependencies:
   ```bash
   poetry install --with dev
   ```

3. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

### Optional Dependencies

For PyTorch support:
```bash
poetry install --extras torch
```

For MPI support:
```bash
poetry install --extras optional
```

For all optional dependencies:
```bash
poetry install --extras all
```

## Code Style Guidelines

We follow strict code quality standards to maintain consistency and readability.

### Code Formatting

- **Line Length:** 88 characters (Black default)
- **Code Formatter:** Black (automatic)
- **Import Sorting:** isort (automatic, Black-compatible)
- **Style Guide:** PEP 8

### Naming Conventions

- **Functions and variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_CASE`
- **Private methods/attributes:** `_leading_underscore`

### Type Hints

- All functions must have type hints for parameters and return values
- Use `Optional[Type]` for optional parameters
- Use `Union[Type1, Type2]` for multiple possible types
- For scientific computing, mathematical variable names (x, y, mu, sigma) are acceptable

### Docstrings

We use **Google-style docstrings** throughout the codebase:

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief one-line summary.

    More detailed description of what the function does,
    including any important algorithmic details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.
    """
    pass
```

**Note:** Comprehensive docstring improvements are planned for Phase 3 of the documentation enhancement project. Current docstrings may not all follow this standard yet.

## Running Quality Checks

We use several tools to maintain code quality. All can be run manually:

### Format Code

```bash
# Format with Black
poetry run black .

# Sort imports with isort
poetry run isort .
```

### Type Checking

```bash
# Run mypy type checker
poetry run mypy src/
```

### Linting

```bash
# Run enforced pylint checks on CI-scoped files
poetry run pylint src/dmx/stats/pdist.py --jobs=1 --fail-under=10
poetry run pylint src/dmx/torch_stats/pdist.py --jobs=1 --fail-under=10
poetry run pylint src/dmx/utils/optsutil.py --jobs=1 --fail-under=10
poetry run pylint src/dmx/utils/vector.py --jobs=1 --fail-under=10 --ignored-modules=numpy,scipy,scipy.linalg,scipy.special
```

### Docstring Checking

```bash
# Check docstrings (manual only, not enforced in pre-commit yet)
poetry run pydocstyle src/dmx/
```

### Run All Checks

```bash
# Run all quality checks
poetry run black --check .
poetry run isort --check .
poetry run mypy src/
poetry run pylint src/dmx/stats/pdist.py --jobs=1 --fail-under=10
poetry run pylint src/dmx/torch_stats/pdist.py --jobs=1 --fail-under=10
poetry run pylint src/dmx/utils/optsutil.py --jobs=1 --fail-under=10
poetry run pylint src/dmx/utils/vector.py --jobs=1 --fail-under=10 --ignored-modules=numpy,scipy,scipy.linalg,scipy.special
poetry run pydocstyle src/dmx/
```

## Testing

We use pytest for testing.

### Run All Tests

```bash
poetry run pytest tests/
```

### Run Tests with Coverage

```bash
poetry run pytest tests/ --cov=src/dmx --cov-report=term
```

### Run Specific Test Files

```bash
poetry run pytest tests/stats/gaussian_test.py
```

### Run Tests with PyTorch

```bash
# Install torch extras first
poetry install --extras torch

# Run torch-specific tests
poetry run pytest -m torch
```

### Run MPI Tests

```bash
# Install optional extras first
poetry install --extras optional

# Run MPI tests
poetry run pytest tests/mpi4py/
```

## Pre-commit Hooks

Pre-commit hooks run automatically before each commit to ensure code quality.

### What Runs Automatically

- **Trailing whitespace removal**
- **End-of-file fixer**
- **YAML validation**
- **TOML validation**
- **Large file check**
- **Merge conflict check**
- **Black formatting**
- **isort import sorting**

**Note:** Pydocstyle is configured but temporarily disabled until Phase 3 when docstrings are comprehensively improved.

### Skipping Hooks (Not Recommended)

Only skip hooks if absolutely necessary:

```bash
git commit --no-verify -m "commit message"
```

### Updating Hooks

```bash
poetry run pre-commit autoupdate
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clear, concise code
   - Add tests for new functionality
   - Update documentation as needed
   - Follow the code style guidelines

3. **Run quality checks:**
   ```bash
    poetry run black .
    poetry run isort .
    poetry run pylint src/dmx/stats/pdist.py --jobs=1 --fail-under=10
    poetry run pylint src/dmx/torch_stats/pdist.py --jobs=1 --fail-under=10
    poetry run pylint src/dmx/utils/optsutil.py --jobs=1 --fail-under=10
    poetry run pylint src/dmx/utils/vector.py --jobs=1 --fail-under=10 --ignored-modules=numpy,scipy,scipy.linalg,scipy.special
    poetry run pytest tests/
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

   Pre-commit hooks will run automatically.

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request:**
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template with:
     - Description of changes
     - Related issue numbers
     - Testing performed
     - Breaking changes (if any)

7. **Address review feedback:**
   - Make requested changes
   - Push updates to the same branch
   - PR will update automatically

### PR Requirements

- All tests must pass
- Code coverage should not decrease
- Code must pass Black and isort checks
- Mypy type checking should pass
- The four CI-scoped pylint files must pass at `10.00/10`
- Documentation should be updated if needed

## Documentation

### Building Documentation Locally

```bash
# Install docs dependencies
poetry install --with docs

# Build documentation
cd docs/
poetry run sphinx-build -b html . _build/html

# View documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

### Documentation Style

- Use **reStructuredText** (.rst) for Sphinx documentation
- Follow existing documentation structure
- Include code examples where appropriate
- Use proper cross-references (`:class:`, `:func:`, `:meth:`)

## IDE Configuration

### VS Code

Recommended VS Code settings are available in `.vscode-recommended/`. To use them:

```bash
cp .vscode-recommended/* .vscode/
```

Or merge with your existing `.vscode/settings.json` if you have personal customizations.

The recommended settings include:
- Black formatter with format-on-save
- isort import organization
- Pylint, mypy, and pydocstyle integration
- pytest configuration
- Recommended extensions

### PyCharm / IntelliJ IDEA

See `.pycharm-config.md` for detailed PyCharm configuration instructions including:
- Poetry environment setup
- Black and isort integration
- Linting and type checking configuration
- Testing setup
- File watchers

## Questions or Issues?

- **Bug Reports:** Open an issue on GitHub with a clear description and reproduction steps
- **Feature Requests:** Open an issue describing the feature and use case
- **Questions:** Open a discussion or issue on GitHub

## License

By contributing to dmx-learn, you agree that your contributions will be licensed under the -3-Clause License.

---

Thank you for contributing to dmx-learn! 🎉
