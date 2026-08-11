# VS Code Recommended Settings

This directory contains recommended VS Code settings for the dmx-learn project.

## Setup

To use these settings, copy them to your local `.vscode` directory (which is git-ignored):

```bash
cp .vscode-recommended/* .vscode/
```

Or manually merge with your existing `.vscode/settings.json` if you have personal customizations.

## What's Included

- `settings.json` - Python development settings including:
  - Black formatter with format-on-save
  - isort import organization
  - Pylint, mypy, and pydocstyle integration
  - pytest configuration
  - 88-character ruler
  - File exclusions

- `extensions.json` - Recommended VS Code extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff (optional)
  - Jupyter

## Note

These are recommendations only. Your personal `.vscode` directory is git-ignored, so you can customize settings without affecting the repository.
