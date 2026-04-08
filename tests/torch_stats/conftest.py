"""Pytest configuration for torch_stats tests.

This conftest.py automatically skips all torch_stats tests if PyTorch is not installed.
"""
import os
import sys
import pytest

# Set environment variable for torch tests
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def pytest_ignore_collect(collection_path, config):
    """Prevent collection of torch_stats tests if torch is not available."""
    if not TORCH_AVAILABLE:
        # Skip collection of all test files in torch_stats if torch is missing
        if "torch_stats" in str(collection_path):
            return True
    return False


def pytest_collection_modifyitems(config, items):
    """Add markers to torch_stats tests."""
    if not TORCH_AVAILABLE:
        # This shouldn't be reached due to pytest_ignore_collect,
        # but adding as a safety net
        skip_torch = pytest.mark.skip(
            reason="PyTorch is not installed. Install with: poetry install --with torch"
        )
        for item in items:
            if "torch_stats" in str(item.fspath):
                item.add_marker(skip_torch)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "torch: mark test as requiring PyTorch (skipped if torch not installed)"
    )


def pytest_report_header(config):
    """Add torch availability info to pytest header."""
    if TORCH_AVAILABLE:
        import torch
        return [f"PyTorch: {torch.__version__} (available)"]
    else:
        return ["PyTorch: not installed (torch_stats tests will be skipped)"]


# If torch is not available and we somehow got here during import,
# prevent test discovery by marking the module
if not TORCH_AVAILABLE:
    collect_ignore_glob = ["*.py"]
