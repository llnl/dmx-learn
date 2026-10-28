"""Utilities for configuring and fitting PyTorch-backed models.

This package contains device, tensor-vector, and estimation helpers intended
for :mod:`dmx.torch_stats`. The package root publicly exposes
:func:`detect_device`; import fitting and vector helpers from the
``dmx.torch_utils.estimation`` and ``dmx.torch_utils.vector`` submodules.
These utilities are separate from the NumPy-oriented :mod:`dmx.utils` package
and operate on the tensor protocols of the PyTorch backend.

PyTorch is optional. Importing this package succeeds without it so callers can
inspect the package, but calling :func:`detect_device` raises :class:`ImportError`
until PyTorch is installed.
"""

from dmx.utils.optional import missing_dependency_message

__all__ = ["detect_device"]

# Check if torch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def detect_device() -> str:
    """Detect device available for torch

    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')

    Raises:
        ImportError: If torch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            missing_dependency_message("PyTorch", "torch", "dmx.torch_utils")
        )

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
