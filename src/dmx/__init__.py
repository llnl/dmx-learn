"""Top-level namespace for the dmx-learn modeling toolkit.

The statistical backends live in :mod:`dmx.stats`, :mod:`dmx.bstats`, and
:mod:`dmx.torch_stats`; utilities are split between :mod:`dmx.utils` and the
PyTorch-specific :mod:`dmx.torch_utils`. Import the backend or utility package
needed by an application directly. This namespace deliberately performs no
eager backend imports or runtime setup, so optional PyTorch and MPI dependencies
are not required merely to import :mod:`dmx`.
"""

__all__ = ["stats", "utils", "src"]
