"""Backend-neutral and NumPy-oriented modeling utilities.

The submodules provide estimation loops, automatic estimator selection,
embedding helpers, option handling, and vector operations used primarily with
:mod:`dmx.stats` and, where documented, :mod:`dmx.bstats`. The package root
declares only ``optsutil`` and ``vector`` as its star-import surface; import
other helpers from their defining ``dmx.utils`` submodules.

Tensor device and fitting concerns are outside this package and belong to
:mod:`dmx.torch_utils`. MPI orchestration likewise belongs to
:mod:`dmx.mpi4py.utils`; some optional embedding helpers here may still require
dependencies beyond the core installation.
"""

__all__ = ["optsutil", "vector"]
