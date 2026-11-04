"""Tests for MPI Fisher embeddings against the local fisher_embedding rows.

Run with any number of ranks, for example:

    NUMBA_DISABLE_JIT=1 mpiexec -n 2 python -m pytest \
        tests/mpi4py/test_fisher_embedding.py

Every rank builds the same model and records and embeds its own contiguous
share. Checks are made on rank 0 and broadcast before asserting, so a failure
cannot leave other ranks waiting in a collective.
"""

from typing import Any, List, Optional

import numpy as np
import pytest
from mpi4py import MPI  # pylint: disable=no-name-in-module

from dmx import bstats, stats
from dmx.mpi4py.utils.fisher import fisher_embedding_mpi
from dmx.utils.fisher import FisherEmbedding, fisher_embedding

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def _stats_model() -> stats.MixtureDistribution:
    """Heterogeneous two-component mixture."""
    return stats.MixtureDistribution(
        [
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(-2.0, 1.0),
                    stats.CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
                )
            ),
            stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(2.0, 2.0),
                    stats.CategoricalDistribution({"a": 0.1, "b": 0.3, "c": 0.6}),
                )
            ),
        ],
        [0.4, 0.6],
    )


def _bstats_model() -> bstats.MixtureDistribution:
    """Bayesian Gaussian mixture."""
    return bstats.MixtureDistribution(
        (
            bstats.GaussianDistribution(-3.0, 1.0),
            bstats.GaussianDistribution(0.0, 0.5),
            bstats.GaussianDistribution(3.0, 2.0),
        ),
        [0.3, 0.3, 0.4],
    )


def _bounds(count: int, empty_last: bool) -> List[int]:
    """Return uneven contiguous split points, optionally leaving the last rank empty."""
    ranks = SIZE - 1 if empty_last and SIZE > 1 else SIZE
    sizes = [count // ranks + (1 if r < count % ranks else 0) for r in range(ranks)]
    sizes += [0] * (SIZE - ranks)
    return list(np.concatenate(([0], np.cumsum(sizes))).astype(int))


def _local_share(data: List[Any], empty_last: bool) -> List[Any]:
    """Return this rank's records."""
    bounds = _bounds(len(data), empty_last)
    return data[bounds[RANK] : bounds[RANK + 1]]


def _check(condition: Optional[bool], message: str) -> None:
    """Assert a condition evaluated on rank 0 on every rank."""
    assert COMM.bcast(condition, root=0), message


@pytest.mark.parametrize("make", [_stats_model, _bstats_model])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("chunk_size", [None, 3])
@pytest.mark.parametrize("empty_last", [False, True])
def test_gathered_rows_match_local(
    make: Any, dtype: Any, chunk_size: Optional[int], empty_last: bool
) -> None:
    model = make()
    data = list(model.sampler(5).sample(37))

    result = fisher_embedding_mpi(
        model,
        _local_share(data, empty_last),
        COMM,
        dtype=dtype,
        chunk_size=chunk_size,
    )

    if RANK == 0:
        assert isinstance(result, FisherEmbedding)
        expected = fisher_embedding(model, data, dtype=dtype)
        same = (
            result.scale == expected.scale
            and result.matrix.dtype == dtype
            and result.component_slices == expected.component_slices
            and np.array_equal(result.matrix.toarray(), expected.matrix.toarray())
        )
    else:
        same = result is None
    _check(same, "Gathered MPI rows differ from the local rows.")


def test_rows_stay_on_each_rank_without_gather() -> None:
    model = _stats_model()
    data = list(model.sampler(7).sample(29))
    expected = fisher_embedding(model, data, dtype=np.float64)
    bounds = _bounds(len(data), empty_last=False)

    result = fisher_embedding_mpi(
        model, _local_share(data, False), COMM, dtype=np.float64, gather=False
    )

    assert result is not None
    same = result.scale == expected.scale and np.array_equal(
        result.matrix.toarray(),
        expected.matrix[bounds[RANK] : bounds[RANK + 1]].toarray(),
    )
    assert all(COMM.allgather(same)), "Per-rank MPI rows differ from local rows."


def test_whitening_and_like_match_local() -> None:
    model = _stats_model()
    data = list(model.sampler(9).sample(40))

    result = fisher_embedding_mpi(
        model, _local_share(data, False), COMM, whiten="full", dtype=np.float64
    )
    layout = COMM.bcast(result, root=0)
    reused = fisher_embedding_mpi(model, _local_share(data, True), COMM, like=layout)

    if RANK == 0:
        assert result is not None and reused is not None
        expected = fisher_embedding(model, data, whiten="full", dtype=np.float64)
        close = np.allclose(
            result.matrix.toarray(),
            expected.matrix.toarray(),
            rtol=1e-9,
            atol=1e-12,
        )
        same = np.array_equal(
            reused.matrix.toarray(),
            fisher_embedding(model, data, like=layout).matrix.toarray(),
        )
        ok: Optional[bool] = bool(close and same)
    else:
        ok = None
    _check(ok, "Whitened or reused MPI rows differ from the local rows.")
