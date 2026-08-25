"""Estimate and validate sequence-encodable models from observed data.

The module provides randomized data partitioning and iterative EM helpers for
local encoded chunks or Spark RDDs. Callers may provide raw observations,
pre-encoded data, or a previous model depending on the helper.
"""

import sys
import time
from typing import (
    IO,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.random import RandomState
from pyspark import RDD

from dmx.stats import (
    seq_encode,
    seq_estimate,
    seq_initialize,
    seq_log_density,
    seq_log_density_sum,
)
from dmx.stats.pdist import (
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)

T = TypeVar("T")
E0 = TypeVar("E0")
EncodedChunks = Union[List[Tuple[int, EncodedDataSequence]], RDD[Any]]


def empirical_kl_divergence(
    dist1: SequenceEncodableProbabilityDistribution,
    dist2: SequenceEncodableProbabilityDistribution,
    enc_data: EncodedChunks,
) -> Tuple[float, float, float]:
    """Estimate KL divergence between two models on encoded observations.

    The log densities are normalized over the supplied observations and the
    discrete divergence from ``dist1`` to ``dist2`` is computed. Both models
    must accept the same encoding and at least one observation must have a
    finite, non-NaN log density under both models.

    Args:
        dist1: First distribution, which defines the KL weighting.
        dist2: Second distribution using the same encoded representation.
        enc_data: Local or Spark chunks represented as ``(chunk_size,
            encoded_sequence)`` pairs.

    Returns:
        The empirical KL estimate, followed by the counts of non-finite or NaN
        log densities under ``dist1`` and ``dist2``.

    Raises:
        ValueError: If there are no jointly valid log-density values.
    """
    log_density_chunks = seq_log_density(enc_data, estimate=(dist1, dist2))
    ll = np.hstack(log_density_chunks)

    l1 = ll[0, :]
    l2 = ll[1, :]
    g1 = np.bitwise_and(l1 != -np.inf, ~np.isnan(l1))
    g2 = np.bitwise_and(l2 != -np.inf, ~np.isnan(l2))
    gg = np.bitwise_and(g1, g2)

    max_l1 = np.max(l1[gg])
    max_l2 = np.max(l2[gg])

    p1 = np.exp(l1[gg] - max_l1)
    p1 /= p1.sum()

    p2 = np.exp(l2[gg] - max_l2)
    p2 /= p2.sum()

    r1 = (p1[gg] * (np.log(p1[gg]) - np.log(p2[gg]))).sum()
    r2 = (~g1).sum()
    r3 = (~g2).sum()

    return r1, r2, r3


def k_fold_split_index(sz: int, k: int, rng: RandomState) -> np.ndarray:
    """Assign observations to approximately balanced random folds.

    Random values drawn from ``rng`` determine a permutation, after which fold
    identifiers are assigned round-robin. The random-state is advanced.

    Args:
        sz: Number of observations.
        k: Number of folds.
        rng: Random state controlling the permutation.

    Returns:
        Integer array of shape ``(sz,)`` whose entry is the observation's
        zero-based fold identifier.
    """
    idx = rng.rand(sz)
    sidx = np.argsort(idx)

    rv = np.zeros(sz, dtype=int)
    for i in range(k):
        rv[sidx[np.arange(start=i, stop=sz, step=k, dtype=int)]] = i

    return rv


def partition_data_index(
    sz: int, pvec: Union[List[float], np.ndarray], rng: RandomState
) -> List[np.ndarray]:
    """Randomly partition observation indexes according to proportions.

    The random state is advanced once per observation. Boundaries are obtained
    by rounding cumulative proportions times ``sz``; ``pvec`` is not
    normalized or validated, so proportions summing to less than one omit
    trailing indexes.

    Args:
        sz: Number of observations.
        pvec: One-dimensional sequence of partition proportions.
        rng: Random state controlling the shuffled ordering.

    Returns:
        One integer index array per requested partition.
    """
    idx = rng.rand(sz)
    sidx = np.argsort(idx)

    rv = []
    p_tot = 0.0
    prev_idx = 0

    for p in pvec:
        next_idx = int(round(sz * (p_tot + p), 0))
        rv.append(sidx[prev_idx:next_idx])
        p_tot += p
        prev_idx = next_idx

    return rv


def partition_data(
    data: Sequence[T], pvec: Union[List[float], np.ndarray], rng: RandomState
) -> List[List[T]]:
    """Randomly partition observations according to proportions.

    Args:
        data: Sequence of observations.
        pvec: One-dimensional sequence of partition proportions. Values are
            passed unchanged to :func:`partition_data_index`.
        rng: Random state controlling the shuffled ordering; it is advanced.

    Returns:
        Materialized data partitions in the order of ``pvec``.
    """
    idx_list = partition_data_index(len(data), pvec, rng)

    return [[data[i] for i in u] for u in idx_list]


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def best_of(
    data: Optional[Sequence[T]],
    vdata: Optional[Sequence[T]],
    est: ParameterEstimator,
    trials: int,
    max_its: int,
    init_p: float,
    delta: float,
    rng: RandomState,
    init_estimator: Optional[ParameterEstimator] = None,
    enc_data: Optional[EncodedChunks] = None,
    enc_vdata: Optional[EncodedChunks] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
) -> Tuple[float, SequenceEncodableProbabilityDistribution]:
    """Run EM from multiple randomized starts and return the best fit.

    Every trial initializes from a random subset controlled by ``rng``. A
    proposed EM update is retained only when training log likelihood does not
    decrease (unless ``delta`` is ``None``), and a trial stops when its change
    is less than ``delta``. Models are compared by validation log likelihood
    when validation data is available, otherwise by training log likelihood.
    Progress is written to ``out``.

    Args:
        data: Raw training observations, or ``None`` when ``enc_data`` is
            supplied.
        vdata: Optional raw validation observations.
        est: Estimator used for EM updates.
        trials: Number of randomized starts; values below one are treated as
            one.
        max_its: Maximum updates per trial; values below one are treated as
            one.
        init_p: Fraction of observations participating in randomized
            initialization, in ``(0, 1]``.
        delta: Stop a trial when the signed training log-likelihood change is
            less than this value.
        rng: Random state used for every trial and advanced in place.
        init_estimator: Optional estimator used only for encoding and
            initialization.
        enc_data: Pre-encoded local chunks or Spark RDD. Takes precedence over
            ``data``.
        enc_vdata: Pre-encoded validation chunks. Takes precedence over
            ``vdata``.
        out: Text stream receiving iteration and trial summaries.
        print_iter: Positive interval between iteration summaries.

    Returns:
        Best validation log likelihood and its fitted model.

    Raises:
        ValueError: If both training representations are absent or ``init_p``
            is outside ``(0, 1]``.
        RuntimeError: If no trial produces a selectable model.
    """
    rv_ll = -np.inf
    rv_mm: Optional[SequenceEncodableProbabilityDistribution] = None
    i_est = est if init_estimator is None else init_estimator

    if data is None and enc_data is None:
        raise ValueError("Optimization called with empty data or enc_data.")

    if not 0 < init_p <= 1:
        raise ValueError(
            f"Invalid init_p: {init_p}. It must be greater than 0 and less "
            "than or equal to 1."
        )

    max_its = max(max_its, 1)
    trials = max(trials, 1)

    for kk in range(trials):

        encoded_data = enc_data
        encoded_vdata = enc_vdata
        if encoded_data is None:
            assert data is not None
            encoder = i_est.accumulator_factory().make().acc_to_encoder()
            encoded_data = seq_encode(data, encoder)

            if encoded_vdata is None and vdata is not None:
                encoded_vdata = seq_encode(vdata, encoder)

        mm = seq_initialize(encoded_data, i_est, rng, init_p)
        _, old_ll = seq_log_density_sum(encoded_data, mm)

        for i in range(max_its):

            mm_next = seq_estimate(encoded_data, est, mm)
            _, ll = seq_log_density_sum(encoded_data, mm_next)
            dll = ll - old_ll

            if (i + 1) % print_iter == 0:
                out.write(f"Iteration {i + 1}. LL={ll:f}, delta LL={dll:e}\n")

            if (dll >= 0) or (delta is None):
                mm = mm_next

            if (delta is not None) and (dll < delta):
                break

            old_ll = ll

        validation_data = encoded_vdata if encoded_vdata is not None else encoded_data
        _, vll = seq_log_density_sum(validation_data, mm)
        out.write(f"Trial {kk + 1}. VLL={vll:f}\n")

        if vll > rv_ll:
            rv_mm = mm
            rv_ll = vll

    if rv_mm is None:
        raise RuntimeError("No model was estimated.")

    return rv_ll, rv_mm


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def optimize(
    data: Optional[Sequence[T]],
    estimator: ParameterEstimator,
    max_its: int = 10,
    delta: Optional[float] = 1.0e-9,
    init_estimator: Optional[ParameterEstimator] = None,
    init_p: float = 0.1,
    rng: RandomState = RandomState(),
    prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None,
    vdata: Optional[Sequence[T]] = None,
    enc_data: Optional[EncodedChunks] = None,
    enc_vdata: Optional[EncodedChunks] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
    num_chunks: int = 1,
) -> SequenceEncodableProbabilityDistribution:
    """Fit a model with EM and return the best validation-scoring estimate.

    Raw data is encoded once, using ``prev_estimate`` when supplied or the
    initialization estimator otherwise. Without a previous estimate, model
    initialization samples observations according to ``init_p`` using ``rng``.
    Updates that decrease training log likelihood are rejected unless
    ``delta`` is ``None``. A non-``None`` ``delta`` stops optimization when the
    signed change is smaller than the threshold. Progress is written to
    ``out`` and model selection uses validation likelihood when available.

    Args:
        data: Raw training observations, or ``None`` when ``enc_data`` is
            supplied.
        estimator: Estimator used for each EM update.
        max_its: Maximum number of EM updates.
        delta: Optional stopping threshold for signed training
            log-likelihood improvement. ``None`` disables early stopping and
            accepts decreasing updates.
        init_estimator: Optional estimator used for encoding and randomized
            initialization; ``estimator`` is used by default.
        init_p: Initialization sampling fraction in ``(0, 1]``. Ignored when
            ``prev_estimate`` is supplied.
        rng: Random state used for initialization and advanced in place. The
            default instance is shared across calls.
        prev_estimate: Optional starting model, which also supplies the
            encoder.
        vdata: Optional raw validation observations.
        enc_data: Pre-encoded local chunks or Spark RDD. Takes precedence over
            ``data``.
        enc_vdata: Pre-encoded validation chunks. Takes precedence over
            ``vdata``.
        out: Text stream receiving progress messages.
        print_iter: Positive interval between progress messages.
        num_chunks: Number of local encoded chunks created from raw data.

    Returns:
        The fitted model with the greatest observed validation likelihood, or
        training likelihood when no validation data is supplied.

    Raises:
        ValueError: If both training representations are absent, or if a new
            model is requested with ``init_p`` outside ``(0, 1]``.
    """
    if data is None and enc_data is None:
        raise ValueError("Optimization called with empty data or enc_data.")

    est = estimator if init_estimator is None else init_estimator

    if prev_estimate is None:
        data_encoder = est.accumulator_factory().make().acc_to_encoder()
    else:
        data_encoder = prev_estimate.dist_to_encoder()

    if enc_data is None:
        assert data is not None
        enc_data = seq_encode(data=data, encoder=data_encoder, num_chunks=num_chunks)

    if prev_estimate is None:
        if not 0 < init_p <= 1:
            raise ValueError(
                f"Invalid init_p: {init_p}. It must be greater than 0 and "
                "less than or equal to 1."
            )

        # if isinstance(enc_data, pyspark.rdd.RDD):
        #     mm = initialize(data=data, estimator=est, rng=rng, p=p)
        # else:
        mm = seq_initialize(enc_data=enc_data, estimator=est, rng=rng, p=init_p)

    else:
        mm = prev_estimate

    _, old_ll = seq_log_density_sum(enc_data=enc_data, estimate=mm)

    if enc_vdata is None and vdata is not None:
        enc_vdata = seq_encode(vdata, data_encoder, num_chunks=num_chunks)

    if enc_vdata is not None:
        _, old_vll = seq_log_density_sum(enc_vdata, mm)
    else:
        old_vll = old_ll

    best_model = mm
    best_vll = old_vll

    for i in range(max_its):

        mm_next = seq_estimate(enc_data=enc_data, estimator=estimator, prev_estimate=mm)
        _, ll = seq_log_density_sum(enc_data=enc_data, estimate=mm_next)

        if enc_vdata is not None:
            _, vll = seq_log_density_sum(enc_vdata, mm_next)
        else:
            vll = ll

        dll = ll - old_ll

        if (dll >= 0) or (delta is None):
            mm = mm_next

        if (delta is not None) and (dll < delta):
            if enc_vdata is not None:
                out.write(
                    f"Iteration {i + 1}: ln[p_mat(Data|Model)]={ll:e}, "
                    f"ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]={dll:e}, "
                    f"ln[p_mat(Valid Data|Model)]={vll:e}\n"
                )
            else:
                out.write(
                    f"Iteration {i + 1}: ln[p_mat(Data|Model)]={ll:e}, "
                    f"ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]={dll:e}\n"
                )
            break

        if (i + 1) % print_iter == 0:
            if enc_vdata is not None:
                out.write(
                    f"Iteration {i + 1}: ln[p_mat(Data|Model)]={ll:e}, "
                    f"ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]={dll:e}, "
                    f"ln[p_mat(Valid Data|Model)]={vll:e}\n"
                )
            else:
                out.write(
                    f"Iteration {i + 1}: ln[p_mat(Data|Model)]={ll:e}, "
                    f"ln[p_mat(Data|Model)]-ln[p_mat(Data|PrevModel)]={dll:e}\n"
                )

        old_ll = ll

        if best_vll < vll:
            best_vll = vll
            best_model = mm

    return best_model


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def iterate(
    data: List[T],
    estimator: Optional[ParameterEstimator],
    max_its: int,
    prev_estimate: Optional[SequenceEncodableProbabilityDistribution] = None,
    init_p: float = 0.1,
    rng: Optional[RandomState] = RandomState(),
    out: IO = sys.stdout,
    enc_data: Optional[EncodedChunks] = None,
    init_estimator: Optional[ParameterEstimator] = None,
    print_iter: int = 1,
) -> SequenceEncodableProbabilityDistribution:
    """Perform a fixed number of EM updates and return the final estimate.

    Unlike :func:`optimize`, this helper performs no likelihood-based stopping
    or model selection. It optionally caches Spark encoded data and writes
    average elapsed time at the requested interval.

    Args:
        data: Raw training observations. Ignored when ``enc_data`` is supplied.
        estimator: Estimator used for updates, or ``None`` to use
            ``init_estimator``.
        max_its: Exact number of updates to perform when positive.
        prev_estimate: Optional starting model. Otherwise randomized
            initialization is performed.
        init_p: Initialization sampling fraction in ``(0, 1]``.
        rng: Random state used for initialization and advanced in place.
            ``None`` creates a fresh unseeded state; the default instance is
            shared across calls.
        out: Text stream receiving timing messages.
        enc_data: Pre-encoded local chunks or Spark RDD.
        init_estimator: Fallback estimator used when ``estimator`` is absent.
        print_iter: Positive interval between timing messages.

    Returns:
        The model after the requested updates.

    Raises:
        ValueError: If neither data representation is supplied, no estimator
            is available, or randomized initialization receives an invalid
            ``init_p``.
    """
    if data is None and enc_data is None:
        raise ValueError("Optimization called with empty data or enc_data.")

    active_estimator = estimator if estimator is not None else init_estimator
    if active_estimator is None:
        raise ValueError("estimator or init_estimator is required.")
    rng = rng if rng is not None else RandomState()

    if enc_data is None:
        encoder = active_estimator.accumulator_factory().make().acc_to_encoder()
        enc_data = seq_encode(data, encoder)

    if prev_estimate is None:
        if not 0 < init_p <= 1:
            raise ValueError(
                f"Invalid init_p: {init_p}. It must be greater than 0 and "
                "less than or equal to 1."
            )

        mm = seq_initialize(enc_data, active_estimator, rng, init_p)
    else:
        mm = prev_estimate

    if hasattr(enc_data, "cache"):
        cast(Any, enc_data).cache()

    t0 = time.time()
    for i in range(max_its):
        mm = seq_estimate(enc_data, active_estimator, mm)

        if (i + 1) % print_iter == 0:
            out.write(
                f"Iteration {i + 1}\t E[dT]="
                f"{(time.time() - t0) / float(i + 1):f}.\n"
            )

    return mm


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def hill_climb(
    data: List[T],
    vdata: List[T],
    estimator: ParameterEstimator,
    prev_estimate: SequenceEncodableProbabilityDistribution,
    max_its: int,
    metric_lambda: Callable[[List[T], SequenceEncodableProbabilityDistribution], float],
    best_estimate: Optional[SequenceEncodableProbabilityDistribution] = None,
    enc_data: Optional[EncodedChunks] = None,
    enc_vdata: Optional[EncodedChunks] = None,
    out: IO = sys.stdout,
    print_iter: int = 1,
) -> SequenceEncodableProbabilityDistribution:
    """Run fixed EM updates and retain the best metric-scoring model.

    Every proposal becomes the starting point for the next update, regardless
    of its score. The returned model maximizes ``metric_lambda`` on ``vdata``;
    validation log likelihood breaks exact score ties. Progress is written to
    ``out``.

    Args:
        data: Raw training observations, encoded only when ``enc_data`` is
            absent.
        vdata: Raw validation observations used by ``metric_lambda`` and
            encoded only when ``enc_vdata`` is absent.
        estimator: Estimator used for each EM update.
        prev_estimate: Starting model and source of encoders.
        max_its: Number of EM proposals to evaluate.
        metric_lambda: Callable receiving ``(vdata, model)`` and returning a
            scalar score to maximize.
        best_estimate: Optional incumbent model; defaults to ``prev_estimate``.
        enc_data: Pre-encoded training chunks.
        enc_vdata: Pre-encoded validation chunks.
        out: Text stream receiving progress messages.
        print_iter: Positive interval between progress messages.

    Returns:
        The incumbent or proposal with the best metric and likelihood tie
        break.
    """
    mm = prev_estimate

    if enc_data is None:
        data_enc = mm.dist_to_encoder().seq_encode(data)
        enc_data = [(len(data), data_enc)]
    if enc_vdata is None:
        vdata_enc = mm.dist_to_encoder().seq_encode(vdata)
        enc_vdata = [(len(vdata), vdata_enc)]

    best_model = prev_estimate if best_estimate is None else best_estimate
    _, best_ll = seq_log_density_sum(enc_vdata, best_model)
    best_score = metric_lambda(vdata, best_model)

    for i in range(max_its):

        mm_next = seq_estimate(enc_data, estimator, mm)

        _, next_ll = seq_log_density_sum(enc_vdata, mm_next)
        next_score = metric_lambda(vdata, mm_next)

        if (next_score > best_score) or (
            (next_score == best_score) and (best_ll < next_ll)
        ):
            best_model = mm_next
            best_ll = next_ll
            best_score = next_score

        if i % print_iter == 0:
            out.write(
                f"Iteration {i + 1}. LL={next_ll:f}, Best LL={best_ll:f}, "
                f"Best Score={best_score:f}\n"
            )

        mm = mm_next

    return best_model
