"""Local variational estimation helpers for Bayesian models.

Initialization uses the established per-observation Bernoulli subsample and
delegates randomized allocation to each estimator accumulator. ``optimize``
reports data log likelihood (``LL``), its change (``dLL``), model-prior and
entropy terms (``MLL``/``dMLL``), and validation log likelihood (``VLL``).
For a DPM, ``LL`` is the block variational objective returned by its sequence
scorer. A proposed update is retained only when LL does not decrease, and the
returned model is the retained estimate with the best validation score.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from typing import IO, Any, Optional, TypeVar, cast

import numpy as np
from numpy.random import RandomState

from dmx.bstats import (
    initialize,
    seq_encode,
    seq_estimate,
    seq_log_density,
    seq_log_density_sum,
)
from dmx.bstats.pdist import ParameterEstimator, ProbabilityDistribution

T = TypeVar("T")
Model = ProbabilityDistribution[Any, Any, Any]
Estimator = ParameterEstimator[Any, Any, Any, Any]
EncodedChunk = tuple[int, Any]
EncodedChunks = Sequence[EncodedChunk]


def _random_state(rng: Optional[RandomState]) -> RandomState:
    """Return the supplied random state or a fresh local state."""
    return RandomState() if rng is None else rng


def _validate_init_p(init_p: float) -> None:
    """Validate the initialization sampling proportion."""
    if not 0.0 < init_p <= 1.0:
        raise ValueError("init_p must be greater than 0 and at most 1.")


def empirical_kl_divergence(
    dist1: Model, dist2: Model, enc_data: EncodedChunks
) -> tuple[float, int, int]:
    """Estimate KL divergence on encoded support shared by two models.

    Returns the discrete KL estimate and the number of non-finite log scores
    produced by each distribution. At least one jointly finite score is
    required.

    Args:
        dist1: Distribution defining the empirical reference probabilities.
        dist2: Distribution compared with ``dist1``.
        enc_data: Sequence of ``(count, encoded_data)`` chunks accepted by both
            distributions.

    Returns:
        Tuple containing the empirical divergence, the number of non-finite
        scores from ``dist1``, and the number from ``dist2``.

    Raises:
        ValueError: If no observation has finite log-density under both models.
    """
    chunks = seq_log_density(enc_data, estimate=(dist1, dist2), is_list=True)
    likelihoods = np.hstack(chunks)
    first = likelihoods[0, :]
    second = likelihoods[1, :]
    valid_first = np.isfinite(first)
    valid_second = np.isfinite(second)
    shared = valid_first & valid_second
    if not np.any(shared):
        raise ValueError("Empirical KL requires at least one jointly finite score.")

    first_probability = np.exp(first[shared] - np.max(first[shared]))
    first_probability /= first_probability.sum()
    second_probability = np.exp(second[shared] - np.max(second[shared]))
    second_probability /= second_probability.sum()
    divergence = np.sum(
        first_probability * (np.log(first_probability) - np.log(second_probability))
    )
    return (
        float(divergence),
        int(np.count_nonzero(~valid_first)),
        int(np.count_nonzero(~valid_second)),
    )


def k_fold_split_index(sz: int, k: int, rng: RandomState) -> np.ndarray[Any, Any]:
    """Return a randomized fold identifier for each observation index.

    Args:
        sz: Number of observations.
        k: Number of folds.
        rng: Random state used to permute observations.

    Returns:
        Integer array of shape ``(sz,)`` with fold identifiers in ``[0, k)``.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    sorted_indices = np.argsort(rng.rand(sz))
    result = np.zeros(sz, dtype=int)
    for fold in range(k):
        result[sorted_indices[fold:sz:k]] = fold
    return result


def partition_data_index(
    sz: int, pvec: Sequence[float] | np.ndarray[Any, Any], rng: RandomState
) -> list[np.ndarray[Any, Any]]:
    """Return randomized index partitions with proportions from ``pvec``.

    Partition boundaries are rounded cumulative proportions of ``sz``. The
    proportions are used as supplied and need not sum to one.

    Args:
        sz: Number of observation indices to partition.
        pvec: Proportion assigned to each output partition.
        rng: Random state used to permute the indices.

    Returns:
        One one-dimensional index array per requested proportion.
    """
    sorted_indices = np.argsort(rng.rand(sz))
    result: list[np.ndarray[Any, Any]] = []
    total = 0.0
    previous = 0
    for proportion in pvec:
        next_index = int(round(sz * (total + float(proportion)), 0))
        result.append(sorted_indices[previous:next_index])
        total += float(proportion)
        previous = next_index
    return result


def partition_data(
    data: Sequence[T],
    pvec: Sequence[float] | np.ndarray[Any, Any],
    rng: RandomState,
) -> list[list[T]]:
    """Randomly partition observations with proportions from ``pvec``.

    Args:
        data: Observations to partition.
        pvec: Proportion assigned to each output partition.
        rng: Random state used to permute observations.

    Returns:
        Lists of observations corresponding to :func:`partition_data_index`.
    """
    indices = partition_data_index(len(data), pvec, rng)
    return [[data[int(index)] for index in partition] for partition in indices]


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def best_of(
    data: Sequence[T],
    vdata: Sequence[T],
    est: Estimator,
    trials: int,
    max_its: int,
    init_p: float,
    delta: Optional[float],
    rng: Optional[RandomState],
    init_estimator: Optional[Estimator] = None,
    enc_data: Optional[EncodedChunks] = None,
    enc_vdata: Optional[EncodedChunks] = None,
    out: IO[str] = sys.stdout,
    print_iter: int = 1,
) -> tuple[float, Model]:
    """Run randomized variational fits and return the best validation model.

    Each trial initializes a model from a Bernoulli subsample of the training
    data. Updates stop when training log-likelihood improvement is below
    ``delta``; ``None`` disables both that stopping rule and rejection of
    likelihood-decreasing updates.

    Args:
        data: Raw training observations.
        vdata: Raw validation observations used to select the returned model.
        est: Estimator used for every variational update.
        trials: Number of independent randomized initializations.
        max_its: Maximum updates per trial.
        init_p: Probability that an observation participates in initialization.
        delta: Minimum training log-likelihood improvement, or ``None`` to run
            every update.
        rng: Random state for initialization, or ``None`` for a fresh state.
        init_estimator: Optional estimator used only during initialization.
        enc_data: Optional pre-encoded training chunks of the form
            ``(observation_count, encoded_data)``.
        enc_vdata: Optional pre-encoded validation chunks in the same form.
        out: Text stream receiving progress messages.
        print_iter: Emit training progress every this many updates.

    Returns:
        Best validation log-likelihood and its fitted model.

    Raises:
        ValueError: If ``trials`` is not positive or ``init_p`` is outside
            ``(0, 1]``.
        RuntimeError: If no trial produces a selectable model.
    """
    _validate_init_p(init_p)
    if trials <= 0:
        raise ValueError("trials must be positive.")
    active_rng = _random_state(rng)
    initialization_estimator = est if init_estimator is None else init_estimator
    best_likelihood = float(-np.inf)
    best_model: Optional[Model] = None

    for trial in range(trials):
        model = initialize(data, initialization_estimator, active_rng, init_p)
        training = list(enc_data) if enc_data is not None else seq_encode(data, model)
        validation = (
            list(enc_vdata) if enc_vdata is not None else seq_encode(vdata, model)
        )
        _, old_likelihood = seq_log_density_sum(training, model)
        for iteration in range(max_its):
            proposed = seq_estimate(training, est, model)
            _, likelihood = seq_log_density_sum(training, proposed)
            change = likelihood - old_likelihood
            if (iteration + 1) % print_iter == 0:
                out.write(
                    f"Iteration {iteration + 1}. LL={likelihood:f}, "
                    f"delta LL={change:e}\n"
                )
            if change >= 0.0 or delta is None:
                model = proposed
            if delta is not None and change < delta:
                break
            old_likelihood = likelihood

        _, validation_likelihood = seq_log_density_sum(validation, model)
        out.write(f"Trial {trial + 1}. VLL={validation_likelihood:f}\n")
        if validation_likelihood > best_likelihood:
            best_model = model
            best_likelihood = validation_likelihood

    if best_model is None:
        raise RuntimeError("No model was estimated.")
    return best_likelihood, best_model


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def optimize(
    data: Sequence[T],
    estimator: Estimator,
    max_its: int = 10,
    delta: Optional[float] = 1.0e-6,
    init_estimator: Optional[Estimator] = None,
    init_p: float = 0.1,
    rng: Optional[RandomState] = None,
    prev_estimate: Optional[Model] = None,
    vdata: Optional[Sequence[T]] = None,
    enc_data: Optional[EncodedChunks] = None,
    enc_vdata: Optional[EncodedChunks] = None,
    out: IO[str] = sys.stdout,
    print_iter: int = 1,
) -> Model:
    """Optimize locally until LL improvement is below ``delta``.

    Progress lines expose ``LL``, ``dLL``, ``MLL``, ``dMLL``, and ``VLL``;
    a line prefixed by ``Terminating`` records convergence before returning the
    best retained validation model.

    Args:
        data: Raw training observations.
        estimator: Estimator used for initialization and updates. It must also
            provide ``model_log_density(model)`` for the reported ``MLL``.
        max_its: Maximum number of variational updates.
        delta: Minimum training log-likelihood improvement, or ``None`` to run
            every update and retain likelihood-decreasing proposals.
        init_estimator: Optional estimator used only during initialization.
        init_p: Probability that an observation participates in initialization.
        rng: Random state for initialization, or ``None`` for a fresh state.
        prev_estimate: Existing model to update instead of initializing one.
        vdata: Validation observations, defaulting to ``data``.
        enc_data: Optional pre-encoded training chunks of the form
            ``(observation_count, encoded_data)``.
        enc_vdata: Optional pre-encoded validation chunks in the same form.
        out: Text stream receiving progress messages.
        print_iter: Emit progress every this many updates.

    Returns:
        Model with the highest validation log-likelihood among retained
        estimates.

    Raises:
        ValueError: If initialization is needed and ``init_p`` is outside
            ``(0, 1]``.
        AttributeError: If ``estimator`` has no ``model_log_density`` method.
    """
    active_rng = _random_state(rng)
    initialization_estimator = estimator if init_estimator is None else init_estimator
    if prev_estimate is None:
        _validate_init_p(init_p)
        model = initialize(data, initialization_estimator, active_rng, init_p)
    else:
        model = prev_estimate
    validation_data = data if vdata is None else vdata
    training = list(enc_data) if enc_data is not None else seq_encode(data, model)
    validation = (
        list(enc_vdata) if enc_vdata is not None else seq_encode(validation_data, model)
    )

    _, old_validation_likelihood = seq_log_density_sum(validation, model)
    _, old_likelihood = seq_log_density_sum(training, model)
    model_log_density = cast(
        Callable[[Model], float], getattr(estimator, "model_log_density")
    )
    model_likelihood = model_log_density(model)
    best_model = model
    best_likelihood = old_validation_likelihood

    with np.errstate(divide="ignore"):
        for iteration in range(max_its):
            proposed = seq_estimate(training, estimator, model)
            old_model_likelihood = model_likelihood
            model_likelihood = model_log_density(proposed)
            _, validation_likelihood = seq_log_density_sum(validation, proposed)
            _, likelihood = seq_log_density_sum(training, proposed)
            model_change = model_likelihood - old_model_likelihood
            change = likelihood - old_likelihood

            if change >= 0.0 or delta is None:
                model = proposed
            if delta is not None and change < delta:
                out.write(
                    f"Terminating {iteration + 1}. LL={likelihood:f}, "
                    f"dLL={change:e}, MLL={model_likelihood:f}, "
                    f"dMLL={model_change:e}, VLL={validation_likelihood:f}\n"
                )
                break
            if (iteration + 1) % print_iter == 0:
                out.write(
                    f"Iteration {iteration + 1}. LL={likelihood:f}, "
                    f"dLL={change:e}, MLL={model_likelihood:f}, "
                    f"dMLL={model_change:e}, VLL={validation_likelihood:f}\n"
                )
            old_likelihood = likelihood
            old_validation_likelihood = validation_likelihood
            if best_likelihood < validation_likelihood:
                best_likelihood = validation_likelihood
                best_model = model

    return best_model


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def iterate(
    data: Sequence[T] | EncodedChunks,
    estimator: Estimator,
    max_its: int,
    prev_estimate: Optional[Model] = None,
    init_p: float = 0.1,
    rng: Optional[RandomState] = None,
    out: IO[str] = sys.stdout,
    is_encoded: bool = False,
    init_estimator: Optional[Estimator] = None,
    print_iter: int = 1,
) -> Model:
    """Run exactly ``max_its`` updates and report mean iteration time.

    Args:
        data: Raw observations, or encoded chunks when ``is_encoded`` is true.
        estimator: Estimator used for initialization and every update.
        max_its: Number of variational updates to run.
        prev_estimate: Existing model to update instead of initializing one.
        init_p: Probability that an observation participates in initialization.
        rng: Random state for initialization, or ``None`` for a fresh state.
        out: Text stream receiving timing messages.
        is_encoded: Whether ``data`` already contains encoded chunks.
        init_estimator: Optional estimator used only during initialization.
        print_iter: Emit mean elapsed time every this many updates.

    Returns:
        Model produced by the final update.

    Raises:
        ValueError: If encoded data is supplied without ``prev_estimate``, or
            initialization is needed and ``init_p`` is outside ``(0, 1]``.
    """
    active_rng = _random_state(rng)
    initialization_estimator = estimator if init_estimator is None else init_estimator
    if prev_estimate is None:
        _validate_init_p(init_p)
        if is_encoded:
            raise ValueError("prev_estimate is required when data is encoded.")
        model = initialize(data, initialization_estimator, active_rng, init_p)
    else:
        model = prev_estimate
    encoded = (
        cast(EncodedChunks, data)
        if is_encoded
        else seq_encode(cast(Sequence[Any], data), model)
    )
    if hasattr(encoded, "cache"):
        cast(Any, encoded).cache()

    start = time.time()
    for iteration in range(max_its):
        model = seq_estimate(encoded, estimator, model)
        if (iteration + 1) % print_iter == 0:
            elapsed = (time.time() - start) / float(iteration + 1)
            out.write(f"Iteration {iteration + 1}\t E[dT]={elapsed:f}.\n")
    return model


# Keep the current public call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def hill_climb(
    data: Sequence[T],
    vdata: Sequence[T],
    estimator: Estimator,
    prev_estimate: Model,
    max_its: int,
    metric_lambda: Callable[[Sequence[T], Model], float],
    best_estimate: Optional[Model] = None,
    enc_data: Optional[EncodedChunks] = None,
    enc_vdata: Optional[EncodedChunks] = None,
    out: IO[str] = sys.stdout,
    print_iter: int = 1,
) -> Model:
    """Return the update with best metric, breaking ties by validation LL.

    Args:
        data: Raw training observations.
        vdata: Raw validation observations passed to ``metric_lambda``.
        estimator: Estimator used for each update.
        prev_estimate: Model from which to begin updating.
        max_its: Number of updates to evaluate.
        metric_lambda: Callable returning a score to maximize for a validation
            data and model pair.
        best_estimate: Optional incumbent model included in the comparison.
        enc_data: Optional pre-encoded training chunks of the form
            ``(observation_count, encoded_data)``.
        enc_vdata: Optional pre-encoded validation chunks in the same form.
        out: Text stream receiving progress messages.
        print_iter: Emit progress every this many updates.

    Returns:
        Model with the largest metric value, using validation log-likelihood to
        break exact metric ties.
    """
    model = prev_estimate
    training = (
        list(enc_data)
        if enc_data is not None
        else [(len(data), model.seq_encode(data))]
    )
    validation = (
        list(enc_vdata)
        if enc_vdata is not None
        else [(len(vdata), model.seq_encode(vdata))]
    )
    best_model = prev_estimate if best_estimate is None else best_estimate
    _, best_likelihood = seq_log_density_sum(validation, best_model)
    best_score = metric_lambda(vdata, best_model)

    for iteration in range(max_its):
        proposed = seq_estimate(training, estimator, model)
        _, next_likelihood = seq_log_density_sum(validation, proposed)
        next_score = metric_lambda(vdata, proposed)
        if next_score > best_score or (
            next_score == best_score and best_likelihood < next_likelihood
        ):
            best_model = proposed
            best_likelihood = next_likelihood
            best_score = next_score
        if iteration % print_iter == 0:
            out.write(
                f"Iteration {iteration + 1}. LL={next_likelihood:f}, "
                f"Best LL={best_likelihood:f}, Best Score={best_score:f}\n"
            )
        model = proposed
    return best_model
