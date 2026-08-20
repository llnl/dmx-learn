"""Bayesian statistical distributions and fitting helpers.

The package-level helpers accept local sequences and PySpark RDDs. The scalar
fitting helpers additionally accept pandas DataFrames when the corresponding
accumulator implements the DataFrame protocol.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from typing import Any, Optional, TypeVar, Union, cast, overload

import numpy as np
import pandas as pd
from numpy.random import RandomState
from pyspark import RDD

from dmx.arithmetic import inf
from dmx.bstats.bernoulli import (
    BernoulliDistribution,
    BernoulliEstimator,
    BernoulliSampler,
)
from dmx.bstats.beta import BetaDistribution, BetaSampler
from dmx.bstats.catdirichlet import DictDirichletDistribution
from dmx.bstats.categorical import (
    CategoricalDistribution,
    CategoricalEstimator,
    CategoricalSampler,
)
from dmx.bstats.composite import (
    CompositeDistribution,
    CompositeEstimator,
    CompositeSampler,
)
from dmx.bstats.dirichlet import (
    DirichletDistribution,
    DirichletEstimator,
    DirichletSampler,
)
from dmx.bstats.dmvn import (
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    DiagonalGaussianSampler,
)
from dmx.bstats.dpm import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
    DirichletProcessMixtureSampler,
)
from dmx.bstats.exponential import (
    ExponentialDistribution,
    ExponentialEstimator,
    ExponentialSampler,
)
from dmx.bstats.gamma import GammaDistribution, GammaEstimator, GammaSampler
from dmx.bstats.gaussian import GaussianDistribution, GaussianEstimator, GaussianSampler
from dmx.bstats.geometric import (
    GeometricDistribution,
    GeometricEstimator,
    GeometricSampler,
)
from dmx.bstats.ignored import IgnoredDistribution, IgnoredEstimator, IgnoredSampler
from dmx.bstats.intrange import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
    IntegerCategoricalSampler,
)
from dmx.bstats.mixture import MixtureDistribution, MixtureEstimator, MixtureSampler
from dmx.bstats.mvngamma import (
    MultivariateNormalGammaDistribution,
    MultivariateNormalGammaSampler,
)
from dmx.bstats.normgamma import NormalGammaDistribution, NormalGammaSampler
from dmx.bstats.nulldist import NullDistribution, NullEstimator, NullSampler
from dmx.bstats.optional import OptionalDistribution, OptionalEstimator, OptionalSampler
from dmx.bstats.pdist import (
    DataFrameEncodableAccumulator,
    DataSequenceEncoder,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
)
from dmx.bstats.poisson import PoissonDistribution, PoissonEstimator, PoissonSampler
from dmx.bstats.sequence import SequenceDistribution, SequenceEstimator, SequenceSampler
from dmx.bstats.setdist import (
    BernoulliSetDistribution,
    BernoulliSetEstimator,
    BernoulliSetSampler,
)

__all__ = [
    "BernoulliDistribution",
    "BernoulliEstimator",
    "BernoulliSampler",
    "BernoulliSetDistribution",
    "BernoulliSetEstimator",
    "BernoulliSetSampler",
    "BetaDistribution",
    "BetaSampler",
    "CategoricalDistribution",
    "CategoricalEstimator",
    "CategoricalSampler",
    "CompositeDistribution",
    "CompositeEstimator",
    "CompositeSampler",
    "DiagonalGaussianDistribution",
    "DiagonalGaussianEstimator",
    "DiagonalGaussianSampler",
    "DictDirichletDistribution",
    "DirichletDistribution",
    "DirichletEstimator",
    "DirichletSampler",
    "DirichletProcessMixtureDistribution",
    "DirichletProcessMixtureEstimator",
    "DirichletProcessMixtureSampler",
    "ExponentialDistribution",
    "ExponentialEstimator",
    "ExponentialSampler",
    "GaussianDistribution",
    "GaussianEstimator",
    "GaussianSampler",
    "GammaDistribution",
    "GammaEstimator",
    "GammaSampler",
    "GeometricDistribution",
    "GeometricEstimator",
    "GeometricSampler",
    "IgnoredDistribution",
    "IgnoredEstimator",
    "IgnoredSampler",
    "IntegerCategoricalDistribution",
    "IntegerCategoricalEstimator",
    "IntegerCategoricalSampler",
    "MixtureDistribution",
    "MixtureEstimator",
    "MixtureSampler",
    "MultivariateNormalGammaDistribution",
    "MultivariateNormalGammaSampler",
    "NormalGammaDistribution",
    "NormalGammaSampler",
    "NullDistribution",
    "NullEstimator",
    "NullSampler",
    "OptionalDistribution",
    "OptionalEstimator",
    "OptionalSampler",
    "PoissonDistribution",
    "PoissonEstimator",
    "PoissonSampler",
    "SequenceDistribution",
    "SequenceEstimator",
    "SequenceSampler",
    "estimate",
    "seq_estimate",
    "initialize",
    "seq_log_density_sum",
    "seq_encode",
    "seq_log_density",
]

Model = ProbabilityDistribution[Any, Any, Any]
Estimator = ParameterEstimator[Any, Any, Any, Any]
EncodedChunk = tuple[int, Any]
ModelT = TypeVar("ModelT", bound=Model)


def load_models(x: str) -> Model:
    """Load a model from its constructor-like string representation.

    Args:
        x: String previously returned by :func:`dump_models`.

    Returns:
        The reconstructed probability distribution.
    """
    return cast(Model, eval(x, globals(), {"inf": inf}))  # pylint: disable=eval-used


def dump_models(x: Model) -> str:
    """Return the constructor-like string representation of a model.

    Args:
        x: Probability distribution to serialize.

    Returns:
        A string accepted by :func:`load_models`.
    """
    return str(x)


def _local_estimate(
    data: Sequence[Any],
    estimator: Estimator,
    prev_estimate: Optional[Model] = None,
) -> Model:
    """Estimate a model from a local sequence using the legacy local call form."""
    accumulator = estimator.accumulator_factory().make()
    for value in data:
        accumulator.update(value, 1.0, estimate=prev_estimate)
    stats_dict: dict[str, Any] = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    return estimator.estimate(accumulator.value())


def estimate(
    data: Union[RDD[Any], pd.DataFrame, Sequence[Any]],
    estimator: Estimator,
    prev_estimate: Optional[Model] = None,
) -> Model:
    """Estimate a distribution from unencoded observations.

    Args:
        data: A PySpark RDD, pandas DataFrame, or local sequence. DataFrames require
            an accumulator implementing the DataFrame update protocol.
        estimator: Estimator compatible with the observations.
        prev_estimate: Optional previous model used while accumulating statistics.

    Returns:
        The model produced from the accumulated sufficient statistics.
    """
    if isinstance(data, RDD):
        context = data.context
        factory = estimator.accumulator_factory()
        estimator_broadcast = context.broadcast(estimator)
        estimate_broadcast = context.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def accumulate_partition(
            split_index: int, values: Iterable[Any]
        ) -> Iterable[tuple[float, Any]]:
            del split_index
            local_accumulator = estimator_broadcast.value.accumulator_factory().make()
            count = 0.0
            local_estimate = pickle.loads(estimate_broadcast.value)
            for value in values:
                count += 1.0
                local_accumulator.update(value, 1.0, estimate=local_estimate)
            return iter([(count, local_accumulator.value())])

        partition_stats = data.mapPartitionsWithIndex(accumulate_partition, True)
        nobs = 0.0
        accumulator = factory.make()
        for partition_count, values in partition_stats.collect():
            nobs += partition_count
            accumulator.combine(values)
        return estimator.estimate(nobs, accumulator.value())

    if isinstance(data, pd.DataFrame):
        accumulator = cast(
            DataFrameEncodableAccumulator[Any, Any, Any],
            estimator.accumulator_factory().make(),
        )
        accumulator.df_update(data, np.ones(len(data)), estimate=prev_estimate)
        return estimator.estimate(None, accumulator.value())

    return _local_estimate(data, estimator, prev_estimate)


def initialize(
    data: Union[Sequence[Any], RDD[Any], pd.DataFrame],
    estimator: Estimator,
    rng: RandomState,
    p: float,
) -> Model:
    """Randomly initialize a distribution from observations.

    Args:
        data: A PySpark RDD, pandas DataFrame, or local sequence. DataFrames require
            an accumulator implementing the DataFrame initialization protocol.
        estimator: Estimator compatible with the observations.
        rng: NumPy random state used to initialize statistics.
        p: Probability or weight scale used by the existing input-specific sampler.

    Returns:
        The initialized probability distribution.
    """
    if isinstance(data, RDD):
        factory = estimator.accumulator_factory()
        context = data.context
        seeds = rng.randint(np.iinfo(np.int32).max, size=data.getNumPartitions())
        estimator_broadcast = context.broadcast(estimator)
        seeds_broadcast = context.broadcast(seeds)

        def initialize_partition(
            split_index: int, values: Iterable[Any]
        ) -> Iterable[tuple[float, Any]]:
            local_accumulator = estimator_broadcast.value.accumulator_factory().make()
            count = 0.0
            local_rng = np.random.RandomState(seeds_broadcast.value[split_index])
            for value in values:
                weight = 1.0 if local_rng.rand() <= p else 0.0
                count += weight
                local_accumulator.initialize(value, weight, local_rng)
            return iter([(count, local_accumulator.value())])

        partition_stats = data.mapPartitionsWithIndex(initialize_partition, True)
        nobs = 0.0
        accumulator = factory.make()
        for partition_count, values in partition_stats.collect():
            nobs += partition_count
            accumulator.combine(values)
        stats_dict: dict[str, Any] = {}
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)
        return estimator.estimate(nobs, accumulator.value())

    if isinstance(data, pd.DataFrame):
        dataframe_accumulator = cast(
            DataFrameEncodableAccumulator[Any, Any, Any],
            estimator.accumulator_factory().make(),
        )
        dataframe_accumulator.df_initialize(data, rng.rand(len(data)) * p, rng)
        return estimator.estimate(None, dataframe_accumulator.value())

    accumulator = estimator.accumulator_factory().make()
    for value in data:
        weight = 1.0 if rng.rand() <= p else 0.0
        accumulator.initialize(value, weight, rng)
    stats_dict = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    return estimator.estimate(accumulator.value())


@overload
def seq_encode(
    data: RDD[Any],
    model: Model,
    num_chunks: int = 1,
    chunk_size: Optional[int] = None,
) -> RDD[Any]: ...


@overload
def seq_encode(
    data: Sequence[Any],
    model: Model,
    num_chunks: int = 1,
    chunk_size: Optional[int] = None,
) -> list[EncodedChunk]: ...


def seq_encode(
    data: Union[Sequence[Any], RDD[Any]],
    model: Model,
    num_chunks: int = 1,
    chunk_size: Optional[int] = None,
) -> Union[RDD[Any], list[EncodedChunk]]:
    """Encode observations in chunks for vectorized model operations.

    Args:
        data: A PySpark RDD or local sequence of observations.
        model: Distribution whose sequence encoder is used.
        num_chunks: Number of interleaved chunks for local data.
        chunk_size: Approximate local chunk size; overrides ``num_chunks``.

    Returns:
        An RDD or list of ``(observation_count, encoded_data)`` pairs.
    """
    if isinstance(data, RDD):
        model_broadcast = data.context.broadcast(pickle.dumps(model, protocol=0))
        return (
            data.glom()
            .map(list)
            .map(
                lambda values: (
                    len(values),
                    pickle.loads(model_broadcast.value).seq_encode(values),
                )
            )
        )

    size = len(data)
    chunks = (
        int(np.ceil(float(size) / float(chunk_size)))
        if chunk_size is not None
        else num_chunks
    )
    encoded: list[EncodedChunk] = []
    for chunk_index in range(chunks):
        values = [data[index] for index in range(chunk_index, size, chunks)]
        encoded.append((len(values), model.seq_encode(values)))
    return encoded


def seq_estimate(
    enc_data: Union[RDD[Any], Sequence[EncodedChunk]],
    estimator: Estimator,
    prev_estimate: ModelT,
) -> ModelT:
    """Estimate a distribution from sequence-encoded observations.

    Args:
        enc_data: A PySpark RDD or local sequence of ``(count, encoded_data)`` pairs.
        estimator: Estimator compatible with the encoded observations.
        prev_estimate: Previous model used for vectorized statistic updates.

    Returns:
        The next model estimate, with the same static type as ``prev_estimate``.
    """
    if isinstance(enc_data, RDD):
        context = enc_data.context
        estimator_broadcast = context.broadcast(estimator)
        estimate_broadcast = context.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def accumulate_partition(
            split_index: int, chunks: Iterable[EncodedChunk]
        ) -> list[bytes]:
            del split_index
            accumulator = cast(
                SequenceEncodableAccumulator[Any, Any, Any],
                estimator_broadcast.value.accumulator_factory().make(),
            )
            count = 0.0
            local_estimate = pickle.loads(estimate_broadcast.value)
            for size, encoded in chunks:
                count += size
                accumulator.seq_update(encoded, np.ones(size), local_estimate)
            return [pickle.dumps((count, accumulator.value()), protocol=0)]

        partition_stats = enc_data.mapPartitionsWithIndex(
            accumulate_partition, True
        ).cache()
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        for serialized_stats in partition_stats.collect():
            partition_count, values = pickle.loads(serialized_stats)
            nobs += partition_count
            accumulator.combine(values)
        stats_dict: dict[str, Any] = {}
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)
        estimate_broadcast.destroy()
        estimator_broadcast.destroy()
        partition_stats.unpersist()
        enc_data.localCheckpoint()
        return cast(ModelT, estimator.estimate(nobs, accumulator.value()))

    accumulator = cast(
        SequenceEncodableAccumulator[Any, Any, Any],
        estimator.accumulator_factory().make(),
    )
    for size, encoded in enc_data:
        accumulator.seq_update(encoded, np.ones(size), prev_estimate)
    stats_dict = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    return cast(ModelT, estimator.estimate(accumulator.value()))


# ``estimate`` is a long-standing public keyword for the model being scored.
def seq_log_density(  # pylint: disable=redefined-outer-name
    enc_data: Union[RDD[Any], Sequence[EncodedChunk]],
    estimate: Union[Model, Sequence[Model]],
    is_list: bool = False,
) -> list[np.ndarray[Any, Any]]:
    """Evaluate vectorized log densities for sequence-encoded observations.

    Args:
        enc_data: A PySpark RDD or local sequence of encoded chunks.
        estimate: One distribution, or a sequence when ``is_list`` is true.
        is_list: Interpret ``estimate`` as a sequence and stack model results.

    Returns:
        One NumPy array per chunk, with models stacked first when requested.
    """
    if isinstance(enc_data, RDD):
        estimate_broadcast = enc_data.context.broadcast(
            pickle.dumps(estimate, protocol=0)
        )

        def score_partition(
            chunks: Iterable[EncodedChunk],
        ) -> list[np.ndarray[Any, Any]]:
            local_estimate = pickle.loads(estimate_broadcast.value)
            if is_list:
                models = cast(Sequence[Model], local_estimate)
                return [
                    np.asarray([model.seq_log_density(encoded) for model in models])
                    for _size, encoded in chunks
                ]
            model = cast(Model, local_estimate)
            return [model.seq_log_density(encoded) for _size, encoded in chunks]

        return enc_data.mapPartitions(score_partition).collect()

    if is_list:
        models = cast(Sequence[Model], estimate)
        return [
            np.asarray([model.seq_log_density(encoded) for model in models])
            for _size, encoded in enc_data
        ]
    model = cast(Model, estimate)
    return [model.seq_log_density(encoded) for _size, encoded in enc_data]


def seq_log_density_sum(  # pylint: disable=redefined-outer-name
    enc_data: Union[RDD[Any], Sequence[EncodedChunk]],
    estimate: Model,
) -> tuple[float, float]:
    """Sum vectorized log densities over sequence-encoded observations.

    Args:
        enc_data: A PySpark RDD or local sequence of encoded chunks.
        estimate: Distribution compatible with the encoded observations.

    Returns:
        The total observation count and summed log density.
    """
    if isinstance(enc_data, RDD):
        estimate_broadcast = enc_data.context.broadcast(
            pickle.dumps(estimate, protocol=0)
        )

        def sum_partition(
            chunks: Iterable[EncodedChunk],
        ) -> list[tuple[float, float]]:
            total = 0.0
            count = 0.0
            local_estimate = cast(Model, pickle.loads(estimate_broadcast.value))
            for size, encoded in chunks:
                total += float(local_estimate.seq_log_density(encoded).sum())
                count += size
            return [(count, total)]

        return enc_data.mapPartitions(sum_partition).reduce(
            lambda left, right: (left[0] + right[0], left[1] + right[1])
        )

    return float(sum(size for size, _encoded in enc_data)), float(
        sum(estimate.seq_log_density(encoded).sum() for _size, encoded in enc_data)
    )
