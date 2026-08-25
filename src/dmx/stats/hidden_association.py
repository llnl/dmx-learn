r"""Provide hidden-association models for paired bags of weighted values.

An observation is ``(source, target)``.  Each bag is represented by
``[(value, count), ...]`` with values of generic type ``T``.  Write source
counts as :math:`c_v`, target counts as :math:`d_y`,
:math:`M=\sum_v c_v`, and :math:`N=\sum_y d_y`.  For each target occurrence, a
latent assignment :math:`A` selects a source value with
:math:`P(A=v\mid source)=c_v/M`, then ``cond_dist`` generates the target from
:math:`q(y\mid v)`.  Marginalizing the assignments gives

.. math::

   \log p(source,target) = \log p_G(source) + \log p_N(N)
     + \sum_y d_y\log\left(\sum_v \frac{c_v}{M}q(y\mid v)\right).

``given_dist`` supplies :math:`p_G` and ``len_dist`` supplies :math:`p_N`.
The accumulator computes posterior assignment weights and passes expected
weighted pairs to the conditional-distribution accumulator.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np

from dmx.arithmetic import exp, maxrandint
from dmx.stats.conditional import (
    ConditionalDistribution,
    ConditionalDistributionAccumulator,
    ConditionalDistributionAccumulatorFactory,
    ConditionalDistributionEstimator,
)
from dmx.stats.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDistribution,
    NullEstimator,
)
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.optsutil import count_by_value

T = TypeVar("T")  ### value data type
SS1 = TypeVar("SS1")  ### Data type for suff stats of conditional
SS2 = TypeVar("SS2")  ### Data type for suff stats of given
SS3 = TypeVar("SS3")  ### Data type for suff stats of length


class HiddenAssociationDistribution(SequenceEncodableProbabilityDistribution):
    """Model associations between source and target bags of generic values.

    The latent source assignment for each target occurrence is marginalized in
    density evaluation and sampling returns grouped target counts.

    Attributes:
        cond_dist (ConditionalDistribution): ConditionalDistribution defining
            target-value distributions conditioned on source values.
        given_dist (SequenceEncodableProbabilityDistribution): Distribution for the
            grouped source bag. Defaults to ``NullDistribution``.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution for the length
            of the target bag, measured by its total count.
        name (Optional[str]): Name for object instance.
        keys (Tuple[Optional[str], Optional[str]]): Compatibility keys forwarded to the
            estimator and accumulator factory.

    """

    def __init__(
        self,
        cond_dist: ConditionalDistribution,
        given_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        len_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
    ) -> None:
        """Initialize a generic hidden-association distribution.

        Args:
            cond_dist (ConditionalDistribution): ConditionalDistribution defining
                target-value distributions conditioned on source values.
            given_dist (Optional[SequenceEncodableProbabilityDistribution]):
                Distribution for the previous set. Must
                be compatible with Tuple[T, float].
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution
                for the length of the observed
                emission. (Second set output).
            name (Optional[str]): Name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Compatibility keys
                forwarded to estimation objects.

        """
        super().__init__()
        self.cond_dist = cond_dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.given_dist = given_dist if given_dist is not None else NullDistribution()
        self.name = name
        self.keys = keys if keys is not None else (None, None)

    def __str__(self) -> str:
        """Return an evaluable representation of the distribution."""
        s1 = repr(self.cond_dist)
        s2 = repr(self.given_dist)
        s3 = repr(self.len_dist)
        s4 = repr(self.name)
        s5 = repr(self.keys)

        return (
            f"HiddenAssociationDistribution({s1}, given_dist={s2}, len_dist={s3}, "
            f"name={s4}, keys={s5})"
        )

    def density(self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]) -> float:
        """Evaluate the density of a paired source and target bag."""
        return float(exp(self.log_density(x)))

    def log_density(
        self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]
    ) -> float:
        """Evaluate the log density after marginalizing source assignments."""
        rv = 0.0
        nn = 0.0
        for x1, c1 in x[1]:
            cc = 0.0  # count for counts in given
            nn += c1
            ll = -np.inf
            for x0, c0 in x[0]:
                tt = self.cond_dist.log_density((x0, x1)) + math.log(c0)
                cc += c0

                if tt == -np.inf:
                    continue

                if ll > tt:
                    ll = math.log1p(math.exp(tt - ll)) + ll
                else:
                    ll = math.log1p(math.exp(ll - tt)) + tt

            ll -= math.log(cc)
            rv += ll * c1

        rv += self.given_dist.log_density(x[0])
        rv += self.len_dist.log_density(nn)

        return float(rv)

    def seq_log_density(self, x: "HiddenAssociationEncodedDataSequence") -> np.ndarray:
        """Evaluate log densities for an encoded batch of paired bags."""
        if not isinstance(x, HiddenAssociationEncodedDataSequence):
            raise TypeError("Requires HiddenAssociationEncodedDataSequence.")

        return np.asarray([self.log_density(xx) for xx in x.data])

    def sampler(self, seed: Optional[int] = None) -> "HiddenAssociationSampler":
        """Create a sampler for paired source and target bags."""
        return HiddenAssociationSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "HiddenAssociationEstimator":
        """Create an estimator from the three component estimators.

        The ``pseudo_count`` argument is accepted for protocol compatibility but
        is not forwarded by this implementation.
        """
        return HiddenAssociationEstimator(
            cond_estimator=self.cond_dist.estimator(),
            given_estimator=self.given_dist.estimator(),
            len_estimator=self.len_dist.estimator(),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "HiddenAssociationDataEncoder":
        """Create the pass-through encoder used by vectorized operations."""
        return HiddenAssociationDataEncoder()


class HiddenAssociationSampler(DistributionSampler):
    """Sample paired bags or target bags conditional on a source bag."""

    def __init__(
        self, dist: HiddenAssociationDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize a generic hidden-association sampler."""
        if isinstance(dist.given_dist, NullDistribution):
            raise RuntimeError(
                "HiddenAssociationSampler requires attribute dist.given_dist."
            )
        if isinstance(dist.len_dist, NullDistribution):
            raise RuntimeError(
                "HiddenAssociationSampler requires attribute dist.len_dist."
            )

        super().__init__(dist, seed)
        self.dist = dist

        self.cond_sampler = dist.cond_dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.idx_sampler = np.random.RandomState(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.dist.len_dist.sampler(
            seed=self.rng.randint(0, maxrandint)
        )
        self.given_sampler = self.dist.given_dist.sampler(
            seed=self.rng.randint(0, maxrandint)
        )

    def _sample_single(
        self,
    ) -> Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]]:
        prev_obs = cast(List[Tuple[Any, float]], self.given_sampler.sample())
        cnt = int(self.len_sampler.sample())
        rng = np.random.RandomState(self.idx_sampler.randint(0, maxrandint))
        rv: List[Any] = []
        pp = np.asarray([u[1] for u in prev_obs], dtype=float)
        pp /= pp.sum()

        for i in rng.choice(len(prev_obs), p=pp, size=cnt):
            rv.append(self.cond_sampler.sample_given(prev_obs[int(i)][0]))

        counted = [(k, float(v)) for k, v in count_by_value(rv).items()]

        return prev_obs, counted

    def sample(self, size: Optional[int] = None) -> Union[
        Sequence[Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]]],
        Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]],
    ]:
        """Draw one paired-bag observation or a batch of observations."""
        if size is None:
            return self._sample_single()

        return [self._sample_single() for i in range(size)]

    def sample_given(self, x: List[Tuple[T, float]]) -> List[Tuple[Any, float]]:
        """Draw a grouped target bag conditional on source bag ``x``."""
        cnt = int(self.len_sampler.sample())
        rng = np.random.RandomState(self.idx_sampler.randint(0, maxrandint))
        rv: List[Any] = []
        pp = np.asarray([u[1] for u in x], dtype=float)
        pp /= pp.sum()

        for i in rng.choice(len(x), p=pp, size=cnt):
            rv.append(self.cond_sampler.sample_given(x[int(i)][0]))

        return [(k, float(v)) for k, v in count_by_value(rv).items()]


class HiddenAssociationAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate expected latent assignments and component statistics.

    The sufficient statistic is ``(conditional, source, length)``.  During an
    update, posterior assignment probabilities distribute each target count
    across source values before updating the conditional component.
    """

    def __init__(
        self,
        cond_acc: ConditionalDistributionAccumulator,
        given_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
        size_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
    ) -> None:
        """Initialize a generic hidden-association accumulator."""
        self.cond_accumulator = cond_acc
        self.given_accumulator = (
            given_acc if given_acc is not None else NullAccumulator()
        )
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        key_pair = keys if keys is not None else (None, None)
        self.init_key, self.trans_key = key_pair
        self.name = name

    def update(
        self,
        x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]],
        weight: float,
        estimate: HiddenAssociationDistribution,
    ) -> None:
        """Add a weighted paired-bag observation using posterior assignments."""
        nn = 0.0
        pv = np.zeros(len(x[0]))

        for x1, c1 in x[1]:
            cc = 0.0
            nn += c1
            ll = -np.inf

            for i, (x0, c0) in enumerate(x[0]):
                tt = estimate.cond_dist.log_density((x0, x1)) + math.log(c0)
                cc += c0
                pv[i] = tt

                if tt == -np.inf:
                    continue

                if ll > tt:
                    ll = math.log1p(math.exp(tt - ll)) + ll
                else:
                    ll = math.log1p(math.exp(ll - tt)) + tt

            pv -= ll
            np.exp(pv, out=pv)

            for i, (x0, c0) in enumerate(x[0]):
                self.cond_accumulator.update(
                    (x0, x1), pv[i] * c1 * weight, estimate.cond_dist
                )

        if self.given_accumulator is not None:
            given_dist = None if estimate is None else estimate.given_dist
            self.given_accumulator.update(x[0], weight, given_dist)

        if self.size_accumulator is not None:
            len_dist = None if estimate is None else estimate.len_dist
            self.size_accumulator.update(nn, weight, len_dist)

    def initialize(
        self,
        x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]],
        weight: float,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize statistics with random source assignments per target value."""
        w = rng.dirichlet(np.ones(len(x[0])), size=len(x[1]))
        nn = 0.0
        for j, (x1, c1) in enumerate(x[1]):
            nn += c1
            for i, (x0, _c0) in enumerate(x[0]):
                self.cond_accumulator.initialize((x0, x1), w[j, i] * weight, rng)

        if self.given_accumulator is not None:
            self.given_accumulator.initialize(x[0], weight, rng)

        if self.size_accumulator is not None:
            self.size_accumulator.initialize(nn, weight, rng)

    def seq_initialize(
        self,
        x: "HiddenAssociationEncodedDataSequence",
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize statistics from an encoded weighted batch."""
        for i, xx in enumerate(x.data):
            self.initialize(xx, weights[i], rng)

    def seq_update(
        self,
        x: "HiddenAssociationEncodedDataSequence",
        weights: np.ndarray,
        estimate: HiddenAssociationDistribution,
    ) -> None:
        """Add an encoded weighted batch using posterior assignments."""
        for xx, ww in zip(x.data, weights):
            self.update(xx, ww, estimate)

    def combine(
        self, suff_stat: Tuple[SS1, Optional[SS2], Optional[SS3]]
    ) -> "HiddenAssociationAccumulator":
        """Merge conditional, source, and length sufficient statistics."""
        cond_acc, given_acc, size_acc = suff_stat

        cast(Any, self.cond_accumulator).combine(cond_acc)
        self.given_accumulator.combine(given_acc)
        self.size_accumulator.combine(size_acc)

        return self

    def value(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Return conditional, source, and target-length statistics."""
        return (
            self.cond_accumulator.value(),
            self.given_accumulator.value(),
            self.size_accumulator.value(),
        )

    def from_value(
        self, x: Tuple[SS1, Optional[SS2], Optional[SS3]]
    ) -> "HiddenAssociationAccumulator":
        """Replace all component sufficient statistics with ``x``."""
        cond_acc, given_acc, size_acc = x

        cast(Any, self.cond_accumulator).from_value(cond_acc)
        self.given_accumulator.from_value(given_acc)
        self.size_accumulator.from_value(size_acc)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge keyed component statistics into ``stats_dict``."""
        self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace keyed component statistics from ``stats_dict``."""
        self.size_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "HiddenAssociationDataEncoder":
        """Create the pass-through encoder used by this accumulator."""
        return HiddenAssociationDataEncoder()


class HiddenAssociationAccumulatorFactory(StatisticAccumulatorFactory):
    """Create generic hidden-association accumulators."""

    def __init__(
        self,
        cond_factory: ConditionalDistributionAccumulatorFactory,
        given_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
        len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
    ) -> None:
        """Initialize a hidden-association accumulator factory."""
        self.cond_factory = cond_factory
        self.given_factory = (
            given_factory if given_factory is not None else NullAccumulatorFactory()
        )
        self.len_factory = (
            len_factory if len_factory is not None else NullAccumulatorFactory()
        )
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def make(self) -> "HiddenAssociationAccumulator":
        """Create a new accumulator with fresh component accumulators."""
        return HiddenAssociationAccumulator(
            self.cond_factory.make(),
            self.given_factory.make(),
            self.len_factory.make(),
            self.name,
            self.keys,
        )


class HiddenAssociationEstimator(ParameterEstimator):
    """Estimate a generic hidden-association distribution.

    The conditional, source-bag, and target-length components are estimated
    independently from their corresponding sufficient statistics.  Assignment
    expectations are computed earlier by accumulator updates.

    Attributes:
        cond_estimator (ConditionalDistributionEstimator): Estimator for the conditional
            emission of values in
            set 2 given states.
        given_estimator (ParameterEstimator): Estimator for the given values. Should be
            compatible with
            Tuple[T, float] where T is the type for the values.
        len_estimator (ParameterEstimator): Estimator for the length of the observed set
            2 values.
        pseudo_count (Optional[float]): Kept for consistency.
        name (Optional[str]): Set name for object instance.
        keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for weights and
            transitions.

    """

    def __init__(
        self,
        cond_estimator: ConditionalDistributionEstimator,
        given_estimator: Optional[ParameterEstimator] = NullEstimator(),
        len_estimator: Optional[ParameterEstimator] = NullEstimator(),
        pseudo_count: Optional[float] = None,
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
    ) -> None:
        """Initialize a generic hidden-association estimator.

        Args:
            cond_estimator (ConditionalDistributionEstimator): Estimator for the
                conditional emission of values in
                set 2 given states.
            given_estimator (Optional[ParameterEstimator]): Estimator for the given
                values. Should be compatible with
                Tuple[T, float] where T is the type for the values.
            len_estimator (Optional[ParameterEstimator]): Estimator for the length of
                the observed set 2 values.
            pseudo_count (Optional[float]): Kept for consistency.
            name (Optional[str]): Set name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for weights
                and transitions.

        """
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and all(isinstance(k, (str, type(None))) for k in keys)
        ):
            self.keys = keys
        else:
            raise TypeError(
                "HiddenAssociationEstimator requires keys (Tuple[Optional[str], "
                "Optional[str]])."
            )

        self.keys = keys if keys is not None else (None, None)
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.pseudo_count = pseudo_count
        self.cond_estimator = cond_estimator
        self.given_estimator = (
            given_estimator if given_estimator is not None else NullEstimator()
        )
        self.name = name

    def accumulator_factory(self) -> "HiddenAssociationAccumulatorFactory":
        """Create a factory for compatible sufficient-statistic accumulators."""
        len_factory = self.len_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()
        cond_factory = self.cond_estimator.accumulator_factory()
        return HiddenAssociationAccumulatorFactory(
            cond_factory=cond_factory,
            given_factory=given_factory,
            len_factory=len_factory,
            name=self.name,
            keys=self.keys,
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[SS1, Optional[SS2], Optional[SS3]]
    ) -> "HiddenAssociationDistribution":
        """Estimate all model components from aggregated statistics."""
        cond_stats, given_stats, size_stats = suff_stat

        cond_dist = cast(Any, self.cond_estimator).estimate(None, cond_stats)
        given_dist = self.given_estimator.estimate(nobs, given_stats)
        len_dist = self.len_estimator.estimate(nobs, size_stats)

        return HiddenAssociationDistribution(
            cond_dist=cond_dist,
            given_dist=given_dist,
            len_dist=len_dist,
            name=self.name,
        )


class HiddenAssociationDataEncoder(DataSequenceEncoder):
    """Wrap paired bags without transforming their generic values."""

    def __str__(self) -> str:
        """Return a representation of the encoder."""
        return "HiddenAssociationDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a generic hidden-association encoder."""
        return isinstance(other, HiddenAssociationDataEncoder)

    def seq_encode(
        self, x: Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]
    ) -> "HiddenAssociationEncodedDataSequence":
        """Wrap a batch of paired source and target bags for vectorized calls."""
        return HiddenAssociationEncodedDataSequence(data=x)


class HiddenAssociationEncodedDataSequence(EncodedDataSequence):
    """Store a batch of generic paired-bag observations.

    Attributes:
        data (Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): iid obs.

    """

    def __init__(
        self, data: Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]
    ) -> None:
        """Initialize an encoded batch without transforming its observations.

        Args:
            data (Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): iid
                obs.

        """
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a representation of the encoded batch."""
        return f"HiddenAssociationEncodedDataSequence(data={self.data})"
