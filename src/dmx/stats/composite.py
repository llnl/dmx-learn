"""Create, estimate, and sample from a Composite distribution.

Defines the CompositeDistribution, CompositeSampler, CompositeAccumulatorFactory,
CompositeAccumulator,
CompositeEstimator, and the CompositeDataEncoder classes for use with dmx-learn.

Data type: Tuple[T_0, ... T_{n-1}]: The CompositeDistribution of size 'n' is a joint
distribution for
independent observations of 'n'-tupled data. Each component 'k' of the
CompositeDistribution has data type T_k that
must be compatible with data type T_k.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import maxrandint
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class CompositeDistribution(SequenceEncodableProbabilityDistribution):
    """Model a tuple as independent draws from positional child distributions.

    The support is the Cartesian product of the child supports and the density is
    the product of the child densities. Consequently, the wrapper is normalized
    when every child is normalized. Scoring, encoding, sampling, accumulation, and
    estimation are all delegated position by position; children do not share state
    unless their own estimators or the composite ``keys`` contract says otherwise.
    At least one child is required, and every observation tuple must have the same
    number and order of fields as ``dists``.

    Attributes:
        dists (Tuple[SequenceEncodableProbabilityDistribution, ...]): Distributions for
            each component.
        count (int): Number of components (i.e. len(dists)).
        name (Optional[str]): Name of object.
        keys (Optional[str]): Key for marking shared parameters.
    """

    def __init__(
        self,
        dists: Sequence[SequenceEncodableProbabilityDistribution],
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Create an instance of CompositeDistribution.

        Args:
            dists (Sequence[SequenceEncodableProbabilityDistribution]): Component
                distributions.
            name (Optional[str], optional): Name of object. Defaults to None.
            keys (Optional[str], optional): Key for marking shared parameters. Defaults
                to None.
        """
        super().__init__()
        self.dists: Tuple[SequenceEncodableProbabilityDistribution, ...] = tuple(dists)
        self.count: int = len(dists)
        self.name: Optional[str] = name
        self.keys: Optional[str] = keys

    def __str__(self) -> str:
        """Return string representation of CompositeDistribution with each dist as well.

        Returns:
            str: String representation.
        """
        s0 = ",".join(map(str, self.dists))
        s1 = repr(self.name)
        s2 = repr(self.keys)
        return f"CompositeDistribution(dists=[{s0}], name={s1}, keys={s2})"

    def density(self, x: Tuple[Any, ...]) -> float:
        """Evaluate density of CompositeDistribution for a single observation tuple x.

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k-th data type must
                be consistent with dists[k].

        Returns:
            float: Density value.
        """
        rv = self.dists[0].density(x[0])
        for i in range(1, self.count):
            rv *= self.dists[i].density(x[i])
        return rv

    def log_density(self, x: Tuple[Any, ...]) -> float:
        """Evaluate the sum of child log densities for one observation tuple.

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k-th data type must
                be consistent with dists[k].

        Returns:
            float: Log-density value.
        """
        rv = self.dists[0].log_density(x[0])
        for i in range(1, self.count):
            rv += self.dists[i].log_density(x[i])
        return rv

    def seq_log_density(self, x: "CompositeEncodedDataSequence") -> np.ndarray:
        """Vectorized evaluation of log density for CompositeEncodedDataSequence.

        Args:
            x (CompositeEncodedDataSequence): EncodedDataSequence for Composite
                Distribution.

        Returns:
            np.ndarray: Log-density evaluated at all encoded data points.

        Raises:
            Exception: If input is not a CompositeEncodedDataSequence.
        """
        if not isinstance(x, CompositeEncodedDataSequence):
            raise TypeError(
                "CompositeDistribution.seq_log_density() requires "
                "CompositeEncodedDataSequence."
            )
        rv = self.dists[0].seq_log_density(x.data[0])
        for i in range(1, self.count):
            rv += self.dists[i].seq_log_density(x.data[i])
        return rv

    def sampler(self, seed: Optional[int] = None) -> "CompositeSampler":
        """Create CompositeSampler for sampling from CompositeDistribution instance.

        Args:
            seed (Optional[int], optional): Seed to set for sampling with RandomState.
                Defaults to None.

        Returns:
            CompositeSampler: Sampler object.
        """
        return CompositeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "CompositeEstimator":
        """Create CompositeEstimator for estimating CompositeDistribution.

        Args:
            pseudo_count (Optional[float], optional): Used to inflate sufficient
                statistics in estimation.

        Returns:
            CompositeEstimator: Estimator object.
        """
        return CompositeEstimator(
            [d.estimator(pseudo_count=pseudo_count) for d in self.dists],
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "CompositeDataEncoder":
        """Return a CompositeDataEncoder for this distribution.

        Returns:
            CompositeDataEncoder: Encoder object.
        """
        encoders = tuple(d.dist_to_encoder() for d in self.dists)
        return CompositeDataEncoder(encoders=encoders)


class CompositeSampler(DistributionSampler):
    """CompositeSampler used to generate samples from CompositeDistribution.

    Attributes:
        dist (CompositeDistribution): CompositeDistribution to draw samples from.
        rng (RandomState): RandomState with seed set if provided.
        dist_samplers (List[DistributionSampler]): List of DistributionSamplers for each
            component.
    """

    def __init__(
        self, dist: "CompositeDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize CompositeSampler.

        Args:
            dist (CompositeDistribution): CompositeDistribution to draw samples from.
            seed (Optional[int], optional): Seed to set for sampling with RandomState.
                Defaults to None.
        """
        super().__init__(dist, seed)
        self.dist_samplers: List[DistributionSampler] = [
            d.sampler(seed=self.rng.randint(maxrandint)) for d in dist.dists
        ]

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Tuple[Any, ...]], Tuple[Any, ...]]:
        """Generate independent samples from a CompositeDistribution.

        If size is None, draw one sample and return as Tuple of length = len(dists). If
        size > 0,
        draw size samples and return a list of length size containing tuples of
        len(dists).

        Args:
            size (Optional[int], optional): If None, draw 1 sample. Else, draw size
                number of iid samples.

        Returns:
            Union[List[Tuple[Any, ...]], Tuple[Any, ...]]: A tuple of length =
            len(dists) or a list of length size containing tuples of length =
            len(dists).
        """
        if size is None:
            return tuple(d.sample(size=size) for d in self.dist_samplers)
        return list(zip(*[d.sample(size=size) for d in self.dist_samplers]))


class CompositeAccumulator(SequenceEncodableStatisticAccumulator):
    """Aggregate a separate sufficient statistic for every tuple field.

    Each observation weight is passed unchanged to every positional child. The
    public sufficient statistic is a tuple in the same order as ``accumulators``.

    Attributes:
        accumulators (List[SequenceEncodableStatisticAccumulator]): List of accumulators
            for each component.
        count (int): Number of accumulators.
        key (Optional[str]): All CompositeAccumulators with same key will have
            suff-stats merged.
        name (Optional[str]): Name of the object.
        _init_rng (bool): True if _acc_rng has been set by a single function call to
            initialize.
        _acc_rng (Optional[List[RandomState]]): List of RandomState objects generated
            from seeds set by rng in initialize.
    """

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize CompositeAccumulator.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators
                for each component.
            keys (Optional[str], optional): All CompositeAccumulators with same key will
                have suff-stats merged. Defaults to None.
            name (Optional[str], optional): Name of the object. Defaults to None.
        """
        self.accumulators: List[SequenceEncodableStatisticAccumulator] = list(
            accumulators
        )
        self.count: int = len(accumulators)
        self.key: Optional[str] = keys
        self.name: Optional[str] = name
        self._init_rng: bool = False
        self._acc_rng: Optional[List[RandomState]] = None

    def update(
        self,
        x: Tuple[Any, ...],
        weight: float,
        estimate: Optional["CompositeDistribution"],
    ) -> None:
        """Update accumulators with a new observation.

        Args:
            x (Tuple[Any, ...]): Observation tuple.
            weight (float): Weight for the observation.
            estimate (Optional[CompositeDistribution]): Distribution estimate for
                update.
        """
        if estimate is not None:
            for i in range(self.count):
                self.accumulators[i].update(x[i], weight, estimate.dists[i])
        else:
            for i in range(self.count):
                self.accumulators[i].update(x[i], weight, None)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize random number generators for each accumulator.

        Args:
            rng (RandomState): Random number generator.
        """
        seeds = rng.randint(2**31, size=self.count)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._init_rng = True

    def initialize(self, x: Tuple[Any, ...], weight: float, rng: RandomState) -> None:
        """Initialize accumulators with a new observation.

        The composite wrapper does not create its own sufficient-statistic target.
        It delegates each tuple position to the corresponding child accumulator using
        an independent child random number generator. Pseudo-count and prior-target
        behavior is therefore controlled by the child estimators.

        Args:
            x (Tuple[Any, ...]): Observation tuple.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)
        assert self._acc_rng is not None
        for i in range(self.count):
            self.accumulators[i].initialize(x[i], weight, self._acc_rng[i])

    def seq_initialize(
        self, x: "CompositeEncodedDataSequence", weights: np.ndarray, rng: RandomState
    ) -> None:
        """Vectorized initialization for encoded data.

        Each encoded tuple field is initialized by its matching child accumulator with
        the same observation weights and an independent child random number generator.
        Use child estimator pseudo-counts or child keys when only selected fields need
        regularized or shared initialization.

        Args:
            x (CompositeEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)
        assert self._acc_rng is not None
        for i in range(self.count):
            self.accumulators[i].seq_initialize(x.data[i], weights, self._acc_rng[i])

    def get_seq_lambda(self) -> List[Any]:
        """Get sequence lambda for all accumulators.

        Returns:
            List[Any]: Sequence lambda values.
        """
        rv = []
        for i in range(self.count):
            rv.extend(self.accumulators[i].get_seq_lambda())
        return rv

    def seq_update(
        self,
        x: "CompositeEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional["CompositeDistribution"],
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (CompositeEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[CompositeDistribution]): Distribution estimate for
                update.
        """
        for i in range(self.count):
            self.accumulators[i].seq_update(
                x.data[i], weights, estimate.dists[i] if estimate is not None else None
            )

    def combine(self, suff_stat: Tuple[Any, ...]) -> "CompositeAccumulator":
        """Combine another accumulator's sufficient statistics into this one.

        Args:
            suff_stat (Tuple[Any, ...]): Sufficient statistics to combine.

        Returns:
            CompositeAccumulator: Self after combining.
        """
        for i in range(self.count):
            self.accumulators[i].combine(suff_stat[i])
        return self

    def value(self) -> Tuple[Any, ...]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[Any, ...]: Tuple of sufficient statistics for each accumulator.
        """
        return tuple(x.value() for x in self.accumulators)

    def from_value(self, x: Tuple[Any, ...]) -> "CompositeAccumulator":
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[Any, ...]): Sufficient statistics.

        Returns:
            CompositeAccumulator: Self after setting values.
        """
        self.accumulators = [
            self.accumulators[i].from_value(x[i]) for i in range(len(x))
        ]
        self.count = len(x)
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self
        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())
        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "CompositeDataEncoder":
        """Return a CompositeDataEncoder for this accumulator.

        Returns:
            CompositeDataEncoder: Encoder object.
        """
        encoders = tuple(acc.acc_to_encoder() for acc in self.accumulators)
        return CompositeDataEncoder(encoders=encoders)


class CompositeAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for CompositeAccumulator.

    Attributes:
        factories (Sequence[StatisticAccumulatorFactory]): Factories for each component.
        keys (Optional[str]): Declare keys for merging sufficient statistics of
            CompositeAccumulator objects.
        name (Optional[str]): Name of the object.
    """

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize CompositeAccumulatorFactory.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Factories for each
                component.
            keys (Optional[str], optional): Declare keys for merging sufficient
                statistics. Defaults to None.
            name (Optional[str], optional): Name of the object. Defaults to None.
        """
        self.factories: Sequence[StatisticAccumulatorFactory] = factories
        self.keys: Optional[str] = keys
        self.name: Optional[str] = name

    def make(self) -> "CompositeAccumulator":
        """Create a new CompositeAccumulator.

        Returns:
            CompositeAccumulator: New accumulator instance.
        """
        return CompositeAccumulator(
            [u.make() for u in self.factories], keys=self.keys, name=self.name
        )


class CompositeEstimator(ParameterEstimator):
    """Estimator for CompositeDistribution.

    Notes:
        A Composite estimator-level ``keys`` value shares the whole tuple-shaped
        sufficient statistic with other Composite estimators using the same key. The
        merge is positional: field ``i`` in one composite is pooled with field ``i``
        in another.

        Composite accumulators also recurse into their child accumulators. That means
        child estimator keys can still share individual fields even when the
        Composite estimator itself is unkeyed. Use a Composite-level key only when
        every field position has the same meaning across the estimators being fit;
        otherwise prefer keys on selected child estimators.

    Attributes:
        estimators (Sequence[ParameterEstimator]): Estimators for each component.
        keys (Optional[str]): Key used for sharing the complete positional tuple of
            component sufficient statistics.
        count (int): Number of components.
        name (Optional[str]): Name of the object.
    """

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize CompositeEstimator.

        Args:
            estimators (Sequence[ParameterEstimator]): Estimators for each component.
            keys (Optional[str], optional): Key used to share the complete composite
                sufficient statistic with matching Composite estimators. Sharing is
                positional, so component ``i`` is pooled with component ``i``. Use
                child estimator keys instead when only selected fields should be
                tied. Defaults to None.
            name (Optional[str], optional): Name of the object. Defaults to None.

        Raises:
            TypeError: If keys is not a string or None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("CompositeEstimator requires keys to be of type 'str'.")
        self.estimators: Sequence[ParameterEstimator] = estimators
        self.count: int = len(estimators)
        self.name: Optional[str] = name

    def accumulator_factory(self) -> "CompositeAccumulatorFactory":
        """Return a CompositeAccumulatorFactory for this estimator.

        Returns:
            CompositeAccumulatorFactory: Factory object.
        """
        return CompositeAccumulatorFactory(
            [u.accumulator_factory() for u in self.estimators],
            keys=self.keys,
            name=self.name,
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[Any, ...]
    ) -> "CompositeDistribution":
        """Estimate a CompositeDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations used to form
                suff_stat.
            suff_stat (Tuple[Any, ...]): Tuple of sufficient statistics for each
                estimator.

        Returns:
            CompositeDistribution: Estimated distribution.
        """
        return CompositeDistribution(
            tuple(est.estimate(nobs, ss) for est, ss in zip(self.estimators, suff_stat))
        )


class CompositeDataEncoder(DataSequenceEncoder):
    """Encoder for CompositeDistribution data.

    Data must be of form Sequence[Tuple[Any, ...]]. Each encoder component must be
    compatible with each data
    component of the data.

    Attributes:
        encoders (Tuple[DataSequenceEncoder, ...]): DataSequenceEncoders for each
            component.
    """

    def __init__(self, encoders: Sequence[DataSequenceEncoder]) -> None:
        """Initialize CompositeDataEncoder.

        Args:
            encoders (Sequence[DataSequenceEncoder]): DataSequenceEncoders for each
                component.
        """
        self.encoders: Tuple[DataSequenceEncoder, ...] = tuple(encoders)

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Other object.

        Returns:
            bool: True if encoders are equal.
        """
        if not isinstance(other, CompositeDataEncoder):
            return False
        for i, encoder in enumerate(self.encoders):
            if not encoder == other.encoders[i]:
                return False
        return True

    def __str__(self) -> str:
        """Return string representation.

        Returns:
            str: String representation.
        """
        s = "CompositeDataEncoder(["
        for d in self.encoders[:-1]:
            s += str(d) + ","
        s += str(self.encoders[-1]) + "])"
        return s

    def seq_encode(
        self, x: Sequence[Tuple[Any, ...]]
    ) -> "CompositeEncodedDataSequence":
        """Encode sequence of tuples of data for use with vectorized "seq_" functions.

        The input x must be a Sequence of Tuples of length equal to the length of
        encoders. Each component tuple
        observation of x, say x[i], must be component-wise compatible with encoders.

        Args:
            x (Sequence[Tuple[Any, ...]]): Sequence of tuples of length equal to
                len(encoders).

        Returns:
            CompositeEncodedDataSequence: Encoded data sequence.
        """
        enc_data = []
        for i, encoder in enumerate(self.encoders):
            enc_data.append(encoder.seq_encode([u[i] for u in x]))
        return CompositeEncodedDataSequence(data=tuple(enc_data))


class CompositeEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for CompositeDistribution.

    Data must be of form Sequence[Tuple[Any, ...]]. Each encoder component must be
    compatible with each data
    component of the data.

    Attributes:
        data (Tuple[EncodedDataSequence, ...]): Tuple of EncodedDataSequences.
    """

    def __init__(self, data: Tuple[EncodedDataSequence, ...]):
        """Store the positionally encoded child sequences.

        Args:
            data: One encoded sequence per tuple field, in child order.
        """
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a representation containing the child encodings."""
        return f"CompositeEncodedDataSequence(data={self.data})"
