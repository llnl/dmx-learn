"""Create, estimate, and sample from a Conditional distribution.

Defines the ConditionalDistribution, ConditionalDistributionSampler, ConditionalDistributionAccumulatorFactory,
ConditionalDistributionAccumulator, ConditionalDistributionEstimator, and the ConditionalDistributionDataEncoder
classes for use with dmx-learn.

Data type: (Tuple[T0, T1]): The ConditionalDistribution is given by density,
    P(X0,X1) = P_cond(X1|X0)*P_given(X0).

The ConditionalDistribution allows for user-defined conditional distributions P_cond(X1|X0), and given distributions
P_given(X0).
"""

import numpy as np
import math
from dmx.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    ParameterEstimator,
    DistributionSampler,
    StatisticAccumulatorFactory,
    SequenceEncodableStatisticAccumulator,
    DataSequenceEncoder,
    ConditionalSampler,
    EncodedDataSequence,
)
from dmx.stats.null_dist import (
    NullDistribution,
    NullAccumulator,
    NullDataEncoder,
    NullAccumulatorFactory,
    NullEstimator,
)
from typing import Optional, List, Union, Any, Tuple, Sequence, TypeVar, Dict
from dmx.arithmetic import maxrandint
from numpy.random import RandomState

T0 = TypeVar('T0')
T1 = TypeVar('T1')
E0 = TypeVar('E0')
E1 = TypeVar('E1')
E = Tuple[int, Tuple[T0, ...], Tuple[np.ndarray, ...], Tuple[E0, ...], Optional[E1]]
SS0 = TypeVar('SS0')
SS1 = TypeVar('SS1')
SS2 = TypeVar('SS2')


class ConditionalDistribution(SequenceEncodableProbabilityDistribution):
    """ConditionalDistribution object for data types x=Tuple[T0, T1].

    Attributes:
        dmap (Dict[T0, SequenceEncodableProbabilityDistribution]): Mapping from T0 to distributions.
        default_dist (SequenceEncodableProbabilityDistribution): Default distribution if key not in dmap.
        given_dist (SequenceEncodableProbabilityDistribution): Distribution for given variable.
        has_default (bool): True if default_dist is not NullDistribution.
        has_given (bool): True if given_dist is not NullDistribution.
        name (Optional[str]): Name assigned to object.
        keys (Optional[str]): All ConditionalDistribution objects with same keys value are the same distribution.
    """

    def __init__(
        self,
        dmap: Union[Dict[Any, SequenceEncodableProbabilityDistribution], List[SequenceEncodableProbabilityDistribution]],
        default_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
        given_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize ConditionalDistribution.

        Args:
            dmap (Union[Dict[Any, SequenceEncodableProbabilityDistribution], List[SequenceEncodableProbabilityDistribution]]): Used to create dictionary of distributions.
            default_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for case where x[0] is not a key in dmap.
            given_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the given variable.
            name (Optional[str], optional): Name assigned to object.
            keys (Optional[str], optional): All ConditionalDistribution objects with same keys value are the same distribution.
        """
        if isinstance(dmap, list):
            dmap = dict(zip(range(len(dmap)), dmap))

        self.dmap = dmap
        self.default_dist = default_dist if default_dist is not None else NullDistribution()
        self.given_dist = given_dist if given_dist is not None else NullDistribution()
        self.has_default = not isinstance(self.default_dist, NullDistribution)
        self.has_given = not isinstance(self.given_dist, NullDistribution)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation of ConditionalDistribution."""
        s1 = repr(self.dmap)
        s2 = repr(self.default_dist)
        s3 = repr(self.given_dist)
        s4 = repr(self.name)
        s5 = repr(self.keys)
        return f'ConditionalDistribution({s1}, default_dist={s2}, given_dist={s3}, name={s4}, keys={s5})'

    def density(self, x: Tuple[T0, T1]) -> float:
        """Evaluate density of ConditionalDistribution at Tuple x.

        Args:
            x (Tuple[T0, T1]): T0 data type must match keys of dmap, T1 must match value of dmap distribution for key value.

        Returns:
            float: Density of ConditionalDistribution at Tuple x.
        """
        return math.exp(self.log_density(x))

    def log_density(self, x: Tuple[T0, T1]) -> float:
        """Evaluate log-density of ConditionalDistribution at Tuple x.

        Args:
            x (Tuple[T0, T1]): T0 data type must match keys of dmap, T1 must match value of dmap distribution for key value.

        Returns:
            float: Log-density of ConditionalDistribution at Tuple x.
        """
        if self.has_default:
            rv = self.dmap.get(x[0], self.default_dist).log_density(x[1])
        else:
            if x[0] in self.dmap:
                rv = self.dmap[x[0]].log_density(x[1])
            else:
                return -np.inf

        rv += self.given_dist.log_density(x[0])
        return rv

    def seq_log_density(self, x: 'ConditionalEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (ConditionalEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.

        Raises:
            Exception: If input is not a ConditionalEncodedDataSequence.
        """
        if not isinstance(x, ConditionalEncodedDataSequence):
            raise Exception('Requires ConditionalEncodedDataSequence for ConditionalDistribution.seq_log_density()')
        sz, cond_vals, eobs_vals, idx_vals, given_enc = x.data
        rv = np.zeros(sz, dtype=float)

        for i in range(len(cond_vals)):
            if self.has_default:
                rv[idx_vals[i]] = self.dmap.get(cond_vals[i], self.default_dist).seq_log_density(eobs_vals[i])
            else:
                if cond_vals[i] in self.dmap:
                    rv[idx_vals[i]] += self.dmap[cond_vals[i]].seq_log_density(eobs_vals[i])

        if self.has_given:
            rv += self.given_dist.seq_log_density(given_enc)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'ConditionalDistributionSampler':
        """Create ConditionalDistributionSampler for sampling from this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            ConditionalDistributionSampler: Sampler object.
        """
        return ConditionalDistributionSampler(self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "ConditionalDistributionEstimator":
        """Create ConditionalDistributionEstimator from sufficient statistics.

        Args:
            pseudo_count (Optional[float], optional): Used to inflate the sufficient statistics.

        Returns:
            ConditionalDistributionEstimator: Estimator object.
        """
        est_map = {k: v.estimator(pseudo_count) for k, v in self.dmap.items()}
        default_est = self.default_dist.estimator(pseudo_count)
        given_est = self.given_dist.estimator(pseudo_count)

        return ConditionalDistributionEstimator(
            estimator_map=est_map,
            default_estimator=default_est,
            given_estimator=given_est,
            name=self.name,
            keys=self.keys
        )

    def dist_to_encoder(self) -> 'ConditionalDistributionDataEncoder':
        """Return a ConditionalDistributionDataEncoder for this distribution.

        Returns:
            ConditionalDistributionDataEncoder: Encoder object.
        """
        encoder_map = {k: v.dist_to_encoder() for k, v in self.dmap.items()}
        default_encoder = NullDataEncoder() if not self.has_default else self.default_dist.dist_to_encoder()
        given_encoder = NullDataEncoder() if not self.has_given else self.given_dist.dist_to_encoder()

        return ConditionalDistributionDataEncoder(
            encoder_map=encoder_map,
            default_encoder=default_encoder,
            given_encoder=given_encoder
        )


class ConditionalDistributionSampler(ConditionalSampler, DistributionSampler):
    """Sampler for ConditionalDistribution.

    Attributes:
        dist (ConditionalDistribution): ConditionalDistribution object to draw samples from.
        default_sampler (DistributionSampler): Sampler for default_dist.
        has_default_sampler (bool): True if default sampler is not NullDistribution.
        given_sampler (DistributionSampler): Sampler for given_dist.
        has_given_sampler (bool): True if given sampler is not NullDistribution.
        samplers (Dict[T0, DistributionSampler]): Dictionary of samplers for each key.
    """

    def __init__(self, dist: ConditionalDistribution, seed: Optional[int] = None) -> None:
        """Initialize ConditionalDistributionSampler.

        Args:
            dist (ConditionalDistribution): ConditionalDistribution object to draw samples from.
            seed (Optional[int], optional): Seed for random number generator.
        """
        self.dist = dist
        rng = np.random.RandomState(seed)

        loc_seed = rng.randint(0, maxrandint)
        self.has_default_sampler = dist.has_default
        self.default_sampler = dist.default_dist.sampler(loc_seed)

        loc_seed = rng.randint(0, maxrandint)
        self.given_sampler = dist.given_dist.sampler(loc_seed)
        self.has_given_sampler = not isinstance(dist.given_dist, NullDistribution)

        self.samplers = {k: u.sampler(rng.randint(0, maxrandint)) for k, u in self.dist.dmap.items()}

    def single_sample(self) -> Tuple[Any, Any]:
        """Generate a single sample from the ConditionalDistribution.

        Returns:
            Tuple[Any, Any]: (T0, T1) as defined from dmap and given_distribution types in dist.
        """
        x0 = self.given_sampler.sample()
        if x0 in self.samplers:
            x1 = self.samplers[x0].sample()
        else:
            x1 = self.default_sampler.sample()
        return x0, x1

    def sample(self, size: Optional[int] = None) -> Union[Tuple[Any, Any], List[Tuple[Any, Any]]]:
        """Sample independent samples from ConditionalDistribution.

        Args:
            size (Optional[int], optional): Number of samples to draw. If None, returns a single sample.

        Returns:
            Union[Tuple[Any, Any], List[Tuple[Any, Any]]]: A tuple or list of tuples of (T0, T1).
        """
        if size is None:
            return self.single_sample()
        else:
            return [self.single_sample() for _ in range(size)]

    def sample_given(self, x: T0) -> Any:
        """Sample from conditional distribution given value x.

        Args:
            x (T0): Value of given/conditional variable.

        Returns:
            Any: Single sample from the conditional distribution for x.
        """
        if x in self.samplers:
            return self.samplers[x].sample()
        elif self.has_default_sampler:
            return self.default_sampler.sample()
        else:
            raise Exception('Conditional default distribution unspecified.')


class ConditionalDistributionAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of ConditionalDistribution.

    Attributes:
        accumulator_map (Dict[T0, SequenceEncodableStatisticAccumulator]): Sufficient statistics for each conditional distribution.
        default_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Sufficient statistics for default distribution.
        given_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Sufficient statistics for given distribution.
        has_default (bool): True if default_accumulator is not NullAccumulator.
        has_given (bool): True if given_accumulator is not NullAccumulator.
        key (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.
        name (Optional[str]): Name for object.
        _init_rng (bool): False unless a single call to initialize or seq_initialize has been made.
        _acc_rng (Optional[Dict[T0, RandomState]]): Used to seed RandomState calls of accumulator_map.
        _default_rng (Optional[RandomState]): Used to seed RandomState calls of default_accumulator initialize.
        _given_rng (Optional[RandomState]): Used to seed RandomState calls of given_accumulator initialize.
    """

    def __init__(
        self,
        accumulator_map: Dict[T0, SequenceEncodableStatisticAccumulator],
        default_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
        given_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize ConditionalDistributionAccumulator.

        Args:
            accumulator_map (Dict[T0, SequenceEncodableStatisticAccumulator]): Sufficient statistics for each conditional distribution.
            default_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Sufficient statistics for default distribution.
            given_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Sufficient statistics for given distribution.
            name (Optional[str], optional): Name for object.
            keys (Optional[str], optional): All ConditionalAccumulator objects with same keys value will merge suff stats.
        """
        self.accumulator_map = accumulator_map
        self.default_accumulator = default_accumulator if default_accumulator is not None else NullAccumulator()
        self.given_accumulator = given_accumulator if given_accumulator is not None else NullAccumulator()
        self.has_default = not isinstance(default_accumulator, NullAccumulator)
        self.has_given = not isinstance(given_accumulator, NullAccumulator)
        self.name = name
        self.key = keys
        self._init_rng = False
        self._acc_rng: Optional[Dict[T0, RandomState]] = None
        self._default_rng: Optional[RandomState] = None
        self._given_rng: Optional[RandomState] = None

    def update(
        self,
        x: Tuple[T0, T1],
        weight: float,
        estimate: Optional['ConditionalDistribution']
    ) -> None:
        """Update accumulator with a new observation.

        Args:
            x (Tuple[T0, T1]): Observation.
            weight (float): Weight for the observation.
            estimate (Optional[ConditionalDistribution]): Distribution estimate for update.
        """
        if x[0] in self.accumulator_map:
            if estimate is None:
                self.accumulator_map[x[0]].update(x[1], weight, None)
            else:
                self.accumulator_map[x[0]].update(x[1], weight, estimate.dmap[x[0]])
        else:
            if self.has_default:
                if estimate is None:
                    self.default_accumulator.update(x[1], weight, None)
                else:
                    self.default_accumulator.update(x[1], weight, estimate.default_dist)

        if self.has_given:
            if estimate is None:
                self.given_accumulator.update(x[0], weight, None)
            else:
                self.given_accumulator.update(x[0], weight, estimate.given_dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize random number generators for each accumulator.

        Args:
            rng (RandomState): Random number generator.
        """
        self._acc_rng = dict()
        for acc_key in self.accumulator_map.keys():
            self._acc_rng[acc_key] = RandomState(seed=rng.randint(2 ** 31))
        self._default_rng = RandomState(seed=rng.randint(2 ** 31))
        self._given_rng = RandomState(seed=rng.randint(2 ** 31))

    def initialize(
        self,
        x: Tuple[T0, T1],
        weight: float,
        rng: RandomState
    ) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (Tuple[T0, T1]): Observation.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if x[0] in self.accumulator_map:
            self.accumulator_map[x[0]].initialize(x[1], weight, self._acc_rng[x[0]])
        else:
            if self.has_default:
                self.default_accumulator.initialize(x[1], weight, self._default_rng)

        if self.has_given:
            self.given_accumulator.initialize(x[0], weight, self._given_rng)

    def seq_initialize(
        self,
        x: 'ConditionalEncodedDataSequence',
        weights: np.ndarray,
        rng: RandomState
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (ConditionalEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (RandomState): Random number generator.
        """
        sz, cond_vals, eobs_vals, idx_vals, given_enc = x.data

        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_initialize(
                    eobs_vals[i], weights[idx_vals[i]], self._acc_rng[cond_vals[i]]
                )
            else:
                if self.has_default:
                    self.default_accumulator.seq_initialize(eobs_vals[i], weights[idx_vals[i]], self._default_rng)

        if self.has_given:
            self.given_accumulator.seq_initialize(given_enc, weights, self._given_rng)

    def seq_update(
        self,
        x: 'ConditionalEncodedDataSequence',
        weights: np.ndarray,
        estimate: 'ConditionalDistribution'
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (ConditionalEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (ConditionalDistribution): Distribution estimate for update.
        """
        sz, cond_vals, eobs_vals, idx_vals, given_enc = x.data

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_update(
                    eobs_vals[i], weights[idx_vals[i]], estimate.dmap[cond_vals[i]]
                )
            else:
                if self.has_default:
                    if estimate is None:
                        self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], None)
                    else:
                        self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], estimate.default_dist)

        if self.has_given:
            if estimate is None:
                self.given_accumulator.seq_update(given_enc, weights, None)
            else:
                self.given_accumulator.seq_update(given_enc, weights, estimate.given_dist)

    def combine(
        self,
        suff_stat: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS2]]
    ) -> 'ConditionalDistributionAccumulator':
        """Combine another accumulator's sufficient statistics into this one.

        Args:
            suff_stat (Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS2]]): Sufficient statistics to combine.

        Returns:
            ConditionalDistributionAccumulator: Self after combining.
        """
        for k, v in suff_stat[0].items():
            if k in self.accumulator_map:
                self.accumulator_map[k].combine(v)
            else:
                self.accumulator_map[k].from_value(v)

        if self.has_default and suff_stat[1] is not None:
            self.default_accumulator.combine(suff_stat[1])

        if self.has_given and suff_stat[2] is not None:
            self.given_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[Dict[Any, Any], Optional[Any], Optional[Any]]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[Dict[Any, Any], Optional[Any], Optional[Any]]: Sufficient statistics.
        """
        rv3 = self.given_accumulator.value()
        rv2 = self.default_accumulator.value()
        rv1 = {k: v.value() for k, v in self.accumulator_map.items()}
        return rv1, rv2, rv3

    def from_value(
        self,
        x: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS1]]
    ) -> 'ConditionalDistributionAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS1]]): Sufficient statistics.

        Returns:
            ConditionalDistributionAccumulator: Self after setting values.
        """
        for k, v in x[0].items():
            self.accumulator_map[k].from_value(v)

        if self.has_default and x[1] is not None:
            self.default_accumulator.from_value(x[1])

        if self.has_given and x[2] is not None:
            self.given_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        for k, v in self.accumulator_map.items():
            v.key_merge(stats_dict)

        if self.has_default:
            self.default_accumulator.key_merge(stats_dict)

        if self.has_given:
            self.given_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        for k, v in self.accumulator_map.items():
            v.key_replace(stats_dict)

        if self.has_default:
            self.default_accumulator.key_replace(stats_dict)

        if self.has_given:
            self.given_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'ConditionalDistributionDataEncoder':
        """Return a ConditionalDistributionDataEncoder for this accumulator.

        Returns:
            ConditionalDistributionDataEncoder: Encoder object.
        """
        encoder_map = {k: v.acc_to_encoder() for k, v in self.accumulator_map.items()}
        default_encoder = self.default_accumulator.acc_to_encoder()
        given_encoder = self.given_accumulator.acc_to_encoder()

        return ConditionalDistributionDataEncoder(
            encoder_map=encoder_map,
            default_encoder=default_encoder,
            given_encoder=given_encoder
        )


class ConditionalDistributionAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ConditionalDistributionAccumulator.

    Attributes:
        factory_map (Dict[T0, StatisticAccumulatorFactory]): Factories for each conditional distribution.
        default_factory (StatisticAccumulatorFactory): Factory for default_accumulator.
        given_factory (StatisticAccumulatorFactory): Factory for given_accumulator.
        name (Optional[str]): Name for object.
        keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.
    """

    def __init__(
        self,
        factory_map: Dict[T0, StatisticAccumulatorFactory],
        default_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        given_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize ConditionalDistributionAccumulatorFactory.

        Args:
            factory_map (Dict[T0, StatisticAccumulatorFactory]): Factories for each conditional distribution.
            default_factory (StatisticAccumulatorFactory): Factory for default_accumulator.
            given_factory (StatisticAccumulatorFactory): Factory for given_accumulator.
            name (Optional[str], optional): Name for object.
            keys (Optional[str], optional): All ConditionalAccumulator objects with same keys value will merge suff stats.
        """
        self.factory_map = factory_map
        self.default_factory = default_factory
        self.given_factory = given_factory
        self.name = name
        self.keys = keys

    def make(self) -> 'ConditionalDistributionAccumulator':
        """Create a new ConditionalDistributionAccumulator.

        Returns:
            ConditionalDistributionAccumulator: New accumulator instance.
        """
        acc = {k: v.make() for k, v in self.factory_map.items()}
        def_acc = self.default_factory.make()
        given_acc = self.given_factory.make()

        return ConditionalDistributionAccumulator(
            accumulator_map=acc,
            default_accumulator=def_acc,
            given_accumulator=given_acc,
            keys=self.keys,
            name=self.name
        )


class ConditionalDistributionEstimator(ParameterEstimator):
    """Estimator for ConditionalDistribution.

    Attributes:
        estimator_map (Dict[T0, ParameterEstimator]): Estimators for each conditional distribution.
        default_estimator (ParameterEstimator): Estimator for default_distribution.
        given_estimator (ParameterEstimator): Estimator for given_distribution.
        name (Optional[str]): Name for object.
        keys (Optional[str]): ConditionalDistributionEstimator with matching 'keys' will be aggregated.
    """

    def __init__(
        self,
        estimator_map: Dict[T0, ParameterEstimator],
        default_estimator: Optional[ParameterEstimator] = NullEstimator(),
        given_estimator: Optional[ParameterEstimator] = NullEstimator(),
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize ConditionalDistributionEstimator.

        Args:
            estimator_map (Dict[T0, ParameterEstimator]): Estimators for each conditional distribution.
            default_estimator (Optional[ParameterEstimator]): Estimator for default_distribution.
            given_estimator (Optional[ParameterEstimator]): Estimator for given_distribution.
            name (Optional[str], optional): Name for object.
            keys (Optional[str], optional): ConditionalDistributionEstimator with matching 'keys' will be aggregated.

        Raises:
            TypeError: If keys is not a string or None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("ConditionalDistributionEstimator requires keys to be of type 'str'.")

        self.estimator_map = estimator_map
        self.default_estimator = default_estimator if default_estimator is not None else NullEstimator()
        self.given_estimator = given_estimator if given_estimator is not None else NullEstimator()
        self.name = name

    def accumulator_factory(self) -> 'ConditionalDistributionAccumulatorFactory':
        """Return a ConditionalDistributionAccumulatorFactory for this estimator.

        Returns:
            ConditionalDistributionAccumulatorFactory: Factory object.
        """
        emap_items = {k: v.accumulator_factory() for k, v in self.estimator_map.items()}
        def_factory = self.default_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()

        return ConditionalDistributionAccumulatorFactory(
            factory_map=emap_items,
            default_factory=def_factory,
            given_factory=given_factory,
            keys=self.keys,
            name=self.name
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS2]]
    ) -> 'ConditionalDistribution':
        """Estimate a ConditionalDistribution from aggregated data.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency.
            suff_stat (Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS2]]): Sufficient statistics.

        Returns:
            ConditionalDistribution: Estimated distribution.
        """
        default_dist = self.default_estimator.estimate(None, suff_stat[1])
        given_dist = self.given_estimator.estimate(None, suff_stat[2])
        dist_map = {k: self.estimator_map[k].estimate(None, v) for k, v in suff_stat[0].items()}

        return ConditionalDistribution(
            dist_map,
            default_dist=default_dist,
            given_dist=given_dist,
            name=self.name,
            keys=self.keys
        )


class ConditionalDistributionDataEncoder(DataSequenceEncoder):
    """Encoder for ConditionalDistribution data.

    Attributes:
        encoder_map (Dict[T0, DataSequenceEncoder]): Encoders for each conditional value.
        default_encoder (DataSequenceEncoder): Encoder for default distribution.
        given_encoder (DataSequenceEncoder): Encoder for given distribution.
        null_default_encoder (bool): True if default_encoder is NullDataEncoder.
        null_given_encoder (bool): True if given_encoder is NullDataEncoder.
    """

    def __init__(
        self,
        encoder_map: Dict[T0, DataSequenceEncoder],
        default_encoder: DataSequenceEncoder = NullDataEncoder(),
        given_encoder: DataSequenceEncoder = NullDataEncoder()
    ) -> None:
        """Initialize ConditionalDistributionDataEncoder.

        Args:
            encoder_map (Dict[T0, DataSequenceEncoder]): Encoders for each conditional value.
            default_encoder (DataSequenceEncoder): Encoder for default distribution.
            given_encoder (DataSequenceEncoder): Encoder for given distribution.
        """
        self.encoder_map = encoder_map
        self.default_encoder = default_encoder
        self.given_encoder = given_encoder
        self.null_default_encoder = isinstance(self.default_encoder, NullDataEncoder)
        self.null_given_encoder = isinstance(self.given_encoder, NullDataEncoder)

    def __str__(self) -> str:
        """Return string representation."""
        encoder_items = list(self.encoder_map.items())
        encoder_str = 'ConditionalDataEncoder('
        for k, v in encoder_items[:-1]:
            encoder_str += str(k) + ':' + str(v) + ','
        encoder_str += str(encoder_items[-1][0]) + ':' + str(encoder_items[-1][1])

        if not self.null_default_encoder:
            encoder_str += ',default=' + str(self.default_encoder)
        else:
            encoder_str += ',default=None'

        if not self.null_given_encoder:
            encoder_str += ',given=' + str(self.given_encoder)
        else:
            encoder_str += ',given=None)'

        return encoder_str

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Other object.

        Returns:
            bool: True if encoders are equal.
        """
        if not isinstance(other, ConditionalDistributionDataEncoder):
            return False
        if not self.encoder_map == other.encoder_map:
            return False
        if not self.default_encoder == other.default_encoder:
            return False
        if not self.given_encoder == other.given_encoder:
            return False
        return True

    def seq_encode(self, x: List[Tuple[T0, T1]]) -> 'ConditionalEncodedDataSequence':
        """Encode a sequence of data for vectorized "seq_" function calls.

        Args:
            x (List[Tuple[T0, T1]]): List of data observations.

        Returns:
            ConditionalEncodedDataSequence: Encoded data sequence.
        """
        cond_enc = dict()
        given_vals = []

        for i in range(len(x)):
            xx = x[i]
            given_vals.append(xx[0])
            if xx[0] not in cond_enc:
                cond_enc[xx[0]] = [[xx[1]], [i]]
            else:
                cond_enc_loc = cond_enc[xx[0]]
                cond_enc_loc[0].append(xx[1])
                cond_enc_loc[1].append(i)

        cond_enc_items = list(cond_enc.items())
        cond_vals = tuple([u[0] for u in cond_enc_items])

        eobs_vals = []
        idx_vals = []

        for u in cond_enc_items:
            if self.null_default_encoder:
                if u[0] in self.encoder_map:
                    eobs_vals.append(self.encoder_map[u[0]].seq_encode(u[1][0]))
            else:
                eobs_vals.append(self.encoder_map.get(u[0], self.default_encoder).seq_encode(u[1][0]))

            idx_vals.append(np.asarray(u[1][1]))

        given_enc = self.given_encoder.seq_encode(given_vals)

        return ConditionalEncodedDataSequence(data=(len(x), cond_vals, tuple(eobs_vals), tuple(idx_vals), given_enc))


class ConditionalEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for ConditionalDistribution.

    Attributes:
        data (Tuple[int, Tuple[T0, ...], Tuple[EncodedDataSequence], Tuple[np.ndarray], Optional[EncodedDataSequence]]): Encoded data.
    """

    def __init__(
        self,
        data: Tuple[int, Tuple[T0, ...], Tuple[EncodedDataSequence], Tuple[np.ndarray], Optional[EncodedDataSequence]]
    ) -> None:
        """Initialize ConditionalEncodedDataSequence.

        Args:
            data (Tuple[int, Tuple[T0, ...], Tuple[EncodedDataSequence], Tuple[np.ndarray], Optional[EncodedDataSequence]]): Encoded data.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'ConditionalEncodedDataSequence(data={self.data})'



