"""Mixture of a Dirac delta at ``v`` and a length distribution.

The DiracMixtureDistribution is defined by the density:

.. math::

    P(Y) = p \cdot P_1(Y) + (1-p) \cdot \Delta_{v}(Y)

where:

- :math:`P_1()` is a length distribution with support on non-negative integers (or a subset thereof)
- :math:`\Delta_{v}(x) = 1` if :math:`x = v`, else :math:`0`
- :math:`p` is the probability of drawing from the length distribution, with :math:`0 < p \leq 1`
"""

from typing import List, Union, Tuple, Any, Optional, TypeVar, Sequence, Dict

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import maxrandint
from dmx.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    ParameterEstimator,
    DistributionSampler,
    StatisticAccumulatorFactory,
    SequenceEncodableStatisticAccumulator,
    DataSequenceEncoder,
    EncodedDataSequence,
)

SS0 = TypeVar('SS0')  # Type of component suff_stat
key_type = Union[Tuple[str, str], Tuple[None, None]]


class DiracMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """DiracMixtureDistribution object defined by a length distribution, choice of Dirac value, and p.

    Attributes:
        p (float): Probability of being drawn from length distribution. Must be between 0 and 1.
        dist (SequenceEncodableProbabilityDistribution): Distribution with support on non-negative integers.
        v (int): Dirac spike value.
        name (Optional[str]): Name for object instance.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components of mixture.
    """

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        p: float,
        v: int = 0,
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """Initialize DiracMixtureDistribution.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Distribution with support on non-negative integers.
            p (float): Probability of being drawn from length distribution. Must be between 0 and 1.
            v (int, optional): Dirac spike value. Defaults to 0.
            name (Optional[str], optional): Name for object instance. Defaults to None.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and components of mixture. Defaults to (None, None).

        Raises:
            Exception: If p is not in (0, 1].
        """
        if not 0 < p <= 1:
            raise Exception('p must be between (0,1].')
        with np.errstate(divide='ignore'):
            self.p = p
            self.v = v
            self.log_p = np.log(p)
            self.log_1p = np.log1p(-p)
            self.dist = dist
            self.name = name
            self.keys = keys

    def __str__(self) -> str:
        """Return string representation."""
        s1 = repr(self.dist)
        s2 = repr(self.p)
        s3 = repr(self.v)
        s4 = repr(self.name)
        s5 = repr(self.keys)
        return f'DiracMixtureDistribution(dist={s1}, p={s2}, v={s3}, name={s4}, keys={s5})'

    def density(self, x: int) -> float:
        """Evaluate density of Dirac mixture distribution at observation x.

        Args:
            x (int): Integer value.

        Returns:
            float: Density at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of Dirac mixture distribution at observation x.

        log(P(x)) = log( p*P_1(x) + (1-p)*Delta_{v}(x) )

        Args:
            x (int): Integer value.

        Returns:
            float: log-density at x.
        """
        rv0 = self.log_p + self.dist.log_density(x)

        if x == self.v:
            c1 = self.log_1p
            if c1 > rv0:
                rv = np.log1p(np.exp(rv0 - c1)) + c1
            else:
                rv = np.log1p(np.exp(c1 - rv0)) + rv0
        else:
            rv = rv0

        return rv

    def component_log_density(self, x: int) -> np.ndarray:
        """Evaluate the log density for the components of the Dirac mixture.

        The components are Dirac spike and `dist`.

        Args:
            x (int): Integer value with support on mixture components.

        Returns:
            np.ndarray: Log-densities for each component.
        """
        rv = np.zeros(2, dtype=np.float64)
        rv[0] = self.dist.log_density(x)
        if x != self.v:
            rv[1] = -np.inf
        return rv

    def posterior(self, x: int) -> np.ndarray:
        """Evaluate the posterior for the components of the Dirac mixture.

        The components are Dirac spike and `dist`.

        Args:
            x (int): Integer value with support on mixture components.

        Returns:
            np.ndarray: Posterior probabilities for each component.
        """
        comp_log_density = self.component_log_density(x)
        if comp_log_density[1] == -np.inf:
            return np.array([1, 0], dtype=np.float64)
        else:
            comp_log_density[0] += self.log_p
            comp_log_density[1] += self.log_1p

        max_val = np.max(comp_log_density)
        comp_log_density -= max_val
        np.exp(comp_log_density, out=comp_log_density)
        comp_log_density /= comp_log_density.sum()

        return comp_log_density

    def seq_component_log_density(self, x: 'DiracMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation of the log density for the components of the Dirac mixture.

        The components are Dirac spike and `dist`.

        Args:
            x (DiracMixtureEncodedDataSequence): EncodedDataSequence for DiracMixtureDistribution.

        Returns:
            np.ndarray: Log-densities for each component.
        """
        if not isinstance(x, DiracMixtureEncodedDataSequence):
            raise Exception('DiracMixtureEncodedDataSequence required for `seq_` function calls.')

        sz, idx_v, idx_nv, enc_x = x.data
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        ll_mat[:, 0] += self.dist.seq_log_density(enc_x)
        ll_mat[idx_nv, 1] = -np.inf

        return ll_mat

    def seq_log_density(self, x: 'DiracMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for DiracMixtureEncodedDataSequence.

        Args:
            x (DiracMixtureEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.
        """
        if not isinstance(x, DiracMixtureEncodedDataSequence):
            raise Exception('DiracMixtureEncodedDataSequence required for `seq_` function calls.')

        sz, idx_v, idx_nv, enc_x = x.data
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        ll_mat[:, 0] += self.dist.seq_log_density(enc_x) + self.log_p
        ll_mat[idx_nv, 1] = -np.inf
        ll_mat[idx_v, 1] += self.log_1p

        ll_max = ll_mat.max(axis=1, keepdims=True)
        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max
            return ll_sum.flatten()
        else:
            ll_mat = ll_mat[good_rows, :]
            ll_max = ll_max[good_rows]
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max
            rv = np.zeros(good_rows.shape, dtype=float)
            rv[good_rows] = ll_sum.flatten()
            rv[~good_rows] = -np.inf
            return rv

    def seq_posterior(self, x: 'DiracMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation of the posterior for the components of the Dirac mixture.

        The components are Dirac spike and `dist`.

        Args:
            x (DiracMixtureEncodedDataSequence): EncodedDataSequence for DiracMixtureDistribution.

        Returns:
            np.ndarray: Posterior probabilities for each component.
        """
        if not isinstance(x, DiracMixtureEncodedDataSequence):
            raise Exception('DiracMixtureEncodedDataSequence required for `seq_` function calls.')

        sz, idx_v, idx_nv, enc_x = x.data
        rv = np.zeros((sz, 2), dtype=np.float64)
        rv[:, 0] += self.dist.seq_log_density(enc_x) + self.log_p
        rv[:, 1] = self.log_1p

        if len(idx_v) > 0:
            ll_mat = rv[idx_v, :]
            ll_max = np.max(ll_mat, axis=1, keepdims=True)
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_mat /= np.sum(ll_mat, axis=1, keepdims=True)
            rv[idx_v, :] = ll_mat

        rv[idx_nv, 0] = 1.0
        rv[idx_nv, 1] = 0.0

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'DiracMixtureSampler':
        """Return a DiracMixtureSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            DiracMixtureSampler: Sampler object.
        """
        return DiracMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DiracMixtureEstimator':
        """Return a DiracMixtureEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            DiracMixtureEstimator: Estimator object.
        """
        if pseudo_count is not None:
            est = self.dist.estimator(pseudo_count)
            return DiracMixtureEstimator(
                estimator=est,
                v=self.v,
                pseudo_count=pseudo_count,
                suff_stat=self.p,
                name=self.name,
                keys=self.keys
            )
        else:
            est = self.dist.estimator()
            return DiracMixtureEstimator(estimator=est, v=self.v, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'DiracMixtureDataEncoder':
        """Return a DiracMixtureDataEncoder for this distribution.

        Returns:
            DiracMixtureDataEncoder: Encoder object.
        """
        dist_encoder = self.dist.dist_to_encoder()
        return DiracMixtureDataEncoder(encoder=dist_encoder, v=self.v)


class DiracMixtureSampler(DistributionSampler):
    """DiracMixtureSampler used to generate samples.

    Attributes:
        rng (RandomState): Seeded RandomState for sampling.
        p (float): Probability of drawing from length distribution.
        dist_sampler (DistributionSampler): Sampler for the length distribution.
        v (int): Dirac location.
    """

    def __init__(self, dist: DiracMixtureDistribution, seed: Optional[int] = None) -> None:
        """Initialize DiracMixtureSampler.

        Args:
            dist (DiracMixtureDistribution): DiracMixtureDistribution to draw samples from.
            seed (Optional[int], optional): Seed for random number generator.
        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.p = np.exp(dist.log_p)
        self.dist_sampler = dist.dist.sampler(seed=self.rng.randint(maxrandint))
        self.v = dist.v

    def sample(self, size: Optional[int] = None) -> Union[List[int], int]:
        """Draw iid samples from a DiracMixture distribution.

        Args:
            size (Optional[int], optional): Number of iid samples to draw.

        Returns:
            Union[int, List[int]]: Single sample if size is None, else list of samples.
        """
        comp_state = self.rng.binomial(n=1, size=size, p=self.p)

        if size is None:
            if comp_state == 0:
                return self.v
            else:
                return self.dist_sampler.sample()
        else:
            rv = np.zeros(size, dtype=np.int32)
            rv.fill(self.v)
            idx = np.flatnonzero(comp_state == 1)
            if len(idx) > 0:
                rv[idx] = np.asarray(self.dist_sampler.sample(size=len(idx)), dtype=np.int32)
            return list(rv)


class DiracMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """DiracMixtureAccumulator object for aggregating sufficient statistics.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator object for distribution with integer support.
        comp_counts (np.ndarray): Sufficient statistics for mixture components.
        weight_key (Optional[str]): Key for weights of mixture.
        comp_key (Optional[str]): Key for components of mixture.
        v (int): Dirac spike value.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        accumulator: SequenceEncodableStatisticAccumulator,
        v: int = 0,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """Initialize DiracMixtureAccumulator.

        Args:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator object for distribution with integer support.
            v (int, optional): Dirac spike value. Defaults to 0.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and components. Defaults to (None, None).
            name (Optional[str], optional): Name for object. Defaults to None.
        """
        self.accumulator = accumulator
        self.comp_counts = np.zeros(2, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.v = v
        self.name = name

        # Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: Optional[RandomState] = None
        self._w_rng: Optional[RandomState] = None

    def seq_update(
        self,
        x: 'DiracMixtureEncodedDataSequence',
        weights: np.ndarray,
        estimate: 'DiracMixtureDistribution'
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (DiracMixtureEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (DiracMixtureDistribution): Distribution estimate for update.
        """
        sz, idx_v, idx_nv, enc_x = x.data
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        if len(idx_v) == 0:
            ll_mat[:, 0] += weights
        else:
            ll_mat[:, 0] += estimate.dist.seq_log_density(enc_x) + estimate.log_p
            ll_mat[idx_nv, 0] = weights[idx_nv].copy()

            rv = ll_mat[idx_v, :]
            rv[:, 1] += estimate.log_1p

            rv_max = rv.max(axis=1, keepdims=True)
            bad_rows = np.isinf(rv.flatten())

            if np.any(bad_rows):
                rv[bad_rows, :] = np.array([estimate.log_p, estimate.log_1p], dtype=np.float64)
                rv_max[bad_rows] = np.max(np.asarray([estimate.log_p, estimate.log_1p]))
            rv -= rv_max

            np.exp(rv, out=rv)
            np.sum(rv, axis=1, keepdims=True, out=rv_max)
            np.divide(weights[idx_v, None], rv_max, out=rv_max)
            rv *= rv_max

            ll_mat[idx_v, :] = rv

        self.comp_counts += ll_mat.sum(axis=0)
        self.accumulator.seq_update(enc_x, ll_mat[:, 0], estimate.dist)

    def update(self, x: int, weight: float, estimate: 'DiracMixtureDistribution') -> None:
        """Update accumulator with a new observation.

        Args:
            x (int): Observation.
            weight (float): Weight for the observation.
            estimate (DiracMixtureDistribution): Distribution estimate for update.
        """
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior
        self.accumulator.update(x, posterior[0], estimate.dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize random number generators for accumulator and weights.

        Args:
            rng (RandomState): Random number generator.
        """
        seeds = rng.randint(2 ** 31, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: int, weight: float, rng: np.random.RandomState) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (int): Observation.
            weight (float): Weight for the observation.
            rng (np.random.RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if x == self.v:
            ww = self._w_rng.dirichlet(np.ones(2) / 4)
            self.accumulator.initialize(x, weight * ww[0], rng=self._acc_rng)
            self.comp_counts += ww
        else:
            self.accumulator.initialize(x, weight, rng=self._acc_rng)
            self.comp_counts[0] += weight

    def seq_initialize(
        self,
        x: 'DiracMixtureEncodedDataSequence',
        weights: np.ndarray,
        rng: np.random.RandomState
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (DiracMixtureEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (np.random.RandomState): Random number generator.
        """
        sz, xi_v, xi_nv, enc_x = x.data

        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        keep_len = len(xi_v)
        ww = np.ones((sz, 2))

        if keep_len > 0:
            ww[xi_v, :] = self._w_rng.dirichlet(alpha=np.ones(2) / 4, size=keep_len)

        ww *= np.reshape(weights, (sz, 1))

        self.accumulator.seq_initialize(enc_x, weights=ww[:, 0], rng=self._acc_rng)
        self.comp_counts[0] += np.sum(ww[:, 0])
        self.comp_counts[1] += np.sum(ww[xi_v, 1])

    def combine(self, suff_stat: Tuple[np.ndarray, SS0]) -> 'DiracMixtureAccumulator':
        """Combine another accumulator's sufficient statistics into this one.

        Args:
            suff_stat (Tuple[np.ndarray, SS0]): Sufficient statistics to combine.

        Returns:
            DiracMixtureAccumulator: Self after combining.
        """
        self.comp_counts += suff_stat[0]
        self.accumulator.combine(suff_stat[1])
        return self

    def value(self) -> Tuple[np.ndarray, Any]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[np.ndarray, Any]: (comp_counts, accumulator value)
        """
        return self.comp_counts, self.accumulator.value()

    def from_value(self, x: Tuple[np.ndarray, SS0]) -> 'DiracMixtureAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[np.ndarray, SS0]): Sufficient statistics.

        Returns:
            DiracMixtureAccumulator: Self after setting values.
        """
        self.comp_counts = x[0]
        self.accumulator.from_value(x[1])
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                stats_dict[self.comp_key].combine(self.accumulator.value())
            else:
                stats_dict[self.comp_key] = self.accumulator

        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulator = acc

        self.accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'DiracMixtureDataEncoder':
        """Return a DiracMixtureDataEncoder for this accumulator.

        Returns:
            DiracMixtureDataEncoder: Encoder object.
        """
        acc_encoder = self.accumulator.acc_to_encoder()
        return DiracMixtureDataEncoder(encoder=acc_encoder, v=self.v)


class DiracMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """DiracMixtureAccumulatorFactory object for creating DiracMixtureAccumulator objects.

    Attributes:
        factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for mixture components.
        v (int): Dirac integer value.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and mixture components.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        factory: StatisticAccumulatorFactory,
        v: int = 0,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """Initialize DiracMixtureAccumulatorFactory.

        Args:
            factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for mixture components.
            v (int, optional): Dirac integer value. Defaults to 0.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and mixture components. Defaults to (None, None).
            name (Optional[str], optional): Name for object. Defaults to None.
        """
        self.factory = factory
        self.v = v
        self.keys = keys
        self.name = name

    def make(self) -> 'DiracMixtureAccumulator':
        """Create a new DiracMixtureAccumulator.

        Returns:
            DiracMixtureAccumulator: New accumulator instance.
        """
        return DiracMixtureAccumulator(accumulator=self.factory.make(), v=self.v, keys=self.keys, name=self.name)


class DiracMixtureEstimator(ParameterEstimator):
    """DiracMixtureEstimator object for estimating DiracMixtureDistribution.

    Attributes:
        estimator (ParameterEstimator): Estimator for components of mixture. Should have support on integers.
        v (int): Spiked value.
        pseudo_count (Optional[float]): Regularize sufficient statistics.
        suff_stat (Optional[float]): Regularize estimation on the Dirac probability.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components of mixture.
        name (Optional[str]): Assign a name to the object.
        fixed_p_vec (Optional[np.ndarray]): Fixed Dirac spike probability.
    """

    def __init__(
        self,
        estimator: ParameterEstimator,
        v: int = 0,
        fixed_p: Optional[int] = None,
        suff_stat: Optional[float] = None,
        pseudo_count: Optional[float] = None,
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """Initialize DiracMixtureEstimator.

        Args:
            estimator (ParameterEstimator): Estimator for components of mixture. Should have support on integers.
            v (int, optional): Spiked value. Defaults to 0.
            fixed_p (Optional[int], optional): Fixed Dirac spike probability. Defaults to None.
            suff_stat (Optional[float], optional): Regularize estimation on the Dirac probability. Defaults to None.
            pseudo_count (Optional[float], optional): Regularize sufficient statistics. Defaults to None.
            name (Optional[str], optional): Assign a name to the object. Defaults to None.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and components of mixture. Defaults to (None, None).

        Raises:
            TypeError: If keys is not a tuple of two strings or None.
        """
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and all(isinstance(k, (str, type(None))) for k in keys)
        ):
            self.keys = keys
        else:
            raise TypeError("DiracMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]]).")

        self.estimator = estimator
        self.v = v
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_p_vec = (
            np.asarray([fixed_p, 1 - fixed_p]) if fixed_p is not None and 0 < fixed_p <= 1 else None
        )

    def accumulator_factory(self) -> 'DiracMixtureAccumulatorFactory':
        """Return a DiracMixtureAccumulatorFactory for this estimator.

        Returns:
            DiracMixtureAccumulatorFactory: Factory object.
        """
        factory = self.estimator.accumulator_factory()
        return DiracMixtureAccumulatorFactory(factory=factory, v=self.v, keys=self.keys, name=self.name)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, SS0]
    ) -> 'DiracMixtureDistribution':
        """Estimate a DiracMixtureDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Not used.
            suff_stat (Tuple[np.ndarray, SS0]): Sufficient statistics.

        Returns:
            DiracMixtureDistribution: Estimated distribution.
        """
        counts, comp_suff_stats = suff_stat
        dist = self.estimator.estimate(counts[0], comp_suff_stats)

        if self.fixed_p_vec is not None:
            p = self.fixed_p_vec[0]
        elif self.pseudo_count is not None and self.suff_stat is None:
            w = counts + self.pseudo_count / 2
            w /= w.sum()
            p = w[0]
        elif self.pseudo_count is not None and self.suff_stat is not None:
            ss = np.array([self.suff_stat, 1 - self.suff_stat])
            w = (counts + ss * self.pseudo_count) / (counts.sum() + self.pseudo_count)
            p = w[0]
        else:
            nobs_loc = counts.sum()
            if nobs_loc == 0:
                p = 0.5
            else:
                w = counts / counts.sum()
                p = w[0]

        return DiracMixtureDistribution(dist=dist, p=p, v=self.v, name=self.name)


class DiracMixtureDataEncoder(DataSequenceEncoder):
    """DiracMixtureDataEncoder object for encoding sequences of Dirac mixture observations.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for distribution with support on the integers.
        v (int): Dirac spike value.
    """

    def __init__(self, encoder: DataSequenceEncoder, v: int = 0) -> None:
        """Initialize DiracMixtureDataEncoder.

        Args:
            encoder (DataSequenceEncoder): DataSequenceEncoder for distribution with support on the integers.
            v (int, optional): Dirac spike value. Defaults to 0.
        """
        self.encoder = encoder
        self.v = v

    def __str__(self) -> str:
        """Return string representation."""
        return f'DiracMixtureDataEncoder(encoder={repr(self.encoder)}, v={repr(self.v)})'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Other object.

        Returns:
            bool: True if encoders are equal.
        """
        if isinstance(other, DiracMixtureDataEncoder):
            if other.encoder == self.encoder:
                return other.v == self.v
            else:
                return False
        else:
            return False

    def seq_encode(self, x: Sequence[int]) -> 'DiracMixtureEncodedDataSequence':
        """Encode a sequence of integers for DiracMixtureDistribution.

        Args:
            x (Sequence[int]): Sequence of integer observations.

        Returns:
            DiracMixtureEncodedDataSequence: Encoded data sequence.
        """
        x = np.asarray(x, dtype=np.int32)
        xi_v = np.flatnonzero(x == self.v).astype(np.int32)
        xi_nv = np.flatnonzero(x != self.v).astype(np.int32)
        return DiracMixtureEncodedDataSequence(data=(len(x), xi_v, xi_nv, self.encoder.seq_encode(x)))


class DiracMixtureEncodedDataSequence(EncodedDataSequence):
    """DiracMixtureEncodedDataSequence object for use with vectorized function calls.

    Attributes:
        data (Tuple[int, np.ndarray, np.ndarray, EncodedDataSequence]): Encoded sequence of iid Dirac mixture observations.
    """

    def __init__(self, data: Tuple[int, np.ndarray, np.ndarray, EncodedDataSequence]) -> None:
        """Initialize DiracMixtureEncodedDataSequence.

        Args:
            data (Tuple[int, np.ndarray, np.ndarray, EncodedDataSequence]): Encoded sequence of iid Dirac mixture observations.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'DiracMixtureEncodedDataSequence(data={self.data})'

