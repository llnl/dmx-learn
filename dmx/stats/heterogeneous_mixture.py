"""Create, estimate, and sample from a heterogeneous mixture distribution.

Defines the HeterogeneousMixtureDistribution, HeterogeneousMixtureSampler, HeterogeneousMixtureAccumulatorFactory,
HeterogeneousMixtureAccumulator, HeterogeneousMixtureEstimator, and the HeterogeneousMixtureDataEncoder classes for use
with dmx-learn.

A heterogeneous mixture distribution with data type ``T`` is defined by the density:

.. math::

    p(Y) = \sum_{k=1}^{K} p(Y|Z=k) \cdot p(Z=k)

where :math:`p(Z=k)` is the mixture weight for component :math:`k`, and :math:`p(Y|Z=k)` is the density of the
:math:`k`-th component distribution. All component distributions must be compatible in data type ``T``.

**Example:**  
A heterogeneous mixture with weights ``[0.5, 0.5]`` and component distributions ``Exponential(beta)`` and ``Gamma(k, theta)``
has the form

.. math::

    p(x) = 0.5 \cdot P_0(x; \beta) + 0.5 \cdot P_1(x; k, \theta), \quad x > 0

where :math:`P_0(x; \beta)` is an exponential density and :math:`P_1(x; k, \theta)` is a gamma density.
"""
import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from math import exp
from dmx.arithmetic import maxrandint
import dmx.utils.vector as vec
from dmx.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    StatisticAccumulatorFactory,
    SequenceEncodableStatisticAccumulator,
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    EncodedDataSequence,
)
from typing import Optional, Union, Tuple, Any, TypeVar, List, Dict, Sequence

T = TypeVar('T')


class HeterogeneousMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """HeterogeneousMixtureDistribution object defined by component distributions and weights.

    Attributes:
        components (Sequence[SequenceEncodableProbabilityDistribution]): List of component distributions (data type T).
        w (np.ndarray): Mixture weights assigned from args (w).
        name (Optional[str]): String name for the HeterogeneousMixtureDistribution object.
        zw (np.ndarray): True if a weight is 0.0, else False.
        log_w (np.ndarray): Log of weights (w). Set to -np.inf where zw is True.
        num_components (int): Number of components in HeterogeneousMixtureDistribution instance.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components.
    """

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        w: Union[Sequence[float], np.ndarray],
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """Initialize HeterogeneousMixtureDistribution.

        Args:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Set component distributions.
                Must all be compatible with type T.
            w (Union[Sequence[float], np.ndarray]): Mixture weights, must sum to 1.0.
            name (Optional[str], optional): Assign string name to HeterogeneousMixtureDistribution object.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and components.
        """
        self.w = np.asarray(w, dtype=float)
        self.zw = (self.w == 0.0)
        self.log_w = np.log(self.w + self.zw)
        self.log_w[self.zw] = -np.inf

        self.components = components
        self.num_components = len(components)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation."""
        s1 = ','.join([str(u) for u in self.components])
        s2 = repr(list(self.w))
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return f'HeterogeneousMixtureDistribution(components=[{s1}], w={s2}, name={s3}, keys={s4})'

    def density(self, x: T) -> float:
        """Evaluate density of heterogeneous mixture distribution at observation x.

        Args:
            x (T): Single observation from heterogeneous mixture distribution. T is data type of components.

        Returns:
            float: Density at x.
        """
        return exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        """Evaluate log-density of heterogeneous mixture distribution at observation x.

        .. math::

           \\log{f(x)} = \\log{\\left(\\sum_{k=1}^{K} f_k(x) \\pi_k\\right)}.

        Args:
            x (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            float: Log-density at x.
        """
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def component_log_density(self, x: T) -> np.ndarray:
        """Evaluate component-wise log-density of heterogeneous mixture distribution at observation x.

        Returns a num_components-dimensional array with :math:`\\log(f_k(x))` in each entry.

        Args:
            x (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            np.ndarray: Component-wise log-density at x.
        """
        return np.asarray([m.log_density(x) for m in self.components], dtype=np.float64)

    def posterior(self, x: T) -> np.ndarray:
        """Obtain the posterior distribution for each heterogeneous mixture component at observation x.

        .. math::

           f(z=k \\vert x ) = \\frac{f_k(x) \\pi_k}{\\sum_{k=1}^{K} f_k(x) \\pi_k}

        Args:
            x (T): Single observation from mixture distribution. T is data type of components.

        Returns:
            np.ndarray: Posterior distribution at observation x.
        """
        comp_log_density = np.asarray([m.log_density(x) for m in self.components])
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -np.inf

        max_val = np.max(comp_log_density)

        if max_val == -np.inf:
            return self.w.copy()
        else:
            comp_log_density -= max_val
            np.exp(comp_log_density, out=comp_log_density)
            comp_log_density /= comp_log_density.sum()
            return comp_log_density

    def seq_log_density(self, x: 'HeterogeneousMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation of log-density for encoded sequence x.

        Args:
            x (HeterogeneousMixtureEncodedDataSequence): EncodedDataSequence for Heterogeneous Mixture.

        Returns:
            np.ndarray: log_density of each observation in encoded sequence.
        """
        tag_list, enc_data = x.data
        ll_mat_init = False

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    if not ll_mat_init:
                        ll_mat = np.zeros((len(temp), self.num_components))
                        ll_mat.fill(-np.inf)
                        ll_mat_init = True
                    ll_mat[:, i] = temp
                    ll_mat[:, i] += self.log_w[i]

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

    def seq_component_log_density(self, x: 'HeterogeneousMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation of component-wise log-density for encoded sequence x.

        Args:
            x (HeterogeneousMixtureEncodedDataSequence): EncodedDataSequence for Heterogeneous Mixture.

        Returns:
            np.ndarray: 2-d array of shape (n_samples, n_components).
        """
        tag_list, enc_data = x.data
        ll_mat_init = False

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    if not ll_mat_init:
                        ll_mat = np.zeros((len(temp), self.num_components))
                        ll_mat.fill(-np.inf)
                        ll_mat_init = True
                    ll_mat[:, i] = temp

        return ll_mat

    def seq_posterior(self, x: 'HeterogeneousMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation of posterior of HeterogeneousMixtureDistribution for encoded sequence x.

        Args:
            x (HeterogeneousMixtureEncodedDataSequence): EncodedDataSequence for Heterogeneous Mixture.

        Returns:
            np.ndarray: Posterior probabilities for each observation in encoded sequence.
        """
        if not isinstance(x, HeterogeneousMixtureEncodedDataSequence):
            raise Exception('Input must be HeterogeneousMixtureEncodedDataSequence.')

        tag_list, enc_data = x.data
        ll_mat_init = False

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not self.zw[i]:
                    temp = self.components[i].seq_log_density(enc_data[tag])
                    if not ll_mat_init:
                        ll_mat = np.zeros((len(temp), self.num_components))
                        ll_mat.fill(-np.inf)
                        ll_mat_init = True
                    ll_mat[:, i] = temp
                    ll_mat[:, i] += self.log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w.copy()
        ll_max[bad_rows] = np.max(self.log_w)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        ll_mat /= ll_max

        return ll_mat

    def sampler(self, seed: Optional[int] = None) -> 'HeterogeneousMixtureSampler':
        """Return a HeterogeneousMixtureSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            HeterogeneousMixtureSampler: Sampler object.
        """
        return HeterogeneousMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'HeterogeneousMixtureEstimator':
        """Return a HeterogeneousMixtureEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            HeterogeneousMixtureEstimator: Estimator object.
        """
        if pseudo_count is not None:
            return HeterogeneousMixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count, name=self.name, keys=self.keys)
        else:
            return HeterogeneousMixtureEstimator([u.estimator() for u in self.components], name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'HeterogeneousMixtureDataEncoder':
        """Return a HeterogeneousMixtureDataEncoder for this distribution.

        Returns:
            HeterogeneousMixtureDataEncoder: Encoder object.
        """
        encoders = [comp.dist_to_encoder() for comp in self.components]
        return HeterogeneousMixtureDataEncoder(encoders=encoders)


class HeterogeneousMixtureSampler(DistributionSampler):
    """Sampler for HeterogeneousMixtureDistribution.

    Attributes:
        dist (HeterogeneousMixtureDistribution): Distribution to sample from.
        rng (RandomState): Seeded RandomState for sampling.
        comp_samplers (List[DistributionSampler]): List of DistributionSampler objects for each mixture component.
    """

    def __init__(self, dist: HeterogeneousMixtureDistribution, seed: Optional[int] = None) -> None:
        """Initialize HeterogeneousMixtureSampler.

        Args:
            dist (HeterogeneousMixtureDistribution): Distribution to draw samples from.
            seed (Optional[int], optional): Seed for sampling with RandomState.
        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size: Optional[int] = None) -> Union[Any, Sequence[Any]]:
        """Draw iid samples from a heterogeneous mixture distribution.

        Args:
            size (Optional[int], optional): Number of iid samples to draw.

        Returns:
            Any or Sequence[Any]: Single sample or list of samples.
        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)
        if size is None:
            return self.comp_samplers[comp_state].sample()
        else:
            return [self.comp_samplers[i].sample() for i in comp_state]


class HeterogeneousMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of the heterogeneous mixture distribution.

    Attributes:
        accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the components.
        num_components (int): Number of mixture components.
        comp_counts (np.ndarray): Array for accumulating component weights.
        weight_key (Optional[str]): Key for weights of mixture.
        comp_key (Optional[str]): Key for components of mixture.
        name (Optional[str]): Name for object.
        _init_rng (bool): False if rng for accumulators has not been set.
        _acc_rng (Optional[Sequence[RandomState]]): List of RandomState objects for accumulator initialization.
    """

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """Initialize HeterogeneousMixtureAccumulator.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the components.
            name (Optional[str], optional): Name for object.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and components.
        """
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.name = name
        self._init_rng: bool = False
        self._acc_rng: Optional[Sequence[RandomState]] = None

    def update(self, x: T, weight: float, estimate: HeterogeneousMixtureDistribution) -> None:
        """Update accumulator with a new observation.

        Args:
            x (T): Observation.
            weight (float): Weight for the observation.
            estimate (HeterogeneousMixtureDistribution): Distribution for posterior calculation.
        """
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior
        for i in range(self.num_components):
            self.accumulators[i].update(x, posterior[i], estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize random number generators for accumulators.

        Args:
            rng (RandomState): Random number generator.
        """
        seeds = rng.randint(2 ** 31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._init_rng = True

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (T): Observation.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if weight != 0:
            ww = rng.dirichlet(np.ones(self.num_components) / (self.num_components * self.num_components))
        else:
            ww = np.zeros(self.num_components)

        for i in range(self.num_components):
            w = weight * ww[i]
            self.accumulators[i].initialize(x, w, self._acc_rng[i])
            self.comp_counts[i] += w

    def seq_initialize(
        self,
        x: 'HeterogeneousMixtureEncodedDataSequence',
        weights: np.ndarray,
        rng: RandomState
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (HeterogeneousMixtureEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        tag_list, enc_data = x.data
        sz = len(weights)

        keep_idx = weights > 0.0
        keep_len = np.sum(keep_idx)
        ww = np.zeros((sz, self.num_components))

        if keep_len > 0:
            ww[keep_idx, :] = rng.dirichlet(alpha=np.ones(self.num_components) / (self.num_components ** 2),
                                            size=keep_len)
        ww *= np.reshape(weights, (sz, 1))

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                self.accumulators[i].seq_initialize(enc_data[tag], ww[:, i], self._acc_rng[i])
                self.comp_counts[i] += np.sum(ww[:, i])

    def seq_update(
        self,
        x: 'HeterogeneousMixtureEncodedDataSequence',
        weights: np.ndarray,
        estimate: 'HeterogeneousMixtureDistribution'
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (HeterogeneousMixtureEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (HeterogeneousMixtureDistribution): Distribution for posterior calculation.
        """
        tag_list, enc_data = x.data
        ll_mat_init = False

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                if not estimate.zw[i]:
                    temp = estimate.components[i].seq_log_density(enc_data[tag])
                    if not ll_mat_init:
                        ll_mat = np.zeros((len(temp), self.num_components), dtype=np.float64)
                        ll_mat.fill(-np.inf)
                        ll_mat_init = True
                    ll_mat[:, i] = temp
                    ll_mat[:, i] += estimate.log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = estimate.log_w.copy()
        ll_max[bad_rows] = np.max(estimate.log_w)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        np.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max

        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                w_loc = ll_mat[:, i]
                self.comp_counts[i] += w_loc.sum()
                self.accumulators[i].seq_update(enc_data[tag], w_loc, estimate.components[i])

    def combine(self, suff_stat: Tuple[np.ndarray, Tuple[Any, ...]]) -> 'HeterogeneousMixtureAccumulator':
        """Aggregate sufficient statistics with this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, Tuple[Any, ...]]): Sufficient statistics to combine.

        Returns:
            HeterogeneousMixtureAccumulator: Self after combining.
        """
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])
        return self

    def value(self) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[np.ndarray, Tuple[Any, ...]]: (component counts, tuple of component accumulator values)
        """
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: Tuple[np.ndarray, Tuple[Any, ...]]) -> 'HeterogeneousMixtureAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[np.ndarray, Tuple[Any, ...]]): (component counts, tuple of component accumulator values)

        Returns:
            HeterogeneousMixtureAccumulator: Self after setting values.
        """
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
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
                acc = stats_dict[self.comp_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

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
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'HeterogeneousMixtureDataEncoder':
        """Return a HeterogeneousMixtureDataEncoder for this accumulator.

        Returns:
            HeterogeneousMixtureDataEncoder: Encoder object.
        """
        encoders = [comp.acc_to_encoder() for comp in self.accumulators]
        return HeterogeneousMixtureDataEncoder(encoders=encoders)


class HeterogeneousMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating HeterogeneousMixtureAccumulator objects.

    Attributes:
        factories (Sequence[StatisticAccumulatorFactory]): Factories for the mixture components.
        dim (int): Number of mixture components.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """Initialize HeterogeneousMixtureAccumulatorFactory.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Factories for the mixture components.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and component aggregations.
            name (Optional[str], optional): Name for object.
        """
        self.factories = factories
        self.dim = len(factories)
        self.keys = keys
        self.name = name

    def make(self) -> 'HeterogeneousMixtureAccumulator':
        """Create a new HeterogeneousMixtureAccumulator.

        Returns:
            HeterogeneousMixtureAccumulator: New accumulator instance.
        """
        return HeterogeneousMixtureAccumulator(
            [self.factories[i].make() for i in range(self.dim)],
            keys=self.keys,
            name=self.name
        )


class HeterogeneousMixtureEstimator(ParameterEstimator):
    """Estimator for HeterogeneousMixtureDistribution from aggregated sufficient statistics.

    Attributes:
        estimators (Sequence[ParameterEstimator]): Estimators for the mixture components.
        fixed_weights (Optional[np.ndarray]): Fixed weights for the mixture (if any).
        suff_stat (Optional[np.ndarray]): Sufficient statistics for the weights.
        pseudo_count (Optional[float]): Pseudo-count for regularization.
        name (Optional[str]): Name for the estimator.
        keys (Tuple[Optional[str], Optional[str]]): Keys for the weights and component distributions.
    """

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        fixed_weights: Optional[np.ndarray] = None,
        suff_stat: Optional[np.ndarray] = None,
        pseudo_count: Optional[float] = None,
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """Initialize HeterogeneousMixtureEstimator.

        Args:
            estimators (Sequence[ParameterEstimator]): Estimators for the mixture components.
            fixed_weights (Optional[np.ndarray], optional): Fixed weights for the mixture.
            suff_stat (Optional[np.ndarray], optional): Sufficient statistics for the weights.
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.
            name (Optional[str], optional): Name for the estimator.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for the weights and component distributions.

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
            raise TypeError("HeterogeneousMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]]).")

        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_weights = fixed_weights

    def accumulator_factory(self) -> 'HeterogeneousMixtureAccumulatorFactory':
        """Return a HeterogeneousMixtureAccumulatorFactory for this estimator.

        Returns:
            HeterogeneousMixtureAccumulatorFactory: Factory object.
        """
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return HeterogeneousMixtureAccumulatorFactory(est_factories, keys=self.keys, name=self.name)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, Tuple[Any, ...]]
    ) -> 'HeterogeneousMixtureDistribution':
        """Estimate a HeterogeneousMixtureDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations (not used).
            suff_stat (Tuple[np.ndarray, Tuple[Any, ...]]): Sufficient statistics.

        Returns:
            HeterogeneousMixtureDistribution: Estimated distribution.
        """
        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i]) for i in range(num_components)]

        if self.fixed_weights is not None:
            w = np.asarray(self.fixed_weights)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / num_components
            w = counts + p
            w /= w.sum()
        elif self.pseudo_count is not None and self.suff_stat is not None:
            w = (counts + self.suff_stat * self.pseudo_count) / (counts.sum() + self.pseudo_count)
        else:
            nobs_loc = counts.sum()
            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        return HeterogeneousMixtureDistribution(components, w, name=self.name)


class HeterogeneousMixtureDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of heterogeneous mixture observations.

    Attributes:
        encoder_dict (Dict[str, DataSequenceEncoder]): Dictionary of distinct DataSequenceEncoder objects
            found in encoders list. Value of encoder_dict is a list of ids for the components that are encoded by
            the encoder_dict key.
        idx_dict (Dict[str, Sequence[int]]): Dictionary mapping encoder string to component indices.
    """

    def __init__(self, encoders: Sequence[DataSequenceEncoder]) -> None:
        """Initialize HeterogeneousMixtureDataEncoder.

        Args:
            encoders (Sequence[DataSequenceEncoder]): List of DataSequenceEncoder objects for each heterogeneous mixture
                component.
        """
        encoder_dict: Dict[str, DataSequenceEncoder] = dict()
        idx_dict: Dict[str, Sequence[int]] = dict()

        for encoder_idx, encoder in enumerate(encoders):
            enc_str = str(encoder)
            if enc_str not in encoder_dict:
                encoder_dict[enc_str] = encoder
                idx_dict[enc_str] = []
            idx_dict[enc_str].append(encoder_idx)

        self.encoder_dict: Dict[str, DataSequenceEncoder] = encoder_dict
        self.idx_dict: Dict[str, Sequence[int]] = idx_dict

    def __str__(self) -> str:
        """Return string representation."""
        s = 'HeterogeneousMixtureDataEncoder(['
        item_list = list(self.idx_dict.items())
        for enc_str, comp_list in item_list[:-1]:
            s += enc_str + ',comps=' + str(comp_list) + ','
        s += item_list[-1][0] + ',comps=' + str(item_list[-1][1]) + '])'
        return s

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if encoders are equal.
        """
        if not isinstance(other, HeterogeneousMixtureDataEncoder):
            return False
        else:
            return other.idx_dict == self.idx_dict and other.encoder_dict == self.encoder_dict

    def seq_encode(self, x: Sequence[T]) -> 'HeterogeneousMixtureEncodedDataSequence':
        """Encode a sequence of heterogeneous mixture observations.

        Args:
            x (Sequence[T]): Sequence of observations.

        Returns:
            HeterogeneousMixtureEncodedDataSequence: Encoded data sequence.
        """
        enc_data = []
        tag_list = []

        for enc_str, encoder_idx in self.idx_dict.items():
            tag_list.append(np.asarray(encoder_idx, dtype=int))
            enc_data.append(self.encoder_dict[enc_str].seq_encode(x))

        return HeterogeneousMixtureEncodedDataSequence(data=(tag_list, enc_data))


class HeterogeneousMixtureEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for use with vectorized function calls.

    Attributes:
        data (Tuple[List[np.ndarray], List[EncodedDataSequence]]): Sequence encoding of component tags and data
            encodings.
    """

    def __init__(self, data: Tuple[List[np.ndarray], List[EncodedDataSequence]]) -> None:
        """Initialize HeterogeneousMixtureEncodedDataSequence.

        Args:
            data (Tuple[List[np.ndarray], List[EncodedDataSequence]]): Encoded tags and data.
        """
        super().__init__(data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'HeterogeneousMixtureEncodedDataSequence(data={self.data})'


