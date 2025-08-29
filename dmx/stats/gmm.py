"""Create, estimate, and sample from a univariate Gaussian mixture distribution.

This module defines the GaussianMixtureDistribution, GaussianMixtureSampler, GaussianMixtureAccumulatorFactory,
GaussianMixtureAccumulator, GaussianMixtureEstimator, and GaussianMixtureDataEncoder classes for use with dmx-learn.

The GaussianMixtureDistribution allows users to key the variance parameter across all components. This differs from
MixtureDistribution([GaussianDistribution()]*K), as you cannot key the variance parameter while allowing for different
means in components.
"""
import numpy as np
from dmx.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    ParameterEstimator,
    DistributionSampler,
    StatisticAccumulatorFactory,
    SequenceEncodableStatisticAccumulator,
    DataSequenceEncoder,
    EncodedDataSequence,
)
from numpy.random import RandomState
import dmx.utils.vector as vec
from dmx.arithmetic import maxrandint
from typing import List, Union, Tuple, Any, Optional, TypeVar, Sequence, Dict

E = np.ndarray
SS = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class GaussianMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """GaussianMixtureDistribution for a mixture of univariate Gaussian distributions.

    Attributes:
        mu (np.ndarray): Means of each mixture component.
        sigma2 (Union[float, np.ndarray]): Variance for each Gaussian.
        w (np.ndarray): Weights for each mixture component.
        zw (np.ndarray): Indicator for zero weights.
        log_w (np.ndarray): Log of mixture weights.
        log_c (Union[float, np.ndarray]): Constant for Gaussian distributions.
        num_components (int): Number of mixture components.
        _tied (bool): True if sigma2 is the same across all components.
        name (Optional[str]): Name for the object.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and parameters.
    """

    def __init__(
        self,
        mu: Union[Sequence[float], np.ndarray],
        sigma2: Union[Sequence[float], np.ndarray, float],
        w: Union[np.ndarray, List[float]],
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """Initialize GaussianMixtureDistribution.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Means of each mixture component.
            sigma2 (Union[Sequence[float], np.ndarray, float]): Variance for each Gaussian.
            w (Union[np.ndarray, List[float]]): Weights for each mixture component.
            name (Optional[str], optional): Name for the object.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and parameters.
        """
        if isinstance(sigma2, float):
            self._tied = True
            self.sigma2 = sigma2
            self.log_c = -0.5 * np.log(2 * np.pi)
        else:
            self._tied = False
            self.sigma2 = sigma2 if isinstance(sigma2, np.ndarray) else np.asarray(sigma2, dtype=np.float64)
            self.sigma2 = np.reshape(self.sigma2, (1, -1))
            self.log_c = -0.5 * np.log(2 * np.pi)

        self.mu = mu if isinstance(mu, np.ndarray) else np.asarray(mu, dtype=np.float64)
        self.mu = np.reshape(self.mu, (1, -1))
        self.w = w if isinstance(w, np.ndarray) else np.asarray(w, dtype=np.float64)
        self.zw = (self.w == 0.0)
        self.log_w = np.log(self.w + self.zw)
        self.log_w[self.zw] = -np.inf

        self.name = name
        self.keys = keys
        self.num_components = len(self.w)

    def __repr__(self) -> str:
        """Return string representation."""
        s1 = repr(list(self.mu.flatten()))
        s2 = repr(list(self.sigma2.flatten())) if not self._tied else repr(self.sigma2)
        s3 = repr(list(self.w))
        s4 = repr(self.name)
        s5 = repr(self.keys)
        return f'GaussianMixtureDistribution(mu={s1}, sigma2={s2}, w={s3}, name={s4}, keys={s5})'

    def density(self, x: float) -> float:
        """Evaluate the density of the mixture at x.

        Args:
            x (float): Observation.

        Returns:
            float: Density at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Evaluate the log-density of the mixture at x.

        Args:
            x (float): Observation.

        Returns:
            float: Log-density at x.
        """
        return vec.log_sum(
            -0.5 * (x - self.mu) ** 2 / self.sigma2 - 0.5 * np.log(self.sigma2) + self.log_c + self.log_w
        )

    def component_log_density(self, x: float) -> np.ndarray:
        """Evaluate the log-density for each component at x.

        Args:
            x (float): Observation.

        Returns:
            np.ndarray: Log-density for each component.
        """
        return -0.5 * (x - self.mu) ** 2 / self.sigma2 - 0.5 * np.log(self.sigma2) + self.log_c

    def posterior(self, x: float) -> np.ndarray:
        """Compute the posterior probabilities for each component given x.

        Args:
            x (float): Observation.

        Returns:
            np.ndarray: Posterior probabilities for each component.
        """
        comp_log_density = -0.5 * (x - self.mu) ** 2 / self.sigma2 - 0.5 * np.log(self.sigma2) + self.log_c
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

    def seq_component_log_density(self, x: 'GaussianMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for each component for encoded data.

        Args:
            x (GaussianMixtureEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density matrix (n_samples, n_components).
        """
        if not isinstance(x, GaussianMixtureEncodedDataSequence):
            raise Exception("GaussianMixtureEncodedDataSequence required for seq_component_log_density().")

        rv = -0.5 * (x.data[:, None] - self.mu) ** 2 / self.sigma2 + self.log_c
        rv[:, self.zw] = -np.inf
        return rv

    def seq_log_density(self, x: 'GaussianMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (GaussianMixtureEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values for each observation.
        """
        if not isinstance(x, GaussianMixtureEncodedDataSequence):
            raise Exception("GaussianMixtureEncodedDataSequence required for seq_log_density().")

        ll_mat = -0.5 * (x.data[:, None] - self.mu) ** 2 / self.sigma2 + self.log_c - 0.5 * np.log(self.sigma2)
        ll_mat += self.log_w[None, :]
        ll_mat[:, self.zw] = -np.inf

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

    def seq_posterior(self, x: 'GaussianMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized posterior probabilities for encoded data.

        Args:
            x (GaussianMixtureEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Posterior probabilities (n_samples, n_components).
        """
        if not isinstance(x, GaussianMixtureEncodedDataSequence):
            raise Exception("GaussianMixtureEncodedDataSequence required for seq_posterior().")

        ll_mat = -0.5 * (x.data[:, None] - self.mu) ** 2 / self.sigma2 + self.log_c
        ll_mat += self.log_w[None, :]
        ll_mat[:, self.zw] = -np.inf

        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w.copy()
        ll_max[bad_rows] = np.max(self.log_w)
        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        ll_mat /= ll_max

        return ll_mat

    def sampler(self, seed: Optional[int] = None) -> 'GaussianMixtureSampler':
        """Return a GaussianMixtureSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            GaussianMixtureSampler: Sampler object.
        """
        return GaussianMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GaussianMixtureEstimator':
        """Return a GaussianMixtureEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            GaussianMixtureEstimator: Estimator object.
        """
        pc = (pseudo_count, pseudo_count, pseudo_count)
        return GaussianMixtureEstimator(
            num_components=self.num_components,
            pseudo_count=pc,
            tied=self._tied,
            name=self.name,
            keys=self.keys
        )

    def dist_to_encoder(self) -> 'GaussianMixtureDataEncoder':
        """Return a GaussianMixtureDataEncoder for this distribution.

        Returns:
            GaussianMixtureDataEncoder: Encoder object.
        """
        return GaussianMixtureDataEncoder()


class GaussianMixtureSampler(DistributionSampler):
    """Sampler for GaussianMixtureDistribution.

    Attributes:
        rng (RandomState): Random number generator.
        dist (GaussianMixtureDistribution): Distribution to sample from.
        _tied (bool): True if variance is tied across components.
    """

    def __init__(self, dist: GaussianMixtureDistribution, seed: Optional[int] = None) -> None:
        """Initialize GaussianMixtureSampler.

        Args:
            dist (GaussianMixtureDistribution): Distribution to sample from.
            seed (Optional[int], optional): Seed for random number generator.
        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self._tied = isinstance(self.dist.sigma2, float)

    def sample(self, size: Optional[int] = None) -> Union[float, List[float]]:
        """Draw iid samples from the Gaussian mixture distribution.

        Args:
            size (Optional[int], optional): Number of samples to draw. If None, returns a single sample.

        Returns:
            Union[float, List[float]]: Single sample or list of samples.
        """
        comp_state = self.rng.choice(
            range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w
        )
        if self._tied:
            z = self.rng.normal(
                loc=self.dist.mu[0, comp_state], scale=np.sqrt(self.dist.sigma2), size=size
            )
        else:
            z = self.rng.normal(
                loc=self.dist.mu[0, comp_state],
                scale=np.sqrt(self.dist.sigma2[0, comp_state]),
                size=size
            )
        return z if size is None else z.tolist()


class GaussianMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of the Gaussian mixture distribution.

    Attributes:
        num_components (int): Number of mixture components.
        tied (bool): If variance is the same across all mixture components.
        comp_counts (np.ndarray): Sufficient statistics for component counts.
        xw (np.ndarray): Sufficient statistics for means.
        x2w (np.ndarray): Sufficient statistics for variances.
        wcnt (np.ndarray): Sufficient statistics for means.
        wcnt2 (np.ndarray): Sufficient statistics for variances.
        weight_key (Optional[str]): Key for weights.
        comp_key (Optional[str]): Key for components.
        name (Optional[str]): Name for the accumulator.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and parameters.
    """

    def __init__(
        self,
        num_components: int,
        tied: bool = False,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """Initialize GaussianMixtureAccumulator.

        Args:
            num_components (int): Number of mixture components.
            tied (bool, optional): If variance is the same across all mixture components.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and parameters.
            name (Optional[str], optional): Name for the accumulator.
        """
        self.num_components = num_components
        self.tied = tied
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.xw = np.zeros(self.num_components, dtype=float)
        self.x2w = np.zeros(self.num_components, dtype=float)
        self.wcnt = np.zeros(self.num_components, dtype=float)
        self.wcnt2 = np.zeros(self.num_components, dtype=float)

        self.weight_key = keys[0]
        self.comp_key = keys[1]

        self._init_rng: bool = False
        self._acc_rng: Optional[List[RandomState]] = None
        self.name = name

    def update(self, x: float, weight: float, estimate: 'GaussianMixtureDistribution') -> None:
        """Update accumulator with a new observation.

        Args:
            x (float): Observation.
            weight (float): Weight for the observation.
            estimate (GaussianMixtureDistribution): Distribution for posterior calculation.
        """
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior
        self.wcnt += posterior
        self.wcnt2 += posterior
        self.xw = x * posterior
        self.x2w = x ** 2 * posterior

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize random number generators for accumulator.

        Args:
            rng (RandomState): Random number generator.
        """
        seeds = rng.randint(2 ** 31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: float, weight: float, rng: RandomState) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (float): Observation.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if weight != 0:
            ww = self._w_rng.dirichlet(np.ones(self.num_components) / (self.num_components * self.num_components))
            ww *= weight
            self.xw = x * ww
            self.x2w = x ** 2 * ww
            self.wcnt += ww
            self.wcnt2 += ww
            self.comp_counts += ww

    def seq_initialize(
        self,
        x: 'GaussianMixtureEncodedDataSequence',
        weights: np.ndarray,
        rng: RandomState
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (GaussianMixtureEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        keep_idx = weights > 0
        keep_len = np.count_nonzero(keep_idx)
        ww = np.zeros((sz, self.num_components))

        c = 20 ** 2 if self.num_components > 20 else self.num_components ** 2
        if keep_len > 0:
            ww[keep_idx, :] = self._w_rng.dirichlet(alpha=np.ones(self.num_components), size=keep_len)
        ww *= np.reshape(weights, (sz, 1))

        w_sum = ww.sum(axis=0)
        self.comp_counts += w_sum
        self.wcnt += w_sum
        self.wcnt2 += w_sum

        self.xw += np.dot(x.data, ww)
        self.x2w += np.dot(x.data ** 2, ww)

    def seq_update(
        self,
        x: 'GaussianMixtureEncodedDataSequence',
        weights: np.ndarray,
        estimate: 'GaussianMixtureDistribution'
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (GaussianMixtureEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (GaussianMixtureDistribution): Distribution for posterior calculation.
        """
        ll_mat = -0.5 * (x.data[:, None] - estimate.mu) ** 2 / estimate.sigma2 + estimate.log_c
        ll_mat += estimate.log_w[None, :]
        ll_mat[:, estimate.zw] = -np.inf

        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = estimate.log_w.copy()
        ll_max[bad_rows] = np.max(estimate.log_w)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        np.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max

        w_sum = ll_mat.sum(axis=0)
        self.comp_counts += w_sum
        self.wcnt += w_sum
        self.wcnt2 += w_sum

        self.xw += np.dot(x.data, ll_mat)
        self.x2w += np.dot(x.data ** 2, ll_mat)

    def combine(self, suff_stat: SS) -> 'GaussianMixtureAccumulator':
        """Aggregate sufficient statistics with this accumulator.

        Args:
            suff_stat (SS): Sufficient statistics to combine.

        Returns:
            GaussianMixtureAccumulator: Self after combining.
        """
        self.comp_counts += suff_stat[0]
        self.xw += suff_stat[1]
        self.x2w += suff_stat[2]
        self.wcnt += suff_stat[3]
        self.wcnt2 += suff_stat[4]
        return self

    def value(self) -> SS:
        """Return the sufficient statistics as a tuple.

        Returns:
            SS: Tuple of sufficient statistics.
        """
        return self.comp_counts, self.xw, self.x2w, self.wcnt, self.wcnt2

    def from_value(self, x: SS) -> 'GaussianMixtureAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (SS): Sufficient statistics.

        Returns:
            GaussianMixtureAccumulator: Self after setting values.
        """
        self.comp_counts, self.xw, self.x2w, self.wcnt, self.wcnt2 = x
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
                x = stats_dict[self.comp_key]
                self.xw += x[0]
                self.x2w += x[1]
                self.wcnt += x[2]
                self.wcnt2 += x[3]
            else:
                stats_dict[self.comp_key] = (self.xw, self.x2w, self.wcnt, self.wcnt2)

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
                self.xw, self.x2w, self.wcnt, self.wcnt2 = stats_dict[self.comp_key]

    def acc_to_encoder(self) -> 'GaussianMixtureDataEncoder':
        """Return a GaussianMixtureDataEncoder for this accumulator.

        Returns:
            GaussianMixtureDataEncoder: Encoder object.
        """
        return GaussianMixtureDataEncoder()


class GaussianMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating GaussianMixtureAccumulator objects.

    Attributes:
        num_components (int): Number of mixture components.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and parameters.
        tied (bool): If variance is the same across all mixture components.
        name (Optional[str]): Name for the factory.
    """

    def __init__(
        self,
        num_components: int,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None,
        tied: bool = False
    ) -> None:
        """Initialize GaussianMixtureAccumulatorFactory.

        Args:
            num_components (int): Number of mixture components.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and parameters.
            name (Optional[str], optional): Name for the factory.
            tied (bool, optional): If variance is the same across all mixture components.
        """
        self.keys = keys
        self.num_components = num_components
        self.tied = tied
        self.name = name

    def make(self) -> 'GaussianMixtureAccumulator':
        """Create a new GaussianMixtureAccumulator.

        Returns:
            GaussianMixtureAccumulator: New accumulator instance.
        """
        return GaussianMixtureAccumulator(
            keys=self.keys, num_components=self.num_components, tied=self.tied, name=self.name
        )


class GaussianMixtureEstimator(ParameterEstimator):
    """Estimator for GaussianMixtureDistribution from aggregated sufficient statistics.

    Notes:
        Set equal variance with `tied` parameter.

    Attributes:
        num_components (int): Number of mixture components.
        pseudo_count (Tuple[Optional[float], Optional[float], Optional[float]]): Pseudo-counts for weights, mean, variance.
        suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]): Sufficient statistics for weights, mean, and variance.
        tied (bool): If True, assume each Gaussian mixture has the same variance.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and mixture components.
        fixed_weights (Optional[np.ndarray]): If not None, weights of the mixture are assumed fixed.
        name (Optional[str]): Name for the estimator.
    """

    def __init__(
        self,
        num_components: int,
        fixed_weights: Optional[Union[List[float], np.ndarray]] = None,
        suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]] = (None, None, None),
        pseudo_count: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None),
        tied: bool = False,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """Initialize GaussianMixtureEstimator.

        Args:
            num_components (int): Number of mixture components.
            fixed_weights (Optional[Union[List[float], np.ndarray]], optional): If not None, weights are fixed.
            suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]], optional): Sufficient statistics for weights, mean, and variance.
            pseudo_count (Tuple[Optional[float], Optional[float], Optional[float]], optional): Pseudo-counts for weights, mean, variance.
            tied (bool, optional): If True, assume each Gaussian mixture has the same variance.
            keys (Tuple[Optional[str], Optional[str]], optional): Keys for weights and mixture components.
            name (Optional[str], optional): Name for the estimator.

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
            raise TypeError("GaussianMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]]).")

        self.num_components = num_components
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.tied = tied
        self.fixed_weights = np.asarray(fixed_weights) if fixed_weights is not None else None
        self.name = name

    def accumulator_factory(self) -> 'GaussianMixtureAccumulatorFactory':
        """Return a GaussianMixtureAccumulatorFactory for this estimator.

        Returns:
            GaussianMixtureAccumulatorFactory: Factory object.
        """
        return GaussianMixtureAccumulatorFactory(
            keys=self.keys, tied=self.tied, num_components=self.num_components, name=self.name
        )

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'GaussianMixtureDistribution':
        """Estimate a GaussianMixtureDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations (not used).
            suff_stat (SS): Sufficient statistics.

        Returns:
            GaussianMixtureDistribution: Estimated distribution.
        """
        num_components = self.num_components
        counts, xw, x2w, wcnts, _ = suff_stat

        # Estimate mean
        if self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            mu = (xw + self.pseudo_count[1] * self.suff_stat[1]) / (wcnts + self.pseudo_count[1])
        else:
            nobs_loc = wcnts.copy()
            nobs_loc[nobs_loc == 0.0] = 1.0
            mu = xw / nobs_loc

        # Estimate variance
        if self.tied:
            if self.pseudo_count[2] is not None and self.suff_stat[2] is not None:
                sigma2 = np.sum(x2w - mu * mu * wcnts + self.pseudo_count[2] * self.suff_stat[2])
                sigma2 /= np.sum(wcnts + self.pseudo_count[2])
            else:
                sigma2 = np.sum(x2w - mu * mu * wcnts) / np.sum(wcnts)
        else:
            if self.pseudo_count[2] is not None and self.suff_stat[2] is not None:
                sigma2 = (x2w - mu * mu * wcnts + self.pseudo_count[2] * self.suff_stat[2]) / (
                    wcnts + self.pseudo_count[2]
                )
            else:
                nobs_loc = wcnts.copy()
                nobs_loc[nobs_loc == 0.0] = 1.0
                sigma2 = x2w / nobs_loc - mu * mu

        # Estimate mixture weights
        if self.fixed_weights is not None:
            w = np.asarray(self.fixed_weights)
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is None:
            p = self.pseudo_count[0] / num_components
            w = counts + p
            w /= w.sum()
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            w = (counts + self.suff_stat[0] * self.pseudo_count[0]) / (
                counts.sum() + self.pseudo_count[0] * self.suff_stat[0].sum()
            )
        else:
            nobs_loc = counts.sum()
            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        return GaussianMixtureDistribution(mu=mu, sigma2=sigma2, w=w)


class GaussianMixtureDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of iid Gaussian mixture observations."""

    def __str__(self) -> str:
        """Return string representation."""
        return 'GaussianMixtureDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if encoders are equal.
        """
        return isinstance(other, GaussianMixtureDataEncoder)

    def seq_encode(self, x: Union[Sequence[float], np.ndarray]) -> 'GaussianMixtureEncodedDataSequence':
        """Encode a sequence of Gaussian mixture observations.

        Args:
            x (Union[Sequence[float], np.ndarray]): Sequence of observations.

        Returns:
            GaussianMixtureEncodedDataSequence: Encoded data sequence.
        """
        return GaussianMixtureEncodedDataSequence(x if isinstance(x, np.ndarray) else np.asarray(x, dtype=float))


class GaussianMixtureEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for use with vectorized function calls.

    Attributes:
        data (np.ndarray): Sequence of iid Gaussian mixture observations.
    """

    def __init__(self, data: np.ndarray):
        """Initialize GaussianMixtureEncodedDataSequence.

        Args:
            data (np.ndarray): Sequence of iid Gaussian mixture observations.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'GaussianMixtureEncodedDataSequence(data={self.data})'




