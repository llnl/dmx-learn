"""Create, estimate, and sample from a mixture distribution with homogeneous components.

Defines DiagonalGaussianMixtureDistribution and related classes for use with dmx-learn.

MixtureEstimator([DiagonalGaussianEstimator()], keys=(None, 'comps')) keys both means and variances.
DiagonalGaussianEstimator(tied=True) sets the covariance of each mixture component to be the same for each mixture component.
"""

import numba
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


class DiagonalGaussianMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Represents a diagonal Gaussian mixture distribution.

    Attributes:
        mu (np.ndarray): Means of the mixture components (K x D).
        covar (np.ndarray): Covariances of the mixture components (K x D).
        w (np.ndarray): Mixture weights (length K).
        tied (bool): If True, the covariance of each mixture component is tied.
        name (Optional[str]): Name for object.
        keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and parameters.
        dim (int): Dimensionality of the data.
        num_components (int): Number of mixture components.
        zw (np.ndarray): Boolean mask for zero-weight components.
        log_w (np.ndarray): Logarithm of mixture weights.
        log_c (float): Constant term in log-density.
    """

    def __init__(
        self,
        mu: Union[Sequence[Sequence[float]], np.ndarray],
        covar: Union[Sequence[float], np.ndarray],
        w: Union[np.ndarray, List[float]],
        tied: bool = False,
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """
        Initialize the diagonal Gaussian mixture distribution.

        Args:
            mu (Union[Sequence[Sequence[float]], np.ndarray]): Means of the mixture components (K x D).
            covar (Union[Sequence[float], np.ndarray]): Covariances of the mixture components (K x D).
            w (Union[np.ndarray, List[float]]): Mixture weights (length K).
            tied (bool, optional): If True, the covariance of each mixture component is tied. Defaults to False.
            name (Optional[str], optional): Name for object. Defaults to None.
            keys (Tuple[Optional[str], Optional[str]], optional): Set keys for weights and parameters. Defaults to (None, None).
        """
        self.mu = np.asarray(mu, dtype=np.float64)
        self.covar = np.asarray(covar, dtype=np.float64)
        self.dim = self.mu.shape[1]

        self.w = np.asarray(w, dtype=np.float64)
        self.zw = (self.w == 0.0)
        self.log_w = np.log(self.w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.num_components = len(self.w)

        self.tied = tied
        self.log_c = -0.5 * self.dim * np.log(2 * np.pi)
        self.keys = keys
        self.name = name

    def __repr__(self) -> str:
        """Return string representation."""
        s1 = ','.join([repr(u.tolist()) for u in self.mu])
        s2 = ','.join([repr(u.tolist()) for u in self.covar])
        s3 = repr(self.w.tolist())
        s4 = repr(self.tied)
        s5 = repr(self.name)
        s6 = repr(self.keys)
        return f'DiagonalGaussianMixtureDistribution(mu=[{s1}], covar=[{s2}], w={s3}, tied={s4}, name={s5}, keys={s6})'

    def density(self, x: Union[Sequence[float], np.ndarray]) -> float:
        """Compute the density of the distribution at a given point.

        Args:
            x (Union[Sequence[float], np.ndarray]): Input data point.

        Returns:
            float: Density value.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[float], np.ndarray]) -> float:
        """Compute the log density of the distribution at a given point.

        Args:
            x (Union[Sequence[float], np.ndarray]): Input data point.

        Returns:
            float: Log density value.
        """
        z2 = (np.asarray(x)[None, :] - self.mu) ** 2 / self.covar
        ll = -0.5 * np.sum(z2, axis=1) - 0.5 * np.sum(np.log(self.covar), axis=1) + self.log_c
        ll += self.log_w
        ll_max = np.max(ll)
        return np.log(np.sum(np.exp(ll - ll_max))) + ll_max

    def component_log_density(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Compute the log density of the individual mixture components at a given point.

        Args:
            x (Union[Sequence[float], np.ndarray]): Input data point.

        Returns:
            np.ndarray: Log density values for each mixture component.
        """
        z2 = (np.asarray(x)[None, :] - self.mu) ** 2 / self.covar
        ll = -0.5 * np.sum(z2, axis=1) - 0.5 * np.sum(np.log(self.covar), axis=1) + self.log_c
        return ll

    def posterior(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Compute the posterior probabilities of the mixture components given a data point.

        Args:
            x (Union[Sequence[float], np.ndarray]): Input data point.

        Returns:
            np.ndarray: Posterior probabilities for each mixture component.
        """
        ll = self.component_log_density(np.asarray(x))
        ll += self.log_w
        ll_max = np.max(ll)
        ll = np.exp(ll - ll_max)
        return ll / np.sum(ll)

    def seq_component_log_density(self, x: 'DiagonalGaussianMixtureEncodedDataSequence') -> np.ndarray:
        """Compute the log density of the individual mixture components for a sequence of data points.

        Args:
            x (DiagonalGaussianMixtureEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log density values for each mixture component.
        """
        if not isinstance(x, DiagonalGaussianMixtureEncodedDataSequence):
            raise Exception("DiagonalGaussianMixtureEncodedDataSequence required for seq_component_log_density().")
        ll_mat = -0.5 * np.sum((x.data[:, None, :] - self.mu) ** 2 / self.covar, axis=2)
        ll_mat += -0.5 * np.sum(np.log(self.covar), axis=1)
        ll_mat += self.log_c
        return ll_mat

    def seq_log_density(self, x: 'DiagonalGaussianMixtureEncodedDataSequence') -> np.ndarray:
        """Compute the log density for a sequence of data points.

        Args:
            x (DiagonalGaussianMixtureEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log density values for the sequence.
        """
        if not isinstance(x, DiagonalGaussianMixtureEncodedDataSequence):
            raise Exception("DiagonalGaussianMixtureEncodedDataSequence required for seq_log_density().")

        ll_mat = -0.5 * np.sum((x.data[:, None, :] - self.mu) ** 2 / self.covar, axis=2)
        ll_mat += -0.5 * np.sum(np.log(self.covar), axis=1)
        ll_mat += self.log_c
        ll_mat += self.log_w
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

    def seq_posterior(self, x: 'DiagonalGaussianMixtureEncodedDataSequence') -> np.ndarray:
        """Compute posterior probabilities for a sequence of data points.

        Args:
            x (DiagonalGaussianMixtureEncodedDataSequence): Encoded data sequence containing input data points.

        Returns:
            np.ndarray: Posterior probabilities with shape (N, K), where N is the number of samples and K is the number of mixture components.
        """
        if not isinstance(x, DiagonalGaussianMixtureEncodedDataSequence):
            raise Exception("DiagonalGaussianMixtureEncodedDataSequence required for seq_posterior().")

        ll_mat = -0.5 * np.sum((x.data[:, None, :] - self.mu) ** 2 / self.covar, axis=2)
        ll_mat += -0.5 * np.sum(np.log(self.covar), axis=1)
        ll_mat += self.log_c
        ll_mat += self.log_w
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

    def fast_seq_posterior(self, x: 'DiagonalGaussianMixtureEncodedDataSequence') -> np.ndarray:
        """Compute posterior probabilities for a sequence of data points using numba.

        Args:
            x (DiagonalGaussianMixtureEncodedDataSequence): Encoded data sequence containing input data points.

        Returns:
            np.ndarray: Posterior probabilities with shape (N, K), where N is the number of samples and K is the number of mixture components.
        """
        if not isinstance(x, DiagonalGaussianMixtureEncodedDataSequence):
            raise Exception('Requires DiagonalGaussianMixtureEncodedDataSequence for `seq_` calls.')

        rv = np.zeros((x.data.shape[0], self.num_components), dtype=np.float64)
        fast_seq_posterior(x=x.data, mu=self.mu, covar=self.covar, log_w=self.log_w, zw=self.zw, out=rv)
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'DiagonalGaussianMixtureSampler':
        """Return a DiagonalGaussianMixtureSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            DiagonalGaussianMixtureSampler: Sampler object.
        """
        return DiagonalGaussianMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DiagonalGaussianMixtureEstimator':
        """Return a DiagonalGaussianMixtureEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            DiagonalGaussianMixtureEstimator: Estimator object.
        """
        if pseudo_count is not None:
            pseudo_count = (pseudo_count, pseudo_count, pseudo_count)
            return DiagonalGaussianMixtureEstimator(
                pseudo_count=pseudo_count,
                num_components=self.num_components,
                dim=self.dim,
                tied=self.tied,
                name=self.name,
                keys=self.keys
            )
        else:
            return DiagonalGaussianMixtureEstimator(
                tied=self.tied,
                dim=self.dim,
                num_components=self.num_components,
                name=self.name,
                keys=self.keys
            )

    def dist_to_encoder(self) -> 'DiagonalGaussianMixtureDataEncoder':
        """Return a DiagonalGaussianMixtureDataEncoder for this distribution.

        Returns:
            DiagonalGaussianMixtureDataEncoder: Encoder object.
        """
        return DiagonalGaussianMixtureDataEncoder()


class DiagonalGaussianMixtureSampler(DistributionSampler):
    """Sampler for the diagonal Gaussian mixture distribution.

    Attributes:
        dist (DiagonalGaussianMixtureDistribution): The distribution to sample from.
        rng (RandomState): Random number generator for sampling.
    """

    def __init__(self, dist: DiagonalGaussianMixtureDistribution, seed: Optional[int] = None) -> None:
        """
        Initialize the sampler for the diagonal Gaussian mixture distribution.

        Args:
            dist (DiagonalGaussianMixtureDistribution): The distribution to sample from.
            seed (Optional[int], optional): Seed for random number generation.
        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Sample from the diagonal Gaussian mixture distribution.

        Args:
            size (Optional[int], optional): Number of samples to generate. If None, generates a single sample.

        Returns:
            Union[float, np.ndarray]: Generated sample(s).
        """
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)
        sz = (1 if size is None else size, self.dist.dim)
        z = self.rng.normal(loc=0.0, scale=1.0, size=sz)
        z *= np.sqrt(self.dist.covar[comp_state])
        z += self.dist.mu[comp_state]
        return z if size is None else z.tolist()


class DiagonalGaussianMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of the diagonal Gaussian mixture distribution.

    Attributes:
        num_components (int): Number of mixture components.
        dim (int): Dimensionality of the data.
        tied (bool): If True, the covariance of each mixture component is tied.
        keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and parameters.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        num_components: int,
        dim: int,
        tied: bool = False,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """
        Initialize the accumulator for sufficient statistics.

        Args:
            num_components (int): Number of mixture components.
            dim (int): Dimensionality of the data.
            tied (bool, optional): If True, the covariance of each mixture component is tied. Defaults to False.
            keys (Tuple[Optional[str], Optional[str]], optional): Set keys for weights and parameters. Defaults to (None, None).
            name (Optional[str], optional): Name for object. Defaults to None.
        """
        self.num_components = num_components
        self.tied = tied
        self.dim = dim
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.xw = np.zeros((self.num_components, self.dim), dtype=float)
        self.x2w = np.zeros((self.num_components, self.dim), dtype=float)
        self.wcnt = np.zeros(self.num_components, dtype=float)
        self.wcnt2 = np.zeros(self.num_components, dtype=float)

        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.name = name

        # Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: Optional[List[RandomState]] = None

    def update(
        self,
        x: Union[List[float], np.ndarray],
        weight: float,
        estimate: 'DiagonalGaussianMixtureDistribution'
    ) -> None:
        """Update accumulator with a new observation.

        Args:
            x (Union[List[float], np.ndarray]): Observation.
            weight (float): Weight for the observation.
            estimate (DiagonalGaussianMixtureDistribution): Distribution estimate for update.
        """
        x = np.asarray(x)
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior
        self.wcnt += posterior
        self.wcnt2 += posterior
        self.xw += x[None, :] * posterior[:, None]
        self.x2w += x[None, :] ** 2 * posterior[:, None]

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize random number generators for each accumulator.

        Args:
            rng (RandomState): Random number generator.
        """
        seeds = rng.randint(2 ** 31, size=self.num_components)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(
        self,
        x: Union[List[float], np.ndarray],
        weight: float,
        rng: np.random.RandomState
    ) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (Union[List[float], np.ndarray]): Observation.
            weight (float): Weight for the observation.
            rng (np.random.RandomState): Random number generator.
        """
        x = np.asarray(x)
        if not self._init_rng:
            self._rng_initialize(rng)

        if weight != 0:
            ww = self._w_rng.dirichlet(np.ones(self.num_components) / (self.num_components * self.num_components))
            ww *= weight
            self.xw += x[None, :] * ww[:, None]
            self.x2w += x[None, :] ** 2 * ww[:, None]
            self.wcnt += ww
            self.wcnt2 += ww
            self.comp_counts += ww

    def seq_initialize(
        self,
        x: 'DiagonalGaussianMixtureEncodedDataSequence',
        weights: np.ndarray,
        rng: np.random.RandomState
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (DiagonalGaussianMixtureEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (np.random.RandomState): Random number generator.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        keep_idx = weights > 0
        keep_len = np.count_nonzero(keep_idx)
        ww = np.zeros((sz, self.num_components))

        c = 20 ** 2 if self.num_components > 20 else self.num_components ** 2
        if keep_len > 0:
            ww[keep_idx, :] = self._w_rng.dirichlet(alpha=np.ones(self.num_components) / c, size=keep_len)

        ww *= np.reshape(weights, (sz, 1))

        w_sum = ww.sum(axis=0)
        self.comp_counts += w_sum
        self.wcnt += w_sum
        self.wcnt2 += w_sum

        self.xw += np.sum(x.data[:, None, :] * ww[:, :, None], axis=0)
        self.x2w += np.sum(x.data[:, None, :] ** 2 * ww[:, :, None], axis=0)

    def seq_update(
        self,
        x: 'DiagonalGaussianMixtureEncodedDataSequence',
        weights: np.ndarray,
        estimate: 'DiagonalGaussianMixtureDistribution'
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (DiagonalGaussianMixtureEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (DiagonalGaussianMixtureDistribution): Distribution estimate for update.
        """
        ll_mat = -0.5 * np.sum((x.data[:, None, :] - estimate.mu) ** 2 / np.sqrt(estimate.covar), axis=2) + estimate.log_c
        ll_mat += estimate.log_w
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

        self.xw += np.sum(x.data[:, None, :] * ll_mat[:, :, None], axis=0)
        self.x2w += np.sum(x.data[:, None, :] ** 2 * ll_mat[:, :, None], axis=0)

    def combine(self, suff_stat: SS) -> 'DiagonalGaussianMixtureAccumulator':
        """Combine another accumulator's sufficient statistics into this one.

        Args:
            suff_stat (SS): Sufficient statistics to combine.

        Returns:
            DiagonalGaussianMixtureAccumulator: Self after combining.
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

    def from_value(self, x: SS) -> 'DiagonalGaussianMixtureAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (SS): Sufficient statistics.

        Returns:
            DiagonalGaussianMixtureAccumulator: Self after setting values.
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

    def acc_to_encoder(self) -> 'DiagonalGaussianMixtureDataEncoder':
        """Return a DiagonalGaussianMixtureDataEncoder for this accumulator.

        Returns:
            DiagonalGaussianMixtureDataEncoder: Encoder object.
        """
        return DiagonalGaussianMixtureDataEncoder()


class DiagonalGaussianMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating accumulators for diagonal Gaussian mixture distributions.

    Attributes:
        num_components (int): Number of mixture components.
        dim (int): Dimensionality of the data.
        keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and parameters.
        name (Optional[str]): Name for object.
        tied (bool): If True, the covariance of each mixture component is tied.
    """

    def __init__(
        self,
        num_components: int,
        dim: int,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None,
        tied: bool = False
    ) -> None:
        """
        Initialize the factory for creating accumulators.

        Args:
            num_components (int): Number of mixture components.
            dim (int): Dimensionality of the data.
            keys (Tuple[Optional[str], Optional[str]], optional): Set keys for weights and parameters. Defaults to (None, None).
            name (Optional[str], optional): Name for object. Defaults to None.
            tied (bool, optional): If True, the covariance of each mixture component is tied. Defaults to False.
        """
        self.keys = keys
        self.name = name
        self.dim = dim
        self.num_components = num_components
        self.tied = tied

    def make(self) -> 'DiagonalGaussianMixtureAccumulator':
        """Create a new DiagonalGaussianMixtureAccumulator.

        Returns:
            DiagonalGaussianMixtureAccumulator: New accumulator instance.
        """
        return DiagonalGaussianMixtureAccumulator(
            keys=self.keys,
            num_components=self.num_components,
            dim=self.dim,
            tied=self.tied,
            name=self.name
        )


class DiagonalGaussianMixtureEstimator(ParameterEstimator):
    """Estimator for diagonal Gaussian mixture distributions.

    Attributes:
        num_components (int): Number of mixture components.
        dim (int): Dimensionality of the data.
        fixed_weights (Optional[Union[List[float], np.ndarray]]): Fixed mixture weights, if provided.
        suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]): Prior sufficient statistics.
        pseudo_count (Tuple[Optional[float], Optional[float], Optional[float]]): Pseudo counts for estimation.
        tied (bool): If True, the covariance of each mixture component is tied.
        keys (Tuple[Optional[str], Optional[str]]): Set keys for weights and parameters.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        num_components: int,
        dim: int,
        fixed_weights: Optional[Union[List[float], np.ndarray]] = None,
        suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]] = (None, None, None),
        pseudo_count: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None),
        tied: bool = False,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """
        Initialize the estimator for diagonal Gaussian mixture distributions.

        Args:
            num_components (int): Number of mixture components.
            dim (int): Dimensionality of the data.
            fixed_weights (Optional[Union[List[float], np.ndarray]], optional): Fixed mixture weights, if provided.
            suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]], optional): Prior sufficient statistics.
            pseudo_count (Tuple[Optional[float], Optional[float], Optional[float]], optional): Pseudo counts for estimation.
            tied (bool, optional): If True, the covariance of each mixture component is tied. Defaults to False.
            keys (Tuple[Optional[str], Optional[str]], optional): Set keys for weights and parameters. Defaults to (None, None).
            name (Optional[str], optional): Name for object. Defaults to None.

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
            raise TypeError("DiagonalGaussianMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]]).")

        dim_loc = dim if dim is not None else (
            (None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1])))) if suff_stat[0] is None else len(
                suff_stat[0]))

        self.dim = dim_loc
        self.pseudo_count = pseudo_count
        self.prior_weights = None if suff_stat[0] is None else suff_stat[0]
        self.prior_mu = None if suff_stat[1] is None else np.reshape(suff_stat[1], (num_components, dim_loc))
        self.prior_covar = None if suff_stat[2] is None else np.reshape(suff_stat[2], (num_components, dim_loc))
        self.keys = keys

        self.num_components = num_components
        self.dim = dim
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.tied = tied
        self.fixed_weights = np.asarray(fixed_weights) if fixed_weights is not None else None
        self.name = name

    def accumulator_factory(self) -> 'DiagonalGaussianMixtureAccumulatorFactory':
        """Return a DiagonalGaussianMixtureAccumulatorFactory for this estimator.

        Returns:
            DiagonalGaussianMixtureAccumulatorFactory: Factory object.
        """
        return DiagonalGaussianMixtureAccumulatorFactory(
            keys=self.keys,
            tied=self.tied,
            num_components=self.num_components,
            dim=self.dim,
            name=self.name
        )

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'DiagonalGaussianMixtureDistribution':
        """Estimate a DiagonalGaussianMixtureDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations (not used).
            suff_stat (SS): Sufficient statistics.

        Returns:
            DiagonalGaussianMixtureDistribution: Estimated distribution.
        """
        num_components = self.num_components
        counts, xw, x2w, wcnts, _ = suff_stat
        pc0, pc1, pc2 = self.pseudo_count
        wcnts = np.reshape(wcnts, (-1, 1))

        # estimate the mean
        if pc1 is not None and self.prior_mu is not None:
            mu = (xw + pc1 * self.prior_mu) / (wcnts + pc1)
        else:
            nobs_loc = wcnts.copy()
            nobs_loc[nobs_loc == 0.0] = 1.0
            mu = xw / nobs_loc

        # estimate the covar
        if self.tied:
            if pc2 is not None and self.prior_covar is not None:
                tmp = np.sum(x2w + (pc2 * self.prior_covar) - (mu * mu * wcnts), axis=0)
                tmp /= np.sum(wcnts + pc2)
            else:
                tmp = np.sum(x2w - mu * mu * wcnts, axis=0)
                tmp /= np.sum(wcnts)
            covar = np.ones_like(x2w) * tmp
        else:
            if pc2 is not None and self.prior_covar is not None:
                covar = (x2w + (pc2 * self.prior_covar) - (mu * mu * wcnts))
                covar /= np.sum(wcnts + pc2)
            else:
                nobs_loc = wcnts.copy()
                covar = np.zeros_like(x2w)
                wz = wcnts == 0.0
                nobs_loc[wz] = 1.0
                covar = x2w / nobs_loc - mu * mu
                covar[wz.flatten(), :] = 0.0

        # estimate mixture weights
        if self.fixed_weights is not None:
            w = np.asarray(self.fixed_weights)
        elif pc0 is not None and self.prior_weights is None:
            p = pc0 / num_components
            w = counts + p
            w /= w.sum()
        elif pc0 is not None and self.prior_weights is not None:
            w = (counts + self.prior_weights * pc0) / (counts.sum() + pc0 * self.prior_weights.sum())
        else:
            nobs_loc = counts.sum()
            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        return DiagonalGaussianMixtureDistribution(mu=mu, covar=covar, w=w, tied=self.tied)


class DiagonalGaussianMixtureDataEncoder(DataSequenceEncoder):
    """Encoder for data sequences in diagonal Gaussian mixture distributions."""

    def __str__(self) -> str:
        """Return the string representation of the encoder.

        Returns:
            str: String representation of the encoder.
        """
        return 'DiagonalGaussianMixtureDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Another encoder to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        return isinstance(other, DiagonalGaussianMixtureDataEncoder)

    def seq_encode(
        self,
        x: Union[Sequence[Sequence[float]], np.ndarray]
    ) -> 'DiagonalGaussianMixtureEncodedDataSequence':
        """Encode a sequence of data points.

        Args:
            x (Union[Sequence[Sequence[float]], np.ndarray]): Sequence of data points.

        Returns:
            DiagonalGaussianMixtureEncodedDataSequence: Encoded data sequence.
        """
        return DiagonalGaussianMixtureEncodedDataSequence(
            x if isinstance(x, np.ndarray) else np.asarray(x, dtype=float)
        )


class DiagonalGaussianMixtureEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for diagonal Gaussian mixture distributions.

    Attributes:
        data (np.ndarray): Encoded data points.
    """

    def __init__(self, data: np.ndarray):
        """Initialize the encoded data sequence.

        Args:
            data (np.ndarray): Encoded data points.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return the string representation of the encoded data sequence.

        Returns:
            str: String representation of the encoded data sequence.
        """
        return f'DiagonalGaussianMixtureEncodedDataSequence(data={self.data})'


@numba.njit(
    'void(float64[:, :], float64[:, :], float64[:, :], float64[:], bool_[:], float64[:, :])',
    fastmath=True,
    parallel=True,
    cache=True,
)
def fast_seq_posterior(
    x: np.ndarray,
    mu: np.ndarray,
    covar: np.ndarray,
    log_w: np.ndarray,
    zw: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute posterior probabilities for a sequence of data points using Numba for optimization.

    Args:
        x (np.ndarray): Input data points (N x D).
        mu (np.ndarray): Means of the mixture components (K x D).
        covar (np.ndarray): Covariances of the mixture components (K x D).
        log_w (np.ndarray): Log mixture weights (length K).
        zw (np.ndarray): Boolean array indicating zero-weight components.
        out (np.ndarray): Output array for posterior probabilities (N x K).
    """
    sz, dim = x.shape[0], x.shape[1]
    ncomps = mu.shape[0]
    for i in numba.prange(sz):
        ll = out[i, :]
        for k in range(ncomps):
            if not zw[k]:
                tmp = 0.0
                for d in range(dim):
                    tmp += -0.5 * (x[i, d] - mu[k, d]) ** 2 / covar[k, d] - 0.5 * np.log(covar[k, d])
                ll[k] = tmp + log_w[k]
            else:
                ll[k] = -np.inf
