"""Dirichlet distribution: estimation, sampling, and sufficient statistics.

This module defines the DirichletDistribution and related classes for use with dmx-learn:

- DirichletDistribution
- DirichletSampler
- DirichletAccumulatorFactory
- DirichletAccumulator
- DirichletEstimator
- DirichletDataEncoder

Provides methods for parameter estimation, sampling, and encoding of Dirichlet-distributed data.
"""

import numpy as np
import sys
from numpy.random import RandomState
from dmx.utils.special import digamma, digammainv, one
from scipy.special import gammaln
from dmx.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    ParameterEstimator,
    DistributionSampler,
    SequenceEncodableStatisticAccumulator,
    DataSequenceEncoder,
    StatisticAccumulatorFactory,
    EncodedDataSequence,
)
from typing import Union, List, Any, Optional, Dict, Sequence, Tuple, Callable


def dirichlet_param_solve(
    alpha: np.ndarray, mean_log_p: np.ndarray, delta: float
) -> Tuple[np.ndarray, int]:
    """Iteratively solve for alpha of a Dirichlet distribution.

    Args:
        alpha (np.ndarray): Numpy array of Dirichlet parameters (all entries should be non-negative).
        mean_log_p (np.ndarray): Sufficient statistic (1/N) sum_{i=1}^{N} log(x_{i,k}), where N is the number of observations.
        delta (float): Tolerance for convergence of Newton-Method.

    Returns:
        Tuple[np.ndarray, int]: Estimates of alpha and number of iterations in solver.
    """
    dim = len(alpha)
    valid = np.bitwise_and(np.isfinite(alpha), alpha > 0)
    valid = np.bitwise_and(valid, np.isfinite(mean_log_p))

    alpha = alpha[valid]
    mlp = mean_log_p[valid]

    count = 0
    a_sum = alpha.sum()
    d_alpha = (2 * delta) + 1

    while d_alpha > delta:
        count += 1
        da_sum = digamma(a_sum)
        old_alpha = alpha
        adj_alpha = mlp + da_sum
        alpha = digammainv(adj_alpha)
        a_sum = np.sum(alpha)
        d_alpha = np.abs(alpha - old_alpha).sum()
        d_alpha /= a_sum

    if dim != alpha.size:
        rv = np.zeros(dim, dtype=float)
        rv[valid] = alpha
    else:
        rv = alpha

    return rv, count


def mpe(
    x0: np.ndarray, f: Callable[[np.ndarray], np.ndarray], eps: float
) -> Tuple[np.ndarray, int]:
    """Maximum posterior estimate for iterative update.

    Args:
        x0 (np.ndarray): Initial estimate.
        f (Callable[[np.ndarray], np.ndarray]): Update function.
        eps (float): Convergence threshold.

    Returns:
        Tuple[np.ndarray, int]: Estimate and number of iterations.
    """
    x1 = f(x0)
    x2 = f(x1)
    x3 = f(x2)
    X = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 2

    while res > eps:
        y = f(X[-1, :])
        dy = y - X[-1, :]
        U = (X[1:, :] - X[:-1, :]).T
        X2 = X[1:, :].T
        c = np.dot(np.linalg.pinv(U), dy)
        c *= -1
        s = (np.dot(X2, c) + y) / (c.sum() + 1)

        res = np.abs(s - s0).sum()
        s0 = s
        X = np.concatenate((X, np.reshape(y, (1, -1))), axis=0)
        its_cnt += 1

    return s, its_cnt


def alpha_seq_lambda(mean_log_p: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return update function for alpha given mean_log_p.

    Args:
        mean_log_p (np.ndarray): Mean log probabilities.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Update function.
    """
    def next_alpha(current_alpha: np.ndarray) -> np.ndarray:
        return digammainv(mean_log_p + digamma(current_alpha.sum()))
    return next_alpha


def find_alpha(
    current_alpha: np.ndarray, mlp: np.ndarray, thresh: float
) -> Tuple[np.ndarray, int]:
    """Find alpha using maximum posterior estimate.

    Args:
        current_alpha (np.ndarray): Initial alpha.
        mlp (np.ndarray): Mean log probabilities.
        thresh (float): Convergence threshold.

    Returns:
        Tuple[np.ndarray, int]: Estimated alpha and number of iterations.
    """
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh)


class DirichletDistribution(SequenceEncodableProbabilityDistribution):
    """DirichletDistribution object defining Dirichlet distribution with parameter alpha.

    Attributes:
        dim (int): Number of categories in Dirichlet.
        alpha (np.ndarray): Concentration parameters of length dim.
        alpha_ma (np.ndarray): Boolean mask for positive alpha entries.
        log_const (float): Normalizing constant for distribution.
        has_invalid (bool): True if any alpha are less than or equal to 0.
        name (Optional[str]): Optional name for object instance.
        keys (Optional[str]): Optional key for merging sufficient statistics.
    """

    def __init__(
        self,
        alpha: Union[List[float], np.ndarray],
        name: Optional[str] = None,
        keys: Optional[str] = None
    ) -> None:
        """Initialize DirichletDistribution.

        Args:
            alpha (Union[List[float], np.ndarray]): Array of alpha values. Determines size of Dirichlet distribution.
            name (Optional[str], optional): Name for distribution.
            keys (Optional[str], optional): Key for merging sufficient statistics.
        """
        temp_alpha = np.asarray(alpha)
        temp_mask = temp_alpha <= 0

        self.dim: int = len(alpha)
        self.alpha: np.ndarray = temp_alpha
        self.alpha_ma: np.ndarray = ~temp_mask
        self.log_const: float = sum(gammaln(alpha)) - gammaln(sum(alpha))
        self.has_invalid: bool = np.any(temp_mask)
        self.name: Optional[str] = name
        self.keys: Optional[str] = keys

    def __str__(self) -> str:
        """Return string representation."""
        s1 = repr(self.alpha.tolist())
        s2 = repr(self.name)
        s3 = repr(self.keys)
        return f'DirichletDistribution({s1}, name={s2}, keys={s3})'

    def density(self, x: Union[List[float], np.ndarray]) -> float:
        """Evaluate the density of a Dirichlet observation.

        Args:
            x (Union[List[float], np.ndarray]): A single Dirichlet observation.

        Returns:
            float: Density evaluated at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Union[List[float], np.ndarray]) -> float:
        """Evaluate the log-density of a Dirichlet observation.

        The log-density of a Dirichlet with dim = K, is given by

            log(p_mat(x)) = -log(Const) + sum_{k=0}^{K-1} (alpha_k -1)*log(x_k), for sum_k x_k = 1.0,

        where

            log(Const) = sum_{k=0}^{K-1} log(Gamma(alpha_k)) - log(Gamma(sum_{k=0}^{K-1} alpha_k)).

        Args:
            x (Union[List[float], np.ndarray]): A single Dirichlet observation.

        Returns:
            float: Log-density evaluated at x.
        """
        xx = np.asarray(x)
        zz = np.bitwise_or(xx > 0, self.alpha_ma)
        cnt = np.count_nonzero(zz)

        if cnt == self.dim:
            rv = np.dot(np.log(x), self.alpha - 1.0)
            rv -= self.log_const
        elif cnt == 0:
            rv = 0.0
        else:
            rv = np.dot(np.log(xx[zz]), self.alpha[zz] - 1.0)
            rv -= self.log_const

        return rv

    def seq_log_density(self, x: 'DirichletEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (DirichletEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.
        """
        if not isinstance(x, DirichletEncodedDataSequence):
            raise Exception('DirichletEncodedDataSequence required for DirichletDistribution.seq_log_density().')

        rv = np.dot(x.data[0], self.alpha - 1.0)
        rv -= self.log_const
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'DirichletSampler':
        """Return a DirichletSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            DirichletSampler: Sampler object.
        """
        return DirichletSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DirichletEstimator':
        """Return a DirichletEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            DirichletEstimator: Estimator object.
        """
        if pseudo_count is None:
            return DirichletEstimator(dim=self.dim, name=self.name, keys=self.keys)
        else:
            return DirichletEstimator(
                dim=self.dim,
                pseudo_count=pseudo_count,
                suff_stat=np.log(self.alpha / sum(self.alpha)),
                name=self.name,
                keys=self.keys
            )

    def dist_to_encoder(self) -> 'DirichletDataEncoder':
        """Create DirichletDataEncoder object for encoding sequences of iid Dirichlet observations.

        Returns:
            DirichletDataEncoder: Encoder object.
        """
        return DirichletDataEncoder()


class DirichletSampler(DistributionSampler):
    """DirichletSampler object for drawing samples from Dirichlet distribution.

    Attributes:
        rng (RandomState): RandomState object for generating seeded samples.
        dist (DirichletDistribution): DirichletDistribution object to draw samples from.
    """

    def __init__(self, dist: DirichletDistribution, seed: Optional[int] = None) -> None:
        """Initialize DirichletSampler.

        Args:
            dist (DirichletDistribution): DirichletDistribution object to draw samples from.
            seed (Optional[int], optional): Optional seed for sampler.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Draw samples from Dirichlet distribution.

        Args:
            size (Optional[int], optional): Number of samples to draw.

        Returns:
            np.ndarray: Array of samples (size, dim).
        """
        alpha = self.dist.alpha
        has_invalid = self.dist.has_invalid
        alpha_ma = self.dist.alpha_ma

        if has_invalid:
            if size is None:
                rv = np.zeros(alpha.size)
                rv[alpha_ma] = self.rng.dirichlet(alpha=alpha[alpha_ma])
            else:
                rv = np.zeros((size, alpha.size))
                rv[:, alpha_ma] = self.rng.dirichlet(alpha=alpha[alpha_ma], size=size)
            return rv
        else:
            return self.rng.dirichlet(alpha=self.dist.alpha, size=size)


class DirichletAccumulator(SequenceEncodableStatisticAccumulator):
    """DirichletAccumulator object for accumulating sufficient statistics.

    Attributes:
        dim (int): Dimension of the Dirichlet distribution.
        sum_of_logs (np.ndarray): Sufficient statistic for log(x).
        sum (np.ndarray): Sufficient statistic for x.
        sum2 (np.ndarray): Sufficient statistic for x^2.
        counts (float): Number of observations.
        keys (Optional[str]): Key for the shape parameter of distribution.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        dim: int,
        keys: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        """Initialize DirichletAccumulator.

        Args:
            dim (int): Dimension of the Dirichlet distribution.
            keys (Optional[str], optional): Key for the shape parameter of distribution.
            name (Optional[str], optional): Name for object.
        """
        self.dim = dim
        self.sum_of_logs = np.zeros(dim)
        self.sum = np.zeros(dim)
        self.sum2 = np.zeros(dim)
        self.counts = 0
        self.keys = keys
        self.name = name

    def update(
        self,
        x: Union[np.ndarray, List[float]],
        weight: float,
        estimate: Optional['DirichletDistribution']
    ) -> None:
        """Update accumulator with a new observation.

        Args:
            x (Union[np.ndarray, List[float]]): Observation.
            weight (float): Weight for the observation.
            estimate (Optional[DirichletDistribution]): Not used.
        """
        xx = np.asarray(x)
        z = xx > 0
        if np.all(z):
            self.sum_of_logs += np.log(xx) * weight
            self.sum += weight * xx
            self.sum2 += weight * xx * xx
            self.counts += weight
        else:
            self.sum_of_logs[z] += np.log(xx[z]) * weight
            self.sum += weight * xx
            self.sum2 += weight * xx * xx
            self.counts += weight

    def initialize(
        self,
        x: Union[np.ndarray, List[float]],
        weight: float,
        rng: Optional[RandomState]
    ) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (Union[np.ndarray, List[float]]): Observation.
            weight (float): Weight for the observation.
            rng (Optional[RandomState]): Not used.
        """
        self.update(x, weight, None)

    def get_seq_lambda(self) -> List[Callable]:
        """Return list of sequence update functions.

        Returns:
            List[Callable]: List containing seq_update.
        """
        return [self.seq_update]

    def seq_update(
        self,
        x: 'DirichletEncodedDataSequence',
        weights: np.ndarray,
        estimate: Optional['DirichletDistribution']
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (DirichletEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[DirichletDistribution]): Not used.
        """
        self.sum_of_logs += np.dot(weights, x.data[0])
        self.counts += weights.sum()
        self.sum += np.dot(weights, x.data[1])
        self.sum2 += np.dot(weights, x.data[2])

    def seq_initialize(
        self,
        x: 'DirichletEncodedDataSequence',
        weights: np.ndarray,
        rng: Optional[RandomState]
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (DirichletEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (Optional[RandomState]): Not used.
        """
        self.seq_update(x, weights, None)

    def combine(
        self,
        suff_stat: Tuple[float, np.ndarray, np.ndarray, np.ndarray]
    ) -> 'DirichletAccumulator':
        """Combine another accumulator's sufficient statistics into this one.

        Args:
            suff_stat (Tuple[float, np.ndarray, np.ndarray, np.ndarray]): Sufficient statistics to combine.

        Returns:
            DirichletAccumulator: Self after combining.
        """
        self.sum_of_logs += suff_stat[1]
        self.sum += suff_stat[2]
        self.sum2 += suff_stat[3]
        self.counts += suff_stat[0]
        return self

    def value(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[float, np.ndarray, np.ndarray, np.ndarray]: (counts, sum_of_logs, sum, sum2)
        """
        return self.counts, self.sum_of_logs, self.sum, self.sum2

    def from_value(
        self,
        x: Tuple[float, np.ndarray, np.ndarray, np.ndarray]
    ) -> 'DirichletAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[float, np.ndarray, np.ndarray, np.ndarray]): Sufficient statistics.

        Returns:
            DirichletAccumulator: Self after setting values.
        """
        self.counts = x[0]
        self.sum_of_logs = x[1]
        self.sum = x[2]
        self.sum2 = x[3]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> 'DirichletDataEncoder':
        """Return a DirichletDataEncoder for this accumulator.

        Returns:
            DirichletDataEncoder: Encoder object.
        """
        return DirichletDataEncoder()


class DirichletAccumulatorFactory(StatisticAccumulatorFactory):
    """DirichletAccumulatorFactory object.

    Attributes:
        dim (int): Shape of Dirichlet.
        keys (Optional[str]): Optional key for sufficient statistics.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        dim: int,
        keys: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        """Initialize DirichletAccumulatorFactory.

        Args:
            dim (int): Shape of Dirichlet.
            keys (Optional[str], optional): Optional key for sufficient statistics.
            name (Optional[str], optional): Name for object.
        """
        self.dim = dim
        self.keys = keys
        self.name = name

    def make(self) -> 'DirichletAccumulator':
        """Create a new DirichletAccumulator.

        Returns:
            DirichletAccumulator: New accumulator instance.
        """
        return DirichletAccumulator(dim=self.dim, keys=self.keys, name=self.name)


class DirichletEstimator(ParameterEstimator):
    """DirichletEstimator object.

    Attributes:
        dim (int): Dimension of Dirichlet distribution to estimate.
        pseudo_count (Optional[float]): Pseudo count for sufficient statistics.
        delta (Optional[float]): Tolerance for shape estimation from sufficient statistics.
        suff_stat (Optional[np.ndarray]): Sufficient statistics.
        keys (Optional[str]): Optional key string for shape parameter.
        use_mpe (bool): If True, use max posterior estimate.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        dim: int,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[np.ndarray] = None,
        delta: Optional[float] = 1.0e-8,
        keys: Optional[str] = None,
        use_mpe: bool = False,
        name: Optional[str] = None
    ) -> None:
        """Initialize DirichletEstimator.

        Args:
            dim (int): Dimension of Dirichlet distribution to estimate.
            pseudo_count (Optional[float], optional): Pseudo count for sufficient statistics.
            suff_stat (Optional[np.ndarray], optional): Sufficient statistics.
            delta (Optional[float], optional): Tolerance for shape estimation from sufficient statistics.
            keys (Optional[str], optional): Optional key string for shape parameter.
            use_mpe (bool, optional): If True, use max posterior estimate.
            name (Optional[str], optional): Name for object.

        Raises:
            TypeError: If keys is not a string or None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("DirichletEstimator requires keys to be of type 'str'.")

        self.dim = dim
        self.pseudo_count = pseudo_count
        self.delta = delta
        self.suff_stat = suff_stat
        self.keys = keys
        self.use_mpe = use_mpe
        self.name = name

    def accumulator_factory(self) -> 'DirichletAccumulatorFactory':
        """Return a DirichletAccumulatorFactory for this estimator.

        Returns:
            DirichletAccumulatorFactory: Factory object.
        """
        return DirichletAccumulatorFactory(dim=self.dim, keys=self.keys, name=self.name)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[float, np.ndarray, np.ndarray, np.ndarray]
    ) -> DirichletDistribution:
        """Estimate a DirichletDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations.
            suff_stat (Tuple[float, np.ndarray, np.ndarray, np.ndarray]): Sufficient statistics.

        Returns:
            DirichletDistribution: Estimated distribution.
        """
        nobs, sum_of_logs, sum_v, sum_v2 = suff_stat
        dim = len(sum_of_logs)

        if self.pseudo_count is not None and self.suff_stat is None:
            c1 = digamma(one) - digamma(dim)
            c2 = sum_of_logs + c1 * self.pseudo_count
            initial_estimate = c2 * (dim / sum(c2))
            mean_log_p = c2 / (nobs + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is not None:
            c2 = sum_of_logs + self.suff_stat * self.pseudo_count
            initial_estimate = c2 * (dim / sum(c2))
            mean_log_p = c2 / (nobs + self.pseudo_count)
        else:
            sum_v = sum_v / nobs
            sum_v2 = sum_v2 / nobs
            sum_v[-1] = 1.0 - sum_v[:-1].sum()
            initial_estimate = sum_v
            mean_log_p = sum_of_logs / nobs

        if nobs == 1.0:
            return DirichletDistribution(initial_estimate, name=self.name)
        else:
            if self.use_mpe:
                alpha, its_cnt = find_alpha(np.asarray(initial_estimate), mean_log_p, self.delta)
            else:
                alpha, its_cnt = dirichlet_param_solve(np.asarray(initial_estimate), mean_log_p, self.delta)
            return DirichletDistribution(alpha, name=self.name)


class DirichletDataEncoder(DataSequenceEncoder):
    """DirichletDataEncoder object for encoding iid observations."""

    def __str__(self) -> str:
        """Return string representation."""
        return 'DirichletDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Other object.

        Returns:
            bool: True if encoders are equal.
        """
        return isinstance(other, DirichletDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[float]]) -> 'DirichletEncodedDataSequence':
        """Create DirichletEncodedDataSequence for vectorized functions.

        Args:
            x (Sequence[Sequence[float]]): Sequence of iid dirichlet observations.

        Returns:
            DirichletEncodedDataSequence

        """
        rv = np.asarray(x).copy()

        rv2 = np.maximum(rv, sys.float_info.min)
        np.log(rv2, out=rv2)

        return DirichletEncodedDataSequence(data=(rv2, rv, rv * rv))

class DirichletEncodedDataSequence(EncodedDataSequence):
    """DirichletEncodedDataSequence for vectorized function calls.

    Attributes:
        data (Tuple[np.ndarray, np.ndarray, np.ndarray]): Log-max, sequence of values, and sequence values squared.

    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """DirichletEncodedDataSequence for vectorized function calls.

        Args:
            data (Tuple[np.ndarray, np.ndarray, np.ndarray]): Log-max, sequence of values, and sequence values squared.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'DirichletEncodedDataSequence(data={self.data})'
