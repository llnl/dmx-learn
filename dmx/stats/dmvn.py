"""Create, estimate, and sample from a diagonal Gaussian distribution (independent-multivariate Gaussian).

Defines the DiagonalGaussianDistribution, DiagonalGaussianSampler, DiagonalGaussianAccumulatorFactory,
DiagonalGaussianAccumulator, DiagonalGaussianEstimator, and the DiagonalGaussianDataEncoder classes for use with
dmx-learn.

The log-density of an ``n``-dimensional diagonal Gaussian observation :math:`x = (x_1, x_2, ..., x_n)` with mean
:math:`\mu = (m_1, m_2, ..., m_n)` and diagonal covariance matrix :math:`\mathrm{covar} = \mathrm{diag}(s^2_1, ..., s^2_n)` is:

.. math::

    \log p(x) = -0.5 \sum_{i=1}^n \frac{(x_i - m_i)^2}{s^2_i} - 0.5 \log(s^2_i) - \frac{n}{2} \log(2\pi)

Data type: ``x`` (List[float], np.ndarray).
"""

from dmx.arithmetic import *
from dmx.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    ParameterEstimator,
    DistributionSampler,
    DataSequenceEncoder,
    StatisticAccumulatorFactory,
    EncodedDataSequence,
)
from numpy.random import RandomState
import dmx.utils.vector as vec
import numpy as np
from typing import Sequence, Optional, Dict, Any, Tuple, List, Union


class DiagonalGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Diagonal Gaussian distribution with mean ``mu`` and diagonal covariance ``covar``.

    Attributes:
        dim (int): Dimension of the multivariate Gaussian.
        mu (np.ndarray): Mean of the Gaussian.
        covar (np.ndarray): Variance for each component.
        name (Optional[str]): Name of object instance.
        log_c (float): Normalizing constant for diagonal Gaussian.
        ca (np.ndarray): Term for likelihood calculation.
        cb (np.ndarray): Term for likelihood calculation.
        cc (np.ndarray): Term for likelihood calculation.
        keys (Tuple[Optional[str], Optional[str]]): Key for mean and covariance.
    """

    def __init__(
        self,
        mu: Union[Sequence[float], np.ndarray],
        covar: Union[Sequence[float], np.ndarray],
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """Initialize a DiagonalGaussianDistribution object.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Mean of Gaussian distribution.
            covar (Union[Sequence[float], np.ndarray]): Variance of each component.
            name (Optional[str], optional): Name for object instance.
            keys (Tuple[Optional[str], Optional[str]], optional): Key for mean and covariance.
        """
        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.name = name
        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + np.log(self.covar).sum())
        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation."""
        s1 = repr(list(self.mu.flatten()))
        s2 = repr(list(self.covar.flatten()))
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return f'DiagonalGaussianDistribution({s1}, {s2}, name={s3}, keys={s4})'

    def density(self, x: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate the density of the DiagonalGaussianDistribution.

        Args:
            x (Union[Sequence[float], np.ndarray]): Observation from Diagonal Gaussian.

        Returns:
            float: Density at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate the log-density of the DiagonalGaussianDistribution.

        Args:
            x (Union[Sequence[float], np.ndarray]): Observation from Diagonal Gaussian.

        Returns:
            float: Log-density at x.
        """
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

    def seq_log_density(self, x: 'DiagonalGaussianEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (DiagonalGaussianEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.
        """
        if not isinstance(x, DiagonalGaussianEncodedDataSequence):
            raise Exception('DiagonalGaussianDistribution.seq_log_density() requires DiagonalGaussianEncodedDataSequence.')

        rv = np.dot(x.data * x.data, self.ca)
        rv += np.dot(x.data, self.cb)
        rv += self.cc
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'DiagonalGaussianSampler':
        """Return a DiagonalGaussianSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for random number generator.

        Returns:
            DiagonalGaussianSampler: Sampler object.
        """
        return DiagonalGaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DiagonalGaussianEstimator':
        """Return a DiagonalGaussianEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for regularization.

        Returns:
            DiagonalGaussianEstimator: Estimator object.
        """
        if pseudo_count is None:
            return DiagonalGaussianEstimator(dim=self.dim, name=self.name, keys=self.keys)
        else:
            return DiagonalGaussianEstimator(
                dim=self.dim,
                pseudo_count=(pseudo_count, pseudo_count),
                name=self.name,
                keys=self.keys
            )

    def dist_to_encoder(self) -> 'DiagonalGaussianDataEncoder':
        """Return a DiagonalGaussianDataEncoder for this distribution.

        Returns:
            DiagonalGaussianDataEncoder: Encoder object.
        """
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianSampler(DistributionSampler):
    """Sampler for DiagonalGaussianDistribution.

    Attributes:
        dist (DiagonalGaussianDistribution): Object instance to sample from.
        rng (RandomState): Random number generator.
    """

    def __init__(self, dist: DiagonalGaussianDistribution, seed: Optional[int] = None) -> None:
        """Initialize DiagonalGaussianSampler.

        Args:
            dist (DiagonalGaussianDistribution): Object instance to sample from.
            seed (Optional[int], optional): Seed for random number generator.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate samples from Diagonal Gaussian distribution.

        Args:
            size (Optional[int], optional): Number of samples to generate.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Single sample or list of samples.
        """
        if size is None:
            rv = self.rng.randn(self.dist.dim)
            rv *= np.sqrt(self.dist.covar)
            rv += self.dist.mu
            return rv
        else:
            return [self.sample() for _ in range(size)]


class DiagonalGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for aggregating sufficient statistics from iid observations.

    Attributes:
        dim (Optional[int]): Optional dimension of Gaussian.
        count (float): Used for tracking weighted observation counts.
        sum (np.ndarray): Sum of observation vectors.
        sum2 (np.ndarray): Sum of squared observation vectors.
        keys (Tuple[Optional[str], Optional[str]]): Key for mean and covariance.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """Initialize DiagonalGaussianAccumulator.

        Args:
            dim (Optional[int], optional): Optional dimension of Gaussian.
            keys (Tuple[Optional[str], Optional[str]], optional): Key for mean and covariance.
            name (Optional[str], optional): Name for object.
        """
        self.dim = dim
        self.count = 0.0
        self.count2 = 0.0
        self.sum = vec.zeros(dim) if dim is not None else None
        self.sum2 = vec.zeros(dim) if dim is not None else None
        self.keys = keys
        self.name = name

    def update(
        self,
        x: Union[Sequence[float], np.ndarray],
        weight: float,
        estimate: Optional['DiagonalGaussianDistribution']
    ) -> None:
        """Update accumulator with a new observation.

        Args:
            x (Union[Sequence[float], np.ndarray]): Observation.
            weight (float): Weight for the observation.
            estimate (Optional[DiagonalGaussianDistribution]): Not used.
        """
        if self.dim is None:
            self.dim = len(x)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        x_weight = x * weight
        self.count += weight
        self.count2 += weight
        self.sum += x_weight
        x_weight *= x
        self.sum2 += x_weight

    def initialize(
        self,
        x: Union[Sequence[float], np.ndarray],
        weight: float,
        rng: RandomState
    ) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (Union[Sequence[float], np.ndarray]): Observation.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator (not used).
        """
        self.update(x, weight, None)

    def seq_update(
        self,
        x: 'DiagonalGaussianEncodedDataSequence',
        weights: np.ndarray,
        estimate: Optional['DiagonalGaussianDistribution']
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (DiagonalGaussianEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[DiagonalGaussianDistribution]): Not used.
        """
        if self.dim is None:
            self.dim = len(x.data[0])
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        x_weight = np.multiply(x.data.T, weights)
        self.count += weights.sum()
        self.count2 += weights.sum()
        self.sum += x_weight.sum(axis=1)
        x_weight *= x.data.T
        self.sum2 += x_weight.sum(axis=1)

    def seq_initialize(
        self,
        x: 'DiagonalGaussianEncodedDataSequence',
        weights: np.ndarray,
        rng: RandomState
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (DiagonalGaussianEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (RandomState): Random number generator (not used).
        """
        self.seq_update(x, weights, None)

    def combine(
        self,
        suff_stat: Tuple[np.ndarray, np.ndarray, float, float]
    ) -> 'DiagonalGaussianAccumulator':
        """Combine another accumulator's sufficient statistics into this one.

        Args:
            suff_stat (Tuple[np.ndarray, np.ndarray, float, float]): Sufficient statistics to combine.

        Returns:
            DiagonalGaussianAccumulator: Self after combining.
        """
        if suff_stat[0] is not None and self.sum is not None:
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]
            self.count2 += suff_stat[3]
        elif suff_stat[0] is not None and self.sum is None:
            self.sum = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]
            self.count2 = suff_stat[3]
        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, float]: (sum, sum2, count, count2)
        """
        return self.sum, self.sum2, self.count, self.count2

    def from_value(
        self,
        x: Tuple[np.ndarray, np.ndarray, float, float]
    ) -> 'DiagonalGaussianAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[np.ndarray, np.ndarray, float, float]): Sufficient statistics.

        Returns:
            DiagonalGaussianAccumulator: Self after setting values.
        """
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        mean_key, cov_key = self.keys
        if mean_key is not None:
            if mean_key in stats_dict:
                xw, wcnt = stats_dict[mean_key]
                self.sum += xw
                self.count += wcnt
            else:
                stats_dict[mean_key] = (self.sum, self.count)

        if cov_key is not None:
            if cov_key in stats_dict:
                x2w, wcnt2 = stats_dict[cov_key]
                self.sum2 += x2w
                self.count2 += wcnt2
            else:
                stats_dict[cov_key] = (self.sum2, self.count2)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        mean_key, cov_key = self.keys
        if mean_key is not None:
            if mean_key in stats_dict:
                xw, wcnt = stats_dict[mean_key]
                self.sum = xw
                self.count = wcnt

        if cov_key is not None:
            if cov_key in stats_dict:
                x2w, wcnt2 = stats_dict[cov_key]
                self.sum2 = x2w
                self.count2 = wcnt2

    def acc_to_encoder(self) -> 'DiagonalGaussianDataEncoder':
        """Return a DiagonalGaussianDataEncoder for this accumulator.

        Returns:
            DiagonalGaussianDataEncoder: Encoder object.
        """
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating DiagonalGaussianAccumulator objects.

    Attributes:
        dim (Optional[int]): Optional dimension of Gaussian.
        keys (Tuple[Optional[str], Optional[str]]): Key for mean and covariance.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None
    ) -> None:
        """Initialize DiagonalGaussianAccumulatorFactory.

        Args:
            dim (Optional[int], optional): Optional dimension of Gaussian.
            keys (Tuple[Optional[str], Optional[str]], optional): Key for mean and covariance.
            name (Optional[str], optional): Name for object.
        """
        self.dim = dim
        self.keys = keys
        self.name = name

    def make(self) -> 'DiagonalGaussianAccumulator':
        """Create a new DiagonalGaussianAccumulator.

        Returns:
            DiagonalGaussianAccumulator: New accumulator instance.
        """
        return DiagonalGaussianAccumulator(dim=self.dim, keys=self.keys, name=self.name)


class DiagonalGaussianEstimator(ParameterEstimator):
    """Estimator for diagonal Gaussian distributions from aggregated sufficient statistics.

    Attributes:
        name (Optional[str]): Name for object instance.
        dim (int): Dimension of Gaussian, either set or determined from suff_stat arg.
        prior_mu (Optional[np.ndarray]): Set from suff_stat[0].
        prior_covar (Optional[np.ndarray]): Set from suff_stat[1].
        pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight the sum of observations and sum of squared observations in estimation.
        keys (Tuple[Optional[str], Optional[str]]): Key for mean and covariance.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
        suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str]] = (None, None)
    ) -> None:
        """Initialize DiagonalGaussianEstimator.

        Args:
            dim (Optional[int], optional): Optional dimension of Gaussian.
            pseudo_count (Tuple[Optional[float], Optional[float]], optional): Re-weight the sum of observations and sum of squared observations in estimation.
            suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray]], optional): Sum of observations and sum of squared observations both having same dimension.
            name (Optional[str], optional): Name for object instance.
            keys (Tuple[Optional[str], Optional[str]], optional): Key for mean and covariance.

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
            raise TypeError("DiagonalGaussianEstimator requires keys (Tuple[Optional[str], Optional[str]]).")

        dim_loc = dim if dim is not None else (
            (None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1])))) if suff_stat[0] is None else len(
                suff_stat[0]))

        self.name = name
        self.dim = dim_loc
        self.pseudo_count = pseudo_count
        self.prior_mu = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        self.prior_covar = None if suff_stat[1] is None else np.reshape(suff_stat[1], dim_loc)
        self.keys = keys

    def accumulator_factory(self) -> 'DiagonalGaussianAccumulatorFactory':
        """Return a DiagonalGaussianAccumulatorFactory for this estimator.

        Returns:
            DiagonalGaussianAccumulatorFactory: Factory object.
        """
        return DiagonalGaussianAccumulatorFactory(dim=self.dim, keys=self.keys, name=self.name)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, np.ndarray, float]
    ) -> 'DiagonalGaussianDistribution':
        """Estimate a DiagonalGaussianDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations.
            suff_stat (Tuple[np.ndarray, np.ndarray, float]): Sufficient statistics.

        Returns:
            DiagonalGaussianDistribution: Estimated distribution.
        """
        nobs = suff_stat[2]
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (suff_stat[1] + (pc2 * self.prior_covar) - (mu * mu * nobs)) / (nobs + pc2)
        else:
            covar = (suff_stat[1] / nobs) - (mu * mu)

        return DiagonalGaussianDistribution(mu, covar, name=self.name)


class DiagonalGaussianDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of iid diagonal-Gaussian observations.

    Attributes:
        dim (Optional[int]): Dimension of Gaussian distribution.
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        """Initialize DiagonalGaussianDataEncoder.

        Args:
            dim (Optional[int], optional): Dimension of Gaussian distribution.
        """
        self.dim = dim

    def __str__(self) -> str:
        """Return string representation."""
        return f'DiagonalGaussianDataEncoder(dim={self.dim})'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Other encoder.

        Returns:
            bool: True if encoders are equal.
        """
        if isinstance(other, DiagonalGaussianDataEncoder):
            return self.dim == other.dim
        else:
            return False

    def seq_encode(
        self,
        x: Sequence[Union[List[float], np.ndarray]]
    ) -> 'DiagonalGaussianEncodedDataSequence':
        """Create DiagonalGaussianEncodedDataSequence object.

        Args:
            x (Sequence[Union[List[float], np.ndarray]]): Sequence of iid multivariate Gaussian observations.

        Returns:
            DiagonalGaussianEncodedDataSequence: Encoded data sequence.
        """
        if self.dim is None:
            self.dim = len(x[0])
        xv = np.reshape(x, (-1, self.dim))
        return DiagonalGaussianEncodedDataSequence(data=xv)


class DiagonalGaussianEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for vectorized functions.

    Attributes:
        data (np.ndarray): Numpy array of observations.
    """

    def __init__(self, data: np.ndarray):
        """Initialize DiagonalGaussianEncodedDataSequence.

        Args:
            data (np.ndarray): Numpy array of observations.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'DiagonalGaussianEncodedDataSequence(data={self.data})'

