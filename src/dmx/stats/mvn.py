""""Create, estimate, and sample from a multivariate normal distribution with mean vector 'mu' (length n), and
covariance matrix 'covar' (n by n).

Defines the MultivariateGaussianDistribution, MultivariateGaussianSampler, MultivariateGaussianAccumulatorFactory,
MultivariateGaussianAccumulator, MultivariateGaussianEstimator, and the MultivariateGaussianDataEncoder classes for use
with dmx-learn.

"""
import numpy as np
import scipy.linalg
from dmx.arithmetic import *
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, StatisticAccumulatorFactory, DistributionSampler, EncodedDataSequence
from numpy.random import RandomState
import dmx.utils.vector as vec

from typing import Union, List, Dict, Optional, Any, Sequence, Tuple


class MultivariateGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """MultivariateGaussianDistribution object for multivariate Gaussian with mean mu and covaraince 'covar'.

    Attributes:
        dim (int): N is the dim of multivariate normal.
        mu (np.ndarray): Length N numpy array
        covar (np.ndarray): N by N numpy array for Covariance matrix.
        chol (np.ndarray): Cholesky decomposition of covar.
        lower (bool): Flag for lower (False for upper)
        name (Optional[str]): Set name to object.
        keys (Optional[str]): Set keys for distribution.
        self.use_lstsq (bool): Cholesky does not exist so use least squares approx.
        self.chol_const (float): det from covar if lstsq is to be used.

    """

    def __init__(self, mu: Union[List[float], np.ndarray], covar: Union[List[List[float]], np.ndarray],
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """MultivariateGaussianDistribution object.

        Args:
            mu (Union[List[float], np.ndarray]): N-dimensional mean.
            covar (Union[List[List[float]], np.ndarray]): Covariance matrix, should be N by N and positive definite.
            name (Optional[str]): Set name to object.
            keys (Optional[str]): Set keys for distribution.

        """
        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.covar = np.reshape(self.covar, (len(self.mu), len(self.mu)))
        self.chol, self.lower = scipy.linalg.cho_factor(self.covar)
        self.name = name
        self.keys = keys

        if self.chol is None:
            raise RuntimeError('Cannot obtain Choleskey factorization for covariance matrix.')
        else:
            self.use_lstsq = False
            self.chol_const = -0.5 * (len(self.mu) * np.log(2.0 * pi) + 2.0 * np.log(vec.diag(self.chol)).sum())

    def __str__(self) -> str:
        s1 = repr(self.mu.tolist())
        s2 = repr([u.tolist() for u in self.covar])
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return 'MultivariateGaussianDistribution(mu=%s, covar=%s, name=%s, keys=%s)' % (s1, s2, s3, s4)

    def density(self, x: np.ndarray) -> float:
        """Evaluate the density at x.

        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            float: Density at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Evaluate the log-density at x.

        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            float: Log-density at x.

        """
        if self.use_lstsq:
            raise RuntimeError('Least-squares log-likelihood evaluation not supported.')
        else:
            try:
                diff = self.mu - x
                soln = scipy.linalg.cho_solve((self.chol, self.lower), diff.T).T
                rv = self.chol_const - 0.5 * ((diff * soln).sum())
                return rv
            except Exception as e:
                raise e

    def seq_log_density(self, x: 'MultivariateGaussianEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, MultivariateGaussianEncodedDataSequence):
            raise Exception('MultivariateGaussianEncodedDataSequence required for seq_log_density().')

        if self.use_lstsq:
            return np.ones(x.data.shape[0])
        else:
            diff = self.mu - x.data
            soln = scipy.linalg.cho_solve((self.chol, self.lower), diff.T).T
            rv = self.chol_const - 0.5 * ((diff * soln).sum(axis=1))
            return rv

    def sampler(self, seed: Optional[int] = None):
        return MultivariateGaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None):
        if pseudo_count is None:
            return MultivariateGaussianEstimator(dim=self.dim, name=self.name, keys=self.keys)
        else:
            pseudo_count = (pseudo_count, pseudo_count)
            return MultivariateGaussianEstimator(dim=self.dim, 
                                                 pseudo_count=pseudo_count, 
                                                 suff_stat=(self.mu, self.covar),
                                                 name=self.name,
                                                 keys=self.keys)

    def dist_to_encoder(self) -> 'MultivariateGaussianDataEncoder':
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianSampler(DistributionSampler):
    """MultivariateGaussianSampler object for sampling from MultivariateGaussianDistribution.

    Attributes:
        rng (RandomState): Sets seed for generating samples.
        dist (MultivariateGaussianDistribution): MultivariateGaussianDistribution to sample from.

    """

    def __init__(self, dist: 'MultivariateGaussianDistribution', seed: Optional[int] = None) -> None:
        """MultivariateGaussianSampler object.

        Args:
            seed (Optional[int]): Set the seed for sampling.
            dist (MultivariateGaussianDistribution): MultivariateGaussianDistribution to sample from.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Generate samples from MultivariateGaussianDistribution.

        Args:
            size (Optional[int]): Number of samples to generate.

        Returns:
            np.ndarray: Size by dim number of samples.

        """
        return self.rng.multivariate_normal(mean=self.dist.mu, cov=self.dist.covar, size=size)


class MultivariateGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """MultivariateGaussianAccumulator object for aggregating sufficient statistics.

    Attributes:
        dim (Optional[int]): Dimension of the mvn.
        count (float): weight counter suff stat
        sum (Optional[np.ndarray]): Suff stat, weighted sum of obs
        sum2 (Optional[np.ndarray]): Suff stat, weights sum of squared obs.
        key (Optional[str]): Key for the mean and covariance.
        name (Optional[str]): Name of distribution.

    """
    def __init__(self, dim: Optional[int] = None, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """MultivariateGaussianAccumulator object.

        Args:
            dim (Optional[int]): Dimension of the mvn.
            keys (Optional[str]): Key for the mean and covariance.
            name (Optional[str]): Name of distribution.

        """
        self.dim = dim
        self.count = 0.0
        self.key = keys
        self.name = name

        if dim is not None:
            self.sum = vec.zeros(dim)
            self.sum2 = vec.zeros((dim, dim))
        else:
            self.sum = None
            self.sum2 = None

    def update(self, x: np.ndarray, weight: float, estimate: Optional[MultivariateGaussianDistribution]) -> None:
        if self.dim is None:
            self.dim = len(x)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros((self.dim, self.dim))

        x_weight = x * weight
        self.sum += x_weight
        self.sum2 += vec.outer(x, x_weight)
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: 'MultivariateGaussianEncodedDataSequence', weights: np.ndarray, estimate: Optional[RandomState]) -> None:
        if self.dim is None:
            self.dim = x.data.shape[1]
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros((self.dim, self.dim))

        x_weight = np.multiply(x.data.T, weights)
        self.count += weights.sum()
        self.sum += x_weight.sum(axis=1)
        self.sum2 += np.einsum('ji,ik->jk', x_weight, x.data)

    def seq_initialize(self, x: 'MultivariateGaussianEncodedDataSequence', weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[np.ndarray, np.ndarray, float]) -> 'MultivariateGaussianAccumulator':
        if suff_stat[0] is not None and self.sum is not None:
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            self.sum = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, float]:
        return self.sum, self.sum2, self.count

    def from_value(self, x: Tuple[np.ndarray, np.ndarray, float]) -> 'MultivariateGaussianAccumulator':
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key])

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key])

    def acc_to_encoder(self) -> 'MultivariateGaussianDataEncoder':
        return MultivariateGaussianDataEncoder(dim=self.dim)

class MultivariateGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """MultivariateGaussianAccumulatorFactory object for creating MultivariateGaussianAccumulator objects.

    Attributes:
        dim (Optional[int]): Dimension of the mvn.
        keys (Optional[str]): Key for the mean and covariance.
        name (Optional[str]): Name of distribution.

    """

    def __init__(self, dim: Optional[int], keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """MultivariateGaussianAccumulatorFactory object.

        Args:
            dim (Optional[int]): Dimension of the mvn.
            keys (Optional[str]): Key for the mean and covariance.
            name (Optional[str]): Name of distribution.

        """
        self.dim = dim
        self.key = keys
        self.name = name

    def make(self) -> 'MultivariateGaussianAccumulator':
        return MultivariateGaussianAccumulator(dim=self.dim, keys=self.key, name=self.name)


class MultivariateGaussianEstimator(ParameterEstimator):
    """MultivariateGaussianEstimator object for estimating multivariate normal distribution from sufficient stats.

    Attributes:
        dim (int): Dimension of multivariate normal.
        pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Regularize mean and/or covariance.
        prior_mu (Optional[np.ndarray]): Mean from prior data or used to regularize.
        prior_covar (Optional[np.ndarray]): Covariance matrix from prior data or used to regularize.
        name (Optional[str]): Set name to object.
        keys (Optional[str]): Keys for merging sufficient statistics.

    """

    def __init__(self, dim: Optional[int] = None,
                 pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
                 suff_stat: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """MultivariateGaussianEstimator object.

        Args:
            dim (Optional[int]): Dimension of multivariate normal. Inferred from 'suff_stat' if None.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Regularize mean and/or covariance.
            suff_stat (Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]): Mean and covariance estimated
                from previous data or used to regularize.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for estimator.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("MultivariateGaussianEstimator requires keys to be of type 'str'.")

        dim_loc = dim if dim is not None else (
            (None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1])))) if suff_stat[0] is None else len(
                suff_stat[0]))

        self.dim = dim_loc
        self.pseudo_count = pseudo_count
        self.prior_mu = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        self.prior_covar = None if suff_stat[1] is None else np.reshape(suff_stat[1], (dim_loc, dim_loc))
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> 'MultivariateGaussianAccumulatorFactory':
        return MultivariateGaussianAccumulatorFactory(dim=self.dim, keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, np.ndarray, float]) \
            -> 'MultivariateGaussianDistribution':

        nobs = suff_stat[2]
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (suff_stat[1] + (pc2 * self.prior_covar) - vec.outer(mu, mu * nobs)) / (nobs + pc2)
        else:
            covar = (suff_stat[1] / nobs) - vec.outer(mu, mu)

        return MultivariateGaussianDistribution(mu, covar, name=self.name)

class MultivariateGaussianDataEncoder(DataSequenceEncoder):
    """MultivariateGaussianDataEncoder object for sequence encoding iid mvn observations.

    Attributes:
        dim (Optional[int]): dimension of mvn.

    """

    def __init__(self, dim: Optional[int] = None) -> None:
        """MultivariateGaussianDataEncoder object.

        Args:
            dim (Optional[int]): dimension of mvn.

        """
        self.dim = dim

    def __str__(self) -> str:
        return 'MultivariateGaussianDataEncoder(dim=' + str(self.dim) + ')'

    def __eq__(self, other: object) -> bool:
        return other.dim == self.dim if isinstance(other, MultivariateGaussianDataEncoder) else False

    def seq_encode(self, x: Union[Sequence[List[float]], Sequence[List[np.ndarray]], np.ndarray]) \
            -> 'MultivariateGaussianEncodedDataSequence':
        self.dim = len(x[0]) if self.dim is None else self.dim

        return MultivariateGaussianEncodedDataSequence(data=np.reshape(np.asarray(x), (-1, self.dim)))


class MultivariateGaussianEncodedDataSequence(EncodedDataSequence):
    """MultivariateEncodedDataSequence object for vectorized function calls.

    Attributes:
        data (np.ndarray): Encoded sequence of mvn obs. sz by dim.

    """

    def __init__(self, data: np.ndarray):
        """MultivariateEncodedDataSequence object.

        Args:
            data (np.ndarray): Encoded sequence of mvn obs. sz by dim.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'MultivariateGaussianEncodedDataSequence(data={self.data})'


