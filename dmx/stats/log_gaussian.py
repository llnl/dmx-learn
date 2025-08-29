"""Evaluate, estimate, and sample from a log-gaussian distribution with location mu and scale sigma2.

Defines the LogGaussianDistribution, LogGaussianSampler, LogGaussianAccumulatorFactory, LogGaussianAccumulator,
LogGaussianEstimator, and the LogGaussianDataEncoder classes for use with dmx-learn.

"""
import numpy as np
from numpy.random import RandomState
from dmx.arithmetic import *
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from typing import Optional, Tuple, List, Callable, Dict, Union, Any


class LogGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """LogGaussianDistribution object defines Gaussian distribution with mean mu and variance sigma2.

    Attributes:
        mu (float): Location parameter for log-Gaussian distribution.
        sigma2 (float): Scale for log-Gaussian distribution.
        const (float): Normalizing constant (depends on sigma2).
        log_const (float): Log of above.
        name (Optional[str]): String for name of object.
        keys (Optional[str]): Key for parameters of dist. 

    """

    def __init__(self, mu: float, sigma2: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """LogGaussianDistribution object.

        Args:
            mu (float): Real-valued number.
            sigma2 (float): Positive real-valued number.
            name (Optional[str]): String for name of object.
            keys (Optional[str]): Key for parameters of dist. 

        """
        self.mu = mu
        self.sigma2 = 1.0 if (sigma2 <= 0 or isnan(sigma2) or isinf(sigma2)) else sigma2
        self.log_const = -0.5 * log(2.0 * pi * self.sigma2)
        self.const = 1.0 / sqrt(2.0 * pi * self.sigma2)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'LogGaussianDistribution(%s, %s, name=%s, keys=%s)' % (repr(self.mu), repr(self.sigma2), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        """Density of Log-Gaussian distribution at observation x.

        See log_density() for details.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Density of Log-Gaussian at x.

        """
        return self.const * exp(-0.5 * (np.log(x) - self.mu) ** 2 / self.sigma2) / x

    def log_density(self, x: float) -> float:
        """Log-density of log-Gaussian distribution at observation x.

        Args:
            x (float): Positive valued observation of log-Gaussian.

        Returns:
            float: Log-density at observation x.

        """
        return self.log_const - 0.5 * (np.log(x) - self.mu) ** 2 / self.sigma2 - np.log(x)

    def seq_ld_lambda(self) -> List[Callable]:
        return [self.seq_log_density]

    def seq_log_density(self, x: 'LogGaussianEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, LogGaussianEncodedDataSequence):
            raise Exception('LogGaussianEncodedDataSequence required for seq_log_density().')

        rv = x.data - self.mu
        rv *= rv
        rv *= -0.5 / self.sigma2
        rv += self.log_const
        rv -= x.data

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'LogGaussianSampler':
        return LogGaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'LogGaussianEstimator':
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return LogGaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name, keys=self.keys)
        else:
            return LogGaussianEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'LogGaussianDataEncoder':
        return LogGaussianDataEncoder()


class LogGaussianSampler(DistributionSampler):
    """LogGaussianSampler for drawing samples from LogGaussianSampler instance.

    Attributes:
        dist (LogGaussianDistribution): LogGaussianDistribution instance to sample from.
        rng (RandomState): RandomState with seed set to seed if passed in args.

    """

    def __init__(self, dist: LogGaussianDistribution, seed: Optional[int] = None) -> None:
        """LogGaussianSampler object.

        Args:
            dist (LogGaussianDistribution): LogGaussianDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from LogGaussianSampler object.

        Numpy array of length 'size' from log-Gaussian distribution with scale beta if size not None. Else a single
        sample is returned as float.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            'size' iid samples from Gaussian distribution.

        """
        return np.exp(self.rng.normal(loc=self.dist.mu, scale=np.sqrt(self.dist.sigma2), size=size))


class LogGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """LogGaussianAccumulator object used to accumulate sufficient statistics from observed data.

    Attributes:
        log_sum (float): Sum of weighted observations (sum_i w_i*X_i).
        log_sum2 (float): Sum of weighted squared observations (sum_i w_i*X_i^2)
        count (float): Sum of weights for observations (sum_i w_i).
        count2 (float): Sum of weights for squared observations (sum_i w_i).
        count (float): Tracks the sum of weighted observations used to form sum.
        keys (Optional[str]): Key string used to aggregate all sufficient statistics with same keys values.
        name (Optional[str]): Name for GaussianAccumulator object.

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """LogGaussianAccumulator object.

        Args:
            keys (Optional[str]): Set key for LogGaussianAccumulator object.
            name (Optional[str]): Set name for LogGaussianAccumulator object.

        """
        self.log_sum = 0.0
        self.log_sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional['LogGaussianDistribution']) -> None:
        x_weight = np.log(x) * weight
        self.log_sum += x_weight
        self.log_sum2 += np.log(x) * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, x: 'LogGaussianEncodedDataSequence', weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def seq_update(self, x: 'LogGaussianEncodedDataSequence', weights: np.ndarray, estimate: Optional[LogGaussianDistribution]) -> None:

        self.log_sum += np.dot(x.data, weights)
        self.log_sum2 += np.dot(x.data * x.data, weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: Tuple[float, float, float, float]) -> 'LogGaussianAccumulator':
        self.log_sum += suff_stat[0]
        self.log_sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        return self

    def value(self) -> Tuple[float, float, float, float]:
        return self.log_sum, self.log_sum2, self.count, self.count2

    def from_value(self, x: Tuple[float, float, float, float]) -> 'LogGaussianAccumulator':
        self.log_sum = x[0]
        self.log_sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> 'LogGaussianDataEncoder':
        return LogGaussianDataEncoder()


class LogGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """LogGaussianAccumulatorFactory object for creating LogGaussianAccumulator.

    Attributes:
        name (Optional[str]): Name of the LogGaussianAccumulatorFactory object.
        keys (Optional[str]): String id for merging sufficient statistics of LogGaussianAccumulator.

    """

    def __init__(self, name: Optional[str] = None, keys:  Optional[str] = None) -> None:
        """LogGaussianAccumulatorFactory object.

        Args:
            name (Optional[str]): Assign a name to LogGaussianAccumulatorFactory object.
            keys (Optional[str]): Assign keys member for LogGaussianAccumulators.

        """
        self.keys = keys
        self.name = name

    def make(self) -> 'LogGaussianAccumulator':
        return LogGaussianAccumulator(name=self.name, keys=self.keys)


class LogGaussianEstimator(ParameterEstimator):
    """LogGaussianEstimator object used to estimate LogGaussianDistribution. 

    Attributes:
        pseudo_count (Tuple[Optional[float], Optional[float]]): Weights for suff_stat.
        suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean (mu) and variance (sigma2).
        name (Optional[str]): String name of LogGaussianEstimator instance.
        keys (Optional[str]): String keys of LogGaussianEstimator instance for combining sufficient statistics.

    """

    def __init__(self,
                 pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
                 suff_stat: Tuple[Optional[float], Optional[float]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[str] = None):
        """LogGaussianEstimator object.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Tuple of two positive floats.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of float and positive float.
            name (Optional[str]): Assign a name to LogGaussianEstimator.
            keys (Optional[str]): Assign keys to LogGaussianEstimator for combining sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("LogGaussianEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'LogGaussianAccumulatorFactory':

        return LogGaussianAccumulatorFactory(self.name, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float, float]) \
            -> 'LogGaussianDistribution':

        log_x, log_x2 = suff_stat[0], suff_stat[1]
        nobs_loc1, nobs_loc2 = suff_stat[2], suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (log_x + self.pseudo_count[0] * self.suff_stat[0]) / (nobs_loc1 + self.pseudo_count[0])
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = 0.0
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (suff_stat[1] - mu * mu * nobs_loc2 + self.pseudo_count[1] * self.suff_stat[1]) / (
                        nobs_loc2 + self.pseudo_count[1])
        else:
            sigma2 = np.sum(log_x2 - np.sum(log_x)**2 / nobs_loc1 ) / nobs_loc2

        return LogGaussianDistribution(mu, sigma2, name=self.name)


class LogGaussianDataEncoder(DataSequenceEncoder):
    """LogGaussianDataEncoder object for encoding sequences of iid Gaussian observations with data type float."""

    def __str__(self) -> str:
        return 'LogGaussianDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, LogGaussianDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> 'LogGaussianEncodedDataSequence':
        """Encode sequence of iid Log-Gaussian observations.

        Data type must be List[float] or np.ndarray[float].

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid log-Gaussian observations.

        Returns:
            A numpy array of floats.

        """
        rv = np.asarray(np.log(x), dtype=float)

        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise Exception('LogGaussianDistribution requires support x in (0,inf).')
        return LogGaussianEncodedDataSequence(data=rv)

class LogGaussianEncodedDataSequence(EncodedDataSequence):
    """LogGaussianEncodedDataSequence object for vectorized function calls.

    Attributes:
        data (np.ndarray): IID log Gaussian observations.

    """

    def __init__(self, data: np.ndarray):
        """LogGaussianEncodedDataSequence object.

        Args:
            data (np.ndarray): IID log Gaussian observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'LogGaussianEncodedDataSequence(data={self.data})'
