"""Create, estimate, and sample from a Poisson distribution with rate lam > 0.0.

Defines the PoissonDistribution, PoissonSampler, PoissonAccumulatorFactory, PoissonAccumulator,
PoissonEstimator, and the PoissonDataEncoder classes for use with dmx-learn.

Data type (int): The Poisson distribution with rate lam, has log-density

    log(p_mat(x_mat=x; lam) = -x*log(lam) - log(x!) - lam,

for x in {0,1,2,...}, and

    log(p_mat(x_mat=x)) = -np.inf,

else.

"""
import numpy as np
from numpy.random import RandomState
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from dmx.utils.vector import gammaln
from math import log
from typing import Tuple, List, Union, Optional, Any, Dict, Sequence


class PoissonDistribution(SequenceEncodableProbabilityDistribution):
    """PoissonDistribution object defining Poisson distribution with mean lam > 0.0.

    Attributes:
        lam (float): Mean of Poisson distribution.
        name (Optional[str]): String name for object instance.
        log_lambda (float): Log of attribute lam.
        keys (Optional[str]): Keys for lambda. 

    """

    def __init__(self, lam: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """PoissonDistribution object.

        Args:
            lam (float): Positive real-valued number.
            name (Optional[str]): String name for object instance.
            keys (Optional[str]): Key for lambda.

        """
        self.lam = lam
        self.log_lambda = log(lam)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        s0=repr(float(self.lam))
        s1=repr(self.name)
        s2 = repr(self.keys)

        return 'PoissonDistribution(%s, name=%s, keys=%s)' % (s0, s1, s2) 

    def density(self, x: int) -> float:
        """Evaluate the density of Poisson distribution at observation x.

        Notes:
            See log_density().

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            float: Density of Poisson distribution evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of Poisson distribution evaluated at x.
        
        .. math::
            \\log{f(x | \\lambda)} = -x \\log{\\lambda} - \\log{x!} - \\lambda.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            float: Log-density of Poisson distribution evaluated at x.

        """
        if x < 0:
            return -np.inf
        else:
            return x * self.log_lambda - gammaln(x + 1.0) - self.lam

    def seq_log_density(self, x: 'PoissonEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, PoissonEncodedDataSequence):
            raise Exception('PoissonEncodedDataSequence required for seq_log_density().')

        rv = x.data[0] * self.log_lambda
        rv -= x.data[1]
        rv -= self.lam
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'PoissonSampler':
        return PoissonSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'PoissonEstimator':
        if pseudo_count is None:
            return PoissonEstimator(name=self.name, keys=self.keys)
        else:
            return PoissonEstimator(pseudo_count=pseudo_count, suff_stat=self.lam, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'PoissonDataEncoder':
        return PoissonDataEncoder()


class PoissonSampler(DistributionSampler):
    """PoissonSampler object used to draw samples from PoissonDistribution.

      Attributes:
          rng (RandomState): RandomState with seed set for sampling.
          dist (GeometricDistribution): PoissonDistribution to sample from.

      """

    def __init__(self, dist: 'PoissonDistribution', seed: Optional[int] = None) -> None:
        """PoissonSampler object.

        Args:
            dist (PoissonDistribution): Set PoissonDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, Sequence[int]]:
        """Generate iid samples from Poisson distribution.

        Generates a single Poisson sample (int) if size is None, else a numpy array of integers of length size
        containing iid samples, from the Poisson distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, assumed to be 1.

        Returns:
            If size is None, int, else size length numpy array of ints.

        """
        if size:
            return self.rng.poisson(lam=self.dist.lam, size=size).tolist()
        else:
            return int(self.rng.poisson(lam=self.dist.lam))


class PoissonAccumulator(SequenceEncodableStatisticAccumulator):
    """PoissonAccumulator object used to accumulate sufficient statistics from observed data.

    Attributes:
         sum (float): Aggregate sum of weighted observations.
         count (float): Aggregate sum of observation weights.
         name (Optional[str]): name for object
         keys (Optional[str]): Key for combining sufficient statistics with object instance containing the same key.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """PoissonAccumulator object.

        Args:
            name (Optional[str]): Name for object
            keys (Optional[str]): Assign a string valued to key to object instance.

        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys

    def initialize(self, x: int, weight: float, rng: Optional[np.random.RandomState] = None) -> None:
        self.update(x, weight, None)

    def update(self, x: int, weight: float, estimate: Optional['PoissonDistribution'] = None) -> None:
        self.sum += x * weight
        self.count += weight

    def seq_initialize(self, x: 'PoissonEncodedDataSequence', weights: np.ndarray,
                       rng: Optional[np.random.RandomState] = None) -> None:
        self.seq_update(x, weights, None)

    def seq_update(self, x: 'PoissonEncodedDataSequence', weights: np.ndarray,
                   estimate: Optional['PoissonDistribution'] = None) -> None:

        self.sum += np.dot(x.data[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat: Tuple[float, float]) -> 'PoissonAccumulator':

        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'PoissonAccumulator':

        self.count = x[0]
        self.sum = x[1]

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

    def acc_to_encoder(self) -> 'PoissonDataEncoder':
        return PoissonDataEncoder()


class PoissonAccumulatorFactory(StatisticAccumulatorFactory):
    """PoissonAccumulatorFactory object used for constructing PoissonAccumulator objects.

    Attributes:
        name (Optional[str]): Name for object
        keys (Optional[str]): Tag for combining sufficient statistics of PoissonAccumulator objects when
            constructed.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """PoissonAccumulatorFactory object.

        Args:
            name (Optional[str]): Name for object
            keys (Optional[str]): Assign keys to PoissonAccumulatorFactory object.

        """
        self.name = name
        self.keys = keys

    def make(self) -> 'PoissonAccumulator':
        return PoissonAccumulator(name=self.name, keys=self.keys)


class PoissonEstimator(ParameterEstimator):
    """PoissonEstimator object for estimating PoissonDistribution object from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Optional[float]): Re-weight suff_stat.
        suff_stat (Optional[float]): Mean of Poisson if not None.
        name (Optional[str]): String name of PoissonEstimator instance.
        keys (Optional[str]): String keys of PoissonEstimator instance for combining sufficient statistics.

    """

    def __init__(self, pseudo_count: Optional[float] = None, suff_stat: Optional[float] = None,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """PoissonEstimator object.

        Args:
            pseudo_count (Optional[float]): Optional non-negative float.
            suff_stat (Optional[float]): Optional non-negative float.
            name (Optional[str]): Assign a name to PoissonEstimator.
            keys (Optional[str]): Assign keys to PoissonEstimator for combining sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("PoissonEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> 'PoissonAccumulatorFactory':
        return PoissonAccumulatorFactory(keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> 'PoissonDistribution':

        nobs, psum = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:
            return PoissonDistribution((psum + self.suff_stat * self.pseudo_count) / (nobs + self.pseudo_count),
                                       name=self.name)
        else:
            return PoissonDistribution(psum / nobs, name=self.name)


class PoissonDataEncoder(DataSequenceEncoder):
    """GeometricDataEncoder object for encoding sequences of iid Poisson observations with data type int."""

    def __str__(self) -> str:
        return 'PoissonDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, PoissonDataEncoder)

    def seq_encode(self, x: Union[np.ndarray, Sequence[int]]) -> 'PoissonEncodedDataSequence':
        rv1 = np.asarray(x)

        if np.any(rv1 < 0) or np.any(np.isnan(rv1)):
            raise Exception('Poisson requires non-negative integer values of x.')
        else:
            rv2 = gammaln(rv1 + 1.0)

            return PoissonEncodedDataSequence(data=(rv1, rv2))

class PoissonEncodedDataSequence(EncodedDataSequence):
    """PoissonEncodedDataSequence object for vectorized function calls.

    Attributes:
        data (Tuple[np.ndarray, np.ndarray]): Poisson observations, and the log-gamma value of the obs.

    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray]):
        """PoissonEncodedDataSequence object.

        Args:
            data (Tuple[np.ndarray, np.ndarray]): Poisson observations, and the log-gamma value of the obs.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'PoissonEncodedDataSequence(data={self.data})'

