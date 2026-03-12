"""Create, estimate, and sample from a Poisson distribution with rate lam > 0.0.

Defines the PoissonDistribution, PoissonSampler, PoissonAccumulatorFactory, PoissonAccumulator,
PoissonEstimator, and the PoissonDataEncoder classes for use with pysparkplug.

Data type (int): The Poisson distribution with rate lam, has log-density

    log(p_mat(x_mat=x; lam) = -x*log(lam) - log(x!) - lam,

for x in {0,1,2,...}, and

    log(p_mat(x_mat=x)) = -np.inf,

else.

"""
import torch as tn
import numpy as np
from numpy.random import RandomState
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence
from typing import Optional, Tuple, List, Callable, Dict, Union, Any, Sequence
import dmx.torch_utils.vector as vec

from dmx.utils.vector import gammaln
from math import log
from typing import Tuple, List, Union, Optional, Any, Dict, Sequence


class PoissonDistribution(TorchProbabilityDistribution):
    """PoissonDistribution object defining Poisson distribution with mean lam > 0.0.

    Attributes:
        lam (float): Mean of Poisson distribution.
        log_lam (float): Log of attribute lam.

    """

    def __init__(self, lam: float, device: Optional[tn.device] = None) -> None:
        """PoissonDistribution object.

        Args:
            lam (float): Positive real-valued number.
            device: Define device for Tensor calculations.

        """
        super().__init__(device)
        self.lam = lam
        self.log_lam = log(lam)

    def to(self, device: str) -> None:
        self._device = device

    def __repr__(self) -> str:
        return f'PoissonDistribution({repr(self.lam)})'

    def density(self, x: int) -> float:
        """Evaluate the density of Poisson distribution at observation x.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            float: Density of Poisson distribution evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of Poisson distribution evaluated at x.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            float: Log-density of Poisson distribution evaluated at x.

        """
        if x < 0:
            return -np.inf
        else:
            return x * self.log_lam - gammaln(x + 1.0) - self.lam

    def seq_log_density(self, x: 'PoissonTorchSequence') -> tn.tensor:
        if not isinstance(x, PoissonTorchSequence):
            raise Exception('Requires PoissonTorchSequence for `seq_` function calls.')

        rv = x.data[0] * self.log_lam
        rv -= x.data[1]
        rv -= self.lam

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'PoissonSampler':
        return PoissonSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'PoissonEstimator':
        if pseudo_count is None:
            return PoissonEstimator()
        else:
            return PoissonEstimator(pseudo_count=pseudo_count, suff_stat=self.lam)

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

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate iid samples from Poisson distribution.

        Generates a single Poisson sample (int) if size is None, else a numpy array of integers of length size
        containing iid samples, from the Poisson distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, assumed to be 1.

        Returns:
            If size is None, int, else size length numpy array of ints.

        """
        return self.rng.poisson(lam=self.dist.lam, size=size)


class PoissonAccumulator(TorchStatisticAccumulator):
    """PoissonAccumulator object used to accumulate sufficient statistics from observed data.

    Attributes:
         sum (float): Aggregate sum of weighted observations.
         count (float): Aggregate sum of observation weights.
         key (Optional[str]): Key for combining sufficient statistics with object instance containing the same key.

    """

    def __init__(self, keys: Optional[str] = None, device: Optional[tn.device] = None) -> None:
        """PoissonAccumulator object.

        Args:
            keys (Optional[str]): Assign a string valued to key to object instance.
            device: Set device for Tensor calculations.

        """
        super().__init__(device)
        self.sum = 0.0
        self.count = 0.0
        self.key = keys

    def seq_initialize(self, x: 'PoissonTorchSequence', weights: tn.Tensor,
                       tng: Optional[tn.Generator] = None) -> None:
        self.seq_update(x, weights, None)

    def seq_update(self, x: 'PoissonTorchSequence', weights: tn.Tensor,
                   estimate: Optional['PoissonDistribution'] = None) -> None:
        self.sum += float(tn.dot(x.data[0], weights))
        self.count += float(weights.sum())

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
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'PoissonDataEncoder':
        return PoissonDataEncoder()


class PoissonAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """PoissonAccumulatorFactory object used for constructing PoissonAccumulator objects.

     Attributes:
          keys (Optional[str]): Tag for combining sufficient statistics of PoissonAccumulator objects when
             constructed.

     """

    def __init__(self, keys: Optional[str] = None) -> None:
        """PoissonAccumulatorFactory object.

        Args:
            keys (Optional[str]): Assign keys to PoissonAccumulatorFactory object.

        """
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> 'PoissonAccumulator':
        return PoissonAccumulator(keys=self.keys, device=device)


class PoissonEstimator(TorchParameterEstimator):
    """PoissonEstimator object for estimating PoissonDistribution object from aggregated sufficient statistics.

       Attributes:
           pseudo_count (Optional[float]): Re-weight suff_stat.
           suff_stat (Optional[float]): Mean of Poisson if not None.
           keys (Optional[str]): String keys of PoissonEstimator instance for combining sufficient statistics.

       """

    def __init__(self, pseudo_count: Optional[float] = None, suff_stat: Optional[float] = None,
                 keys: Optional[str] = None) -> None:
        """PoissonEstimator object.

        Attributes:
            pseudo_count (Optional[float]): Re-weight suff_stat.
            suff_stat (Optional[float]): Mean of Poisson if not None.
            keys (Optional[str]): String keys of PoissonEstimator instance for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

    def accumulator_factory(self) -> 'PoissonAccumulatorFactory':
        return PoissonAccumulatorFactory(self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float], device: Optional[tn.device] = None) -> 'PoissonDistribution':
        nobs, psum = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:
            return PoissonDistribution((psum + self.suff_stat * self.pseudo_count) / (nobs + self.pseudo_count),
                                       device=device)
        else:
            return PoissonDistribution(psum / nobs, device=device)


class PoissonDataEncoder(TorchSequenceEncoder):
    """GeometricDataEncoder object for encoding sequences of iid Poisson observations with data type int."""

    def __str__(self) -> str:
        return 'PoissonDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, PoissonDataEncoder)

    def seq_encode(self, x: Union[np.ndarray, Sequence[int]], device: Optional[tn.device] = None) -> 'PoissonTorchSequence':
        rv1 = vec.tensor(x, device=device)

        if tn.any(rv1 < 0) or tn.any(tn.isnan(rv1)):
            raise Exception('Poisson requires non-negative integer values of x.')
        else:
            rv2 = tn.lgamma(rv1 + 1.0)
            return PoissonTorchSequence(data=(rv1, rv2), device=device)


class PoissonTorchSequence(TorchEncodedSequence):

    def __init__(self, data: Tuple[tn.tensor, tn.tensor], device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'PoissonTorchSequence(device={repr(self.device)})'
