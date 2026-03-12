"""Create, estimate, and sample from a gamma distribution with shape k and scale theta.

Defines the GammaDistribution, GammaSampler, GammaAccumulatorFactory, GammaAccumulator, GammaEstimator,
and the GammaDataEncoder classes for use with pysparkplug.

Data type: (float): The GammaDistribution with shape k > 0.0 and scale theta > 0.0, has log-density
    log(f(x;k,theta)) = -gammaln(k) - k*log(theta) + (k-1) * log(x) - x / theta, for x > 0.0, else -np.inf

"""

from dmx.utils.special import gammaln, digamma, trigamma

import torch as tn
import numpy as np
from numpy.random import RandomState
from dmx.arithmetic import *
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence
from typing import Optional, Tuple, List, Dict, Union, Any
import dmx.torch_utils.vector as vec


class GammaDistribution(TorchProbabilityDistribution):
    """GammaDistribution for shape k and scale theta.

    Attributes:
        k (float): Positive real-valued number.
        theta (float): Positive real-valued number.
        log_const (float): Normalizing constant of gamma distribution.

    """

    def __init__(self, k: float, theta: float, device: Optional[tn.device] = None) -> None:
        """GammaDistribution object.

        Args:
            k (float): Positive real-valued number.
            theta (float): Positive real-valued number.
            device (Optional[device]): Device for GPU calculations.

        """
        super().__init__(device)
        self.k = k
        self.theta = theta
        self.log_const = -(gammaln(k) + k * log(theta))

    def to(self, device: tn.device) -> None:
        self._device = device

    def __repr__(self) -> str:
        s0, s1 = repr(self.k), repr(self.theta)

        return 'GammaDistribution(%s, %s)' % (s0, s1)

    def density(self, x: float) -> float:
        """Density of gamma distribution evaluated at x.

        See log_density() for details.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Density of gamma distribution evaluated at x.

        """
        return exp(self.log_const + (self.k - one) * log(x) - x / self.theta)

    def log_density(self, x: float) -> float:
        """Log-density of gamma distribution evaluated at x.

        Log-density given by,
        If x > 0.0,
            log(f(x;k,theta)) = -gammaln(k) - k*log(theta) + (k-1) * log(x) - x / theta,
        else,
            -np.inf
        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Log-density of gamma distribution evaluated at x.

        """
        return self.log_const + (self.k - one) * log(x) - x / self.theta

    def seq_log_density(self, x: 'GammaTorchEncodedSequence') -> tn.Tensor:

        if not isinstance(x, GammaTorchEncodedSequence):
            raise Exception('Requires GammaTorchEncodedSequence for `seq_` function calls. ')

        rv = x.data[0] * (-1.0 / self.theta)
        if self.k != 1.0:
            rv += x.data[1] * (self.k - 1.0)
        rv += self.log_const

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GammaSampler':
        return GammaSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GammaEstimator':
        if pseudo_count is None:
            return GammaEstimator()
        else:
            suff_stat = (self.k * self.theta, exp(digamma(self.k) + log(self.theta)))
            return GammaEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat)

    def dist_to_encoder(self) -> 'GammaDataEncoder':
        return GammaDataEncoder()


class GammaSampler(DistributionSampler):
    """GammaSampler object used to draw samples from GammaDistribution.

    Attributes:
        rng (RandomState): RandomState with seed set for sampling.
        dist (GammaDistribution): GammaDistribution to sample from.
        seed (Optional[int]): Used to set seed on random number generator used in sampling.

    """

    def __init__(self, dist: 'GammaDistribution', seed: Optional[int] = None) -> None:
        """GammaSampler object.

        Args:
            dist (GammaDistribution): GammaDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size'-iid observations from GammaSampler.

        Args:
            size (Optional[int]): Number of iid samples to draw from GammaSampler.

        Returns:
            Single sample (float) if size is None, else a numpy array of floats containing iid samples from
            GammaDistribution.

        """
        return self.rng.gamma(shape=self.dist.k, scale=self.dist.theta, size=size)


class GammaAccumulator(TorchStatisticAccumulator):
    """GammaAccumulator object used to accumulate sufficient statistics from observations.

    Attributes:
        nobs (float): Number of observations accumulated.
        sum (float): Weighted-sum of observations accumulated.
        sum_of_logs (float): log weighted sum of weighted log(observations).
        key (Optional[str]): GammaAccumulator objects with same key merge sufficient statistics.

    """

    def __init__(self, keys: Optional[str] = None, device: Optional[tn.device] = None) -> None:
        """GammaAccumulator object used to accumulate sufficient statistics from observations.

        Args:
            keys (Optional[str]): GammaAccumulator objects with same key merge sufficient statistics.
            device (Optional[tn.device]): Set device for tensor calculations.

        """
        super().__init__(device)
        self.nobs = zero
        self.sum = zero
        self.sum_of_logs = zero
        self.key = keys

    def seq_initialize(self, x: 'GammaTorchEncodedSequence', weights: tn.Tensor, tng: Optional[tn.Generator]) -> None:
        self.seq_update(x, weights, None)

    def seq_update(self, x: 'GammaTorchEncodedSequence', weights: tn.Tensor, estimate: Optional['GammaDistribution']) -> None:
        self.sum += float(tn.dot(x.data[0], weights))
        self.sum_of_logs += float(tn.dot(x.data[1], weights))
        self.nobs += float(tn.sum(weights))

    def combine(self, suff_stat: Tuple[float, float, float]) -> 'GammaAccumulator':

        self.nobs += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_of_logs += suff_stat[2]

        return self

    def value(self) -> Tuple[float, float, float]:
        return self.nobs, self.sum, self.sum_of_logs

    def from_value(self, x: Tuple[float, float, float]) -> 'GammaAccumulator':

        self.nobs = x[0]
        self.sum = x[1]
        self.sum_of_logs = x[2]

        return self

    def key_merge(self, stats_dict: Dict[str,  Any]) -> None:

        if self.key is not None:
            if self.key in stats_dict:
                x0, x1, x2 = stats_dict[self.key]
                self.nobs += x0
                self.sum += x1
                self.sum_of_logs += x2

            else:
                stats_dict[self.key] = (self.nobs, self.sum, self.sum_of_logs)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.key is not None:
            if self.key in stats_dict:
                x0, x1, x2 = stats_dict[self.key]
                self.nobs = x0
                self.sum = x1
                self.sum_of_logs = x2

    def acc_to_encoder(self) -> 'GammaDataEncoder':
        return GammaDataEncoder()


class GammaAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """GammaAccumulatorFactory object for creating GammaAccumulator objects.

    Attributes:
        keys (Optional[str]): Used for merging sufficient statistics of GammaAccumulator.

    """

    def __init__(self, keys: Optional[str] = None) -> None:
        """GammaAccumulatorFactory object.

        Args:
            keys (Optional[str]): Used for merging sufficient statistics of GammaAccumulator.

        """
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> 'GammaAccumulator':
        return GammaAccumulator(keys=self.keys, device=device)


class GammaEstimator(TorchParameterEstimator):
    """GammaEstimator object used for estimating GammaDistribution from aggregated data.

    Attributes:
        pseudo_count (Tuple[float, float]): Values used to re-weight member instances of sufficient statistics.
        suff_stat (Tuple[float, float]):  shape 'k' and scale 'theta'.
        threshold (float): Threshold used for estimating the shape of gamma.
        keys (Optional[str]): Assign keys to GammaEstimator for combining sufficient statistics.

    """

    def __init__(self, pseudo_count: Tuple[float, float] = (0.0, 0.0), suff_stat: Tuple[float, float] = (1.0, 0.0),
                 threshold: float = 1.0e-8, keys: Optional[str] = None) -> None:
        """GammaEstimator object.

        Args:
            pseudo_count (Tuple[float, float]): Values used to re-weight member instances of sufficient statistics.
            suff_stat (Tuple[float, float]):  shape 'k' and scale 'theta'.
            threshold (float): Threshold used for estimating the shape of gamma.
            keys (Optional[str]): Assign keys to GammaEstimator for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.keys = keys

    def accumulator_factory(self) -> 'GammaAccumulatorFactory':
        return GammaAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float], device: Optional[tn.device] = None) -> 'GammaDistribution':
        """Obtain GammaDistribution from aggregated sufficient statistics of observed data.

        Takes sufficient statistic aggregated from observed data:
            suff_stat[0]: weighted sum of observations
            suff_stat[1]: weighted sum of log-observations
            suff_stat[2]: weighted observation count.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: See description above for details.
            device (Optional[tn.device]): Device to declare estiamte on.

        Returns:
            GammaDistribution object.

        """
        pc1, pc2 = self.pseudo_count
        ss1, ss2 = self.suff_stat

        if suff_stat[0] == 0:
            return GammaDistribution(1.0, 1.0)

        adj_sum = suff_stat[1] + ss1 * pc1
        adj_cnt = suff_stat[0] + pc1
        adj_mean = adj_sum / adj_cnt

        adj_lsum = suff_stat[2] + ss2 * pc2
        adj_lcnt = suff_stat[0] + pc2
        adj_lmean = adj_lsum / adj_lcnt

        k = self.estimate_shape(adj_mean, adj_lmean, self.threshold)

        return GammaDistribution(k, adj_sum / (k * adj_lcnt), device=device)

    @staticmethod
    def estimate_shape(avg_sum: float, avg_sum_of_logs: float, threshold: float) -> float:
        """Estimates the shape parameter of GammaDistribution.

        Args:
            avg_sum (float): Weighted sum of gamma observations.
            avg_sum_of_logs (float): Weighted log sum of gamma observations.
            threshold (float): Threshold used for assessing convergence of shape estimation.

        Returns:
            Estimate of shape parameter 'k'.

        """
        s = log(avg_sum) - avg_sum_of_logs
        old_k = inf
        k = (3 - s + sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s)
        while abs(old_k - k) > threshold:
            old_k = k
            k -= (log(k) - digamma(k) - s) / (one / k - trigamma(k))
        return k


class GammaDataEncoder(TorchSequenceEncoder):
    """GammaDataEncoder object for encoding sequences of iid Gamma observations with data type float."""

    def __str__(self) -> str:
        return 'GammaDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GammaDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray], device: Optional[tn.device] = None) -> 'GammaTorchEncodedSequence':
        """Encode iid sequence of gamma observations for vectorized "seq_" function calls.

        Note: Each entry of x must be positive float.

        Args:
            x (Union[List[float], np.ndarray]): IID sequence of gamma distributed observations.
            device (Optional[tn.device]): Device to encode tensors on.

        Returns:
            Tuple Tensors containing x and log(x).

        """
        rv1 = vec.tensor(x, device=device)

        if tn.any(rv1 <= 0) or tn.any(tn.isnan(rv1)):
            raise Exception('GammaDistribution has support x > 0.')
        else:
            rv2 = tn.log(rv1)
            return GammaTorchEncodedSequence(data=(rv1, rv2))


class GammaTorchEncodedSequence(TorchEncodedSequence):

    def __init__(self, data: Tuple[tn.tensor, tn.tensor], device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'GammaTorchEncodedSequence(device={repr(self.device)})'

