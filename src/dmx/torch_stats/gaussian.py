import torch as tn
import numpy as np
from numpy.random import RandomState
from dmx.arithmetic import *
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence
from typing import Optional, Tuple, List, Callable, Dict, Union, Any, Sequence
import dmx.torch_utils.vector as vec


class GaussianDistribution(TorchProbabilityDistribution):

    def __init__(self, mu: float, sigma2: float, device: Optional[tn.device] = None):
        super().__init__(device)
        self.mu = mu
        self.sigma2 = 1.0 if (sigma2 <= 0 or isnan(sigma2) or isinf(sigma2)) else sigma2
        self.log_const = -0.5 * log(2.0 * pi * self.sigma2)
        self.const = 1.0 / sqrt(2.0 * pi * self.sigma2)

    def to(self, device: tn.device) -> None:
        self._device = device

    def __repr__(self) -> str:
        s0, s1 = repr(float(self.mu)), repr(float(self.sigma2))
        return 'GaussianDistribution(mu=%s, sigma2=%s)' % (s0, s1)

    def density(self, x: float) -> float:
        """Density of Gaussian distribution at observation x.

        Args:
            x (float): Real-valued observation of Gaussian.

        Returns:
            float: Density of Gaussian at x.

        """
        return self.const * exp(-0.5 * (x - self.mu) * (x - self.mu) / self.sigma2)

    def log_density(self, x: float) -> float:
        """Log-density of Gaussian distribution at observation x.

        Args:
            x (float): Real-valued observation of Gaussian.

        Returns:
            float: Log-density at observation x.

        """
        return self.log_const - 0.5 * (x - self.mu) * (x - self.mu) / self.sigma2

    def seq_log_density(self, x: 'GaussianTorchEncodedSequence') -> tn.Tensor:
        if not isinstance(x, GaussianTorchEncodedSequence):
            raise Exception('Requires GaussianTorchEncodedSequence for `seq_` calls.')

        rv = (x.data - self.mu) / np.sqrt(self.sigma2)
        rv *= rv
        rv *= -0.5
        rv += self.log_const

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GaussianSampler':

        return GaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GaussianEstimator':

        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return GaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat)
        else:
            return GaussianEstimator()

    def dist_to_encoder(self) -> 'GaussianDataEncoder':
        return GaussianDataEncoder()


class GaussianSampler(DistributionSampler):
    """GaussianSampler for drawing samples from GaussianSampler instance.

    Attributes:
        dist (GaussianDistribution): GaussianDistribution instance to sample from.
        tng (tn.Generator): RandomState with seed set to seed if passed in args.

    """

    def __init__(self, dist: 'GaussianDistribution', seed: Optional[int] = None) -> None:
        """GaussianSampler object.

        Args:
            dist (GaussianDistribution): GaussianDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        """
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from GaussianSampler object.

        Numpy array of length 'size' from Gaussian distribution with mean mu and scale sigma2 if size not None.
        Else a single sample is returned as float.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            'size' iid samples from Gaussian distribution.

        """
        return self.dist.mu + np.sqrt(self.dist.sigma2)*self.rng.normal(size=size)


class GaussianAccumulator(TorchStatisticAccumulator):
    """GaussianAccumulator object used to accumulate sufficient statistics from observed data.

    Attributes:
        sum (float): Sum of weighted observations (sum_i w_i*X_i).
        sum2 (float): Sum of weighted squared observations (sum_i w_i*X_i^2)
        count (float): Sum of weights for observations (sum_i w_i).
        count2 (float): Sum of weights for squared observations (sum_i w_i).
        count (float): Tracks the sum of weighted observations used to form sum.
        key (Optional[str]): Key string used to aggregate all sufficient statistics with same keys values.

    """

    def __init__(self, keys: Optional[str] = None, device: Optional[tn.device] = None) -> None:
        """GaussianAccumulator object.

        Args:
            keys (Optional[str]): Set key for GaussianAccumulator object.
            device (Optional[tn.device]): Device to declare accumulator on.

        """
        super().__init__(device)
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys

    def seq_initialize(self, x: 'GaussianTorchEncodedSequence', weights: tn.Tensor, tng: Optional[tn.Generator]) -> None:
        self.seq_update(x, weights, None)

    def seq_update(self, x: 'GaussianTorchEncodedSequence', weights: tn.Tensor, estimate: Optional[GaussianDistribution]) -> None:
        self.sum += float(tn.dot(x.data, weights))
        self.sum2 += float(tn.dot(x.data * x.data, weights))
        w_sum = float(weights.sum())
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: Tuple[float, float, float, float]) -> 'GaussianAccumulator':
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        return self

    def value(self, device: Optional[str] = None) -> Tuple[float, float, float, float]:
        return self.sum, self.sum2, self.count, self.count2

    def from_value(self, x: Tuple[float, float, float, float]) -> 'GaussianAccumulator':
        self.sum, self.sum2, self.count, self.count2 = x

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.sum, self.sum2, self.count, self.count2 = stats_dict[self.keys]
            else:
                stats_dict[self.keys] = (self.sum, self.sum2, self.count, self.count2)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.sum, self.sum2, self.count, self.count2 = stats_dict[self.keys]

    def acc_to_encoder(self) -> 'GaussianDataEncoder':
        return GaussianDataEncoder()


class GaussianAccumulatorFactory(TorchStatisticAccumulatorFactory):

    def __init__(self, keys:  Optional[str] = None):
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> 'GaussianAccumulator':
        return GaussianAccumulator(keys=self.keys, device=device if device is not None else None)


class GaussianEstimator(TorchParameterEstimator):
    """GaussianEstimator object for estimating GaussianDistribution with torch tensors.

    Attributes:
        pseudo_count (Tuple[Optional[float], Optional[float]]): Pseudo count to regularize suff stats.
        suff_stat (Tuple[Optional[float], Optional[float]]): Suff stats for Gaussian.
        keys (Optional[str]): Key for distribution.

    """

    def __init__(self,
                 pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
                 suff_stat: Tuple[Optional[float], Optional[float]] = (None, None),
                 keys: Optional[str] = None):
        """GaussianEstimator object.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Pseudo count to regularize suff stats.
            suff_stat (Tuple[Optional[float], Optional[float]]): Suff stats for Gaussian.
            keys (Optional[str]): Key for distribution.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

    def accumulator_factory(self) -> 'GaussianAccumulatorFactory':
        return GaussianAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float, float], device: Optional[tn.device] = None) -> 'GaussianDistribution':

        nobs_loc1 = suff_stat[2]
        nobs_loc2 = suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (suff_stat[0] + self.pseudo_count[0] * self.suff_stat[0]) / (nobs_loc1 + self.pseudo_count[0])
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = 0.0
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (suff_stat[1] - mu * mu * nobs_loc2 + self.pseudo_count[1] * self.suff_stat[1]) / (
                        nobs_loc2 + self.pseudo_count[1])
        else:
            sigma2 = suff_stat[1] / nobs_loc2 - mu * mu

        return GaussianDistribution(mu, sigma2, device=device)


class GaussianDataEncoder(TorchSequenceEncoder):
    """GaussianDataEncoder object for encoding sequences of iid Gaussian observations with data type float."""

    def __str__(self) -> str:
        return 'GaussianDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, GaussianDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray, tn.Tensor], device: Optional[tn.device] = None) -> 'GaussianTorchEncodedSequence':
        rv = vec.tensor(x, device=device)

        if tn.any(tn.isnan(rv)) or tn.any(tn.isinf(rv)):
            raise Exception('GaussianDistribution requires support x in (-inf,inf).')

        return GaussianTorchEncodedSequence(data=rv, device=device)


class GaussianTorchEncodedSequence(TorchEncodedSequence):
    """GaussianTorchEncodedSequence object for use with `seq_` function calls.

    Attributes:
        data (tn.tensor): iid observations of Gaussian
        device (Optional[tn.device]): Device that data lives on.

    """

    def __init__(self, data: tn.tensor, device: Optional[tn.device] = None):
        """GaussianTorchEncodedSequence object.

        Args:
            data (tn.tensor): iid observations of Gaussian
            device (Optional[tn.device]): Device that data lives on.

        """
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'GaussianTorchEncodedSequence(device={repr(self.device)})'


