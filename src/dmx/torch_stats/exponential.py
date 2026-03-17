import torch as tn
import numpy as np
from numpy.random import RandomState
from dmx.arithmetic import *
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence, TorchDevice
from typing import Optional, Tuple, List, Callable, Dict, Union, Any, Sequence
from typing import cast
import dmx.torch_utils.vector as vec


class ExponentialDistribution(TorchProbabilityDistribution):
    """ExponentialDistribution for creating exponential distribution with mean beta.

    Attributes:
        beta (float): Scale of exponential.
        log_beta (float): Log of scale.

    """

    def __init__(self, beta: float, device: Optional[TorchDevice] = None):
        """ExponentialDistribution object.

        Args:
            beta (float): Scale of Exponential random variable.
            device (Optional[TorchDevice]): Device for model calculations.

        """
        super(ExponentialDistribution, self).__init__(device)
        self.beta = beta
        self.log_beta = np.log(beta)

    def to(self, device: TorchDevice) -> None:
        self._device = device

    def __repr__(self) -> str:
        return f'ExponentialDistribution(beta={repr(self.beta)})'

    def density(self, x: float) -> float:
        """Density of Exponential distribution at observation x.

        See log_density() for details.

        Args:
            x (float): Real-valued observation of Exponential.

        Returns:
            float: Density of Exponential at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Log-density of Exponential distribution at observation x.

        Log-density of Exponential with mean mu and variance sigma2 given by,
            log(f(x;mu, sigma2)) = -log(2*pi*sigma2) - (x-mu)^2/sigma2, for real-valued x.

        Args:
            x (float): Real-valued observation of Exponential.

        Returns:
            float: Log-density at observation x.

        """
        if x < 0:
            return -inf
        else:
            return -x / self.beta - self.log_beta

    def seq_log_density(self, x: 'ExponentialTorchEncodedSequence') -> tn.Tensor:

        if not isinstance(x, ExponentialTorchEncodedSequence):
            raise Exception('Requires ExponentialTorchEncodedSequence for `seq_` function calls.')

        rv = x.data * (-1.0 / self.beta)
        rv -= self.log_beta
        
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'ExponentialSampler':
        return ExponentialSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'ExponentialEstimator':
        if pseudo_count is None:
            return ExponentialEstimator()
        else:
            return ExponentialEstimator(pseudo_count=pseudo_count, suff_stat=self.beta)

    def dist_to_encoder(self) -> 'ExponentialDataEncoder':
        """Returns a ExponentialDataEncoder object for encoding sequences of data."""
        return ExponentialDataEncoder()


class ExponentialSampler(DistributionSampler):

    def __init__(self, dist: 'ExponentialDistribution', seed: Optional[int] = None) -> None:
        """ExponentialSampler for drawing samples from ExponentialSampler instance.

        Args:
            dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
            tng (tn.Generator): RandomState with seed set to seed if passed in args.

        """
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.beta = dist.beta

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from ExponentialSampler object.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            Numpy array of length 'size' from exponential distribution with scale beta if size not None. Else a single
            sample is returned as float.


        """
        return self.rng.exponential(scale=self.beta, size=size)


class ExponentialAccumulator(TorchStatisticAccumulator):
    """ExponentialAccumulator object used to accumulate sufficient statistics.

    Attributes:
        sum (float): Tracks the sum of observation values.
        count (float): Tracks the sum of weighted observations used to form sum.
        key (Optional[str]): Aggregate all sufficient statistics with same key.
        _device (TorchDevice): Device for tensor operations.

    """

    def __init__(self, keys: Optional[str] = None, device: Optional[TorchDevice] = None) -> None:
        """ExponentialAccumulator object.

        Args:
            keys (Optional[str]): Aggregate all sufficient statistics with same keys values.
            device: Optional[device]: Sets device for GPU calculations

        """
        super(ExponentialAccumulator, self).__init__(cast(Optional[str], device))
        self.sum = 0.0
        self.count = 0.0
        self.key = keys

    def seq_initialize(self, x: 'ExponentialTorchEncodedSequence', weights: tn.Tensor, tng: Optional[tn.Generator]) -> None:
        self.seq_update(x, weights, None)

    def seq_update(self, x: 'ExponentialTorchEncodedSequence', weights: tn.Tensor, estimate: Optional[ExponentialDistribution]) -> None:
        self.sum += float(tn.dot(x.data, weights))
        self.count += float(tn.sum(weights))

    def combine(self, suff_stat: Tuple[float, float]) -> 'ExponentialAccumulator':
        self.sum += suff_stat[0]
        self.count += suff_stat[1]

        return self

    def value(self, device: Optional[str] = None) -> Tuple[float, float]:
        return self.sum, self.count

    def from_value(self, x: Tuple[float, float]) -> 'ExponentialAccumulator':
        self.sum, self.count = x

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.sum, self.count = stats_dict[self.key]
            else:
                stats_dict[self.key] = (self.sum, self.count)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.sum, self.count = stats_dict[self.key]

    def acc_to_encoder(self) -> 'ExponentialDataEncoder':
        return ExponentialDataEncoder()


class ExponentialAccumulatorFactory(TorchStatisticAccumulatorFactory):

    def __init__(self, keys:  Optional[str] = None):
        self.keys = keys

    def make(self, device: Optional[TorchDevice] = None) -> 'ExponentialAccumulator':
        return ExponentialAccumulator(keys=self.keys, device=device if device is not None else None)


class ExponentialEstimator(TorchParameterEstimator):

    def __init__(self,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[float] = None,
                 keys: Optional[str] = None):
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

    def accumulator_factory(self) -> 'ExponentialAccumulatorFactory':
        return ExponentialAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float], device: Optional[TorchDevice] = None) -> 'ExponentialDistribution':
        """Estimate ExponentialDistribution from suff_stat arg.

        Estimate ExponentialDistribution from sufficient statistic tuple suff_stat,
        storing the weighted observation sum followed by the weighted count. If
        pseudo_count is set, this is used to re-weight the member value
        "suff_stat", which is the scale of ExponentialEstimator object.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat (Tuple[float, float]): Tuple of (sum, count). Both are
                positive real-valued floats.
            device (Optional[TorchDevice]): Set for estimating model on GPU device

        Returns:
            ExponentialDistribution object.

        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[0] + self.suff_stat * self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[0] + self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        else:
            if suff_stat[1] > 0:
                p = suff_stat[0] / suff_stat[1]
            else:
                p = 1.0

        return ExponentialDistribution(beta=p, device=device)


class ExponentialDataEncoder(TorchSequenceEncoder):
    """ExponentialDataEncoder object for encoding sequences of iid exponential observations with data type float."""

    def __str__(self) -> str:
        return 'ExponentialDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExponentialDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray], device: Optional[TorchDevice] = None) -> 'ExponentialTorchEncodedSequence':
        rv = vec.tensor(x, device=device)

        if tn.any(rv <= 0) or tn.any(tn.isnan(rv)):
            raise Exception('Exponential requires x > 0.')

        return ExponentialTorchEncodedSequence(data=rv, device=device)


class ExponentialTorchEncodedSequence(TorchEncodedSequence):

    def __init__(self, data: tn.Tensor, device: Optional[TorchDevice] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'ExponentialTorchEncodedSequence(device={repr(self.device)})'
