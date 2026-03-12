"""Create, estimate, and sample from the binomial distribution.

Defines the BinomialDistribution, BinomialSampler, BinomialAccumulatorFactory, BinomialAccumulator, BinomialEstimator,
and the BinomialDataEncoder classes for use with pysparkplug.

Data type: int.

"""
import torch as tn
import numpy as np
from numpy.random import RandomState
from dmx.utils.vector import gammaln
from dmx.arithmetic import *
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence
from typing import Optional, Tuple, List, Callable, Dict, Union, Any, Sequence
import dmx.torch_utils.vector as vec


E = Tuple[tn.Tensor, tn.Tensor, tn.Tensor, int, int]


class BinomialDistribution(TorchProbabilityDistribution):
    """BinomialDistribution object used for x~Binomial(n,p) with support (min_val, n-min_val-1).

    Notes:
        Supports data types of int between (0, n-1) or (min_val, n-min_val-1) if min_val is not None.
        Log-probability mass for BinomialDistribution(n,p),

        log(f(x|n,p)) = log(n!) - log((n-x)!) - log(x!) + x*log(p) + (1-x)*log(1-p),

        for x in [0,n-1) or [min_val, n-1-min_val), else -inf.

    Attributes:
        p (float): Proportion for binomial distribution, between (0,1.0].
        log_p (float): Logrithm of p above.
        log_1p (float): Logrithm of 1-p, p defined above.
        n (int): Number of trials in binomial distribution, n > 0.
        min_val (Optional[int]): Change domain of binomial from (0,n-1) to (min_val, n-min_val).
        keys (Optional[str]): All BinomialDistributions with same keys are same distributions.

    """

    def __init__(self, p: float, n: int, min_val: Optional[int] = None, keys: Optional[str] = None, device: Optional[tn.device] = None) -> None:
        """BinomialDistribution object.

        Args:
            p (float): Proportion for binomial distribution, between (0,1.0].
            n (int): Number of trials in binomial distribution, n > 0.
            min_val (Optional[int]): Change domain of binomial from (0,n-1) to (min_val, n-min_val-1).
            keys (Optional[str]): All BinomialDistributions with same keys are same distributions.
            device: Device for Tensor calculations.

        """
        super().__init__(device)
        if p <= 0.0 or p >= 1.0:
            raise Exception('Binomial distribution requires p in [0,1]')
        else:
            self.p = p

        if n < 0 or np.isinf(n):
            raise Exception('Binomial distribution requires n > 0.')
        else:
            self.n = n

        self.log_p = np.log(p)
        self.log_1p = np.log1p(-p)
        self.keys = keys
        self.min_val = min_val

    def to(self, device: tn.device) -> None:
        self._device = device

    def __repr__(self) -> str:
        return 'BinomialDistribution(p=%s, n=%s, min_val=%s, keys=%s)' % (
            repr(self.p), repr(self.n), repr(self.min_val), repr(self.keys))

    def density(self, x: int) -> float:
        """Returns the probability mass of integer value x.

        If x is not an integer between [0,n) or [min_val, n-1-min_val), density is 0.0.

        Args:
            x (int): Integer value for density evaluation.

        Returns:
            float: Probability mass of x for binomial(n,p) with min_val=min_val. 0.0 if x is not in support.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Returns the log-probability mass of integer value x.

        If x is not an integer between [0,n) or [min_val, n-1-min_val), log-density is -inf.

        Args:
            x (int): Integer value for density evaluation.

        Returns:
            float: Log-probability mass of x for binomial(n,p) with min_val=min_val. -inf if x is not in support.

        """
        return float(self.seq_log_density(self.dist_to_encoder().seq_encode([x])))

    def seq_log_density(self, x: 'BinomialTorchEncodedSequence') -> tn.Tensor:

        if not isinstance(x, BinomialTorchEncodedSequence):
            raise Exception('Required BinomialTorchEncodedSequence for `seq_` function calls.')

        ux, ix, _, _, _ = x.data
        n = self.n
        gn = tn.lgamma(vec.tensor([n+1], device=self.model_device()))

        if self.min_val is not None:
            xx = ux - self.min_val
        else:
            xx = ux

        cc = (gn - tn.lgamma(xx + 1) - tn.lgamma((n + 1) - xx)) + self.log_1p * (n - xx) + self.log_p * xx
        return cc[ix]

    def sampler(self, seed: Optional[int] = None) -> 'BinomialSampler':
        """Returns BinomialSampler for generating samples from BinomialDistribution(n,p,min_val).

        Args:
            seed Optional[int]: Used to set seed on random number generator for sampling.

        Returns:
            BinomialSampler for BinomialDistribution with seed.
        """
        return BinomialSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'BinomialEstimator':
        """Creates a BinomialEstimator for estimating parameters of BinomialDistribution.

        Args:
            pseudo_count (Optional[float]): If set, inflates counts for currently set sufficient statistic (p).

        Returns:
            BinomialEstimator object.
        """
        if pseudo_count is None:
            return BinomialEstimator(keys=self.keys)
        else:
            return BinomialEstimator(max_val=self.n, min_val=self.min_val, pseudo_count=pseudo_count,
                                     suff_stat=self.p * self.n * pseudo_count)

    def dist_to_encoder(self) -> 'BinomialDataEncoder':
        """Creates a BinomialDataEncoder object for sequence encoding data.

        Returns:
            BinomialDataEncoder object.
        """
        return BinomialDataEncoder()


class BinomialSampler(DistributionSampler):

    def __init__(self, dist: BinomialDistribution, seed: Optional[int] = None) -> None:
        """BinomialSampler object used to draw samples from BinomialDistribution.

        Args:
            dist (BinomialDistribution): BinomialDistribution to sample from.
            seed (Optional[int]): Seed for setting random number generator.

        Attributes:
            dist (BinomialDistribution): BinomialDistribution to sample from.
            seed (Optional[int]): Seed for setting random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, List[int]]:
        """Draw samples from BinomialSampler.

        Args:
            size (Optional[int]): Number of samples to draw from BinomialSampler (1 if size is None).

        Returns:
            An integer sample from BinomialDistribution(n,p,min_val), or List[int] of samples with length = size.

        """
        rv = self.rng.binomial(n=self.dist.n, p=self.dist.p, size=size)

        if size is None:
            if self.dist.min_val is not None:
                return int(rv) + self.dist.min_val
            else:
                return int(rv)
        else:
            if self.dist.min_val is not None:
                return list(rv + self.dist.min_val)
            else:
                return list(rv)


class BinomialAccumulator(TorchStatisticAccumulator):
    """BinomialAccumulator object used for aggregating sufficient statistics of BinomialDistribution.

    Attributes:
        sum (float): Aggregates the sum of all data observations.
        count (float): Aggregates the number of weighted-data observations used in accumulating sum.
        max_val (Optional[int]): Largest integer value encountered while accumulating sufficient statistics.
        min_val (Optional[int]): Smallest integer value encountered while accumulating sufficient statistics.
        key (Optional[str]): All BinomialAccumulators with same key will have suff-stats merged.

    """

    def __init__(self, max_val: Optional[int] = None, min_val: Optional[int] = None,
                 keys: Optional[str] = None, device: Optional[tn.device] = None) -> None:
        """BinomialAccumulator object.

        Args:
            max_val (Optional[int]): Largest integer value encountered while accumulating sufficient statistics.
            min_val (Optional[int]): Smallest integer value encountered while accumulating sufficient statistics.
            keys (Optional[str]): All BinomialAccumulators with same keys will have suff-stats merged.
            device (device): Set device for tensor calculations.

        """
        super().__init__(device)
        self.sum = 0.0
        self.count = 0.0
        self.key = keys
        self.max_val = max_val
        self.min_val = min_val

    def seq_update(self, x: 'BinomialTorchEncodedSequence', weights: tn.Tensor, estimate: Optional['BinomialDistribution']) -> None:
        _, _, xx, min_val, max_val = x.data

        self.sum += float(tn.sum(xx * weights))
        self.count += float(tn.sum(weights))

        if self.min_val is not None:
            self.min_val = min(self.min_val, min_val)
        else:
            self.min_val = min_val

        if self.max_val is not None:
            self.max_val = max(self.max_val, max_val)
        else:
            self.max_val = max_val

    def seq_initialize(self, x: 'BinomialTorchEncodedSequence', weights: tn.Tensor, tng: Optional[tn.Generator]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float, Optional[int], Optional[int]]) -> 'BinomialAccumulator':
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        if self.min_val is None:
            self.min_val = suff_stat[2]
        elif self.min_val is not None and suff_stat[2] is not None:
            self.min_val = min(self.min_val, suff_stat[2])

        if self.max_val is None:
            self.max_val = suff_stat[3]
        elif self.max_val is not None and suff_stat[3] is not None:
            self.max_val = max(self.max_val, suff_stat[3])

        return self

    def value(self) -> Tuple[float, float, Optional[int], Optional[int]]:
        return self.count, self.sum, self.min_val, self.max_val

    def from_value(self, x: Tuple[float, float, Optional[int], Optional[int]]) -> 'BinomialAccumulator':
        self.count = x[0]
        self.sum = x[1]
        self.min_val = x[2]
        self.max_val = x[3]

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

    def acc_to_encoder(self) -> 'BinomialDataEncoder':
        return BinomialDataEncoder()


class BinomialAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Creates BinomialAccumulatorFactory object.

    Attributes:
        max_val (Optional[int]): Max value for binomial observations.
        min_val (Optional[int]): min value for binomial observations.
        keys (Optional[str]): Declare BinomialAccumulatorFactory objects for merging suff_stats.

    """

    def __init__(self, max_val: Optional[int] = None, min_val: Optional[int] = 0, keys: Optional[str] = None, device: Optional[tn.device] = None) -> None:
        """BinomialAccumulatorFactory object.

        Args:
            max_val (Optional[int]): Max value for binomial observations.
            min_val (Optional[int]): min value for binomial observations.
            keys (Optional[str]): Declare BinomialAccumulatorFactory objects for merging suff_stats.

        """

        self.max_val = max_val
        self.min_val = min_val
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> 'BinomialAccumulator':
        return BinomialAccumulator(max_val=self.max_val, min_val=self.min_val, keys=self.keys, device=device)


class BinomialEstimator(TorchParameterEstimator):
    """Create a BinomialEstimator object for estimating BinomialDistribution.

    Attributes:
        max_val (Optional[int]): Set max value encountered.
        min_val (Optional[int]): Set min value for BinomialDistribution.
        pseudo_count (Optional[float]): Inflate sufficient statistic (p).
        suff_stat (Optional[float]): Set p from prior observations.
        keys (Optional[str]): Assign key to BinomialEstimator designating all same key estimators to later be combined,
            in aggregation.

    """

    def __init__(self, max_val: Optional[int] = None, min_val: Optional[int] = 0, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[float] = None, keys: Optional[str] = None) -> None:
        """BinomialEstimator object.

        Args:
            max_val (Optional[int]): Set max value encountered.
            min_val (Optional[int]): Set min value for BinomialDistribution.
            pseudo_count (Optional[float]): Inflate sufficient statistic (p).
            suff_stat (Optional[float]): Set p from prior observations.
            keys (Optional[str]): Assign key to BinomialEstimator designating all same key estimators to later be combined,
                in accumualtation.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.min_val = min_val if min_val is not None else 0
        self.max_val = max_val
        self.keys = keys

    def accumulator_factory(self) -> BinomialAccumulatorFactory:
        return BinomialAccumulatorFactory(self.max_val, self.min_val, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, Optional[int], Optional[int]], device: Optional[tn.device] = None) -> 'BinomialDistribution':
        """Estimate a BinomialDistribution from BinomialEstimator using sufficient statistics in suff_stat.

        Note: nobs is not used here. Kept for consistency with other ParameterEstimators.

        Memeber variable suff_stat is simply the proportion (p) of the BinomialDistributon passed to BinomalEstimator.
        The pseudo_count is used to inflate (p) in estimation.

        Args:
            nobs (Optional[float]): Not used.
            suff_stat (Tuple[float, float, Optional[int], Optional[int]]): Tuple of count, sum, min_val max_val,
                obtained from aggregation of data.
            device: Set the device for the estimate to be returned to.

        Returns:
            BinomialDistribution estimated from suff_stat input and member variables suff_stat and pseudo_count.

        """
        count, sum, min_val, max_val = suff_stat

        if min_val is not None:
            if self.min_val is not None:
                min_val = min(min_val, self.min_val)
        else:
            if self.min_val is not None:
                min_val = self.min_val
            else:
                min_val = 0

        if max_val is not None:
            if self.max_val is not None:
                max_val = max(max_val, self.max_val)
        else:
            if self.max_val is not None:
                max_val = self.max_val
            else:
                max_val = 0

        n = max_val - min_val

        if self.pseudo_count is not None and self.suff_stat is not None:
            pn = self.pseudo_count
            pp = self.suff_stat
            p = (sum - min_val * count + pp) / ((count + pn) * n)

        elif self.pseudo_count is not None and self.suff_stat is None:
            pn = self.pseudo_count
            pp = self.pseudo_count * 0.5 * n
            p = (sum - min_val * count + pp) / ((count + pn) * n)

        else:
            if count > 0 and n > 0:
                p = (sum - min_val * count) / (count * n)
            else:
                p = 0.5

        return BinomialDistribution(p, max_val - min_val, min_val=min_val, keys=self.keys, device=device)


class BinomialDataEncoder(TorchSequenceEncoder):
    """BinomialDataEncoder object used to encode Sequence[int] or ndarray[int]."""

    def __str__(self) -> str:
        """Creates string name of BinomialDataEncoder.

        Returns:
            String name BinomialDataEncoder

        """
        return 'BinomialDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Define equality for BinomialDataEncoder objects.

        Args:
            other (object): Any object to be compares to BinomialDataEncoder.

        Returns:
            True is other is BinomialDataEncoder, else False.

        """
        return isinstance(other, BinomialDataEncoder)

    def seq_encode(self, x: Sequence[int], device: Optional[tn.device] = None) -> 'BinomialTorchEncodedSequence':
        """Encode List[int] for vectorized seq calls in Accumulator and Distribution.

        Args:
            x (List[int]): List of integers.
            device (Optional[device]): Set device for data to be encoded to.

        Returns:
            Tuple[tn.Tensor, tn.Tensor, tn.Tensor, int, int] containing unique values in x, indices of ux to
                reconstruct x, numpy array of x, min value of x, and max value of x.

        """
        xx = vec.int_tensor(x, device=device)

        if tn.any(xx < 0) or tn.any(tn.isnan(xx)):
            raise Exception('BinomialDistribution requires non-negative integer values for x.')

        ux, ix = tn.unique(xx, return_inverse=True)
        min_val = int(tn.min(ux))
        max_val = int(tn.max(ux))

        return BinomialTorchEncodedSequence(data=(ux, ix, xx, min_val, max_val), device=device)


class BinomialTorchEncodedSequence(TorchEncodedSequence):
    def __init__(self, data: Tuple[tn.Tensor, tn.Tensor, tn.Tensor, int, int], device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'BinomialTorchEncodedSequence(device={repr(self.device)})'



