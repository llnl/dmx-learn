"""Create, estimate, and sample from the binomial distribution.

Defines the BinomialDistribution, BinomialSampler, BinomialAccumulatorFactory, BinomialAccumulator, BinomialEstimator,
and the BinomialDataEncoder classes for use with dmx-learn.

Data type: int.
"""

import numpy as np
from numpy.random import RandomState
from dmx.utils.vector import gammaln
from dmx.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    ParameterEstimator,
    DistributionSampler,
    StatisticAccumulatorFactory,
    SequenceEncodableStatisticAccumulator,
    DataSequenceEncoder,
    EncodedDataSequence,
)
from typing import Optional, Dict, List, Union, Tuple, Any, Sequence

E = Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]


class BinomialDistribution(SequenceEncodableProbabilityDistribution):
    """BinomialDistribution object used for Binomial

    Attributes:
        p (float): Probability of success, between (0, 1.0].
        log_p (float): Logarithm of p.
        log_1p (float): Logarithm of 1-p.
        n (int): Number of trials, n > 0.
        min_val (Optional[int]): Minimum value of the support.
        name (Optional[str]): Name of the distribution.
        keys (Optional[str]): Key for identifying equivalent distributions.
    """

    def __init__(
        self,
        p: float,
        n: int,
        min_val: Optional[int] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize BinomialDistribution.

        Args:
            p (float): Probability of success, between (0, 1.0].
            n (int): Number of trials, n > 0.
            min_val (Optional[int], optional): Minimum value of the support. Defaults to None.
            name (Optional[str], optional): Name of the distribution. Defaults to None.
            keys (Optional[str], optional): Key for identifying equivalent distributions. Defaults to None.

        Raises:
            Exception: If p is not in (0, 1) or n is not positive.
        """
        if p <= 0.0 or p >= 1.0 or np.isnan(p):
            raise Exception('Binomial distribution requires p in [0,1]')
        self.p = p

        if n < 0 or np.isinf(n) or np.isnan(n):
            raise Exception('Binomial distribution requires n > 0.')
        self.n = n

        self.log_p = np.log(p)
        self.log_1p = np.log1p(-p)
        self.name = name
        self.keys = keys
        self.min_val = min_val

    def __str__(self) -> str:
        """Return string representation of BinomialDistribution."""
        return (
            f'BinomialDistribution(p={self.p!r}, n={self.n!r}, min_val={self.min_val!r}, '
            f'name={self.name!r}, keys={self.keys!r})'
        )

    def density(self, x: int) -> float:
        """Return the probability mass at integer value x.

        Args:
            x (int): Value for density evaluation.

        Returns:
            float: Probability mass at x. 0.0 if x is not in support.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Return the log-probability mass at integer value x.

        Args:
            x (int): Value for log-density evaluation.

        Returns:
            float: Log-probability mass at x. -inf if x is not in support.
        """
        n = self.n
        if self.min_val is not None:
            xx = x - self.min_val
        else:
            xx = x

        return (
            gammaln(n + 1)
            - gammaln(xx + 1)
            - gammaln(n - xx + 1)
            + self.log_1p * (n - xx)
            + self.log_p * xx
        )

    def seq_log_density(self, x: 'BinomialEncodedDataSequence') -> np.ndarray:
        """Vectorized log-density for encoded data.

        Args:
            x (BinomialEncodedDataSequence): Encoded data sequence.

        Returns:
            np.ndarray: Log-density values.
        """
        if not isinstance(x, BinomialEncodedDataSequence):
            raise Exception('BinomialDistribution.seq_log_density() requires BinomialEncodedDataSequence.')

        ux, ix, _, _, _ = x.data
        n = self.n
        gn = gammaln(n + 1)

        if self.min_val is not None:
            xx = ux - self.min_val
        else:
            xx = ux

        cc = (
            gn
            - gammaln(xx + 1)
            - gammaln((n + 1) - xx)
            + self.log_1p * (n - xx)
            + self.log_p * xx
        )
        return cc[ix]

    def sampler(self, seed: Optional[int] = None) -> 'BinomialSampler':
        """Return a BinomialSampler for this distribution.

        Args:
            seed (Optional[int], optional): Seed for RNG. Defaults to None.

        Returns:
            BinomialSampler: Sampler for this distribution.
        """
        return BinomialSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'BinomialEstimator':
        """Create a BinomialEstimator for this distribution.

        Args:
            pseudo_count (Optional[float], optional): Pseudo-count for prior. Defaults to None.

        Returns:
            BinomialEstimator: Estimator object.
        """
        if pseudo_count is None:
            return BinomialEstimator(name=self.name, keys=self.keys)
        else:
            return BinomialEstimator(
                max_val=self.n,
                min_val=self.min_val,
                pseudo_count=pseudo_count,
                suff_stat=self.p * self.n * pseudo_count,
                name=self.name,
                keys=self.keys,
            )

    def dist_to_encoder(self) -> 'BinomialDataEncoder':
        """Return a BinomialDataEncoder."""
        return BinomialDataEncoder()


class BinomialSampler(DistributionSampler):
    """Sampler for BinomialDistribution.

    Attributes:
        dist (BinomialDistribution): Distribution to sample from.
        rng (RandomState): Random number generator.
    """

    def __init__(self, dist: BinomialDistribution, seed: Optional[int] = None) -> None:
        """Initialize BinomialSampler.

        Args:
            dist (BinomialDistribution): Distribution to sample from.
            seed (Optional[int], optional): Seed for RNG. Defaults to None.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, List[int]]:
        """Draw samples from the distribution.

        Args:
            size (Optional[int], optional): Number of samples to draw. Defaults to None.

        Returns:
            Union[int, List[int]]: Single sample or list of samples.
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


class BinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of BinomialDistribution.

    Attributes:
        sum (float): Sum of data observations.
        count (float): Number of weighted observations.
        max_val (Optional[int]): Largest value encountered.
        min_val (Optional[int]): Smallest value encountered.
        name (Optional[str]): Name of the accumulator.
        key (Optional[str]): Key for merging accumulators.
    """

    def __init__(
        self,
        max_val: Optional[int] = None,
        min_val: Optional[int] = 0,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize BinomialAccumulator.

        Args:
            max_val (Optional[int], optional): Largest value encountered. Defaults to None.
            min_val (Optional[int], optional): Smallest value encountered. Defaults to 0.
            name (Optional[str], optional): Name of the accumulator. Defaults to None.
            keys (Optional[str], optional): Key for merging accumulators. Defaults to None.
        """
        self.sum: float = 0.0
        self.count: float = 0.0
        self.key: Optional[str] = keys
        self.name: Optional[str] = name
        self.max_val: Optional[int] = max_val
        self.min_val: Optional[int] = min_val

    def update(
        self, x: int, weight: float, estimate: Optional['BinomialDistribution']
    ) -> None:
        """Update accumulator with a new observation.

        Args:
            x (int): Observation.
            weight (float): Weight for the observation.
            estimate (Optional[BinomialDistribution]): Not used.
        """
        self.sum += x * weight
        self.count += weight

        if self.min_val is None:
            self.min_val = x
        else:
            self.min_val = min(self.min_val, x)

        if self.max_val is None:
            self.max_val = x
        else:
            self.max_val = max(self.max_val, x)

    def initialize(
        self, x: int, weight: float, rng: Optional[RandomState]
    ) -> None:
        """Initialize accumulator with a new observation.

        Args:
            x (int): Observation.
            weight (float): Weight for the observation.
            rng (Optional[RandomState]): Not used.
        """
        self.update(x, weight, None)

    def seq_update(
        self,
        x: 'BinomialEncodedDataSequence',
        weights: np.ndarray,
        estimate: Optional['BinomialDistribution'],
    ) -> None:
        """Vectorized update for encoded data.

        Args:
            x (BinomialEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            estimate (Optional[BinomialDistribution]): Not used.
        """
        _, _, xx, min_val, max_val = x.data

        self.sum += np.sum(xx * weights)
        self.count += np.sum(weights)

        if self.min_val is not None:
            self.min_val = min(self.min_val, min_val)
        else:
            self.min_val = min_val

        if self.max_val is not None:
            self.max_val = max(self.max_val, max_val)
        else:
            self.max_val = max_val

    def seq_initialize(
        self,
        x: 'BinomialEncodedDataSequence',
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Vectorized initialization for encoded data.

        Args:
            x (BinomialEncodedDataSequence): Encoded data sequence.
            weights (np.ndarray): Weights for each observation.
            rng (Optional[RandomState]): Not used.
        """
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[float, float, Optional[int], Optional[int]]
    ) -> 'BinomialAccumulator':
        """Combine another accumulator's sufficient statistics into this one.

        Args:
            suff_stat (Tuple[float, float, Optional[int], Optional[int]]): Sufficient statistics to combine.

        Returns:
            BinomialAccumulator: Self after combining.
        """
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
        """Return the sufficient statistics as a tuple.

        Returns:
            Tuple[float, float, Optional[int], Optional[int]]: (count, sum, min_val, max_val)
        """
        return self.count, self.sum, self.min_val, self.max_val

    def from_value(
        self, x: Tuple[float, float, Optional[int], Optional[int]]
    ) -> 'BinomialAccumulator':
        """Set the sufficient statistics from a tuple.

        Args:
            x (Tuple[float, float, Optional[int], Optional[int]]): Sufficient statistics.

        Returns:
            BinomialAccumulator: Self after setting values.
        """
        self.count = x[0]
        self.sum = x[1]
        self.min_val = x[2]
        self.max_val = x[3]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's values with those from a dictionary by key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of accumulators.
        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'BinomialDataEncoder':
        """Return a BinomialDataEncoder."""
        return BinomialDataEncoder()


class BinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for BinomialAccumulator.

    Attributes:
        max_val (Optional[int]): Max value for binomial observations.
        min_val (Optional[int]): Min value for binomial observations.
        name (Optional[str]): Name of the factory.
        keys (Optional[str]): Key for merging accumulators.
    """

    def __init__(
        self,
        max_val: Optional[int] = None,
        min_val: Optional[int] = 0,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize BinomialAccumulatorFactory.

        Args:
            max_val (Optional[int], optional): Max value for binomial observations. Defaults to None.
            min_val (Optional[int], optional): Min value for binomial observations. Defaults to 0.
            name (Optional[str], optional): Name of the factory. Defaults to None.
            keys (Optional[str], optional): Key for merging accumulators. Defaults to None.
        """
        self.max_val = max_val
        self.min_val = min_val if min_val is not None else 0
        self.name = name
        self.keys = keys

    def make(self) -> BinomialAccumulator:
        """Create a new BinomialAccumulator.

        Returns:
            BinomialAccumulator: New accumulator instance.
        """
        return BinomialAccumulator(self.max_val, self.min_val, self.name, self.keys)


class BinomialEstimator(ParameterEstimator):
    """Estimator for BinomialDistribution.

    Attributes:
        max_val (Optional[int]): Max value encountered.
        min_val (Optional[int]): Min value for BinomialDistribution.
        pseudo_count (Optional[float]): Pseudo-count for prior.
        suff_stat (Optional[float]): Sufficient statistic for prior.
        name (Optional[str]): Name of the estimator.
        keys (Optional[str]): Key for merging estimators.
    """

    def __init__(
        self,
        max_val: Optional[int] = None,
        min_val: Optional[int] = 0,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[float] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize BinomialEstimator.

        Args:
            max_val (Optional[int], optional): Max value encountered. Defaults to None.
            min_val (Optional[int], optional): Min value for BinomialDistribution. Defaults to 0.
            pseudo_count (Optional[float], optional): Pseudo-count for prior. Defaults to None.
            suff_stat (Optional[float], optional): Sufficient statistic for prior. Defaults to None.
            name (Optional[str], optional): Name of the estimator. Defaults to None.
            keys (Optional[str], optional): Key for merging estimators. Defaults to None.

        Raises:
            TypeError: If keys is not a string or None.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("BinomialEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_val = min_val if min_val is not None else 0
        self.max_val = max_val

    def accumulator_factory(self) -> BinomialAccumulatorFactory:
        """Return a BinomialAccumulatorFactory."""
        return BinomialAccumulatorFactory(self.max_val, self.min_val, self.name, self.keys)

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[float, float, Optional[int], Optional[int]]
    ) -> BinomialDistribution:
        """Estimate a BinomialDistribution from sufficient statistics.

        Args:
            nobs (Optional[float]): Not used.
            suff_stat (Tuple[float, float, Optional[int], Optional[int]]): (count, sum, min_val, max_val).

        Returns:
            BinomialDistribution: Estimated distribution.
        """
        count, sum_, min_val, max_val = suff_stat

        if min_val is not None:
            if self.min_val is not None:
                min_val = min(min_val, self.min_val)
        else:
            min_val = self.min_val if self.min_val is not None else 0

        if max_val is not None:
            if self.max_val is not None:
                max_val = max(max_val, self.max_val)
        else:
            max_val = self.max_val if self.max_val is not None else 0

        n = max_val - min_val

        if self.pseudo_count is not None and self.suff_stat is not None:
            pn = self.pseudo_count
            pp = self.suff_stat
            p = (sum_ - min_val * count + pp) / ((count + pn) * n)
        elif self.pseudo_count is not None and self.suff_stat is None:
            pn = self.pseudo_count
            pp = self.pseudo_count * 0.5 * n
            p = (sum_ - min_val * count + pp) / ((count + pn) * n)
        else:
            if count > 0 and n > 0:
                p = (sum_ - min_val * count) / (count * n)
            else:
                p = 0.5

        return BinomialDistribution(
            p, max_val - min_val, min_val=min_val, name=self.name, keys=self.keys
        )


class BinomialDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of integers for BinomialDistribution."""

    def __str__(self) -> str:
        """Return string representation."""
        return 'BinomialDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other (object): Other object.

        Returns:
            bool: True if other is a BinomialDataEncoder.
        """
        return isinstance(other, BinomialDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> 'BinomialEncodedDataSequence':
        """Encode a sequence of integers for vectorized operations.

        Args:
            x (Sequence[int]): Sequence of integers.

        Returns:
            BinomialEncodedDataSequence: Encoded data sequence.

        Raises:
            Exception: If any value is negative or NaN.
        """
        xx = np.array(x)

        if np.any(xx < 0) or np.any(np.isnan(xx)):
            raise Exception('BinomialDistribution requires non-negative integer values for x.')

        xx = np.asarray(x, dtype=np.int32)
        ux, ix = np.unique(xx, return_inverse=True)
        min_val = np.min(ux)
        max_val = np.max(ux)

        return BinomialEncodedDataSequence(data=(ux, ix, xx, min_val, max_val))


class BinomialEncodedDataSequence(EncodedDataSequence):
    """Encoded data sequence for BinomialDistribution.

    Attributes:
        data (Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]): Unique values, inverse mapping, original values, min, max.
    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]) -> None:
        """Initialize BinomialEncodedDataSequence.

        Args:
            data (Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]): Encoded data.
        """
        super().__init__(data)

    def __repr__(self) -> str:
        """Return string representation."""
        return f'BinomialEncodedDataSequence(data={self.data})'

