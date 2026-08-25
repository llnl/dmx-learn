r"""Provide a finite distribution over complete rankings.

The support is all permutations of ``range(dim)``. A ranking :math:`x` has
mass proportional to :math:`\exp(-\rho\lVert x-\sigma\rVert^2)`, where
``sigma`` is a reference rank vector and ``rho`` controls concentration.
Normalization and sampling enumerate every permutation, so this implementation
is suitable only for small dimensions. The estimator accumulates weighted rank
sums, sets ``sigma`` to their ordering, and fixes ``rho`` to one when data are
present; it is a ranking-center estimator, not a general correlation fit.
"""

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState

from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class SpearmanRankingDistribution(SequenceEncodableProbabilityDistribution):
    """Represent a squared-rank-distance distribution over permutations.

    Attributes:
        sigma (np.ndarray]): Numpy array of means for the rank variables.
        rho (float): Decay rate on variance of ranks.
        name (Optional[str]): Name for object instance.
        dim (int): Dimension of the rank variable.
        keys (Optional[str]): Set keys for object instance.

    """

    def __init__(
        self,
        sigma: Union[Sequence[float], np.ndarray],
        rho: float = 1.0,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a complete-ranking distribution.

        Args:
            sigma (np.ndarray]): Numpy array of means for the rank variables.
            rho (float): Decay rate on variance of ranks.
            name (Optional[str]): Name for object instance.
            keys (Optional[str]): Set keys for object instance.

        """
        super().__init__()
        self.sigma = np.asarray(sigma)
        self.rho = rho
        self.name = name
        self.dim = len(sigma)
        self.keys = keys

        perms = map(np.asarray, map(list, itertools.permutations(range(self.dim))))
        self.log_const = np.log(
            sum(np.exp(-rho * np.dot(self.sigma - u, self.sigma - u)) for u in perms)
        )

    def __str__(self) -> str:
        """Return an evaluable distribution representation."""
        return (
            f"SpearmanRankingDistribution(sigma={repr(self.sigma.tolist())}, "
            f"rho={repr(self.rho)}, name={repr(self.name)}, keys={repr(self.keys)})"
        )

    def density(self, x: List[int]) -> float:
        """Return the probability of one complete ranking."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: List[int]) -> float:
        """Return the log probability of one complete ranking."""
        temp = np.subtract(x, self.sigma)
        return float(-self.rho * np.dot(temp, temp) - self.log_const)

    def seq_log_density(self, x: "SpearmanRankingEncodedDataSequence") -> np.ndarray:
        """Return log probabilities for an encoded ranking batch."""
        if not isinstance(x, SpearmanRankingEncodedDataSequence):
            raise TypeError(
                "SpearmanRankingEncodedDataSequence required for seq_log_density()."
            )

        temp = x.data - self.sigma
        temp *= temp
        rv = np.sum(temp, axis=1) * -self.rho
        rv -= self.log_const
        return np.asarray(rv, dtype=float)

    def sampler(self, seed: Optional[int] = None) -> "SpearmanRankingSampler":
        """Create an exact enumerative ranking sampler."""
        return SpearmanRankingSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "SpearmanRankingEstimator":
        """Create the ranking-center estimator."""
        return SpearmanRankingEstimator(
            self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "SpearmanRankingDataEncoder":
        """Return the encoder for complete-ranking batches."""
        return SpearmanRankingDataEncoder()


class SpearmanRankingSampler(DistributionSampler):
    """Sample rankings by enumerating the finite permutation support.

    Attributes:
        rng (RandomState): Seed samples.
        dist (SpearmanRankingDistribution): Distribution to draw samples from.
        perms (List[List[int]]): List of all possible rankings.
        probs (np.ndarray): Probability of each permutation.

    """

    def __init__(
        self, dist: SpearmanRankingDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize an enumerative ranking sampler.

        Args:
            dist (SpearmanRankingDistribution): Distribution to draw samples from.
            seed (Optional[int]): Set seed for generating samples.

        """
        super().__init__(dist, seed)

        self.perms = list(map(list, itertools.permutations(range(dist.dim))))
        encoder = self.dist.dist_to_encoder()
        self.probs = np.exp(dist.seq_log_density(encoder.seq_encode(self.perms)))

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[int], Sequence[List[int]]]:
        """Draw one ranking, or ``size`` independently drawn rankings."""
        idx = self.rng.choice(len(self.perms), p=self.probs, replace=True, size=size)

        if size is None:
            return self.perms[int(idx)]
        return [self.perms[int(u)] for u in np.asarray(idx, dtype=int)]


class SpearmanRankingAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted rank sums for the ranking-center estimator.

    Attributes:
        sum (np.ndarray): Suff stat counts
        count (float): Suff stat total weight count.
        keys (Optional[str]): Key for distribution.
        name (Optional[str]): Name for object.
    """

    def __init__(
        self, dim: int, name: Optional[str] = None, keys: Optional[str] = None
    ) -> None:
        """Initialize a weighted-ranking accumulator.

        Args:
            dim (int): Dimension of rankings.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Key for distribution.

        """
        self.sum = np.zeros(dim, dtype=np.float64)
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(
        self,
        x: Union[List[int], np.ndarray],
        weight: float,
        estimate: Optional[SpearmanRankingDistribution],
    ) -> None:
        """Initialize from one ranking without randomization."""
        self.sum += np.multiply(x, weight)
        self.count += weight

    def initialize(
        self, x: Union[List[int], np.ndarray], weight: float, rng: RandomState
    ) -> None:
        """Accumulate a weighted encoded ranking batch."""
        del rng
        if weight != 0:
            self.sum += np.multiply(x, weight)
            self.count += 0

    def seq_update(
        self,
        x: "SpearmanRankingEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[SpearmanRankingDistribution],
    ) -> None:
        """Initialize from an encoded ranking batch."""
        self.sum += np.dot(x.data.T, weights)
        self.count += weights.sum()

    def seq_initialize(
        self,
        x: "SpearmanRankingEncodedDataSequence",
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Initialize from an encoded ranking batch."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[float, np.ndarray]
    ) -> "SpearmanRankingAccumulator":
        """Combine rank-sum sufficient statistics."""
        self.sum += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self) -> Tuple[float, np.ndarray]:
        """Return total weight and rank-sum vector."""
        return self.count, self.sum

    def from_value(self, x: Tuple[float, np.ndarray]) -> "SpearmanRankingAccumulator":
        """Restore total weight and rank sums from a tuple."""
        self.sum = x[1]
        self.count = x[0]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge keyed rank-sum sufficient statistics."""
        if self.keys is not None:
            if self.keys in stats_dict:
                vals = stats_dict[self.keys]
                stats_dict[self.keys] = (vals[0] + self.count, vals[1] + self.sum)
            else:
                stats_dict[self.keys] = (self.count, self.sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace keyed rank-sum sufficient statistics."""
        if self.keys is not None:
            if self.keys in stats_dict:
                vals = stats_dict[self.keys]
                self.count = vals[0]
                self.sum = vals[1]

    def acc_to_encoder(self) -> "SpearmanRankingDataEncoder":
        """Return the ranking encoder."""
        return SpearmanRankingDataEncoder()


class SpearmanRankingAccumulatorFactory(StatisticAccumulatorFactory):
    """Create rank-sum accumulators for a fixed ranking dimension.

    Attributes:
        dim (int): Dimension of rankings.
        keys (Optional[str]): Key for distribution.
        name (Optional[str]): Name for object.

    """

    def __init__(
        self, dim: int, name: Optional[str] = None, keys: Optional[str] = None
    ) -> None:
        """Initialize a ranking accumulator factory.

        Args:
            dim (int): Dimension of rankings.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Key for distribution.

        """
        self.dim = dim
        self.keys = keys
        self.name = name

    def make(self) -> "SpearmanRankingAccumulator":
        """Create a fresh ranking accumulator."""
        return SpearmanRankingAccumulator(dim=self.dim, name=self.name, keys=self.keys)


class SpearmanRankingEstimator(ParameterEstimator):
    """Estimate a ranking center from weighted rank sums.

    Attributes:
        dim (int): Dimension of rankings.
        psuedo_count (Optional[float]): Regularize suff stat for estimates.
        suff_stat (Optional[Tuple[float, np.ndarray]]): Suff stat for regularization.
        keys (Optional[str]): Key for distribution.
        name (Optional[str]): Name for object.

    """

    def __init__(
        self,
        dim: int,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Tuple[float, np.ndarray]] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a ranking-center estimator.

        Args:
            dim (int): Dimension of rankings.
            pseudo_count (Optional[float]): Regularize suff stat for estimates.
            suff_stat (Optional[Tuple[float, np.ndarray]]): Suff stat for
                regularization.
            name (Optional[str]): Name for object.
            keys (Optional[str]): Key for distribution.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError(
                "SpearmanRankingEstimator requires keys to be of type 'str'."
            )

        self.dim = dim
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "SpearmanRankingAccumulatorFactory":
        """Return a factory for rank-sum statistics."""
        return SpearmanRankingAccumulatorFactory(self.dim, self.name, self.keys)

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[float, np.ndarray]
    ) -> "SpearmanRankingDistribution":
        """Estimate the reference ranking and fixed concentration parameter."""
        count, vsum = suff_stat

        if count > 0:
            sigma = np.argsort(vsum)
            rho = 1.0
        else:
            sigma = vsum
            rho = 0.0

        return SpearmanRankingDistribution(sigma, rho, name=self.name, keys=self.keys)


class SpearmanRankingDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of Spearman rho observations."""

    def __str__(self) -> str:
        """Return an evaluable encoder representation."""
        return "SpearmanRankingDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a ranking encoder."""
        return isinstance(other, SpearmanRankingDataEncoder)

    def seq_encode(
        self, x: Sequence[List[int]]
    ) -> "SpearmanRankingEncodedDataSequence":
        """Encode rankings as a two-dimensional integer array."""
        rv = np.asarray(x)

        return SpearmanRankingEncodedDataSequence(data=rv)


class SpearmanRankingEncodedDataSequence(EncodedDataSequence):
    """Hold encoded rankings with shape ``(N, dim)``.

    Attributes:
        data (np.ndarray): Iid observations from spearman rho ranking distribution.

    """

    def __init__(self, data: np.ndarray):
        """Initialize encoded rankings.

        Args:
            data (np.ndarray): Iid observations from spearman rho ranking distribution.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing encoded rankings."""
        return f"SpearmanRankingEncodedDataSequence(data={self.data})"
