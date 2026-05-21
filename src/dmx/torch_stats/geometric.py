# pylint: disable=line-too-long
"""Create, estimate, and sample from a geometric distribution with probability of success p.

Defines the GeometricDistribution, GeometricSampler, GeometricAccumulatorFactory, GeometricAccumulator,
GeometricEstimator, and the GeometricDataEncoder classes for use with pysparkplug.

Data type (int): The geometric distribution with probability of success p, has density

    P(x=k) = (k-1)*log(1-p) + log(p), for k = 1,2,...

"""

# pylint: disable=line-too-long,too-many-positional-arguments,duplicate-code
# pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin
# pylint: disable=broad-exception-raised,consider-using-f-string,no-else-return
# pylint: disable=no-else-raise,consider-using-enumerate,consider-using-generator
# pylint: disable=use-dict-literal,super-with-arguments,unnecessary-comprehension
# pylint: disable=simplifiable-if-statement,nested-min-max

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tn
from numpy.random import RandomState

import dmx.torch_utils.vector as vec
from dmx.arithmetic import *
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)


class GeometricDistribution(TorchProbabilityDistribution):
    """GeometricDistribution object defining geometric distribution with probability of success p.

    Notes:
        Mean: 1/p, Variance: (1-p)/p^2.

    Attributes:
        p (float): Probability of success, must between (0,1).
        log_p (float): Log of probability of success p.
        log_1p (float): Log of 1-p (prob of failure).
        _device (device): Device for tensors.

    """

    def __init__(self, p: float, device: Optional[tn.device] = None) -> None:
        """GeometricDistribution object.

        Args:
            p (float): Must between (0,1).
            device (Optional[device]): Device for tensors.

        """
        super().__init__(device)
        self.p = p
        self.log_p = np.log(p)
        self.log_1p = np.log1p(-p)

    def __repr__(self) -> str:
        return f"GeometricDistribution(p={repr(self.p)})"

    def to(self, device: tn.device) -> None:
        self._device = device

    def density(self, x: int) -> float:
        """Density of geometric distribution evaluated at x.

        Notes:

            P(x=k) = (k-1)*log(1-p) + log(p), for x = 1,2,..., else 0.0.

        Args:
            x (int): Observed geometric value (1,2,3,....).


        Returns:
            float: Density of geometric distribution evaluated at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of geometric distribution evaluated at x.

        Notes:
            See density() for details.

        Args:
            x (int): Must be natural number (1,2,3,....).

        Returns:
            float: Log-density of geometric distribution evaluated at x.

        """
        return (x - 1) * self.log_1p + self.log_p

    def seq_log_density(self, x: "GeometricTorchEncodedSequence") -> tn.Tensor:

        if not isinstance(x, GeometricTorchEncodedSequence):
            raise Exception(
                "Requires GeometricTorchEncodedSequence for `seq_` function calls."
            )

        rv = x.data - 1
        rv *= self.log_1p
        rv += self.log_p

        return rv

    def sampler(self, seed: Optional[int] = None) -> "GeometricSampler":
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "GeometricEstimator":
        if pseudo_count is None:
            return GeometricEstimator()
        else:
            return GeometricEstimator(pseudo_count=pseudo_count, suff_stat=self.p)

    def dist_to_encoder(self) -> "GeometricDataEncoder":
        return GeometricDataEncoder()


class GeometricSampler(DistributionSampler):
    """GeometricSampler object used to draw samples from GeometricDistribution.

    Attributes:
        rng (RandomState): RandomState with seed set for sampling.
        dist (GeometricDistribution): GeometricDistribution to sample from.

    """

    def __init__(self, dist: GeometricDistribution, seed: Optional[int] = None) -> None:
        """GeometricSampler object.

        Args:
            dist (GeometricDistribution): GeometricDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate iid samples from geometric distribution.

        Generates a single geometric sample (int) if size is None, else a numpy array of integers of length size,
        iid samples, from the geometric distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, assumed to be 1.

        Returns:
            If size is None, int, else size length numpy array of ints.

        """
        return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(TorchStatisticAccumulator):
    """GeometricAccumulator object used to accumulate sufficient statistics from observations.

    Attributes:
        sum (float): Aggregate weighted sum of observations.
        count (float): Aggregate sum of weighted observation count.
        key (Optional[str]): Assigned from keys arg.

    """

    def __init__(self, keys: Optional[str] = None, device: Optional[tn.device] = None):
        """GeometricAccumulator object.

        Args:
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.
            device (Optional[device]): Device for tensor calculations.

        """
        super().__init__(device)
        self.sum = 0.0
        self.count = 0.0
        self.key = keys

    def seq_update(
        self,
        x: "GeometricTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: Optional["GeometricDistribution"],
    ) -> None:
        self.sum += float(tn.dot(x.data, weights))
        self.count += float(tn.sum(weights))

    def seq_initialize(
        self,
        x: "GeometricTorchEncodedSequence",
        weights: tn.Tensor,
        tng: Optional[tn.Generator],
    ) -> None:
        del tng
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> "GeometricAccumulator":
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> "GeometricAccumulator":
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:

        if self.key is not None:
            if self.key in stats_dict:
                x0, x1 = stats_dict[self.key]
                self.count += x0
                self.sum += x1

            else:
                stats_dict[self.key] = (self.count, self.sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.key is not None:
            if self.key in stats_dict:
                self.count, self.sum = stats_dict[self.key]

    def acc_to_encoder(self) -> "GeometricDataEncoder":
        return GeometricDataEncoder()


class GeometricAccumulatorFactory(TorchStatisticAccumulatorFactory):
    def __init__(self, keys: Optional[str] = None) -> None:
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> "GeometricAccumulator":
        return GeometricAccumulator(device=device, keys=self.keys)


class GeometricEstimator(TorchParameterEstimator):
    """GeometricEstimator object for estimating GeometricDistribution object from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Optional[float]): Assigned from pseudo_count arg.
        suff_stat (Optional[float]): Assigned from suff_stat arg (corrected for [0,1] constraint).
        keys (Optional[str]): Assigned from keys arg.

    """

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[float] = None,
        keys: Optional[str] = None,
    ) -> None:
        """GeometricEstimator object.

        Args:
            pseudo_count (Optional[float]): Float value for re-weighting suff_stat member variable.
            suff_stat (Optional[float]): Probability of success (value between (0,1)).
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = (
            min(min(suff_stat, 1.0), 0.0) if suff_stat is not None else None
        )
        self.keys = keys

    def accumulator_factory(self) -> "GeometricAccumulatorFactory":
        return GeometricAccumulatorFactory(keys=self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[float, float],
        device: Optional[tn.device] = None,
    ) -> "GeometricDistribution":
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[0] + self.pseudo_count * self.suff_stat) / (
                suff_stat[1] + self.pseudo_count
            )
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[0] + self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        else:
            p = suff_stat[0] / suff_stat[1]

        return GeometricDistribution(p, device=device)


class GeometricDataEncoder(TorchSequenceEncoder):
    """GeometricDataEncoder object for encoding sequences of iid geometric observations with data type int."""

    def __str__(self) -> str:
        return "GeometricDataEncoder"

    def __eq__(self, other) -> bool:
        return isinstance(other, GeometricDataEncoder)

    def seq_encode(
        self, x: Union[Sequence[int], np.ndarray], device: Optional[tn.device] = None
    ) -> "GeometricTorchEncodedSequence":
        rv = vec.tensor(x, device=device)
        if tn.any(rv < 1) or tn.any(tn.isnan(rv)):
            raise Exception(
                "GeometricDistribution requires integers greater than 0 for x."
            )

        return GeometricTorchEncodedSequence(data=rv, device=device)


class GeometricTorchEncodedSequence(TorchEncodedSequence):

    def __init__(self, data: tn.tensor, device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f"GeometricTorchEncodedSequence(device={repr(self.device)})"
