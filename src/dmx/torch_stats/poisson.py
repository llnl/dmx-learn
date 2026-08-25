"""Torch-backed Poisson distributions, estimation, and sequence encoding.

The rate ``lam`` and ``(count, sum)`` sufficient-statistic terms match
``dmx.stats.poisson``. Vectorized likelihoods use torch tensors, while scalar
methods and samplers retain their Python/NumPy behavior.
"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from math import log
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tn
from numpy.random import RandomState

import dmx.torch_utils.vector as vec
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)
from dmx.utils.vector import gammaln


class PoissonDistribution(TorchProbabilityDistribution):
    """Represent a Poisson distribution with rate ``lam``.

    ``to`` only records the preferred device because ``lam`` is a Python float.
    """

    def __init__(self, lam: float, device: Optional[tn.device] = None) -> None:
        """Initialize the Poisson distribution.

        Args:
            lam (float): Positive real-valued number.
            device: Define device for Tensor calculations.

        """
        super().__init__(device)
        self.lam = lam
        self.log_lam = log(lam)

    def to(self, device: vec.DeviceLike) -> "PoissonDistribution":
        """Select the device used for subsequent tensor calculations."""
        self._device = self._resolve_device_arg(device)
        return self

    def __repr__(self) -> str:
        """Return an evaluable representation."""
        return f"PoissonDistribution({repr(self.lam)})"

    def density(self, x: int) -> float:
        """Evaluate the density of Poisson distribution at observation x.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            float: Density of Poisson distribution evaluated at x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: int) -> float:
        """Log-density of Poisson distribution evaluated at x.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            float: Log-density of Poisson distribution evaluated at x.

        """
        if x < 0:
            return -np.inf

        return x * self.log_lam - gammaln(x + 1.0) - self.lam

    def seq_log_density(self, x: "PoissonTorchSequence") -> tn.Tensor:
        """Evaluate log densities for an encoded sequence."""
        if not isinstance(x, PoissonTorchSequence):
            raise TypeError("Requires PoissonTorchSequence for `seq_` function calls.")

        rv = x.data[0] * self.log_lam
        rv -= x.data[1]
        rv -= self.lam

        return rv

    def sampler(self, seed: Optional[int] = None) -> "PoissonSampler":
        """Create a NumPy-backed sampler, optionally seeded."""
        return PoissonSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "PoissonEstimator":
        """Create an estimator, optionally regularized toward this rate."""
        if pseudo_count is None:
            return PoissonEstimator()

        return PoissonEstimator(pseudo_count=pseudo_count, suff_stat=self.lam)

    def dist_to_encoder(self) -> "PoissonDataEncoder":
        """Return the encoder for Poisson observations."""
        return PoissonDataEncoder()


class PoissonSampler(DistributionSampler):
    """PoissonSampler object used to draw samples from PoissonDistribution.

    Attributes:
        rng (RandomState): RandomState with seed set for sampling.
        dist (GeometricDistribution): PoissonDistribution to sample from.

    """

    def __init__(self, dist: "PoissonDistribution", seed: Optional[int] = None) -> None:
        """Initialize a sampler for ``dist``.

        Args:
            dist (PoissonDistribution): Set PoissonDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator
                used in sampling.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate iid samples from Poisson distribution.

        Generates a single Poisson sample (int) if size is None, else a numpy
        array of integers of length size containing iid samples from the
        Poisson distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None,
                assumed to be 1.

        Returns:
            If size is None, int, else size length numpy array of ints.

        """
        return self.rng.poisson(lam=self.dist.lam, size=size)


class PoissonAccumulator(TorchStatisticAccumulator):
    """PoissonAccumulator object used to accumulate sufficient statistics.

    Attributes:
         sum (float): Aggregate sum of weighted observations.
         count (float): Aggregate sum of observation weights.
         key (Optional[str]): Key for combining sufficient statistics with an
             object instance containing the same key.

    """

    def __init__(
        self, keys: Optional[str] = None, device: Optional[tn.device] = None
    ) -> None:
        """Initialize the Poisson sufficient-statistic accumulator.

        Args:
            keys (Optional[str]): Assign a string valued to key to object instance.
            device: Set device for Tensor calculations.

        """
        super().__init__(device)
        self.sum = 0.0
        self.count = 0.0
        self.key = keys

    def seq_initialize(
        self,
        x: "PoissonTorchSequence",
        weights: tn.Tensor,
        tng: Optional[tn.Generator] = None,
    ) -> None:
        """Initialize statistics from encoded observations and weights."""
        self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: "PoissonTorchSequence",
        weights: tn.Tensor,
        estimate: Optional["PoissonDistribution"] = None,
    ) -> None:
        """Accumulate encoded observations with their weights."""
        xx = x.data[0].to(device=weights.device, dtype=weights.dtype)
        self.sum += float(tn.dot(xx, weights))
        self.count += float(weights.sum())

    def combine(self, suff_stat: Tuple[float, float]) -> "PoissonAccumulator":
        """Merge a ``(count, sum)`` sufficient statistic."""
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        """Return the ``(count, sum)`` sufficient statistic."""
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> "PoissonAccumulator":
        """Replace state from a ``(count, sum)`` sufficient statistic."""
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator's keyed statistic into ``stats_dict``."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace state from its keyed statistic, when present."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "PoissonDataEncoder":
        """Return the compatible Poisson encoder."""
        return PoissonDataEncoder()


class PoissonAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create PoissonAccumulator objects.

    Attributes:
         keys (Optional[str]): Tag for combining sufficient statistics of
             PoissonAccumulator objects when constructed.

    """

    def __init__(self, keys: Optional[str] = None) -> None:
        """Initialize the factory.

        Args:
            keys (Optional[str]): Assign keys to PoissonAccumulatorFactory object.

        """
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> "PoissonAccumulator":
        """Create an accumulator associated with ``device``."""
        return PoissonAccumulator(keys=self.keys, device=device)


class PoissonEstimator(TorchParameterEstimator):
    """Estimate PoissonDistribution from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Optional[float]): Re-weight suff_stat.
        suff_stat (Optional[float]): Mean of Poisson if not None.
        keys (Optional[str]): String keys of PoissonEstimator instance for
            combining sufficient statistics.

    """

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[float] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize Poisson estimation settings.

        Attributes:
            pseudo_count (Optional[float]): Re-weight suff_stat.
            suff_stat (Optional[float]): Mean of Poisson if not None.
            keys (Optional[str]): String keys of PoissonEstimator instance for
                combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

    def accumulator_factory(self) -> "PoissonAccumulatorFactory":
        """Return a factory for compatible accumulators."""
        return PoissonAccumulatorFactory(self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[float, float],
        device: Optional[tn.device] = None,
    ) -> "PoissonDistribution":
        """Estimate a Poisson model from ``(count, sum)`` statistics."""
        nobs, psum = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:
            return PoissonDistribution(
                (psum + self.suff_stat * self.pseudo_count)
                / (nobs + self.pseudo_count),
                device=device,
            )

        return PoissonDistribution(psum / nobs, device=device)


class PoissonDataEncoder(TorchSequenceEncoder):
    """Encode nonnegative counts as a torch integer tensor of shape ``(n,)``.

    The output follows ``vec.int_tensor`` dtype rules and is placed on the
    requested CPU, CUDA, or MPS device. This is the material difference from
    the NumPy-backed encoder in ``stats``.
    """

    def __str__(self) -> str:
        """Return the encoder name."""
        return "PoissonDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a Poisson encoder."""
        return isinstance(other, PoissonDataEncoder)

    def seq_encode(
        self, x: Union[np.ndarray, Sequence[int]], device: Optional[tn.device] = None
    ) -> "PoissonTorchSequence":
        """Validate and encode nonnegative integer observations on ``device``."""
        rv1 = vec.tensor(x, device=device)

        if tn.any(rv1 < 0) or tn.any(tn.isnan(rv1)):
            raise ValueError("Poisson requires non-negative integer values of x.")

        rv2 = tn.lgamma(rv1 + 1.0)
        return PoissonTorchSequence(data=(rv1, rv2), device=device)


class PoissonTorchSequence(TorchEncodedSequence):
    """Store integer Poisson encoded data and its requested device."""

    data: Tuple[tn.Tensor, tn.Tensor]

    def __init__(
        self, data: Tuple[tn.Tensor, tn.Tensor], device: Optional[tn.device] = None
    ) -> None:
        """Initialize from tensor data and an optional target device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation including the stored device."""
        return f"PoissonTorchSequence(device={repr(self.device)})"
