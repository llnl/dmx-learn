"""Torch-backed exponential distributions, estimation, and sequence encoding.

This module uses the same scale (mean) ``beta`` parameterization as
``dmx.stats.exponential``.  Scalar methods return Python floats and sampling
uses NumPy; encoded sequences and vectorized likelihoods use torch tensors.
"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as tn

import dmx.torch_utils.vector as vec
from dmx.arithmetic import inf
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchDevice,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)


class ExponentialDistribution(TorchProbabilityDistribution):
    """Represent an exponential distribution with mean ``beta``.

    This is the torch counterpart of ``stats.ExponentialDistribution``.  Its
    numerical parameter is kept as a Python float; ``to`` selects the device
    used by scalar-to-tensor and encoded-sequence work without moving a stored
    parameter tensor.

    Attributes:
        beta (float): Scale of exponential.
        log_beta (float): Log of scale.

    """

    def __init__(self, beta: float, device: Optional[TorchDevice] = None):
        """Initialize the distribution.

        Args:
            beta (float): Scale of Exponential random variable.
            device (Optional[TorchDevice]): Device used for torch calculations.

        """
        super().__init__(device)
        self.beta = beta
        self.log_beta = float(np.log(beta))

    def to(self, device: vec.DeviceLike) -> "ExponentialDistribution":
        """Select the device used for subsequent tensor calculations.

        No model parameter is moved because parameters are Python floats.
        """
        self._device = self._resolve_device_arg(device)
        return self

    def __repr__(self) -> str:
        """Return an evaluable representation."""
        return f"ExponentialDistribution(beta={repr(self.beta)})"

    def density(self, x: float) -> float:
        """Density of Exponential distribution at observation x.

        See log_density() for details.

        Args:
            x (float): Real-valued observation of Exponential.

        Returns:
            float: Density of Exponential at x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: float) -> float:
        """Log-density of Exponential distribution at observation x.

        Log-density of Exponential with mean mu and variance sigma2 given by,
            log(f(x;mu, sigma2)) = -log(2*pi*sigma2) - (x-mu)^2/sigma2,
            for real-valued x.

        Args:
            x (float): Real-valued observation of Exponential.

        Returns:
            float: Log-density at observation x.

        """
        if x < 0:
            return -inf

        return float(-x / self.beta - self.log_beta)

    def seq_log_density(self, x: "ExponentialTorchEncodedSequence") -> tn.Tensor:
        """Evaluate log densities for an encoded one-dimensional sequence.

        ``x.data`` and the result have shape ``(n,)`` and retain the encoded
        tensor's device and floating dtype.  Unlike scalar evaluation, this
        method requires the module's encoded-sequence container.
        """
        if not isinstance(x, ExponentialTorchEncodedSequence):
            raise TypeError(
                "Requires ExponentialTorchEncodedSequence for `seq_` function calls."
            )

        rv = x.data * (-1.0 / self.beta)
        rv -= self.log_beta

        return rv

    def sampler(self, seed: Optional[int] = None) -> "ExponentialSampler":
        """Create a NumPy-backed sampler, optionally seeded."""
        return ExponentialSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "ExponentialEstimator":
        """Create an estimator, optionally regularized toward this scale."""
        if pseudo_count is None:
            return ExponentialEstimator()

        return ExponentialEstimator(pseudo_count=pseudo_count, suff_stat=self.beta)

    def dist_to_encoder(self) -> "ExponentialDataEncoder":
        """Returns a ExponentialDataEncoder object for encoding sequences of data."""
        return ExponentialDataEncoder()


class ExponentialSampler(DistributionSampler):
    """Draw NumPy samples from an exponential distribution."""

    def __init__(
        self, dist: "ExponentialDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize a sampler for ``dist``.

        Args:
            dist (ExponentialDistribution): ExponentialDistribution instance to
                sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (ExponentialDistribution): ExponentialDistribution instance to
                sample from.
            tng (tn.Generator): RandomState with seed set to seed if passed in args.

        """
        self.rng = (
            np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        )
        self.beta = dist.beta

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from ExponentialSampler object.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            Numpy array of length 'size' from exponential distribution with
            scale beta if size not None. Else a single sample is returned as
            float.


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

    def __init__(
        self, keys: Optional[str] = None, device: Optional[TorchDevice] = None
    ) -> None:
        """Initialize the exponential sufficient-statistic accumulator.

        Args:
            keys (Optional[str]): Aggregate all sufficient statistics with same
                keys values.
            device: Optional[device]: Sets device for GPU calculations

        """
        super().__init__(device)
        self.sum = 0.0
        self.count = 0.0
        self.key = keys

    def seq_initialize(
        self,
        x: "ExponentialTorchEncodedSequence",
        weights: tn.Tensor,
        tng: Optional[tn.Generator],
    ) -> None:
        """Initialize from encoded values and tensor weights; ``tng`` is unused."""
        self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: "ExponentialTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: Optional[ExponentialDistribution],
    ) -> None:
        """Accumulate encoded values and weights of matching shape ``(n,)``.

        Tensor reductions are converted to Python floats, so accumulator state
        is device- and dtype-independent.
        """
        self.sum += float(tn.dot(x.data, weights))
        self.count += float(tn.sum(weights))

    def combine(self, suff_stat: Tuple[float, float]) -> "ExponentialAccumulator":
        """Merge a ``(sum, count)`` sufficient statistic."""
        self.sum += suff_stat[0]
        self.count += suff_stat[1]

        return self

    def value(self, _device: Optional[str] = None) -> Tuple[float, float]:
        """Return the ``(sum, count)`` sufficient statistic."""
        return self.sum, self.count

    def from_value(self, x: Tuple[float, float]) -> "ExponentialAccumulator":
        """Replace state from a ``(sum, count)`` sufficient statistic."""
        self.sum, self.count = x

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator's keyed statistic into ``stats_dict``."""
        if self.key is not None:
            if self.key in stats_dict:
                self.sum, self.count = stats_dict[self.key]
            else:
                stats_dict[self.key] = (self.sum, self.count)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's state from its keyed statistic."""
        if self.key is not None:
            if self.key in stats_dict:
                self.sum, self.count = stats_dict[self.key]

    def acc_to_encoder(self) -> "ExponentialDataEncoder":
        """Return the encoder accepted by this accumulator."""
        return ExponentialDataEncoder()


class ExponentialAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create exponential accumulators with a shared optional key."""

    def __init__(self, keys: Optional[str] = None):
        """Initialize the factory."""
        self.keys = keys

    def make(self, device: Optional[TorchDevice] = None) -> "ExponentialAccumulator":
        """Create an accumulator associated with ``device``."""
        return ExponentialAccumulator(
            keys=self.keys, device=device if device is not None else None
        )


class ExponentialEstimator(TorchParameterEstimator):
    """Estimate exponential scale from ``(sum, count)`` statistics."""

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[float] = None,
        keys: Optional[str] = None,
    ):
        """Initialize optional pseudo-count regularization."""
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

    def accumulator_factory(self) -> "ExponentialAccumulatorFactory":
        """Return a factory for compatible accumulators."""
        return ExponentialAccumulatorFactory(keys=self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[float, float],
        device: Optional[TorchDevice] = None,
    ) -> "ExponentialDistribution":
        """Estimate ExponentialDistribution from suff_stat arg.

        Estimate ExponentialDistribution from sufficient statistic tuple suff_stat,
        storing the weighted observation sum followed by the weighted count. If
        pseudo_count is set, this is used to re-weight the member value
        "suff_stat", which is the scale of ExponentialEstimator object.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with
                ParameterEstimator.
            suff_stat (Tuple[float, float]): Tuple of (sum, count). Both are
                positive real-valued floats.
            device (Optional[TorchDevice]): Set for estimating model on GPU device

        Returns:
            ExponentialDistribution object.

        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[0] + self.suff_stat * self.pseudo_count) / (
                suff_stat[1] + self.pseudo_count
            )
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[0] + self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        else:
            if suff_stat[1] > 0:
                p = suff_stat[0] / suff_stat[1]
            else:
                p = 1.0

        return ExponentialDistribution(beta=p, device=device)


class ExponentialDataEncoder(TorchSequenceEncoder):
    """Encode positive exponential observations as a floating tensor.

    Input is a list or NumPy array of shape ``(n,)``.  ``vec.tensor`` chooses
    its normal floating dtype and places the tensor on ``device``; callers
    needing a particular dtype should provide data that follows that utility's
    conversion convention.  This differs from ``stats`` only in returning a
    torch-backed encoded container.
    """

    def __str__(self) -> str:
        """Return the encoder name."""
        return "ExponentialDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is an exponential encoder."""
        return isinstance(other, ExponentialDataEncoder)

    def seq_encode(
        self, x: Union[List[float], np.ndarray], device: Optional[TorchDevice] = None
    ) -> "ExponentialTorchEncodedSequence":
        """Validate and encode positive observations on ``device``.

        The returned data tensor has shape ``(n,)``.  CUDA, MPS, and CPU are
        supported to the extent supported by PyTorch and ``vec.tensor``.
        """
        rv = vec.tensor(x, device=device)

        if tn.any(rv <= 0) or tn.any(tn.isnan(rv)):
            raise ValueError("Exponential requires x > 0.")

        return ExponentialTorchEncodedSequence(data=rv, device=device)


class ExponentialTorchEncodedSequence(TorchEncodedSequence):
    """Store a floating encoded exponential sequence of shape ``(n,)``."""

    data: tn.Tensor

    def __init__(self, data: tn.Tensor, device: Optional[TorchDevice] = None) -> None:
        """Initialize from a tensor, optionally moving it to ``device``."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation including the stored device."""
        return f"ExponentialTorchEncodedSequence(device={repr(self.device)})"
