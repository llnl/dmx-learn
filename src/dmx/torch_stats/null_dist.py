"""Provide torch-backed neutral placeholders for absent model components.

Defines the NullDistribution, NullSampler, NullAccumulatorFactory,
NullAccumulator, NullEstimator, and the NullDataEncoder classes for use
with pysparkplug.

The scalar density is one and the log density is zero for any value. Encoding
discards its input, and encoded scoring returns a fixed one-element zero tensor
rather than preserving batch size. ``to`` only updates device metadata; the
placeholder owns no parameter tensors. Sampling always returns ``None`` and
the accumulator's invariant sufficient statistic is ``None``. These contracts
match ``dmx.stats.null_dist`` with torch device-aware wrappers.

"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, Optional

import torch as tn

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


class NullDistribution(TorchProbabilityDistribution):
    """Provide a neutral likelihood factor for an absent distribution."""

    def to(self, device: vec.DeviceLike) -> "NullDistribution":
        """Update device metadata in place and return ``self``."""
        self._device = self._resolve_device_arg(device)
        return self

    def __repr__(self) -> str:
        """Return an evaluable representation of the placeholder."""
        return "NullDistribution()"

    def density(self, x: Optional[Any]) -> float:
        """Return one for any scalar observation."""
        return 1.0

    def log_density(self, x: Optional[Any]) -> float:
        """Return zero for any scalar observation."""
        return 0.0

    def seq_log_density(self, x: "NullTorchEncodedSequence") -> tn.Tensor:
        """Return a one-element zero tensor on the model device."""
        return vec.zeros(1, device=self.model_device())

    def sampler(self, seed: Optional[int] = None) -> "NullSampler":
        """Create a sampler that always returns ``None``."""
        return NullSampler(dist=self, seed=seed)

    def estimator(
        self, pseudo_count: Optional[float] = None, _device: Optional[str] = None
    ) -> "NullEstimator":
        """Create an estimator whose result is another null distribution."""
        if pseudo_count is None:
            return NullEstimator()

        return NullEstimator(pseudo_count=pseudo_count)

    def dist_to_encoder(self) -> "NullDataEncoder":
        """Create an encoder that discards its input."""
        return NullDataEncoder()


class NullSampler(DistributionSampler):
    """Implement the sampler protocol by always returning ``None``."""

    def __init__(self, dist: "NullDistribution", seed: Optional[int] = None) -> None:
        """Store protocol-compatible distribution and seed values."""
        self.seed = seed
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> None:
        """Return ``None`` regardless of ``size``."""
        return None


class NullAccumulator(TorchStatisticAccumulator):
    """Implement accumulation with an invariant ``None`` statistic."""

    def __init__(
        self, keys: Optional[str] = None, device: Optional[tn.device] = None
    ) -> None:
        """Initialize a stateless accumulator with optional key and device."""
        super().__init__(device)
        self.key = keys

    def seq_update(
        self,
        x: "NullTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: Optional["NullDistribution"],
    ) -> None:
        """Ignore an encoded sequence, weights, and estimate."""
        pass

    def seq_initialize(
        self,
        x: "NullTorchEncodedSequence",
        weights: tn.Tensor,
        tng: Optional["tn.Generator"],
    ) -> None:
        """Ignore an encoded initialization sequence and generator."""
        pass

    def combine(self, suff_stat: Optional[Any]) -> "NullAccumulator":
        """Ignore another statistic and return this accumulator."""
        return self

    def value(self) -> None:
        """Return the invariant ``None`` sufficient statistic."""
        return None

    def from_value(self, x: Optional[Any]) -> "NullAccumulator":
        """Ignore a supplied value and return this accumulator."""
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Register ``None`` under the configured key if absent."""
        if self.key is not None:
            if self.key in stats_dict:
                pass
            else:
                stats_dict[self.key] = None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Leave this stateless accumulator unchanged."""
        pass

    def acc_to_encoder(self) -> "NullDataEncoder":
        """Create an encoder that discards every observation."""
        return NullDataEncoder()


class NullAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create stateless null accumulators."""

    def __init__(self, keys: Optional[str] = None) -> None:
        """Initialize the factory with an optional merge key."""
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> "NullAccumulator":
        """Create a null accumulator associated with ``device``."""
        return NullAccumulator(keys=self.keys, device=device)


class NullEstimator(TorchParameterEstimator):
    """Implement estimation by returning a null distribution."""

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Any] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Store protocol-compatible prior and key values."""
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

    def accumulator_factory(self) -> "NullAccumulatorFactory":
        """Create a null accumulator factory retaining the merge key."""
        return NullAccumulatorFactory(self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Optional[Any] = None,
        device: Optional[tn.device] = None,
    ) -> "NullDistribution":
        """Return a null distribution on ``device``."""
        return NullDistribution(device=device)


class NullDataEncoder(TorchSequenceEncoder):
    """Discard observations while retaining encoded device metadata."""

    def __str__(self) -> str:
        """Return the encoder name."""
        return "NullDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a null encoder."""
        return isinstance(other, NullDataEncoder)

    def seq_encode(
        self, x: Any, device: Optional[tn.device] = None
    ) -> "NullTorchEncodedSequence":
        """Discard ``x`` and return an encoding containing ``None``."""
        return NullTorchEncodedSequence(data=None, device=device)


class NullTorchEncodedSequence(TorchEncodedSequence):
    """Store ``None`` plus the device associated with discarded input."""

    data: Optional[Any]

    def __init__(self, data: Optional[Any], device: Optional[tn.device]) -> None:
        """Initialize the null encoding and associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"NullTorchEncodedSequence(device={repr(self.device)})"
