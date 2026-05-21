# pylint: disable=line-too-long
"""Create, estimate, and sample from a null distribution.

Defines the NullDistribution, NullSampler, NullAccumulatorFactory, NullAccumulator,
NullEstimator, and the NullDataEncoder classes for use with pysparkplug.

The NullDistribution object and its related classes are space filling objects meant for consistency in type hints.

Notes:
    The density evaluates to 1.0 for any value (Any data type).
    The sampler generates None for any size input.
    Sequence encodings return None for any input.

"""

# pylint: disable=line-too-long,too-many-positional-arguments,duplicate-code
# pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin
# pylint: disable=broad-exception-raised,consider-using-f-string,no-else-return
# pylint: disable=no-else-raise,consider-using-enumerate,consider-using-generator
# pylint: disable=use-dict-literal,super-with-arguments,unnecessary-comprehension
# pylint: disable=simplifiable-if-statement,nested-min-max

from typing import Any, Dict, Optional

import torch as tn

import dmx.torch_utils.vector as vec
from dmx.torch_stats.pdist import *


class NullDistribution(TorchProbabilityDistribution):

    def to(self, device: tn.device) -> None:
        self._device = device

    def __repr__(self) -> str:
        return "NullDistribution()"

    def density(self, x: Optional[Any]) -> float:
        return 1.0

    def log_density(self, x: Optional[Any]) -> float:
        return 0.0

    def seq_log_density(self, x: "NullTorchEncodedSequence") -> tn.Tensor:
        return vec.zeros(1, device=self.model_device())

    def sampler(self, seed: Optional[int] = None) -> "NullSampler":
        return NullSampler(dist=self, seed=seed)

    def estimator(
        self, pseudo_count: Optional[float] = None, _device: Optional[str] = None
    ) -> "NullEstimator":
        if pseudo_count is None:
            return NullEstimator()

        else:
            return NullEstimator(pseudo_count=pseudo_count)

    def dist_to_encoder(self) -> "NullDataEncoder":
        return NullDataEncoder()


class NullSampler(DistributionSampler):

    def __init__(self, dist: "NullDistribution", seed: Optional[int] = None) -> None:
        self.seed = seed
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> None:
        return None


class NullAccumulator(TorchStatisticAccumulator):

    def __init__(
        self, keys: Optional[str] = None, device: Optional[tn.device] = None
    ) -> None:
        super(NullAccumulator, self).__init__(device)
        self.key = keys

    def seq_update(
        self,
        x: "NullTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: Optional["NullDistribution"],
    ) -> None:
        pass

    def seq_initialize(
        self,
        x: "NullTorchEncodedSequence",
        weights: tn.Tensor,
        tng: Optional["tn.Generator"],
    ) -> None:
        pass

    def combine(self, suff_stat: Optional[Any]) -> "NullAccumulator":
        return self

    def value(self) -> None:
        return None

    def from_value(self, x: Optional[Any]) -> "NullAccumulator":
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                pass
            else:
                stats_dict[self.key] = None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def acc_to_encoder(self) -> "NullDataEncoder":
        return NullDataEncoder()


class NullAccumulatorFactory(TorchStatisticAccumulatorFactory):

    def __init__(self, keys: Optional[str] = None) -> None:
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> "NullAccumulator":
        return NullAccumulator(keys=self.keys, device=device)


class NullEstimator(TorchParameterEstimator):

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Any] = None,
        keys: Optional[str] = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys

    def accumulator_factory(self) -> "NullAccumulatorFactory":
        return NullAccumulatorFactory(self.keys)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Optional[Any] = None,
        device: Optional[tn.device] = None,
    ) -> "NullDistribution":
        return NullDistribution(device=device)


class NullDataEncoder(TorchSequenceEncoder):

    def __str__(self) -> str:
        return "NullDataEncoder"

    def __eq__(self, other) -> bool:
        return isinstance(other, NullDataEncoder)

    def seq_encode(
        self, x: Any, device: Optional[tn.device] = None
    ) -> "NullTorchEncodedSequence":
        return NullTorchEncodedSequence(data=None, device=device)


class NullTorchEncodedSequence(TorchEncodedSequence):

    def __init__(self, data: Optional[Any], device: Optional[tn.device]):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f"NullTorchEncodedSequence(device={repr(self.device)})"
