import math
import numpy as np
import torch as tn
from abc import abstractmethod
from pysp.utils.arithmetic import *
from typing import TypeVar, Optional, Any, Generic, Dict

SS = TypeVar('SS')
from torch import device as TorchDevice

class TorchProbabilityDistribution(object):

    def __init__(self, device: Optional[tn.device] = None) -> None:
        self._device = tn.device('cpu') if device is None else device

    def __repr__(self) -> str:
        return self.__str__()

    def model_device(self) -> TorchDevice:
        return self._device

    @abstractmethod
    def to(self, device: TorchDevice): ...

    @abstractmethod
    def density(self, x: Any) -> float:
        return math.exp(self.log_density(x))

    @abstractmethod
    def log_density(self, x: Any) -> float: ...

    @abstractmethod
    def sampler(self, seed: Optional[int] = None) -> 'DistributionSampler': ...

    @abstractmethod
    def estimator(self, pseudo_count: Optional[float] = None) -> 'TorchParameterEstimator': ...

    @abstractmethod
    def seq_log_density(self, x: Any) -> tn.Tensor:
        return tn.asarray([self.log_density(u) for u in x], dtype=tn.float64)

    @abstractmethod
    def dist_to_encoder(self) -> 'TorchSequenceEncoder': ...


class DistributionSampler(object):

    @abstractmethod
    def sample(self, size: Optional[int] = None) -> Any: ...


class ConditionalSampler(object):
    @abstractmethod
    def sample_given(self, x): ...


class TorchStatisticAccumulator(Generic[SS]):

    def __init__(self, device: Optional[str] = None):
        self._device = TorchDevice('cpu') if device is None else device

    @abstractmethod
    def seq_update(self, x, weights: tn.Tensor, estimate) -> None: ...

    @abstractmethod
    def seq_initialize(self, x, weights: tn.Tensor, tng: tn.Generator) -> None: ...

    @abstractmethod
    def combine(self, suff_stat: SS) -> 'TorchStatisticAccumulator':
        ...

    @abstractmethod
    def value(self) -> SS:
        ...

    @abstractmethod
    def from_value(self, x: SS) -> 'TorchStatisticAccumulator':
        ...

    @abstractmethod
    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def acc_to_encoder(self) -> 'TorchSequenceEncoder': ...


class TorchStatisticAccumulatorFactory(object):

    @abstractmethod
    def make(self, device: Optional[TorchDevice] = None) -> 'TorchStatisticAccumulator': ...


class TorchParameterEstimator(Generic[SS]):

    def __repr__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def estimate(self, nobs: Optional[float], suff_stat: SS, device: Optional[TorchDevice] = None) -> 'TorchProbabilityDistribution': ...

    @abstractmethod
    def accumulator_factory(self) -> 'TorchStatisticAccumulatorFactory': ...


class TorchSequenceEncoder:

    def __str__(self) -> str:
        return self.__str__()

    @abstractmethod
    def seq_encode(self, x: Any, device: Optional[TorchDevice] = None) -> 'TorchEncodedSequence':
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...


class TorchEncodedSequence:

    @abstractmethod
    def __init__(self, data: Any, device: Optional[TorchDevice] = None):
        self.data = data
        self.device = tn.device('cpu') if device is None else device

    @abstractmethod
    def __str__(self) -> str:
        ...





