"""Abstract classes for PyTorch-based probability distributions.

This module provides the torch-based equivalents of the
`dmx.stats.pdist` classes, optimized for GPU computation and PyTorch
tensor operations.

Classes:
    TorchProbabilityDistribution: Abstract base class for torch probability
        distributions.
    DistributionSampler: Abstract sampler for probability distributions.
    ConditionalSampler: Abstract sampler for conditional distributions.
    TorchStatisticAccumulator: Abstract accumulator for sufficient statistics.
    TorchStatisticAccumulatorFactory: Factory for creating statistic accumulators.
    TorchParameterEstimator: Abstract estimator for distribution parameters.
    TorchSequenceEncoder: Abstract encoder for data sequences.
    TorchEncodedSequence: Container for encoded sequence data.

"""

import math
from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

import torch as tn
from torch import device as TorchDevice

SS = TypeVar("SS")


class TorchProbabilityDistribution:
    """Abstract base class for PyTorch-based probability distributions.

    This class provides the interface for torch-based probability distributions
    that support GPU computation and PyTorch tensor operations.

    Attributes:
        _device: The torch device (CPU or CUDA) for computations.

    """

    def __init__(self, device: Optional[tn.device] = None) -> None:
        """Initialize the distribution with a specified device.

        Args:
            device: PyTorch device for computations. Defaults to CPU if None.

        """
        self._device = tn.device("cpu") if device is None else device

    def __repr__(self) -> str:
        """Return string representation of the distribution."""
        return self.__str__()

    def model_device(self) -> TorchDevice:
        """Return the device used for model computations.

        Returns:
            The PyTorch device object.

        """
        return self._device

    @abstractmethod
    def to(self, device: TorchDevice) -> "TorchProbabilityDistribution":
        """Move the distribution to a specified device.

        Args:
            device: Target PyTorch device.

        Returns:
            Distribution instance on the target device.

        """
        raise NotImplementedError

    @abstractmethod
    def density(self, x: Any) -> float:
        """Compute the probability density at x.

        Args:
            x: Input value to evaluate density.

        Returns:
            Probability density value.

        """
        return math.exp(self.log_density(x))

    @abstractmethod
    def log_density(self, x: Any) -> float:
        """Compute the log probability density at x.

        Args:
            x: Input value to evaluate log density.

        Returns:
            Log probability density value.

        """
        raise NotImplementedError

    @abstractmethod
    def sampler(self, seed: Optional[int] = None) -> "DistributionSampler":
        """Create a sampler for this distribution.

        Args:
            seed: Random seed for sampling. Defaults to None.

        Returns:
            A DistributionSampler instance.

        """
        raise NotImplementedError

    @abstractmethod
    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "TorchParameterEstimator":
        """Create a parameter estimator for this distribution.

        Args:
            pseudo_count: Regularization parameter. Defaults to None.

        Returns:
            A TorchParameterEstimator instance.

        """
        raise NotImplementedError

    @abstractmethod
    def seq_log_density(self, x: Any) -> tn.Tensor:
        """Compute log densities for a sequence of values.

        Args:
            x: Sequence of input values.

        Returns:
            Tensor of log density values.

        """
        return tn.asarray([self.log_density(u) for u in x], dtype=tn.float64)

    @abstractmethod
    def dist_to_encoder(self) -> "TorchSequenceEncoder":
        """Create a sequence encoder for this distribution.

        Returns:
            A TorchSequenceEncoder instance.

        """
        raise NotImplementedError


class DistributionSampler:
    """Abstract base class for distribution samplers.

    Samplers generate random samples from probability distributions.

    """

    @abstractmethod
    def sample(self, size: Optional[int] = None) -> Any:
        """Generate random samples from the distribution.

        Args:
            size: Number of samples to generate. Defaults to None (single sample).

        Returns:
            Generated samples.

        """
        raise NotImplementedError


class ConditionalSampler:
    """Abstract base class for conditional distribution samplers.

    Samplers that generate samples conditioned on input values.

    """

    @abstractmethod
    def sample_given(self, x: Any) -> Any:
        """Generate a sample conditioned on x.

        Args:
            x: Conditioning value.

        Returns:
            Generated sample conditioned on x.

        """
        raise NotImplementedError


class TorchStatisticAccumulator(Generic[SS]):
    """Abstract base class for PyTorch-based sufficient statistic accumulators.

    Accumulators maintain and update sufficient statistics for parameter estimation
    using PyTorch tensors for GPU-accelerated computation.

    Type Parameters:
        SS: Type of the sufficient statistics.

    Attributes:
        _device: The torch device for computations.

    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize the accumulator with a specified device.

        Args:
            device: Device string (e.g., 'cpu', 'cuda'). Defaults to CPU if None.

        """
        self._device = TorchDevice("cpu") if device is None else device

    @abstractmethod
    def seq_update(self, x: Any, weights: tn.Tensor, estimate: Any) -> None:
        """Update sufficient statistics with a sequence of observations.

        Args:
            x: Sequence of observations.
            weights: Tensor of weights for each observation.
            estimate: Current parameter estimate.

        """
        raise NotImplementedError

    @abstractmethod
    def seq_initialize(self, x: Any, weights: tn.Tensor, tng: tn.Generator) -> None:
        """Initialize sufficient statistics with a sequence of observations.

        Args:
            x: Sequence of observations.
            weights: Tensor of weights for each observation.
            tng: PyTorch random number generator.

        """
        raise NotImplementedError

    @abstractmethod
    def combine(self, suff_stat: SS) -> "TorchStatisticAccumulator":
        """Combine this accumulator with another's sufficient statistics.

        Args:
            suff_stat: Sufficient statistics to combine with.

        Returns:
            Updated accumulator with combined statistics.

        """
        raise NotImplementedError

    @abstractmethod
    def value(self) -> SS:
        """Return the current sufficient statistics value.

        Returns:
            Current sufficient statistics.

        """
        raise NotImplementedError

    @abstractmethod
    def from_value(self, x: SS) -> "TorchStatisticAccumulator":
        """Create an accumulator from sufficient statistics.

        Args:
            x: Sufficient statistics value.

        Returns:
            Accumulator initialized with the given statistics.

        """
        raise NotImplementedError

    @abstractmethod
    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge statistics from a dictionary by key.

        Args:
            stats_dict: Dictionary of statistics to merge.

        """
        raise NotImplementedError

    @abstractmethod
    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics with values from a dictionary by key.

        Args:
            stats_dict: Dictionary of statistics to use as replacements.

        """
        raise NotImplementedError

    @abstractmethod
    def acc_to_encoder(self) -> "TorchSequenceEncoder":
        """Create a sequence encoder from this accumulator.

        Returns:
            A TorchSequenceEncoder instance.

        """
        raise NotImplementedError


class TorchStatisticAccumulatorFactory:
    """Abstract factory for creating TorchStatisticAccumulator instances.

    Factories provide a standard interface for creating accumulator instances
    on specific devices.

    """

    @abstractmethod
    def make(self, device: Optional[TorchDevice] = None) -> "TorchStatisticAccumulator":
        """Create a new TorchStatisticAccumulator instance.

        Args:
            device: Target PyTorch device. Defaults to CPU if None.

        Returns:
            A new TorchStatisticAccumulator instance.

        """
        raise NotImplementedError


class TorchParameterEstimator(Generic[SS]):
    """Abstract base class for PyTorch-based parameter estimators.

    Estimators compute distribution parameters from sufficient statistics
    using PyTorch for GPU-accelerated computation.

    Type Parameters:
        SS: Type of the sufficient statistics.

    """

    def __repr__(self) -> str:
        """Return string representation of the estimator."""
        return self.__repr__()

    @abstractmethod
    def estimate(
        self, nobs: Optional[float], suff_stat: SS, device: Optional[TorchDevice] = None
    ) -> "TorchProbabilityDistribution":
        """Estimate distribution parameters from sufficient statistics.

        Args:
            nobs: Number of observations.
            suff_stat: Sufficient statistics.
            device: Target PyTorch device. Defaults to None.

        Returns:
            Estimated probability distribution.

        """
        raise NotImplementedError

    @abstractmethod
    def accumulator_factory(self) -> "TorchStatisticAccumulatorFactory":
        """Return the factory for creating compatible accumulators.

        Returns:
            A TorchStatisticAccumulatorFactory instance.

        """
        raise NotImplementedError


class TorchSequenceEncoder:
    """Abstract base class for PyTorch-based sequence encoders.

    Encoders transform raw data sequences into encoded representations
    suitable for PyTorch-based processing.

    """

    def __str__(self) -> str:
        """Return string representation of the encoder."""
        return self.__str__()

    @abstractmethod
    def seq_encode(
        self, x: Any, device: Optional[TorchDevice] = None
    ) -> "TorchEncodedSequence":
        """Encode a data sequence.

        Args:
            x: Input data sequence.
            device: Target PyTorch device. Defaults to None.

        Returns:
            Encoded sequence.

        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check equality with another encoder.

        Args:
            other: Object to compare with.

        Returns:
            True if encoders are equal, False otherwise.

        """
        raise NotImplementedError


class TorchEncodedSequence:
    """Container for encoded data sequences with PyTorch support.

    Stores encoded data and tracks the device for PyTorch computation.

    Attributes:
        data: The encoded data.
        device: PyTorch device where data resides.

    """

    @abstractmethod
    def __init__(self, data: Any, device: Optional[TorchDevice] = None):
        """Initialize the encoded sequence.

        Args:
            data: Encoded data.
            device: PyTorch device. Defaults to CPU if None.

        """
        self.data = data
        self.device = tn.device("cpu") if device is None else device

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of the encoded sequence."""
        raise NotImplementedError
