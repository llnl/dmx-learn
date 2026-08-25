"""Provide torch-backed diagonal Gaussian models and estimation utilities.

Defines the DiagonalGaussianDistribution, DiagonalGaussianSampler,
DiagonalGaussianAccumulatorFactory, DiagonalGaussianAccumulator,
DiagonalGaussianEstimator, and the DiagonalGaussianDataEncoder classes for use
with pysparkplug.

The log-density of an `n`-dimensional diagonal-Gaussian observation
`x = (x_1,x_2,...,x_n)` with mean `mu=(m_1,m_2,..,m_n)` and diagonal
covariance matrix `diag(s2_1, s2_2,...,s2_n)` is

    log(p_mat(x)) = -0.5*sum_{i=1}^{n} (x_i-m_i)^2 / s2_i
        - 0.5*log(s2_i) - (n/2)*log(pi).

Scalar methods accept one vector of shape ``(D,)`` and return Python floats.
Sequence methods consume a ``(N, D)`` floating-point tensor created by
``DiagonalGaussianDataEncoder`` and return tensors of shape ``(N,)``. Model
parameters are tensors on the requested device. They use the vector-helper
default dtype, normally float64 and float32 on MPS; the encoded tensor dtype
controls the calculation dtype in ``seq_log_density``. Unlike
``dmx.stats.dmvn``, this implementation uses one optional key for all
sufficient statistics and its sampler and accumulator retain NumPy arrays on
the CPU.

"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tn

import dmx.torch_utils.vector as vec
from dmx.arithmetic import exp
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)


class DiagonalGaussianDistribution(TorchProbabilityDistribution):
    """Represent a torch-backed Gaussian with diagonal covariance.

    The support is ``R^D``. ``mu`` and ``covar`` are length-``D`` vectors, and
    each covariance entry is a positive variance, not a standard deviation.

    Attributes:
          dim (int): Dimension of the multivariate Gaussian. Determined by the
              mean length.
         mu (tn.Tensor): Mean tensor of shape ``(D,)``.
         covar (tn.Tensor): Variance tensor of shape ``(D,)``.
         log_c (tn.Tensor): Scalar normalizing constant.
         ca (tn.Tensor): Quadratic log-density coefficients of shape ``(D,)``.
         cb (tn.Tensor): Linear log-density coefficients of shape ``(D,)``.
         cc (tn.Tensor): Scalar constant log-density term.
         key (Optional[str]): Key for merging sufficient statistics.

    """

    def __init__(
        self,
        mu: Union[Sequence[float], np.ndarray],
        covar: Union[Sequence[float], np.ndarray],
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Create a DiagonalGaussianDistribution object.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Mean of Gaussian distribution.
            covar (Union[Sequence[float], np.ndarray]): Variance of each component.
            keys (Optional[str]): Set keys for object instance.
            device (Optional[tn.device]): Set device for tensor calculations.

        """
        super().__init__(device=device)
        self.dim = len(mu)
        self.mu = vec.tensor(mu, device=self._device)
        self.covar = vec.tensor(covar, device=self._device)
        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + tn.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c
        self.key = keys

    def to(self, device: vec.DeviceLike) -> "DiagonalGaussianDistribution":
        """Move parameter tensors to ``device`` in place and return ``self``."""
        target_device = self._resolve_device_arg(device)
        self.mu = self.mu.to(target_device)
        self.covar = self.covar.to(target_device)

        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + tn.log(self.covar).sum())
        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c
        self._device = target_device
        return self

    def __repr__(self) -> str:
        """Return a constructor-like CPU representation."""
        s1 = repr(list(self.mu.data.cpu().numpy().flatten()))
        s2 = repr(list(self.covar.data.cpu().numpy().flatten()))

        return f"DiagonalGaussianDistribution({s1}, {s2})"

    def density(self, x: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate the density of one CPU observation of shape ``(D,)``."""
        return float(exp(self.log_density(x)))

    def log_density(self, x: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate the log density using NumPy on the CPU."""
        xx = np.asarray(x)
        rv = np.dot(xx * xx, self.ca.cpu().detach().numpy())
        rv += np.dot(xx, self.cb.cpu().detach().numpy())
        rv += float(self.cc)
        return float(rv)

    def seq_log_density(self, x: "DiagonalGaussianTorchEncodedSequence") -> tn.Tensor:
        """Evaluate ``N`` encoded observations and return a tensor of shape ``(N,)``."""
        if not isinstance(x, DiagonalGaussianTorchEncodedSequence):
            raise TypeError(
                "Requires DiagonalGaussianTorchEncodedSequence for `seq_` "
                "function calls."
            )

        ca = self.ca.to(device=x.data.device, dtype=x.data.dtype)
        cb = self.cb.to(device=x.data.device, dtype=x.data.dtype)
        cc = self.cc.to(device=x.data.device, dtype=x.data.dtype)

        rv = tn.matmul(x.data * x.data, ca)
        rv += tn.matmul(x.data, cb)
        rv += cc
        return rv

    def sampler(self, seed: Optional[int] = None) -> "DiagonalGaussianSampler":
        """Create a CPU NumPy sampler, optionally initialized with ``seed``."""
        return DiagonalGaussianSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "DiagonalGaussianEstimator":
        """Create an estimator with an optional shared mean/variance pseudo-count."""
        if pseudo_count is None:
            return DiagonalGaussianEstimator(keys=self.key)

        return DiagonalGaussianEstimator(
            pseudo_count=(pseudo_count, pseudo_count), keys=self.key
        )

    def dist_to_encoder(self) -> "DiagonalGaussianDataEncoder":
        """Create an encoder fixed to this distribution's dimension."""
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianSampler(DistributionSampler):
    """DiagonalGaussianSampler object for sampling from DiagonalGaussian instance.

    Attributes:
        dist (DiagonalGaussianDistribution): Object instance to sample from.
        seed (Optional[int]): Seed for random number generator.

    """

    def __init__(
        self, dist: DiagonalGaussianDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize a diagonal Gaussian sampler.

        Args:
            dist (DiagonalGaussianDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = np.random.RandomState(seed)
        self.mu = dist.mu.data.cpu().numpy()
        self.covar = dist.covar.data.cpu().numpy()
        self.dim = dist.dim

    def sample(
        self, size: Optional[int] = None
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """Draw one vector or a list of ``size`` vectors, each of shape ``(D,)``."""
        if size is None:
            rv = self.rng.randn(self.dim)
            rv *= np.sqrt(self.covar)
            rv += self.mu
            return rv

        return [np.asarray(self.sample()) for _ in range(size)]


class DiagonalGaussianAccumulator(TorchStatisticAccumulator):
    """Aggregate sufficient statistics from iid observations.

    Attributes:
         dim (Optional[int]): Optional dimension of Gaussian.
         count (float): Used for tracking weighted observations counts.
         sum (np.ndarray): Sum of observation vectors.
         sum2 (np.ndarray): Sum of squared observation vectors.
          key (Optional[str]): If set, merge sufficient statistics with
              objects containing matching keys.

    """

    def __init__(
        self,
        dim: Optional[int] = None,
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize an empty diagonal Gaussian accumulator.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            keys (Optional[str]): Set keys for merging sufficient statistics.
            device (Optional[tn.device]): Device metadata for encoded operations.

        """
        super().__init__(device)
        self.dim = dim
        self.count = 0.0
        self.sum: Optional[np.ndarray] = (
            np.zeros(dim, dtype=np.float64) if dim is not None else None
        )
        self.sum2: Optional[np.ndarray] = (
            np.zeros(dim, dtype=np.float64) if dim is not None else None
        )
        self.key = keys

    def seq_update(
        self,
        x: "DiagonalGaussianTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: Optional[DiagonalGaussianDistribution],
    ) -> None:
        """Add ``N`` encoded vectors with weights of shape ``(N,)``."""
        if self.dim is None:
            self.dim = len(x.data[0])
            self.sum = np.zeros(self.dim, dtype=np.float64)
            self.sum2 = np.zeros(self.dim, dtype=np.float64)

        assert self.sum is not None
        assert self.sum2 is not None
        x_weight = tn.multiply(x.data.T, weights)
        self.count += float(weights.sum())
        self.sum += tn.sum(x_weight, dim=1).data.cpu().numpy()
        x_weight *= x.data.T
        self.sum2 += tn.sum(x_weight, dim=1).data.cpu().numpy()

    def seq_initialize(
        self,
        x: "DiagonalGaussianTorchEncodedSequence",
        weights: tn.Tensor,
        tng: Optional[tn.Generator],
    ) -> None:
        """Add an encoded batch during initialization; ``tng`` is unused."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[np.ndarray, np.ndarray, float]
    ) -> "DiagonalGaussianAccumulator":
        """Merge ``(sum_x, sum_x2, count)`` sufficient statistics."""
        if suff_stat[0] is not None and self.sum is not None:
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            self.sum = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return CPU sufficient statistics ``(sum_x, sum_x2, count)``."""
        assert self.sum is not None
        assert self.sum2 is not None
        return self.sum, self.sum2, self.count

    def from_value(
        self, x: Tuple[np.ndarray, np.ndarray, float]
    ) -> "DiagonalGaussianAccumulator":
        """Replace the accumulator from a sufficient-statistic tuple."""
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under its key."""
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key].value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from the matching entry in ``stats_dict``."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key])

    def acc_to_encoder(self) -> "DiagonalGaussianDataEncoder":
        """Create an encoder using the accumulated dimension."""
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create diagonal Gaussian accumulators with a shared configuration."""

    def __init__(self, dim: Optional[int] = None, keys: Optional[str] = None) -> None:
        """Create DiagonalGaussianAccumulator objects.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            keys (Optional[str]): Set keys for merging sufficient statistics.

        Attributes:
             dim (Optional[int]): Optional dimension of Gaussian.
             key (Optional[str]): If set, merge sufficient statistics with
                 objects containing matching keys.

        """
        self.dim = dim
        self.key = keys

    def make(self, device: Optional[tn.device] = None) -> "DiagonalGaussianAccumulator":
        """Create an accumulator associated with ``device``."""
        return DiagonalGaussianAccumulator(dim=self.dim, keys=self.key, device=device)


class DiagonalGaussianEstimator(TorchParameterEstimator):
    """Estimate diagonal Gaussian distributions from aggregated statistics.

    Attributes:
        dim (int): Dimension of Gaussian, either set of determined from suff_stat arg.
        prior_mu (Optional[np.ndarray]): Set from suff_stat[0].
        prior_covar ((Optional[np.ndarray]): Set from suff_stat[1].
        pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight the
            sum of observations and sum of squared observations in estimation.
        keys (Optional[str]): Key for merging sufficient statistics.

    """

    def __init__(
        self,
        dim: Optional[int] = None,
        pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
        suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a diagonal Gaussian estimator.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight
                the sum of observations and sum of squared observations in
                estimation.
            suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray]]): Sum
                of observations and sum of squared observations, both having
                the same dimension.
            keys (Optional[str]): Set keys for merging sufficient statistics.

        """
        dim_loc = (
            dim
            if dim is not None
            else (
                (None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1]))))
                if suff_stat[0] is None
                else len(suff_stat[0])
            )
        )

        self.dim = dim_loc
        self.pseudo_count = pseudo_count
        if suff_stat[0] is not None:
            if dim_loc is None:
                raise ValueError("dim must be set when suff_stat is provided.")
            self.prior_mu: Optional[np.ndarray] = np.reshape(suff_stat[0], dim_loc)
        else:
            self.prior_mu = None
        if suff_stat[1] is not None:
            if dim_loc is None:
                raise ValueError("dim must be set when suff_stat is provided.")
            self.prior_covar: Optional[np.ndarray] = np.reshape(suff_stat[1], dim_loc)
        else:
            self.prior_covar = None
        self.key = keys

    def accumulator_factory(self) -> "DiagonalGaussianAccumulatorFactory":
        """Create a factory carrying this estimator's dimension and key."""
        return DiagonalGaussianAccumulatorFactory(dim=self.dim, keys=self.key)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, np.ndarray, float],
        device: Optional[tn.device] = None,
    ) -> "DiagonalGaussianDistribution":
        """Estimate parameters from ``(sum_x, sum_x2, count)`` on ``device``."""
        nobs = suff_stat[2]
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (suff_stat[1] + (pc2 * self.prior_covar) - (mu * mu * nobs)) / (
                nobs + pc2
            )
        else:
            covar = (suff_stat[1] / nobs) - (mu * mu)

        return DiagonalGaussianDistribution(mu, covar, device=device)


class DiagonalGaussianDataEncoder(TorchSequenceEncoder):
    """Encode sequences of iid diagonal-Gaussian observations.

    Attributes:
        dim (Optional[int]): Dimension of the Gaussian.

    """

    def __init__(self, dim: Optional[int] = None) -> None:
        """Initialize an encoder for vectors of optional fixed dimension.

        Args:
            dim (Optional[int]): Dimension of the Gaussian.

        """
        self.dim = dim

    def __str__(self) -> str:
        """Return a concise encoder representation."""
        return "DiagonalGaussianDataEncoder(dim=" + str(self.dim) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has the same dimension."""
        if isinstance(other, DiagonalGaussianDataEncoder):
            return self.dim == other.dim

        return False

    def seq_encode(
        self,
        x: Sequence[Union[List[float], np.ndarray]],
        device: Optional[tn.device] = None,
    ) -> "DiagonalGaussianTorchEncodedSequence":
        """Encode ``N`` vectors as a floating tensor of shape ``(N, D)``.

        The tensor is created on ``device`` with the default dtype selected by
        :func:`dmx.torch_utils.vector.tensor`.
        """
        if self.dim is None:
            self.dim = len(x[0])
        dim = self.dim
        xv = np.reshape(np.asarray(x), (-1, dim))
        return DiagonalGaussianTorchEncodedSequence(
            data=vec.tensor(xv, device=device), device=device
        )


class DiagonalGaussianTorchEncodedSequence(TorchEncodedSequence):
    """Store an encoded floating tensor with shape ``(N, D)``."""

    data: tn.Tensor

    def __init__(self, data: tn.Tensor, device: Optional[tn.device] = None) -> None:
        """Initialize the wrapper around ``data`` on its associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"DiagonalGaussianTorchEncodedSequence(device=tn.device({self.device}))"
