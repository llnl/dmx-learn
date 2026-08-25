"""Provide torch-backed multivariate Gaussian models and estimation utilities.

Defines the MultivariateGaussianDistribution, MultivariateGaussianSampler,
MultivariateGaussianAccumulatorFactory, MultivariateGaussianAccumulator,
MultivariateGaussianEstimator, and the MultivariateGaussianDataEncoder classes
for use with pysparkplug.

Data type: np.ndarray[float]

x = (x_1,x_2,..,x_n) ~ MVN(mu, covar), where mu is a length-n numpy array and
covar is an n by n positive definite covariance matrix.

The log-density is given by
    log(p(x)) = -0.5*k*log(2*pi) - 0.5*det(covar) - 0.5*(x-mu)' covar^{-1} (x-mu).

Scalar methods accept one NumPy vector of shape ``(D,)`` and return Python
floats. Encoded methods accept a floating tensor of shape ``(N, D)`` and
return a tensor of shape ``(N,)``. Parameters move in place with ``to`` and use
the vector-helper floating dtype, normally float64 and float32 on MPS. Unlike
``dmx.stats.mvn``, factorization uses torch and MPS likelihood solves run on
CPU before results return to the model device and encoded dtype; sampling and
accumulated sufficient statistics remain CPU NumPy data.

"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tn

import dmx.torch_utils.vector as vec
from dmx.arithmetic import exp, pi
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)


class MultivariateGaussianDistribution(TorchProbabilityDistribution):
    """Represent a torch-backed multivariate Gaussian with full covariance.

    The support is ``R^D``. ``mu`` has shape ``(D,)`` and the symmetric
    positive-definite covariance has shape ``(D, D)``.

    Attributes:
        dim (int): Dimension ``D`` of the multivariate Gaussian.
        mu (tn.Tensor): Mean tensor of shape ``(D,)``.
        covar (tn.Tensor): Covariance tensor of shape ``(D, D)``.
        chol (tn.Tensor): Lower Cholesky factor with shape ``(D, D)``.
        keys (Optional[str]): Set keys for distribution.
        chol_const (tn.Tensor): Scalar log-normalization constant.

    """

    def __init__(
        self,
        mu: Union[List[float], np.ndarray],
        covar: Union[List[List[float]], np.ndarray],
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize a multivariate Gaussian distribution.

        Args:
            mu (Union[List[float], np.ndarray]): N-dimensional mean.
            covar (Union[List[List[float]], np.ndarray]): Covariance matrix;
                should be N by N and positive definite.
            keys (Optional[str]): Set keys for distribution.
            device (Optional[tn.device]): Device for parameter tensors.

        """
        super().__init__(device)
        self.dim = len(mu)
        self.mu: tn.Tensor = vec.tensor(mu, device=self._device)
        self.covar: tn.Tensor = vec.tensor(covar, device=self._device).reshape(
            (self.dim, self.dim)
        )
        self.chol: tn.Tensor = tn.cholesky(self.covar)
        self.keys = keys
        self.chol_const: tn.Tensor = -0.5 * (
            len(self.mu) * np.log(2.0 * pi) + 2.0 * tn.log(tn.diag(self.chol)).sum()
        )

    def to(self, device: vec.DeviceLike = None) -> "MultivariateGaussianDistribution":
        """Move parameters to ``device`` in place and recompute normalization."""
        target_device = self._resolve_device_arg(device)
        self.mu = self.mu.to(target_device)
        self.covar = self.covar.to(target_device)
        self.chol = self.chol.to(target_device)
        self.chol_const = -0.5 * (
            len(self.mu) * np.log(2.0 * pi) + 2.0 * tn.log(tn.diag(self.chol)).sum()
        )
        self._device = target_device
        return self

    def __repr__(self) -> str:
        """Return a constructor-like representation with CPU parameters."""
        s1 = repr(self.mu.data.cpu().tolist())
        s2 = repr([list(u) for u in self.covar.data.cpu().tolist()])
        s3 = repr(self.keys)

        return f"MultivariateGaussianDistribution({s1}, {s2}, keys={s3})"

    def density(self, x: np.ndarray) -> float:
        """Evaluate the density at x.

        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            float: Density at x.

        """
        return float(exp(self.log_density(x)))

    def log_density(self, x: np.ndarray) -> float:
        """Evaluate the log-density at x.

        Notes:
            log(p(x)) = -0.5*k*log(2*pi) - 0.5*det(covar)
            - 0.5*(x-mu)' covar^{-1} (x-mu).
        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            float: Log-density at x.

        """
        if self.model_device().type == "mps":
            x_cpu = vec.tensor(x, device=tn.device("cpu"))
            mu_cpu = self.mu.detach().cpu()
            chol_cpu = self.chol.detach().cpu()
            diff = mu_cpu - x_cpu
            soln = tn.cholesky_solve(diff[:, None], chol_cpu).T
            rv = self.chol_const.detach().cpu() - 0.5 * ((diff * soln).sum())
            return float(rv)

        diff = self.mu - vec.tensor(x, device=self._device)
        soln = tn.cholesky_solve(diff[:, None], self.chol).T

        rv = self.chol_const - 0.5 * ((diff * soln).sum())
        return float(rv)

    def seq_log_density(self, x: "MultivariateGaussianTorchSequence") -> tn.Tensor:
        """Return log densities for an encoded tensor of shape ``(N, D)``."""
        if not isinstance(x, MultivariateGaussianTorchSequence):
            raise TypeError(
                "Requires MultivariateGaussianTorchSequence for `seq_` function calls."
            )
        if self.model_device().type == "mps":
            x_cpu = x.data.detach().cpu()
            mu_cpu = self.mu.detach().cpu()
            chol_cpu = self.chol.detach().cpu()
            diff = mu_cpu - x_cpu
            soln = tn.cholesky_solve(diff.T, chol_cpu).T
            rv = self.chol_const.detach().cpu() - 0.5 * ((diff * soln).sum(dim=1))
            return rv.to(device=self.model_device(), dtype=x.data.dtype)

        diff = self.mu - x.data
        soln = tn.cholesky_solve(diff.T, self.chol).T
        rv = self.chol_const - 0.5 * ((diff * soln).sum(dim=1))
        return rv

    def sampler(self, seed: Optional[int] = None) -> "MultivariateGaussianSampler":
        """Create a CPU NumPy sampler, optionally initialized with ``seed``."""
        return MultivariateGaussianSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "MultivariateGaussianEstimator":
        """Create an estimator centered on this distribution when regularized."""
        if pseudo_count is None:
            return MultivariateGaussianEstimator()

        pseudo_counts = (pseudo_count, pseudo_count)
        return MultivariateGaussianEstimator(
            pseudo_count=pseudo_counts,
            suff_stat=(
                self.mu.detach().cpu().numpy(),
                self.covar.detach().cpu().numpy(),
            ),
        )

    def dist_to_encoder(self) -> "MultivariateGaussianDataEncoder":
        """Create an encoder fixed to this distribution's dimension."""
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianSampler(DistributionSampler):
    """Draw independent multivariate Gaussian vectors using NumPy."""

    def __init__(
        self, dist: "MultivariateGaussianDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize a sampler by copying parameters to CPU NumPy arrays."""
        self.rng = np.random.RandomState(seed)
        self.mu = dist.mu.data.cpu().numpy()
        self.covar = dist.covar.data.cpu().numpy()

    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Draw one ``(D,)`` vector or an array with leading dimension ``size``."""
        return self.rng.multivariate_normal(mean=self.mu, cov=self.covar, size=size)


class MultivariateGaussianAccumulator(TorchStatisticAccumulator):
    """Accumulate weighted first and uncentered second moments on the CPU."""

    def __init__(
        self,
        dim: Optional[int] = None,
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize an empty accumulator with optional fixed dimension."""
        super().__init__(device=device)
        self.dim = dim
        self.count = 0.0
        self.key = keys

        if dim is not None:
            self.sum: Optional[np.ndarray] = np.zeros(dim, dtype=np.float64)
            self.sum2: Optional[np.ndarray] = np.zeros((dim, dim), dtype=np.float64)
        else:
            self.sum = None
            self.sum2 = None

    def seq_update(
        self,
        x: "MultivariateGaussianTorchSequence",
        weights: tn.Tensor,
        estimate: Optional[MultivariateGaussianDistribution],
    ) -> None:
        """Add ``N`` encoded vectors with weights of shape ``(N,)``."""
        if self.dim is None:
            self.dim = x.data.shape[1]
            self.sum = np.zeros(self.dim, dtype=np.float64)
            self.sum2 = np.zeros((self.dim, self.dim), dtype=np.float64)

        assert self.sum is not None
        assert self.sum2 is not None
        x_weight = tn.multiply(x.data.T, weights)
        self.count += float(weights.sum())
        self.sum += x_weight.sum(dim=1).cpu().detach().numpy()
        self.sum2 += tn.einsum("ji,ik->jk", x_weight, x.data).cpu().detach().numpy()

    def seq_initialize(
        self,
        x: "MultivariateGaussianTorchSequence",
        weights: tn.Tensor,
        tng: Optional[tn.Generator],
    ) -> None:
        """Add an encoded batch during initialization; ``tng`` is unused."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[np.ndarray, np.ndarray, float]
    ) -> "MultivariateGaussianAccumulator":
        """Merge ``(sum_x, sum_xx, count)`` sufficient statistics."""
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
        """Return CPU statistics with shapes ``(D,)`` and ``(D, D)`` plus count."""
        assert self.sum is not None
        assert self.sum2 is not None
        return self.sum, self.sum2, self.count

    def from_value(
        self, x: Tuple[np.ndarray, np.ndarray, float]
    ) -> "MultivariateGaussianAccumulator":
        """Replace the accumulator from a sufficient-statistic tuple."""
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under its key."""
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key])

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from the matching entry in ``stats_dict``."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key])

    def acc_to_encoder(self) -> "MultivariateGaussianDataEncoder":
        """Create an encoder using the accumulated dimension."""
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create multivariate Gaussian accumulators with shared configuration."""

    def __init__(self, dim: Optional[int], keys: Optional[str] = None) -> None:
        """Initialize the factory with an optional dimension and merge key."""
        self.dim = dim
        self.key = keys

    def make(
        self, device: Optional[tn.device] = None
    ) -> "MultivariateGaussianAccumulator":
        """Create an accumulator associated with ``device``."""
        return MultivariateGaussianAccumulator(
            dim=self.dim, keys=self.key, device=device
        )


class MultivariateGaussianEstimator(TorchParameterEstimator):
    """Estimate a multivariate normal distribution from sufficient stats.

    Attributes:
        dim (int): Dimension of multivariate normal.
        pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]):
            Regularize mean and/or covariance.
        prior_mu (Optional[np.ndarray]): Mean from prior data or used to regularize.
        prior_covar (Optional[np.ndarray]): Covariance matrix from prior data or
            used to regularize.
        key (Optional[str]): Keys for merging sufficient statistics.

    """

    def __init__(
        self,
        dim: Optional[int] = None,
        pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (
            None,
            None,
        ),
        suff_stat: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = (
            None,
            None,
        ),
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a multivariate Gaussian estimator.

        Args:
            dim (Optional[int]): Dimension of multivariate normal. Inferred
                from `suff_stat` if None.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]):
                Regularize mean and/or covariance.
            suff_stat (Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]):
                Mean and covariance estimated from previous data or used to
                regularize.
            keys (Optional[str]): Set keys for estimator.

        """
        suff_stat_loc = suff_stat if suff_stat is not None else (None, None)
        dim_loc = (
            dim
            if dim is not None
            else (
                (
                    None
                    if suff_stat_loc[1] is None
                    else int(np.sqrt(np.size(suff_stat_loc[1])))
                )
                if suff_stat_loc[0] is None
                else len(suff_stat_loc[0])
            )
        )

        self.dim = dim_loc
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        if suff_stat_loc[0] is not None:
            if dim_loc is None:
                raise ValueError("dim must be set when suff_stat is provided.")
            self.prior_mu: Optional[np.ndarray] = np.reshape(suff_stat_loc[0], dim_loc)
        else:
            self.prior_mu = None
        if suff_stat_loc[1] is not None:
            if dim_loc is None:
                raise ValueError("dim must be set when suff_stat is provided.")
            self.prior_covar: Optional[np.ndarray] = np.reshape(
                suff_stat_loc[1], (dim_loc, dim_loc)
            )
        else:
            self.prior_covar = None
        self.key = keys

    def accumulator_factory(self) -> "MultivariateGaussianAccumulatorFactory":
        """Create a factory carrying this estimator's dimension and key."""
        return MultivariateGaussianAccumulatorFactory(dim=self.dim, keys=self.key)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, np.ndarray, float],
        device: Optional[tn.device] = None,
    ) -> "MultivariateGaussianDistribution":
        """Estimate mean and full covariance on ``device``."""
        nobs = suff_stat[2]
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (
                suff_stat[1] + (pc2 * self.prior_covar) - np.outer(mu, mu * nobs)
            ) / (nobs + pc2)
        else:
            covar = (suff_stat[1] / nobs) - np.outer(mu, mu)

        return MultivariateGaussianDistribution(mu, covar, device=device)


class MultivariateGaussianDataEncoder(TorchSequenceEncoder):
    """Encode vector observations as a two-dimensional floating tensor."""

    def __init__(self, dim: Optional[int] = None) -> None:
        """Initialize an encoder for vectors of optional fixed dimension."""
        self.dim = dim

    def __str__(self) -> str:
        """Return a concise encoder representation."""
        return "MultivariateGaussianDataEncoder(dim=" + str(self.dim) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has the same dimension."""
        return (
            other.dim == self.dim
            if isinstance(other, MultivariateGaussianDataEncoder)
            else False
        )

    def seq_encode(
        self,
        x: Union[Sequence[List[float]], Sequence[List[np.ndarray]], np.ndarray],
        device: Optional[tn.device] = None,
    ) -> "MultivariateGaussianTorchSequence":
        """Encode ``N`` vectors as a floating tensor of shape ``(N, D)``.

        The tensor is created on ``device`` with the default dtype selected by
        :func:`dmx.torch_utils.vector.tensor`.
        """
        self.dim = len(x[0]) if self.dim is None else self.dim
        dim = self.dim

        return MultivariateGaussianTorchSequence(
            data=vec.tensor(np.reshape(np.asarray(x), (-1, dim)), device=device),
            device=device,
        )


class MultivariateGaussianTorchSequence(TorchEncodedSequence):
    """Store an encoded floating tensor with shape ``(N, D)``."""

    data: tn.Tensor

    def __init__(self, data: tn.Tensor, device: Optional[tn.device] = None) -> None:
        """Initialize the wrapper around ``data`` on its associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"MultivariateGaussianTorchSequence(device={repr(self.device)})"
