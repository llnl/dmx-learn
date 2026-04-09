""""Create, estimate, and sample from a multivariate normal distribution with mean vector 'mu' (length n), and
covariance matrix 'covar' (n by n).

Defines the MultivariateGaussianDistribution, MultivariateGaussianSampler, MultivariateGaussianAccumulatorFactory,
MultivariateGaussianAccumulator, MultivariateGaussianEstimator, and the MultivariateGaussianDataEncoder classes for use
with pysparkplug.

Data type: np.ndarray[float]

x = (x_1,x_2,..,x_n) ~ MVN(mu, covar), where mu is a length n numpy array, anc covar is an n by n positive definite
covariance matrix.

The log-density is given by
    log(p(x)) = -0.5*k*log(2*pi) - 0.5*det(covar) - 0.5*(x-mu)' covar^{-1} (x-mu).

"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tn

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


class MultivariateGaussianDistribution(TorchProbabilityDistribution):
    """MultivariateGaussianDistribution object for multivariate Gaussian with mean mu and covaraince 'covar'.

    Attributes:
        dim (int): N is the dim of multivariate normal.
        mu (tn.tensor): Length N numpy array
        covar (tn.tensor): N by N numpy array for Covariance matrix.
        chol (tn.tensor): Cholesky decomposition of covar.
        name (Optional[str]): Set name to object.
        keys (Optional[str]): Set keys for distribution.
        self.use_lstsq (bool): Cholesky does not exist so use least squares approx.
        self.chol_const (float): det from covar if lstsq is to be used.

    """

    def __init__(
        self,
        mu: Union[List[float], np.ndarray],
        covar: Union[List[List[float]], np.ndarray],
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """MultivariateGaussianDistribution object.

        Args:
            mu (Union[List[float], np.ndarray]): N-dimensional mean.
            covar (Union[List[List[float]], np.ndarray]): Covariance matrix, should be N by N and positive definite.
            keys (Optional[str]): Set keys for distribution.

        """
        super().__init__(device)
        self.dim = len(mu)
        self.mu = vec.tensor(mu, device=self._device)
        self.covar = vec.tensor(covar, device=self._device).reshape(
            (self.dim, self.dim)
        )
        self.chol = tn.linalg.cholesky(self.covar)
        self.keys = keys
        self.chol_const = -0.5 * (
            len(self.mu) * np.log(2.0 * pi) + 2.0 * tn.log(tn.diag(self.chol)).sum()
        )

    def to(self, device: Optional[tn.device] = None) -> None:
        self.mu = self.mu.to(device)
        self.covar = self.covar.to(device)
        self.chol = self.chol.to(device)
        self.chol_const = -0.5 * (
            len(self.mu) * np.log(2.0 * pi) + 2.0 * tn.log(tn.diag(self.chol)).sum()
        )
        self._device = device

    def __repr__(self) -> str:
        s1 = repr(self.mu.data.cpu().tolist())
        s2 = repr([list(u) for u in self.covar.data.cpu().tolist()])
        s3 = repr(self.keys)

        return "MultivariateGaussianDistribution(%s, %s, keys=%s)" % (s1, s2, s3)

    def density(self, x: np.ndarray) -> float:
        """Evaluate the density at x.

        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            float: Density at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Evaluate the log-density at x.

        Notes:
            log(p(x)) = -0.5*k*log(2*pi) - 0.5*det(covar) - 0.5*(x-mu)' covar^{-1} (x-mu).
        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            float: Log-density at x.

        """
        try:
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

        except Exception as e:
            raise e

    def seq_log_density(self, x: "MultivariateGaussianTorchSequence") -> tn.Tensor:
        if not isinstance(x, MultivariateGaussianTorchSequence):
            raise Exception(
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

    def sampler(self, seed: Optional[int] = None):
        return MultivariateGaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None):
        if pseudo_count is None:
            return MultivariateGaussianEstimator()
        else:
            pseudo_count = (pseudo_count, pseudo_count)
            return MultivariateGaussianEstimator(
                pseudo_count=pseudo_count, suff_stat=(self.mu, self.covar)
            )

    def dist_to_encoder(self) -> "MultivariateGaussianDataEncoder":
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianSampler(DistributionSampler):

    def __init__(
        self, dist: "MultivariateGaussianDistribution", seed: Optional[int] = None
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.mu = dist.mu.data.cpu().numpy()
        self.covar = dist.covar.data.cpu().numpy()

    def sample(self, size: Optional[int] = None) -> np.ndarray:
        return self.rng.multivariate_normal(mean=self.mu, cov=self.covar, size=size)


class MultivariateGaussianAccumulator(TorchStatisticAccumulator):
    def __init__(
        self,
        dim: Optional[int] = None,
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        super().__init__(device=device)
        self.dim = dim
        self.count = 0.0
        self.key = keys

        if dim is not None:
            self.sum = np.zeros(dim, dtype=np.float64)
            self.sum2 = np.zeros((dim, dim), dtype=np.float64)
        else:
            self.sum = None
            self.sum2 = None

    def seq_update(
        self,
        x: "MultivariateGaussianTorchSequence",
        weights: tn.Tensor,
        estimate: Optional[MultivariateGaussianDistribution],
    ) -> None:
        if self.dim is None:
            self.dim = x.data.shape[1]
            self.sum = np.zeros(self.dim, dtype=np.float64)
            self.sum2 = np.zeros((self.dim, self.dim), dtype=np.float64)

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
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[np.ndarray, np.ndarray, float]
    ) -> "MultivariateGaussianAccumulator":
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
        return self.sum, self.sum2, self.count

    def from_value(
        self, x: Tuple[np.ndarray, np.ndarray, float]
    ) -> "MultivariateGaussianAccumulator":
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key])

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key])

    def acc_to_encoder(self) -> "MultivariateGaussianDataEncoder":
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianAccumulatorFactory(TorchStatisticAccumulatorFactory):

    def __init__(self, dim: Optional[int], keys: Optional[str] = None) -> None:
        self.dim = dim
        self.key = keys

    def make(
        self, device: Optional[tn.device] = None
    ) -> "MultivariateGaussianAccumulator":
        return MultivariateGaussianAccumulator(
            dim=self.dim, keys=self.key, device=device
        )


class MultivariateGaussianEstimator(TorchParameterEstimator):
    """MultivariateGaussianEstimator object for estimating multivariate normal distribution from sufficient stats.

    Attributes:
        dim (int): Dimension of multivariate normal.
        pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Regularize mean and/or covariance.
        prior_mu (Optional[np.ndarray]): Mean from prior data or used to regularize.
        prior_covar (Optional[np.ndarray]): Covariance matrix from prior data or used to regularize.
        key (Optional[str]): Keys for merging sufficient statistics.

    """

    def __init__(
        self,
        dim: Optional[int] = None,
        pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
        suff_stat: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = (
            None,
            None,
        ),
        keys: Optional[str] = None,
    ) -> None:
        """MultivariateGaussianEstimator object.

        Args:
            dim (Optional[int]): Dimension of multivariate normal. Inferred from 'suff_stat' if None.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Regularize mean and/or covariance.
            suff_stat (Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]): Mean and covariance estimated
                from previous data or used to regularize.
            keys (Optional[str]): Set keys for estimator.

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
        self.prior_mu = (
            None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        )
        self.prior_covar = (
            None
            if suff_stat[1] is None
            else np.reshape(suff_stat[1], (dim_loc, dim_loc))
        )
        self.key = keys

    def accumulator_factory(self) -> "MultivariateGaussianAccumulatorFactory":
        return MultivariateGaussianAccumulatorFactory(dim=self.dim, keys=self.key)

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, np.ndarray, float],
        device: Optional[tn.device] = None,
    ) -> "MultivariateGaussianDistribution":
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

    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "MultivariateGaussianDataEncoder(dim=" + str(self.dim) + ")"

    def __eq__(self, other: object) -> bool:
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
        self.dim = len(x[0]) if self.dim is None else self.dim

        return MultivariateGaussianTorchSequence(
            data=vec.tensor(np.reshape(np.asarray(x), (-1, self.dim)), device=device)
        )


class MultivariateGaussianTorchSequence(TorchEncodedSequence):

    def __init__(self, data: tn.tensor, device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:

        return f"MultivariateGaussianTorchSequence(device={repr(self.device)})"
