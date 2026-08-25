r"""Provide multivariate Gaussian distributions and estimation utilities.

``MultivariateGaussianDistribution`` models vectors in :math:`\mathbb{R}^d`
with a mean vector and full covariance matrix. The module also provides its
sampler, weighted sufficient statistics, maximum-likelihood estimator, and
two-dimensional sequence encoding.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.arithmetic import exp, pi
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class MultivariateGaussianDistribution(SequenceEncodableProbabilityDistribution):
    r"""Represent a multivariate Gaussian with full covariance.

    ``mu`` has shape ``(d,)`` and ``covar`` has shape ``(d, d)``. The covariance
    is interpreted directly, rather than as a precision or Cholesky factor, and
    must be symmetric positive definite so SciPy can factor it.

    Attributes:
        dim (int): N is the dim of multivariate normal.
        mu (np.ndarray): Length N numpy array
        covar (np.ndarray): N by N numpy array for Covariance matrix.
        chol (np.ndarray): Cholesky decomposition of covar.
        lower (bool): Flag for lower (False for upper)
        name (Optional[str]): Set name to object.
        keys (Optional[str]): Set keys for distribution.
        self.use_lstsq (bool): Cholesky does not exist so use least squares approx.
        self.chol_const (float): det from covar if lstsq is to be used.

    """

    def __init__(
        self,
        mu: Union[List[float], np.ndarray],
        covar: Union[List[List[float]], np.ndarray],
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a multivariate Gaussian distribution.

        Args:
            mu (Union[List[float], np.ndarray]): N-dimensional mean.
            covar (Union[List[List[float]], np.ndarray]): Covariance matrix, should be N
                by N and positive definite.
            name (Optional[str]): Set name to object.
            keys (Optional[str]): Set keys for distribution.

        """
        super().__init__()
        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.covar = np.reshape(self.covar, (len(self.mu), len(self.mu)))
        self.chol, self.lower = scipy.linalg.cho_factor(self.covar)
        self.name = name
        self.keys = keys

        if self.chol is None:
            raise RuntimeError(
                "Cannot obtain Choleskey factorization for covariance matrix."
            )
        self.use_lstsq = False
        self.chol_const = float(
            -0.5
            * (
                len(self.mu) * np.log(2.0 * pi)
                + 2.0 * np.log(vec.diag(self.chol)).sum()
            )
        )

    def __str__(self) -> str:
        """Return an evaluable representation of the distribution."""
        s1 = repr(self.mu.tolist())
        s2 = repr([u.tolist() for u in self.covar])
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return (
            f"MultivariateGaussianDistribution(mu={s1}, covar={s2}, name={s3}, "
            f"keys={s4})"
        )

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

        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            float: Log-density at x.

        """
        if self.use_lstsq:
            raise RuntimeError("Least-squares log-likelihood evaluation not supported.")
        try:
            diff = self.mu - x
            soln = scipy.linalg.cho_solve((self.chol, self.lower), diff.T).T
            rv = self.chol_const - 0.5 * ((diff * soln).sum())
            return float(rv)
        except Exception as e:
            raise e

    def seq_log_density(
        self, x: "MultivariateGaussianEncodedDataSequence"
    ) -> np.ndarray:
        """Evaluate log densities for an encoded observation matrix.

        Encoded data has shape ``(n, d)`` and the result has shape ``(n,)``.
        """
        if not isinstance(x, MultivariateGaussianEncodedDataSequence):
            raise TypeError(
                "MultivariateGaussianEncodedDataSequence required for "
                "seq_log_density()."
            )

        if self.use_lstsq:
            return np.ones(x.data.shape[0])
        diff = self.mu - x.data
        soln = scipy.linalg.cho_solve((self.chol, self.lower), diff.T).T
        rv = self.chol_const - 0.5 * ((diff * soln).sum(axis=1))
        return np.asarray(rv)

    def sampler(self, seed: Optional[int] = None) -> "MultivariateGaussianSampler":
        """Create a sampler, optionally initialized with ``seed``."""
        return MultivariateGaussianSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "MultivariateGaussianEstimator":
        """Create an estimator centered on this distribution when regularized."""
        if pseudo_count is None:
            return MultivariateGaussianEstimator(
                dim=self.dim, name=self.name, keys=self.keys
            )
        pseudo_counts = (pseudo_count, pseudo_count)
        return MultivariateGaussianEstimator(
            dim=self.dim,
            pseudo_count=pseudo_counts,
            suff_stat=(self.mu, self.covar),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "MultivariateGaussianDataEncoder":
        """Create an encoder for vectors of this distribution's dimension."""
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianSampler(DistributionSampler):
    """Draw independent samples from a multivariate Gaussian.

    Attributes:
        rng (RandomState): Sets seed for generating samples.
        dist (MultivariateGaussianDistribution): MultivariateGaussianDistribution to
            sample from.

    """

    def __init__(
        self, dist: "MultivariateGaussianDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize the sampler."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Generate samples from MultivariateGaussianDistribution.

        Args:
            size (Optional[int]): Number of samples to generate.

        Returns:
            np.ndarray: Size by dim number of samples.

        """
        return self.rng.multivariate_normal(
            mean=self.dist.mu, cov=self.dist.covar, size=size
        )


class MultivariateGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted multivariate Gaussian sufficient statistics.

    The statistic ``(sum, sum2, count)`` contains a vector of shape ``(d,)``,
    the uncentered second-moment sum of shape ``(d, d)``, and total weight.

    Attributes:
        dim (Optional[int]): Dimension of the mvn.
        count (float): weight counter suff stat
        sum (Optional[np.ndarray]): Suff stat, weighted sum of obs
        sum2 (Optional[np.ndarray]): Suff stat, weights sum of squared obs.
        key (Optional[str]): Key for the mean and covariance.
        name (Optional[str]): Name of distribution.

    """

    def __init__(
        self,
        dim: Optional[int] = None,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize an empty accumulator with an optional fixed dimension.

        Args:
            dim (Optional[int]): Dimension of the mvn.
            keys (Optional[str]): Key for the mean and covariance.
            name (Optional[str]): Name of distribution.

        """
        self.dim = dim
        self.count = 0.0
        self.key = keys
        self.name = name
        self.sum: Optional[np.ndarray]
        self.sum2: Optional[np.ndarray]

        if dim is not None:
            self.sum = vec.zeros(dim)
            self.sum2 = vec.zeros((dim, dim))
        else:
            self.sum = None
            self.sum2 = None

    def update(
        self,
        x: np.ndarray,
        weight: float,
        estimate: Optional[MultivariateGaussianDistribution],
    ) -> None:
        """Add one weighted vector observation."""
        if self.dim is None:
            self.dim = len(x)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros((self.dim, self.dim))

        assert self.sum is not None
        assert self.sum2 is not None
        x_weight = x * weight
        self.sum += x_weight
        self.sum2 += vec.outer(x, x_weight)
        self.count += weight

    def initialize(
        self, x: np.ndarray, weight: float, rng: Optional[RandomState]
    ) -> None:
        """Initialize from one vector observation; ``rng`` is unused."""
        del rng
        self.update(x, weight, None)

    def seq_update(
        self,
        x: "MultivariateGaussianEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[RandomState],
    ) -> None:
        """Add an encoded ``(n, d)`` matrix with weights of shape ``(n,)``."""
        if self.dim is None:
            self.dim = x.data.shape[1]
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros((self.dim, self.dim))

        assert self.sum is not None
        assert self.sum2 is not None
        x_weight = np.multiply(x.data.T, weights)
        self.count += weights.sum()
        self.sum += x_weight.sum(axis=1)
        self.sum2 += np.einsum("ji,ik->jk", x_weight, x.data)

    def seq_initialize(
        self,
        x: "MultivariateGaussianEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Initialize from encoded observations; ``rng`` is unused."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray], float]
    ) -> "MultivariateGaussianAccumulator":
        """Add a ``(sum, sum2, count)`` statistic to this accumulator."""
        if suff_stat[0] is not None and self.sum is not None:
            assert suff_stat[1] is not None
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            assert suff_stat[1] is not None
            self.sum = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]

        return self

    def value(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Return the ``(sum, sum2, count)`` sufficient statistic."""
        return self.sum, self.sum2, self.count

    def from_value(
        self, x: Tuple[Optional[np.ndarray], Optional[np.ndarray], float]
    ) -> "MultivariateGaussianAccumulator":
        """Replace this accumulator from a sufficient-statistic tuple."""
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge keyed statistics using the existing accumulator contract."""
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key])

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator from a keyed statistic when present."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key])

    def acc_to_encoder(self) -> "MultivariateGaussianDataEncoder":
        """Create the matching vector data encoder."""
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Create multivariate Gaussian sufficient-statistic accumulators.

    Attributes:
        dim (Optional[int]): Dimension of the mvn.
        keys (Optional[str]): Key for the mean and covariance.
        name (Optional[str]): Name of distribution.

    """

    def __init__(
        self, dim: Optional[int], keys: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        """Initialize the factory."""
        self.dim = dim
        self.key = keys
        self.name = name

    def make(self) -> "MultivariateGaussianAccumulator":
        """Create an empty accumulator."""
        return MultivariateGaussianAccumulator(
            dim=self.dim, keys=self.key, name=self.name
        )


class MultivariateGaussianEstimator(ParameterEstimator):
    """Estimate a full-covariance multivariate Gaussian.

    The two pseudo-count entries independently regularize the mean and
    covariance toward ``suff_stat``. Without them, estimation uses weighted
    maximum likelihood and therefore assumes positive total weight and a
    positive-definite estimated covariance.

    Attributes:
        dim (int): Dimension of multivariate normal.
        pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Regularize
            mean and/or covariance.
        prior_mu (Optional[np.ndarray]): Mean from prior data or used to regularize.
        prior_covar (Optional[np.ndarray]): Covariance matrix from prior data or used to
            regularize.
        name (Optional[str]): Set name to object.
        keys (Optional[str]): Keys for merging sufficient statistics.

    """

    def __init__(
        self,
        dim: Optional[int] = None,
        pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
        suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (
            None,
            None,
        ),
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize the estimator and optional regularization targets.

        Args:
            dim (Optional[int]): Dimension of multivariate normal. Inferred from
                'suff_stat' if None.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Regularize
                mean and/or covariance.
            suff_stat (Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]):
                Mean and covariance estimated
                from previous data or used to regularize.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for estimator.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError(
                "MultivariateGaussianEstimator requires keys to be of type 'str'."
            )

        dim_loc = (
            dim
            if dim is not None
            else (
                (None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1]))))
                if suff_stat[0] is None
                else len(suff_stat[0])
            )
        )
        if dim_loc is None:
            raise ValueError("Cannot infer multivariate Gaussian dimension.")
        dim_int = int(dim_loc)

        self.dim = dim_int
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.prior_mu = (
            None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_int)
        )
        self.prior_covar = (
            None
            if suff_stat[1] is None
            else np.reshape(suff_stat[1], (dim_int, dim_int))
        )
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "MultivariateGaussianAccumulatorFactory":
        """Create a matching accumulator factory."""
        return MultivariateGaussianAccumulatorFactory(
            dim=self.dim, keys=self.keys, name=self.name
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray], float],
    ) -> "MultivariateGaussianDistribution":
        """Estimate a distribution from ``(sum, sum2, count)``.

        ``nobs`` is accepted for protocol compatibility and replaced by the
        statistic count.
        """
        nobs = suff_stat[2]
        assert suff_stat[0] is not None
        assert suff_stat[1] is not None
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (
                suff_stat[1] + (pc2 * self.prior_covar) - vec.outer(mu, mu * nobs)
            ) / (nobs + pc2)
        else:
            covar = (suff_stat[1] / nobs) - vec.outer(mu, mu)

        return MultivariateGaussianDistribution(mu, covar, name=self.name)


class MultivariateGaussianDataEncoder(DataSequenceEncoder):
    """Encode vector observations as an ``(n, d)`` floating-point array.

    Attributes:
        dim (Optional[int]): dimension of mvn.

    """

    def __init__(self, dim: Optional[int] = None) -> None:
        """Initialize an encoder with an optional fixed dimension."""
        self.dim = dim

    def __str__(self) -> str:
        """Return the encoder name and dimension."""
        return "MultivariateGaussianDataEncoder(dim=" + str(self.dim) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` has the same encoder dimension."""
        return (
            other.dim == self.dim
            if isinstance(other, MultivariateGaussianDataEncoder)
            else False
        )

    def seq_encode(
        self, x: Union[Sequence[List[float]], Sequence[List[np.ndarray]], np.ndarray]
    ) -> "MultivariateGaussianEncodedDataSequence":
        """Encode observations, inferring ``d`` from the first when needed."""
        dim = len(x[0]) if self.dim is None else self.dim
        self.dim = dim

        return MultivariateGaussianEncodedDataSequence(
            data=np.reshape(np.asarray(x), (-1, dim))
        )


class MultivariateGaussianEncodedDataSequence(EncodedDataSequence):
    """Store an encoded observation matrix of shape ``(n, d)``.

    Attributes:
        data (np.ndarray): Encoded sequence of mvn obs. sz by dim.

    """

    def __init__(self, data: np.ndarray) -> None:
        """Initialize the encoded observation matrix."""
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded matrix."""
        return f"MultivariateGaussianEncodedDataSequence(data={self.data})"
