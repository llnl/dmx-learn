r"""Provide Poisson distributions, estimation, sampling, and encoding.

``PoissonDistribution`` has rate :math:`\lambda > 0` and support
:math:`\{0, 1, 2, \ldots\}`. The module also provides its weighted sufficient-
statistic accumulator, maximum-likelihood estimator, sampler, and sequence
encoder.
"""

from math import log
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.vector import gammaln


class PoissonDistribution(SequenceEncodableProbabilityDistribution):
    r"""Represent a Poisson distribution with rate ``lam``.

    For nonnegative integer :math:`x`,

    .. math::

       \log p(x \mid \lambda) = x\log\lambda - \log(x!) - \lambda.

    Attributes:
        lam (float): Mean of Poisson distribution.
        name (Optional[str]): String name for object instance.
        log_lambda (float): Log of attribute lam.
        keys (Optional[str]): Keys for lambda.

    """

    def __init__(
        self, lam: float, name: Optional[str] = None, keys: Optional[str] = None
    ) -> None:
        """Initialize a Poisson distribution.

        Args:
            lam (float): Positive real-valued number.
            name (Optional[str]): String name for object instance.
            keys (Optional[str]): Key for lambda.

        """
        super().__init__()
        self.lam = lam
        self.log_lambda = log(lam)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return an evaluable representation of the distribution."""
        s0 = repr(float(self.lam))
        s1 = repr(self.name)
        s2 = repr(self.keys)

        return f"PoissonDistribution({s0}, name={s1}, keys={s2})"

    def density(self, x: int) -> float:
        """Evaluate the density of Poisson distribution at observation x.

        Notes:
            See log_density().

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            float: Density of Poisson distribution evaluated at x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: int) -> float:
        r"""Evaluate the Poisson log density at ``x``.

        .. math::
            \\log{f(x | \\lambda)} = -x \\log{\\lambda} - \\log{x!} - \\lambda.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            float: Log-density of Poisson distribution evaluated at x.

        """
        if x < 0:
            return -np.inf
        return float(x * self.log_lambda - gammaln(x + 1.0) - self.lam)

    def seq_log_density(self, x: "PoissonEncodedDataSequence") -> np.ndarray:
        """Evaluate log densities for an encoded sequence.

        Both encoded arrays and the returned array have shape ``(n,)``.
        """
        if not isinstance(x, PoissonEncodedDataSequence):
            raise TypeError(
                "PoissonEncodedDataSequence required for seq_log_density()."
            )

        rv = x.data[0] * self.log_lambda
        rv -= x.data[1]
        rv -= self.lam
        return np.asarray(rv)

    def sampler(self, seed: Optional[int] = None) -> "PoissonSampler":
        """Create a sampler, optionally initialized with ``seed``."""
        return PoissonSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "PoissonEstimator":
        """Create an estimator centered on this rate when regularized."""
        if pseudo_count is None:
            return PoissonEstimator(name=self.name, keys=self.keys)
        return PoissonEstimator(
            pseudo_count=pseudo_count,
            suff_stat=self.lam,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "PoissonDataEncoder":
        """Create an encoder for nonnegative count observations."""
        return PoissonDataEncoder()


class PoissonSampler(DistributionSampler):
    """Draw independent samples from a Poisson distribution.

    Attributes:
        rng (RandomState): RandomState with seed set for sampling.
        dist (GeometricDistribution): PoissonDistribution to sample from.

    """

    def __init__(self, dist: "PoissonDistribution", seed: Optional[int] = None) -> None:
        """Initialize the sampler."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Union[int, Sequence[int]]:
        """Generate iid samples from Poisson distribution.

        Generates a single Poisson sample (int) if size is None, else a numpy array of
        integers of length size
        containing iid samples, from the Poisson distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, assumed to be
                1.

        Returns:
            If size is None, int, else size length numpy array of ints.

        """
        if size:
            return list(map(int, self.rng.poisson(lam=self.dist.lam, size=size)))
        return int(self.rng.poisson(lam=self.dist.lam))


class PoissonAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted Poisson sufficient statistics.

    The public statistic ``(count, sum)`` contains the total weight and the
    weighted sum of observations.

    Attributes:
         sum (float): Aggregate sum of weighted observations.
         count (float): Aggregate sum of observation weights.
         name (Optional[str]): name for object
         keys (Optional[str]): Key for combining sufficient statistics with object
             instance containing the same key.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initialize an empty accumulator."""
        del name
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys

    def initialize(
        self, x: int, weight: float, rng: Optional[np.random.RandomState] = None
    ) -> None:
        """Initialize from one count observation; ``rng`` is unused."""
        del rng
        self.update(x, weight, None)

    def update(
        self, x: int, weight: float, estimate: Optional["PoissonDistribution"] = None
    ) -> None:
        """Add one weighted count observation."""
        self.sum += x * weight
        self.count += weight

    def seq_initialize(
        self,
        x: "PoissonEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        """Initialize from encoded observations; ``rng`` is unused."""
        self.seq_update(x, weights, None)

    def seq_update(
        self,
        x: "PoissonEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional["PoissonDistribution"] = None,
    ) -> None:
        """Add encoded counts with weights of shape ``(n,)``."""
        self.sum += np.dot(x.data[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat: Tuple[float, float]) -> "PoissonAccumulator":
        """Add a ``(count, sum)`` statistic to this accumulator."""
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        """Return the ``(count, sum)`` sufficient statistic."""
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> "PoissonAccumulator":
        """Replace this accumulator from a ``(count, sum)`` statistic."""
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under ``keys``."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator from ``stats_dict`` when keyed."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "PoissonDataEncoder":
        """Create the matching data encoder."""
        return PoissonDataEncoder()


class PoissonAccumulatorFactory(StatisticAccumulatorFactory):
    """Create Poisson sufficient-statistic accumulators.

    Attributes:
        name (Optional[str]): Name for object
        keys (Optional[str]): Tag for combining sufficient statistics of
            PoissonAccumulator objects when
            constructed.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Initialize the factory."""
        self.name = name
        self.keys = keys

    def make(self) -> "PoissonAccumulator":
        """Create an empty accumulator."""
        return PoissonAccumulator(name=self.name, keys=self.keys)


class PoissonEstimator(ParameterEstimator):
    """Estimate a Poisson rate from weighted sufficient statistics.

    When both regularization arguments are present, ``pseudo_count`` adds that
    much effective weight at the target rate ``suff_stat``.

    Attributes:
        pseudo_count (Optional[float]): Re-weight suff_stat.
        suff_stat (Optional[float]): Mean of Poisson if not None.
        name (Optional[str]): String name of PoissonEstimator instance.
        keys (Optional[str]): String keys of PoissonEstimator instance for combining
            sufficient statistics.

    """

    def __init__(
        self,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[float] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize the estimator and optional rate regularization.

        Args:
            pseudo_count (Optional[float]): Optional non-negative float.
            suff_stat (Optional[float]): Optional non-negative float.
            name (Optional[str]): Assign a name to PoissonEstimator.
            keys (Optional[str]): Assign keys to PoissonEstimator for combining
                sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("PoissonEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "PoissonAccumulatorFactory":
        """Create a matching accumulator factory."""
        return PoissonAccumulatorFactory(keys=self.keys, name=self.name)

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[float, float]
    ) -> "PoissonDistribution":
        """Estimate the rate from a ``(count, sum)`` statistic.

        The ``nobs`` argument is overwritten by the statistic count.
        """
        nobs, psum = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:
            return PoissonDistribution(
                (psum + self.suff_stat * self.pseudo_count)
                / (nobs + self.pseudo_count),
                name=self.name,
            )
        return PoissonDistribution(psum / nobs, name=self.name)


class PoissonDataEncoder(DataSequenceEncoder):
    """Encode sequences of nonnegative Poisson counts."""

    def __str__(self) -> str:
        """Return the encoder name."""
        return "PoissonDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a Poisson encoder."""
        return isinstance(other, PoissonDataEncoder)

    def seq_encode(
        self, x: Union[np.ndarray, Sequence[int]]
    ) -> "PoissonEncodedDataSequence":
        """Encode counts and their log-factorials.

        The returned pair contains two arrays of shape ``(n,)``.
        """
        rv1 = np.asarray(x)

        if np.any(rv1 < 0) or np.any(np.isnan(rv1)):
            raise ValueError("Poisson requires non-negative integer values of x.")
        rv2 = gammaln(rv1 + 1.0)

        return PoissonEncodedDataSequence(data=(rv1, rv2))


class PoissonEncodedDataSequence(EncodedDataSequence):
    """Store Poisson counts and their elementwise log-factorials.

    Attributes:
        data (Tuple[np.ndarray, np.ndarray]): Poisson observations, and the log-gamma
            value of the obs.

    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray]):
        """Initialize an encoded Poisson sequence."""
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded pair."""
        return f"PoissonEncodedDataSequence(data={self.data})"
