"""Bayesian Bernoulli likelihoods for binary observations.

``BernoulliDistribution(p)`` assigns probability ``p`` to ``True`` and
``1 - p`` to ``False``. The default ``Beta(1.000001, 1.000001)`` prior is
nearly uniform while retaining a unique interior MAP estimate. Accumulators
store weighted ``(true_count, false_count)`` statistics. With a beta prior,
expected log-density integrates the parameter under that prior; otherwise it
falls back to the fixed-parameter log-density.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from dmx.bstats.beta import BetaDistribution
from dmx.bstats.nulldist import NullDistribution, null_dist
from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import digamma

# Legacy bstats implementations are intentionally concrete protocol classes.
# pylint: disable=abstract-method

BernoulliEncoded = np.ndarray[Any, np.dtype[np.bool_]]
BernoulliSuffStat = tuple[float, float]
Model = ProbabilityDistribution[Any, Any, Any]

default_prior = BetaDistribution(1.000001, 1.000001)


class BernoulliDistribution(ProbabilityDistribution[bool, float, BernoulliEncoded]):
    """Bernoulli likelihood on the two boolean observations."""

    def __init__(
        self,
        p: float,
        name: Optional[str] = None,
        prior: Model = default_prior,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a Bernoulli likelihood and its parameter prior."""
        super().__init__()
        self.name = name
        self.keys = keys
        self.set_parameters(p)
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"BernoulliDistribution({self.p!r}, name={self.name!r}, "
            f"prior={self.prior}, keys={self.keys!r})"
        )

    def get_parameters(self) -> float:
        """Return the success probability."""
        return self.p

    def set_parameters(self, value: float) -> None:
        """Replace the success probability and cached logarithms."""
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Bernoulli probability must be between zero and one.")
        self.p = float(value)
        with np.errstate(divide="ignore"):
            self.log_p0 = float(np.log(self.p))
            self.log_p1 = float(np.log1p(-self.p))

    def get_prior(self) -> Model:
        """Return the current beta or nonconjugate prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the prior and cache beta expected-log parameters."""
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = prior is not None and not isinstance(prior, NullDistribution)
        if self.has_conj_prior:
            assert isinstance(prior, BetaDistribution)
            a, b = prior.get_parameters()
            self.conj_prior_params: Optional[tuple[float, float, float]] = (
                float(digamma(a)),
                float(digamma(b)),
                float(digamma(a + b)),
            )
        else:
            self.conj_prior_params = None

    def get_data_type(self) -> type[bool]:
        """Return the intended observation type."""
        return bool

    def log_density(self, x: bool) -> float:
        """Evaluate the log mass for a boolean observation."""
        return self.log_p0 if x else self.log_p1

    def expected_log_density(self, x: bool) -> float:
        """Evaluate log mass averaged under a beta prior when available."""
        if self.conj_prior_params is None:
            return self.log_density(x)
        expected_true, expected_false, expected_total = self.conj_prior_params
        return expected_true - expected_total if x else expected_false - expected_total

    def cross_entropy(self, dist: Model) -> float:
        """Return ``-E_self[log(dist)]`` over the two outcomes."""
        return float(
            -(self.p * dist.log_density(True))
            - ((1.0 - self.p) * dist.log_density(False))
        )

    def entropy(self) -> float:
        """Return the Bernoulli Shannon entropy."""
        terms = np.asarray([self.p, 1.0 - self.p], dtype=float)
        positive = terms > 0.0
        return float(-np.dot(terms[positive], np.log(terms[positive])))

    def moment(self, order: int) -> float:
        """Return the raw moment for a nonnegative order."""
        return 1.0 if order == 0 else self.p

    def seq_log_density(self, x: BernoulliEncoded) -> np.ndarray[Any, Any]:
        """Evaluate fixed-parameter log masses for encoded observations."""
        return np.where(x, self.log_p0, self.log_p1).astype(float)

    def seq_expected_log_density(self, x: BernoulliEncoded) -> np.ndarray[Any, Any]:
        """Evaluate expected log masses for encoded observations."""
        if self.conj_prior_params is None:
            return self.seq_log_density(x)
        expected_true, expected_false, expected_total = self.conj_prior_params
        return np.where(
            x, expected_true - expected_total, expected_false - expected_total
        ).astype(float)

    def seq_encode(self, x: Iterable[bool]) -> BernoulliEncoded:
        """Encode observations as a boolean NumPy array."""
        return np.asarray(tuple(x), dtype=bool)

    def dist_to_encoder(self) -> "BernoulliDataEncoder":
        """Create the Bernoulli sequence encoder."""
        return BernoulliDataEncoder()

    def sampler(self, seed: Optional[int] = None) -> "BernoulliSampler":
        """Create a repeatable Bernoulli sampler."""
        return BernoulliSampler(self, seed)

    def estimator(self) -> "BernoulliEstimator":
        """Create an estimator retaining metadata and the current prior."""
        return BernoulliEstimator(name=self.name, keys=self.keys, prior=self.prior)


class BernoulliSampler(DistributionSampler[bool]):
    """Draw independent boolean Bernoulli observations."""

    def __init__(self, dist: BernoulliDistribution, seed: Optional[int] = None) -> None:
        """Initialize the sampler for ``dist``."""
        super().__init__(dist, seed)

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one boolean or a list of ``size`` booleans."""
        if size is None:
            return bool(self.rng.rand() < self.dist.p)
        return (self.rng.rand(size) < self.dist.p).tolist()


class BernoulliEstimatorAccumulator(
    StatisticAccumulator[bool, BernoulliSuffStat, BernoulliEncoded]
):
    """Accumulate weighted true and false counts."""

    def __init__(self, name: Optional[str], keys: Optional[str]) -> None:
        """Initialize empty counts and optional sharing metadata."""
        self.name = name
        self.key = keys
        self.psum = 0.0
        self.nsum = 0.0

    def initialize(self, x: bool, weight: float, rng: np.random.RandomState) -> None:
        """Accumulate one observation during initialization."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: BernoulliEncoded,
        weights: np.ndarray[Any, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Accumulate encoded observations during initialization."""
        del rng
        self.seq_update(x, weights, None)

    def update(self, x: bool, weight: float, estimate: Optional[Model]) -> None:
        """Add one weighted boolean observation."""
        del estimate
        if x:
            self.psum += weight
        else:
            self.nsum += weight

    def seq_update(
        self,
        x: BernoulliEncoded,
        weights: np.ndarray[Any, Any],
        estimate: Optional[Model],
    ) -> None:
        """Add encoded booleans with corresponding weights."""
        del estimate
        positive = float(weights[x].sum())
        self.psum += positive
        self.nsum += float(weights.sum()) - positive

    def combine(self, suff_stat: BernoulliSuffStat) -> "BernoulliEstimatorAccumulator":
        """Merge true and false counts."""
        self.psum += suff_stat[0]
        self.nsum += suff_stat[1]
        return self

    def value(self) -> BernoulliSuffStat:
        """Return ``(true_count, false_count)``."""
        return self.psum, self.nsum

    def from_value(self, x: BernoulliSuffStat) -> "BernoulliEstimatorAccumulator":
        """Restore ``(true_count, false_count)``."""
        self.psum, self.nsum = x
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Merge statistics through the configured sharing key."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Replace statistics through the configured sharing key."""
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "BernoulliDataEncoder":
        """Create the compatible Bernoulli encoder."""
        return BernoulliDataEncoder()


class BernoulliEstimatorAccumulatorFactory(
    StatisticAccumulatorFactory[bool, BernoulliSuffStat, BernoulliEncoded]
):
    """Create Bernoulli accumulators with shared metadata."""

    def __init__(self, name: Optional[str], keys: Optional[str]) -> None:
        """Store metadata copied into each accumulator."""
        self.name = name
        self.keys = keys

    def make(self) -> BernoulliEstimatorAccumulator:
        """Create an empty Bernoulli accumulator."""
        return BernoulliEstimatorAccumulator(self.name, self.keys)


class BernoulliEstimator(
    ParameterEstimator[bool, float, BernoulliEncoded, BernoulliSuffStat]
):
    """Estimate a Bernoulli parameter and update its beta posterior."""

    def __init__(
        self,
        name: Optional[str] = None,
        keys: Optional[str] = None,
        prior: Model = default_prior,
    ) -> None:
        """Initialize estimator metadata and prior."""
        self.name = name
        self.keys = keys
        self.set_prior(prior)

    def accumulator_factory(self) -> BernoulliEstimatorAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return BernoulliEstimatorAccumulatorFactory(self.name, self.keys)

    def set_prior(self, prior: Model) -> None:
        """Replace the estimator prior and update prior flags."""
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = prior is not None and not isinstance(prior, NullDistribution)

    def get_prior(self) -> Model:
        """Return the estimator prior."""
        return self.prior

    # The base estimator exposes overloaded one- and two-argument call forms.
    def estimate(  # pylint: disable=arguments-differ
        self, *args: Any
    ) -> BernoulliDistribution:
        """Estimate from ``(true_count, false_count)`` statistics."""
        psum, nsum = args[-1]
        if self.has_conj_prior:
            assert isinstance(self.prior, BetaDistribution)
            prior_a, prior_b = self.prior.get_parameters()
            posterior_a = prior_a + psum
            posterior_b = prior_b + nsum
            if posterior_a > 1.0 and posterior_b > 1.0:
                probability = (posterior_a - 1.0) / (posterior_a + posterior_b - 2.0)
            elif posterior_a <= 1.0 < posterior_b:
                probability = 0.0
            elif posterior_b <= 1.0 < posterior_a:
                probability = 1.0
            else:
                probability = psum / (psum + nsum) if psum + nsum else 0.5
            return BernoulliDistribution(
                probability,
                name=self.name,
                prior=BetaDistribution(posterior_a, posterior_b),
                keys=self.keys,
            )
        if self.has_prior:

            def objective(value: float) -> float:
                return float(
                    -np.log(value) * psum
                    - np.log1p(-value) * nsum
                    - self.prior.log_density(value)
                )

            solution = minimize_scalar(
                objective,
                bounds=(np.finfo(float).eps, 1.0 - np.finfo(float).eps),
                method="bounded",
            )
            return BernoulliDistribution(
                float(solution.x), name=self.name, prior=self.prior, keys=self.keys
            )
        total = psum + nsum
        probability = psum / total if total else 0.5
        return BernoulliDistribution(
            probability, name=self.name, prior=null_dist, keys=self.keys
        )


class BernoulliDataEncoder(DataSequenceEncoder[bool, BernoulliEncoded]):
    """Encode Bernoulli observations as booleans."""

    def __str__(self) -> str:
        """Return the stable encoder name."""
        return "BernoulliDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has Bernoulli semantics."""
        return isinstance(other, BernoulliDataEncoder)

    def seq_encode(self, x: Iterable[bool]) -> "BernoulliEncodedData":
        """Encode observations in a typed container."""
        return BernoulliEncodedData(np.asarray(tuple(x), dtype=bool))


class BernoulliEncodedData(EncodedDataSequence[BernoulliEncoded]):
    """Contain an encoded Bernoulli sequence."""

    def __init__(self, data: BernoulliEncoded) -> None:
        """Store the boolean array."""
        super().__init__(data)

    def __repr__(self) -> str:
        """Return a concise encoded-data representation."""
        return f"BernoulliEncodedData(data={self.data!r})"
