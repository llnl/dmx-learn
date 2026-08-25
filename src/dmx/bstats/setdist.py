"""Independent-Bernoulli models for set-like observations.

Observation order does not affect scoring or sufficient statistics. For
compatibility, repeated labels are not deduplicated: every occurrence contributes
again. Statistics are ``(weighted_label_occurrences, observation_weight)``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, MutableMapping
from typing import Any, Optional, cast

import numpy as np

from dmx.bstats.beta import BetaDistribution
from dmx.bstats.mixture import MixtureDistribution
from dmx.bstats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import gammaln

# pylint: disable=abstract-method,arguments-differ

Label = Hashable
Model = ProbabilityDistribution[Any, Any, Any]
Array = np.ndarray[Any, Any]
SetEncoded = tuple[int, Array, Array, Array]
SetSuffStat = tuple[dict[Label, float], float]
default_prior = BetaDistribution(1, 1)


class BernoulliSetDistribution(
    ProbabilityDistribution[Iterable[Label], dict[Label, float], SetEncoded]
):
    """Model inclusion of each known label independently.

    Observations are iterables of labels from ``pmap``. Every known label is
    treated as an independent Bernoulli outcome; absent labels contribute
    exclusion mass. Order is irrelevant, but repeated labels are deliberately
    counted repeatedly for legacy compatibility.

    Nonnegative map values are inclusion probabilities. A negative value is
    the legacy complement encoding for a probability near one: its magnitude
    is the exclusion probability. The common prior applies independently to
    every observed label.

    Args:
        pmap: Mapping from known labels to encoded inclusion probabilities.
        name: Optional identifier for the distribution.
        prior: Common prior for every label's inclusion probability.
    """

    def __init__(
        self,
        pmap: dict[Label, float],
        name: Optional[str] = None,
        prior: Optional[Model] = None,
    ) -> None:
        """Initialize inclusion probabilities, metadata, and prior."""
        super().__init__()
        self.name = name
        self.prior = cast(Model, prior)
        self.set_parameters(pmap)

    def __str__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"BernoulliSetDistribution({self.pmap!r}, name={self.name!r}, "
            f"prior={self.prior})"
        )

    def get_parameters(self) -> dict[Label, float]:
        """Return the legacy encoded inclusion probabilities."""
        return self.pmap

    def set_parameters(self, value: dict[Label, float]) -> None:
        """Replace probabilities and refresh cached log masses.

        Nonnegative values encode inclusion directly. Negative values retain the
        legacy complement encoding used for probabilities near one.
        """
        self.pmap = dict(value)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.log_pmap = {
                key: float(np.log1p(prob) if prob < 0 else np.log(prob))
                for key, prob in self.pmap.items()
            }
            self.log_nmap = {
                key: float(np.log(-prob) if prob < 0 else np.log1p(-prob))
                for key, prob in self.pmap.items()
            }
        self.nmap_sum = float(
            sum(item for item in self.log_nmap.values() if item != -np.inf)
        )

    def get_prior(self) -> Model:
        """Return the common inclusion-probability prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the common inclusion-probability prior."""
        self.prior = prior

    def log_density(self, x: Iterable[Label]) -> float:
        """Score an order-insensitive iterable of known labels."""
        return float(
            self.nmap_sum
            + sum(self.log_pmap[value] - self.log_nmap[value] for value in x)
        )

    def seq_log_density(self, x: SetEncoded) -> Array:
        """Score flattened observations and regroup their label occurrences."""
        count, indices, levels, inverse = x
        log_odds = np.asarray(
            [self.log_pmap[value] - self.log_nmap[value] for value in levels]
        )
        return (
            np.bincount(indices, weights=log_odds[inverse], minlength=count)
            + self.nmap_sum
        )

    def seq_encode(self, x: Iterable[Iterable[Label]]) -> SetEncoded:
        """Flatten observations while retaining row and unique-level indices."""
        observations = tuple(tuple(value) for value in x)
        indices = np.asarray(
            [index for index, value in enumerate(observations) for _ in value],
            dtype=int,
        )
        flat = [item for observation in observations for item in observation]
        levels, inverse = np.unique(np.asarray(flat, dtype=object), return_inverse=True)
        return len(observations), indices, levels, inverse

    def sampler(self, seed: Optional[int] = None) -> "BernoulliSetSampler":
        """Create a repeatable inclusion sampler."""
        return BernoulliSetSampler(self, seed)

    def estimator(self) -> "BernoulliSetEstimator":
        """Create an estimator retaining metadata and prior."""
        return BernoulliSetEstimator(self.name, self.prior or default_prior)

    def dist_to_encoder(self) -> "BernoulliSetDataEncoder":
        """Create the set-like sequence encoder."""
        return BernoulliSetDataEncoder()


class BernoulliSetSampler(DistributionSampler[Iterable[Label]]):
    """Sample included labels in probability-map insertion order."""

    def sample(self, size: Optional[int] = None) -> Any:
        """Draw one label list or a list of label lists."""
        if size is None:
            return [
                key
                for key, probability in self.dist.pmap.items()
                if self.rng.rand() <= probability % 1
            ]
        result: list[list[Label]] = [[] for _ in range(size)]
        for key, probability in self.dist.pmap.items():
            for index in np.flatnonzero(self.rng.rand(size) <= probability % 1):
                result[int(index)].append(key)
        return result


class BernoulliSetAccumulator(
    SequenceEncodableAccumulator[Iterable[Label], SetSuffStat, SetEncoded]
):
    """Accumulate weighted label occurrences and observation weight.

    The sufficient statistic is ``(label_occurrences, observation_weight)``.
    Every repeated occurrence adds its observation's weight again; observations
    are not converted to mathematical sets before accumulation.
    """

    def __init__(self) -> None:
        """Initialize empty occurrence and observation counts."""
        self.pmap: defaultdict[Label, float] = defaultdict(float)
        self.tot_sum = 0.0

    def update(
        self, x: Iterable[Label], weight: float, estimate: Optional[Model]
    ) -> None:
        """Add one weighted set-like observation."""
        del estimate
        for value in x:
            self.pmap[value] += weight
        self.tot_sum += weight

    def initialize(
        self, x: Iterable[Label], weight: float, rng: np.random.RandomState
    ) -> None:
        """Initialize from one observation."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self, x: SetEncoded, weights: Array, rng: np.random.RandomState
    ) -> None:
        """Initialize from encoded observations."""
        del rng
        self.seq_update(x, weights, None)

    def seq_update(
        self, x: SetEncoded, weights: Array, estimate: Optional[Model]
    ) -> None:
        """Add weighted label occurrences from encoded observations."""
        del estimate
        _count, indices, levels, inverse = x
        counts = np.bincount(inverse, weights=weights[indices], minlength=len(levels))
        for index, value in enumerate(counts):
            self.pmap[levels[index]] += float(value)
        self.tot_sum += float(weights.sum())

    def combine(self, suff_stat: SetSuffStat) -> "BernoulliSetAccumulator":
        """Merge occurrence counts and total observation weight."""
        for key, value in suff_stat[0].items():
            self.pmap[key] += value
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> SetSuffStat:
        """Return occurrence counts and total observation weight."""
        return dict(self.pmap), self.tot_sum

    def from_value(self, x: SetSuffStat) -> "BernoulliSetAccumulator":
        """Restore occurrence counts and total observation weight."""
        self.pmap = defaultdict(float, x[0])
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged because this model has no key."""
        del stats_dict

    def key_replace(self, stats_dict: MutableMapping[str, Any]) -> None:
        """Leave shared statistics unchanged because this model has no key."""
        del stats_dict

    def acc_to_encoder(self) -> "BernoulliSetDataEncoder":
        """Create the corresponding encoder."""
        return BernoulliSetDataEncoder()


class BernoulliSetAccumulatorFactory(
    StatisticAccumulatorFactory[Iterable[Label], SetSuffStat, SetEncoded]
):
    """Create empty Bernoulli-set accumulators."""

    def make(self) -> BernoulliSetAccumulator:
        """Create an empty accumulator."""
        return BernoulliSetAccumulator()


class BernoulliSetEstimator(
    ParameterEstimator[Iterable[Label], dict[Label, float], SetEncoded, SetSuffStat]
):
    """Estimate per-label inclusion probabilities.

    A beta prior yields per-label posterior modes. A finite mixture of beta
    distributions selects the highest-posterior prior component separately for
    each label before taking its mode. Other priors use empirical inclusion
    frequencies. Results retain the legacy negative complement encoding when
    appropriate.

    Args:
        name: Optional identifier copied to the estimated distribution.
        prior: Common prior for every label's inclusion probability.
        keys: Compatibility placeholder; this estimator does not share
            sufficient statistics by key.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        prior: Model = default_prior,
        keys: tuple[None] = (None,),
    ) -> None:
        """Initialize estimator metadata and common inclusion prior."""
        self.name = name
        self.prior = prior
        self.keys = keys

    def accumulator_factory(self) -> BernoulliSetAccumulatorFactory:
        """Create a compatible accumulator factory."""
        return BernoulliSetAccumulatorFactory()

    def get_prior(self) -> Model:
        """Return the common inclusion-probability prior."""
        return self.prior

    def set_prior(self, prior: Model) -> None:
        """Replace the common inclusion-probability prior."""
        self.prior = prior

    def estimate(
        self, *args: Any
    ) -> BernoulliSetDistribution:  # pylint: disable=arguments-differ
        """Estimate from label-occurrence and observation counts."""
        occurrence_counts, total_count = args[-1]
        if isinstance(self.prior, BetaDistribution):
            probabilities = bernoulli_beta_posterior_mode(
                occurrence_counts, total_count, self.prior.get_parameters()
            )
        elif isinstance(self.prior, MixtureDistribution) and all(
            isinstance(component, BetaDistribution)
            for component in self.prior.components
        ):
            beta_parameters = np.asarray(
                [component.get_parameters() for component in self.prior.components]
            )
            probabilities = bernoulli_betamix_posterior_mode(
                occurrence_counts, total_count, self.prior.w, beta_parameters
            )
        else:
            probabilities = {
                key: (
                    -(total_count - value) / total_count
                    if value * 2 > total_count
                    else value / total_count
                )
                for key, value in occurrence_counts.items()
            }
        return BernoulliSetDistribution(probabilities, self.name, self.prior)


class BernoulliSetDataEncoder(DataSequenceEncoder[Iterable[Label], SetEncoded]):
    """Encode set-like observations into a flattened four-part tuple.

    For ``n`` observations the payload is ``(n, row_indices, levels,
    inverse)``. ``row_indices`` maps each flattened occurrence to its source
    observation, ``levels`` contains the distinct labels, and ``inverse`` maps
    occurrences to levels. Repeated labels remain repeated occurrences.
    """

    def __str__(self) -> str:
        """Return the encoder name."""
        return "BernoulliSetDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether the other object is the stateless set encoder."""
        return isinstance(other, BernoulliSetDataEncoder)

    def seq_encode(self, x: Iterable[Iterable[Label]]) -> "BernoulliSetEncodedData":
        """Flatten observations while retaining row and level indices."""
        observations = tuple(tuple(value) for value in x)
        indices = np.asarray(
            [index for index, value in enumerate(observations) for _ in value],
            dtype=int,
        )
        flat = [item for observation in observations for item in observation]
        levels, inverse = np.unique(np.asarray(flat, dtype=object), return_inverse=True)
        return BernoulliSetEncodedData((len(observations), indices, levels, inverse))


class BernoulliSetEncodedData(  # pylint: disable=too-few-public-methods
    EncodedDataSequence[SetEncoded]
):
    """Contain the stable flattened set-like encoding."""


def bernoulli_beta_posterior_mode(
    obs_cnt: dict[Label, float], tot_cnt: float, beta_params: tuple[float, float]
) -> dict[Label, float]:
    """Return legacy encoded beta-posterior modes."""
    result: dict[Label, float] = {}
    for key, value in obs_cnt.items():
        alpha = beta_params[0] - 1 + value
        beta = beta_params[1] - 1 - value + tot_cnt
        if alpha > beta > 0:
            probability = -beta / (alpha + beta)
        elif beta > alpha > 0:
            probability = (alpha - 1) / (alpha + beta - 2)
        elif alpha == 0 and beta == 0:
            probability = 0.5
        elif beta > alpha:
            probability = 0.0
        else:
            probability = 1.0
        result[key] = probability
    return result


def bernoulli_betamix_posterior_mode(
    obs_cnt: dict[Label, float],
    tot_cnt: float,
    weights: Array,
    beta_params: Array,
) -> dict[Label, float]:
    """Return legacy encoded modes under a beta-mixture prior."""
    constants = (
        -gammaln(beta_params).sum(axis=1)
        + gammaln(beta_params.sum(axis=1))
        - gammaln(beta_params.sum(axis=1) + tot_cnt)
    )
    result: dict[Label, float] = {}
    for key, value in obs_cnt.items():
        scores = (
            np.log(weights)
            + gammaln(beta_params[:, 0] + value)
            + gammaln(beta_params[:, 1] + tot_cnt - value)
            + constants
        )
        component = int(scores.argmax())
        alpha = beta_params[component, 0] - 1 + value
        beta = beta_params[component, 1] - 1 - value + tot_cnt
        if alpha > beta > 0:
            probability = -beta / (alpha + beta)
        elif beta > alpha > 0:
            probability = (alpha - 1) / (alpha + beta - 2)
        elif alpha == 0 and beta == 0:
            probability = 0.5
        elif beta > alpha:
            probability = 0.0
        else:
            probability = 1.0
        result[key] = float(probability)
    return result
