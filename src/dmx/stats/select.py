"""Create, estimate, and sample from a select distribution.

Defines the SelectDistribution, SelectSampler, SelectAccumulatorFactory,
SelectAccumulator,
SelectEstimator, and the SelectDataEncoder classes for use with dmx-learn.

The SelectDistribution samples from a set of SequenceEncodableProbabilityDistribution
objects. The a choice function
maps an observation a distribution from the set of distributions.

"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import maxint, maxrandint, zero
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class SelectDistribution(SequenceEncodableProbabilityDistribution):
    """Route each observation to one of several child distributions.

    ``choice_function(x)`` must return a valid child index. Scoring uses only that
    child, so the support is the union of the routed pieces of the child supports.
    This piecewise density is normalized only when the integrals of the children on
    their routed regions sum to one; normalization of every child alone is not
    sufficient.

    Encoding and estimation partition observations with the same choice function.
    Vectorized operations consume populated route groups positionally; agreement
    with scalar operations therefore requires those groups to occur in child-index
    order starting at zero. This is stricter than scalar scoring, which accepts any
    valid child index directly.
    Sampling is deliberately different: it returns one draw from every child (or a
    zip of sized child draws), because no distribution over route indices exists.
    It therefore does not generate ordinary observations for this routed density.
    """

    def __init__(
        self,
        dists: Sequence[SequenceEncodableProbabilityDistribution],
        choice_function: Callable[[Any], int],
    ) -> None:
        """Initialize a routed distribution.

        Args:
            dists: Child distributions indexed by ``choice_function`` results.
            choice_function: Deterministic mapping from an observation to a valid
                integer position in ``dists``.
        """
        super().__init__()
        self.dists = dists
        self.choice_function = choice_function
        self.count = len(dists)

    def __str__(self) -> str:
        """Return a representation of the child distributions."""
        return "SelectDistribution(" + ",".join([str(u) for u in self.dists]) + ")"

    def density(self, x: Any) -> float:
        """Evaluate the density of the child selected for ``x``."""
        idx = self.choice_function(x)
        return self.dists[idx].density(x)

    def log_density(self, x: Any) -> float:
        """Evaluate the log density of the child selected for ``x``."""
        idx = self.choice_function(x)
        return self.dists[idx].log_density(x)

    def seq_log_density(self, x: "SelectEncodedDataSequence") -> np.ndarray:
        """Score each encoded route group and restore original observation order.

        Raises:
            TypeError: If ``x`` was not produced by a select encoder.
        """
        if not isinstance(x, SelectEncodedDataSequence):
            raise TypeError("Requires SelectEncodedDataSequence for `seq_` calls.")

        xi, idx, enc_tuple = x.data
        rv = np.zeros(len(xi))
        for i, _ in enumerate(idx):
            rv[xi[i]] = self.dists[i].seq_log_density(enc_tuple[i])
        return rv

    def sampler(self, seed: Optional[int] = None) -> "SelectSampler":
        """Create a sampler that draws from every child."""
        return SelectSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "SelectEstimator":
        """Create one child estimator per route."""
        return SelectEstimator(
            [d.estimator(pseudo_count=pseudo_count) for d in self.dists],
            self.choice_function,
        )

    def dist_to_encoder(self) -> "SelectDataEncoder":
        """Create an encoder that partitions observations by route."""
        encoders = [d.dist_to_encoder() for d in self.dists]
        return SelectDataEncoder(
            encoders=encoders, choice_function=self.choice_function
        )


class SelectSampler(DistributionSampler):
    """Draw independently from every child rather than selecting a route."""

    def __init__(self, dist: SelectDistribution, seed: Optional[int] = None) -> None:
        """Initialize independently seeded child samplers."""
        super().__init__(dist, seed)
        self.dist_samplers = [
            d.sampler(seed=self.rng.randint(maxint)) for d in dist.dists
        ]

    def sample(self, size: Optional[int] = None) -> Any:
        """Return a tuple of scalar child draws or a zip of sized child draws."""
        if size is None:
            return tuple(d.sample(size=size) for d in self.dist_samplers)
        return zip(*[d.sample(size=size) for d in self.dist_samplers])


class SelectEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Maintain route weights and one sufficient statistic per child.

    Scalar and encoded observations are sent only to the accumulator selected by
    ``choice_function``. Scalar updates pass ``None`` as the child estimate, whereas
    encoded updates pass the corresponding child distribution. The public value is
    a one-shot zip iterator of
    ``(route_weight, child_statistic)`` pairs. Key merge and replacement are no-ops;
    child key contracts are not traversed by this wrapper.
    """

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        choice_function: Callable[[Any], int],
    ) -> None:
        """Initialize routed child accumulators and zero route weights."""
        self.accumulators = accumulators
        self.choice_function = choice_function
        self.weights = [zero] * len(accumulators)
        self.count = len(accumulators)

        self._rng_init = False
        self._acc_rng: Optional[List[RandomState]] = None

    def update(
        self, x: Any, weight: float, estimate: Optional[SelectDistribution]
    ) -> None:
        """Update the selected child and its total observation weight."""
        # cf  = pickle.loads(self.choice_function)
        idx = self.choice_function(x)
        self.accumulators[idx].update(x, weight, None)
        self.weights[idx] += weight

    def _rng_initialize(self, rng: RandomState) -> None:
        """Create an independent random state for every child accumulator."""
        self._acc_rng = [
            RandomState(seed=rng.randint(0, maxrandint)) for xx in range(self.count)
        ]
        self._rng_init = True

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initialize the selected child and add its observation weight."""
        if not self._rng_init:
            self._rng_initialize(rng)

        assert self._acc_rng is not None
        idx = self.choice_function(x)
        self.accumulators[idx].initialize(x, weight, self._acc_rng[idx])
        self.weights[idx] += weight

    def seq_update(
        self,
        x: "SelectEncodedDataSequence",
        weights: np.ndarray,
        estimate: SelectDistribution,
    ) -> None:
        """Update every encoded route group with its corresponding estimate."""
        xi, idx, enc_tuple = x.data
        for i, _ in enumerate(idx):
            w = weights[xi[i]]
            self.accumulators[i].seq_update(enc_tuple[i], w, estimate.dists[i])
            self.weights[i] += np.sum(w)

    def seq_initialize(
        self, x: "SelectEncodedDataSequence", weights: np.ndarray, rng: RandomState
    ) -> None:
        """Initialize every encoded route group with independent randomness."""
        if not self._rng_init:
            self._rng_initialize(rng)

        assert self._acc_rng is not None
        xi, idx, enc_tuple = x.data
        for i, _ in enumerate(idx):
            w = weights[xi[i]]
            self.accumulators[i].seq_initialize(enc_tuple[i], w, self._acc_rng[i])
            self.weights[i] += np.sum(w)

    def combine(self, suff_stat: Any) -> "SelectEstimatorAccumulator":
        """Combine route weights and child statistics positionally."""
        for i in range(0, self.count):
            self.weights[i] += suff_stat[i][0]
            self.accumulators[i].combine(suff_stat[i][1])

        return self

    def value(self) -> Any:
        """Return a zip iterator of route-weight and child-statistic pairs."""
        return zip(self.weights, [x.value() for x in self.accumulators])

    def from_value(self, x: Any) -> "SelectEstimatorAccumulator":
        """Restore route weights and child statistics from an iterable."""
        for i, u in enumerate(x):
            self.weights[i] = u[0]
            self.accumulators[i].from_value(u[1])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Leave the shared-statistics dictionary unchanged."""
        pass

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Leave all routed child statistics unchanged."""
        pass

    def acc_to_encoder(self) -> "SelectDataEncoder":
        """Build a routed encoder from the child accumulator encoders."""
        encoders = [acc.acc_to_encoder() for acc in self.accumulators]
        return SelectDataEncoder(
            encoders=encoders, choice_function=self.choice_function
        )


class SelectEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Create a routed accumulator from child estimators."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator[Any]],
        choice_function: Callable[[Any], int],
    ) -> None:
        """Initialize a routed accumulator factory."""
        self.estimators = estimators
        self.choice_function = choice_function

    def make(self) -> "SelectEstimatorAccumulator":
        """Create one accumulator for every child estimator."""
        return SelectEstimatorAccumulator(
            [x.accumulator_factory().make() for x in self.estimators],
            self.choice_function,
        )


class SelectEstimator(ParameterEstimator):
    """Estimate each routed child independently from its grouped statistic."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator[Any]],
        choice_function: Callable[[Any], int],
    ) -> None:
        """Initialize child estimators and the shared routing function."""
        self.estimators = estimators
        self.choice_function = choice_function
        self.count = len(estimators)

    def accumulator_factory(self) -> "SelectEstimatorAccumulatorFactory":
        """Create a factory preserving child order and routing."""
        return SelectEstimatorAccumulatorFactory(self.estimators, self.choice_function)

    def estimate(self, nobs: Optional[float], suff_stat: Any) -> "SelectDistribution":
        """Fit each child using its route weight as that child's observation count."""
        return SelectDistribution(
            [est.estimate(ss[0], ss[1]) for est, ss in zip(self.estimators, suff_stat)],
            self.choice_function,
        )


class SelectDataEncoder(DataSequenceEncoder):
    """Partition observations by route and invoke the matching child encoders.

    The encoded representation stores only routes present in the input together
    with their original positions. Equality compares child encoders but, for
    historical compatibility, does not compare the choice functions. Vectorized
    consumers treat populated groups positionally, so batches must populate routes
    contiguously from child zero and in that order.
    """

    def __init__(
        self,
        encoders: Sequence[DataSequenceEncoder],
        choice_function: Callable[[Any], int],
    ) -> None:
        """Initialize child encoders and the routing function."""
        self.encoders = encoders
        self.choice_function = choice_function

    def __eq__(self, other: object) -> bool:
        """Compare child encoders while intentionally ignoring routing functions."""
        ### Asssumes that the choice functions of each encoder are equal
        if isinstance(other, SelectDataEncoder):
            for i, encoder in enumerate(self.encoders):
                if other.encoders[i] != encoder:
                    return False

            return True

        return False

    def seq_encode(self, x: Sequence[Any]) -> "SelectEncodedDataSequence":
        """Group observations by selected index and encode each populated group."""
        cnt = 0
        idx_dict: Dict[int, Tuple[List[int], List[Any]]] = {}

        for i, xx in enumerate(x):
            idx = self.choice_function(xx)
            if idx not in idx_dict:
                idx_dict[idx] = ([], [])
            idx_dict[idx][1].append(xx)
            idx_dict[idx][0].append(i)
            cnt += 1

        idx_keys = []
        idx_xi = []
        idx_enc_vals = []

        for keys, vals in idx_dict.items():
            idx_keys.append(keys)
            idx_xi.append(np.asarray(vals[0]))
            idx_enc_vals.append(self.encoders[keys].seq_encode(vals[1]))

        return SelectEncodedDataSequence(
            data=(tuple(idx_xi), tuple(idx_keys), tuple(idx_enc_vals))
        )


class SelectEncodedDataSequence(EncodedDataSequence):
    """Store route positions, route indices, and routed child encodings."""

    def __init__(
        self,
        data: Tuple[
            Tuple[np.ndarray, ...], Tuple[int, ...], Tuple[EncodedDataSequence, ...]
        ],
    ):
        """Initialize a grouped routed encoding.

        Args:
            data: Original-position arrays, route indices, and child encodings for
                the routes represented in the batch.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return the historical representation of the routed encoding."""
        return f"SelectEncodedDataSequence(data=f{self.data})"
