"""Model observations with optional priors on their mixture labels.

An observation is ``(datum, prior)``. ``prior`` is either ``None`` or a sequence of
``(component_index, probability)`` pairs. With no prior, the latent component ``Z``
has the global mixture weights. With a prior, only listed components are eligible and
their supplied probabilities reweight the global weights before normalization.
Responsibilities combine this conditional label information with component
likelihoods. The sampler returns bare data because label priors are conditioning input,
not generated observations.

Unlike :mod:`heterogeneous_mixture`, all components use one encoder. Unlike
:mod:`hmixture`, each observation has one latent component rather than sequence- and
item-level variables. Unlike :mod:`jmixture`, the second tuple entry is supervision,
not a second modeled view.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.random import RandomState

import dmx.utils.vector as vec
from dmx.arithmetic import exp, maxrandint
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

T0 = TypeVar("T0")  # Data type
T = Sequence[Tuple[T0, Optional[Sequence[Tuple[int, float]]]]]

E0 = TypeVar("E0")  # Encoded data type components
E1 = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # Encoded prior type
E = Tuple[int, EncodedDataSequence, Tuple[E1, np.ndarray, np.ndarray], T]

SS0 = TypeVar("SS0")  # Suff-stat type from components


class SemiSupervisedMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Define a mixture conditioned by optional per-observation label priors.

    Attributes:
        components (Sequence[SequenceEncodableProbabilityDistribution]): Mixture
            components.
        num_components (int): Number of mixture components.
        zw (np.ndarray): Bool numpy array, True where weights are 0.0.
        log_w (np.ndarray): Log of weights. Set to -np.inf where weights are 0.
        w (np.ndarray): Mixture weights. Should sum to 1.0.
        name (Optional[str]): Set name for object.

    """

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        w: Union[List[float], np.ndarray],
        name: Optional[str] = None,
    ) -> None:
        """Initialize a semi-supervised mixture distribution.

        Args:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Mixture
                components.
            w ( Union[List[float], np.ndarray]): Mixture weights. Should sum to 1.0
            name (Optional[str]): Set name for object.

        """
        super().__init__()
        self.components = components
        self.num_components = len(components)
        self.w = np.asarray(w)
        self.zw = self.w == 0.0
        self.log_w = np.log(self.w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.name = name

    def __str__(self) -> str:
        """Return a string representation of the distribution."""
        components = ",".join([str(u) for u in self.components])
        weights = ",".join(map(str, self.w))
        name = ",".join(repr(self.name))
        return (
            f"SemiSupervisedMixtureDistribution([{components}], "
            f"[{weights}], name={name})"
        )

    def density(self, x: Tuple[T0, Optional[Sequence[Tuple[int, float]]]]) -> float:
        """Evaluate density for a datum and optional label prior."""
        return float(exp(self.log_density(x)))

    def log_density(self, x: Tuple[T0, Optional[Sequence[Tuple[int, float]]]]) -> float:
        """Evaluate log-density for a datum and optional label prior."""
        datum, prior = x
        if prior is None:
            return float(
                vec.log_sum(
                    np.asarray([u.log_density(datum) for u in self.components])
                    + self.log_w
                )
            )
        w_loc = np.zeros(self.num_components)
        h_loc = np.zeros(self.num_components, dtype=bool)
        i_loc = np.zeros(self.num_components, dtype=int)

        for idx, val in prior:
            w_loc[idx] += np.log(val)
            h_loc[idx] = True
            i_loc[idx] = idx

        w_loc[h_loc] += self.log_w[h_loc]
        w_loc = vec.log_posterior(w_loc[h_loc])

        return float(
            vec.log_sum(
                np.asarray(
                    [
                        self.components[i].log_density(datum)
                        for i in np.flatnonzero(h_loc)
                    ]
                )
                + w_loc
            )
        )

    def posterior(
        self, x: Tuple[T0, Optional[Sequence[Tuple[int, float]]]]
    ) -> np.ndarray:
        """Compute component responsibilities subject to an optional label prior."""
        datum, prior = x

        if prior is None:
            rv = vec.posterior(
                np.asarray([u.log_density(datum) for u in self.components]) + self.log_w
            )
        else:

            w_loc = np.zeros(self.num_components)
            h_loc = np.zeros(self.num_components, dtype=bool)

            for idx, val in prior:
                w_loc[idx] += np.log(val)
                h_loc[idx] = True

            w_loc[h_loc] += self.log_w[h_loc]
            for i in np.flatnonzero(h_loc):
                w_loc[i] += self.components[i].log_density(datum)

            w_loc[h_loc] = vec.posterior(w_loc[h_loc])
            rv = w_loc

        return np.asarray(rv, dtype=float)

    def seq_log_density(
        self, x: "SemiSupervisedMixtureEncodedDataSequence"
    ) -> np.ndarray:
        """Evaluate log-densities for encoded conditional observations."""
        if not isinstance(x, SemiSupervisedMixtureEncodedDataSequence):
            raise TypeError(
                "Requires SemiSupervisedMixtureEncodedDataSequence for `seq_` calls."
            )

        sz, enc_data, (enc_prior, _enc_prior_sum, enc_prior_flag), _ = x.data
        ll_mat = np.zeros((sz, self.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(
            enc_prior[0], weights=(enc_prior[2] * self.w[enc_prior[1]]), minlength=sz
        )
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = self.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + self.log_w[enc_prior[1]]

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] += self.components[i].seq_log_density(enc_data)
                ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)
        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max

            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return np.asarray(ll_sum.flatten(), dtype=float)

        ll_mat = ll_mat[good_rows, :]
        ll_max = ll_max[good_rows]

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        np.log(ll_sum, out=ll_sum)
        ll_sum += ll_max
        rv = np.zeros(good_rows.shape, dtype=float)
        rv[good_rows] = ll_sum.flatten()
        rv[~good_rows] = -np.inf

        return np.asarray(rv, dtype=float)

    def seq_posterior(
        self, x: "SemiSupervisedMixtureEncodedDataSequence"
    ) -> np.ndarray:
        """Evaluate responsibilities for encoded conditional observations."""
        if not isinstance(x, SemiSupervisedMixtureEncodedDataSequence):
            raise TypeError(
                "Requires SemiSupervisedMixtureEncodedDataSequence for `seq_` calls."
            )

        sz, enc_data, (enc_prior, _enc_prior_sum, enc_prior_flag), _ = x.data
        ll_mat = np.zeros((sz, self.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(
            enc_prior[0], weights=(enc_prior[2] * self.w[enc_prior[1]]), minlength=sz
        )
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = self.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + self.log_w[enc_prior[1]]

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] += self.components[i].seq_log_density(enc_data)
                ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w
        ll_max[bad_rows] = np.max(self.log_w)

        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        ll_mat /= ll_sum

        return np.asarray(ll_mat, dtype=float)

    def sampler(self, seed: Optional[int] = None) -> "SemiSupervisedMixtureSampler":
        """Return a sampler that generates unlabeled component data."""
        return SemiSupervisedMixtureSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "SemiSupervisedMixtureEstimator":
        """Return an estimator for mixture weights and shared-type components."""
        if pseudo_count is not None:
            return SemiSupervisedMixtureEstimator(
                [
                    u.estimator(pseudo_count=1.0 / self.num_components)
                    for u in self.components
                ],
                pseudo_count=pseudo_count,
                name=self.name,
            )
        return SemiSupervisedMixtureEstimator(
            [u.estimator() for u in self.components], name=self.name
        )

    def dist_to_encoder(self) -> "SemiSupervisedMixtureDataEncoder":
        """Return an encoder for data and sparse label priors."""
        return SemiSupervisedMixtureDataEncoder(
            encoder=self.components[0].dist_to_encoder()
        )


class SemiSupervisedMixtureSampler(DistributionSampler):
    """Sample bare data from the mixture, without conditioning priors."""

    def __init__(
        self, dist: SemiSupervisedMixtureDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize a semi-supervised mixture sampler."""
        super().__init__(dist, seed)
        rng_loc = RandomState(seed)
        self.rng = RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [
            d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components
        ]

    def sample(self, size: Optional[int] = None) -> Union[Sequence[Any], Any]:
        """Draw one datum or a collection of data."""
        comp_state = self.rng.choice(
            range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w
        )

        if size is None:
            return self.comp_samplers[comp_state].sample()
        return [self.comp_samplers[i].sample() for i in comp_state]


class SemiSupervisedMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate label-conditioned component responsibilities and statistics."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
        name: Optional[str] = None,
    ) -> None:
        """Initialize a semi-supervised mixture accumulator."""
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key, self.comp_key = keys if keys is not None else (None, None)
        self.name = name

        self._init_rng = False
        self._acc_rng: Optional[List[RandomState]] = None
        self._w_rng: Optional[RandomState] = None
        self._prior_rng: Optional[RandomState] = None

    def update(
        self,
        x: Tuple[T0, Optional[Sequence[Tuple[int, float]]]],
        weight: float,
        estimate: SemiSupervisedMixtureDistribution,
    ) -> None:
        """Update statistics for one conditionally labeled observation."""
        likelihood = estimate.posterior(x)
        datum, _prior = x

        self.comp_counts += likelihood * weight

        for i in range(self.num_components):
            self.accumulators[i].update(
                datum, likelihood[i] * weight, estimate.components[i]
            )

    def _rng_initialize(self, rng: RandomState) -> None:
        if not self._init_rng:

            self._w_rng = RandomState(seed=rng.randint(maxrandint))
            self._prior_rng = RandomState(seed=rng.randint(maxrandint))

            seeds = rng.randint(maxrandint, size=self.num_components)
            self._acc_rng = [
                RandomState(seed=seeds[i]) for i in range(self.num_components)
            ]

            self._init_rng = True

    def initialize(
        self,
        x: Tuple[T0, Optional[Sequence[Tuple[int, float]]]],
        weight: float,
        rng: RandomState,
    ) -> None:
        """Initialize component statistics using priors or a random assignment."""
        datum, prior = x

        if not self._init_rng:
            self._rng_initialize(rng)

        assert self._prior_rng is not None
        assert self._acc_rng is not None

        if prior is None:
            idx = self._prior_rng.choice(self.num_components)
            wc0 = 0.001
            wc1 = wc0 / max((float(self.num_components) - 1.0), 1.0)
            wc2 = 1.0 - wc0

            for i in range(self.num_components):
                w = weight * wc2 if i == idx else wc1
                self.accumulators[i].initialize(datum, w, self._acc_rng[i])
                self.comp_counts[i] += w

        else:
            for i, w in prior:
                ww = weight * w
                self.accumulators[i].initialize(datum, ww, self._acc_rng[i])
                self.comp_counts[i] += ww

    def seq_initialize(
        self,
        x: "SemiSupervisedMixtureEncodedDataSequence",
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Initialize encoded observations through the scalar initialization path."""
        _sz, _enc_data, (_enc_prior, _enc_prior_sum, _enc_prior_flag), xx = x.data
        for i, xx_i in enumerate(xx):
            self.initialize(xx_i, weights[i], rng=rng)

    def seq_update(
        self,
        x: "SemiSupervisedMixtureEncodedDataSequence",
        weights: np.ndarray,
        estimate: SemiSupervisedMixtureDistribution,
    ) -> None:
        """Accumulate posterior-weighted statistics for encoded observations."""
        sz, enc_data, (enc_prior, _enc_prior_sum, enc_prior_flag), _ = x.data
        ll_mat = np.zeros((sz, estimate.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(
            enc_prior[0],
            weights=(enc_prior[2] * estimate.w[enc_prior[1]]),
            minlength=sz,
        )
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = estimate.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + estimate.log_w[enc_prior[1]]

        for i in range(self.num_components):
            ll_mat[:, i] += estimate.components[i].seq_log_density(enc_data)
            ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = estimate.log_w
        ll_max[bad_rows] = np.max(estimate.log_w)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        ll_mat /= ll_sum

        for i in range(self.num_components):
            w_loc = ll_mat[:, i] * weights
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(enc_data, w_loc, estimate.components[i])

    def combine(
        self, suff_stat: Tuple[np.ndarray, Tuple[SS0, ...]]
    ) -> "SemiSupervisedMixtureEstimatorAccumulator":
        """Merge another semi-supervised mixture sufficient statistic."""
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        """Return component counts and child sufficient statistics."""
        return self.comp_counts, tuple(u.value() for u in self.accumulators)

    def from_value(
        self, x: Tuple[np.ndarray, Tuple[SS0, ...]]
    ) -> "SemiSupervisedMixtureEstimatorAccumulator":
        """Replace accumulated statistics from a serialized value."""
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge keyed component counts and child statistics."""
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i, acc_i in enumerate(acc):
                    acc_i = acc_i.combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from matching keyed values."""
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "SemiSupervisedMixtureDataEncoder":
        """Return the encoder required by the child accumulators."""
        return SemiSupervisedMixtureDataEncoder(
            encoder=self.accumulators[0].acc_to_encoder()
        )


class SemiSupervisedMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Create semi-supervised mixture accumulators."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        dim: int,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
        name: Optional[str] = None,
    ):
        """Initialize a semi-supervised mixture accumulator factory."""
        self.factories = factories
        self.dim = dim
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def make(self) -> "SemiSupervisedMixtureEstimatorAccumulator":
        """Create a semi-supervised mixture accumulator."""
        return SemiSupervisedMixtureEstimatorAccumulator(
            [self.factories[i].make() for i in range(self.dim)], self.keys, self.name
        )


class SemiSupervisedMixtureEstimator(ParameterEstimator):
    """Estimate a semi-supervised mixture from sufficient statistics.

    Label priors affect responsibilities during accumulation; estimation then uses
    the resulting component-count vector and one child statistic per component.
    ``keys[0]`` shares counts and ``keys[1]`` shares child statistics positionally.

    Attributes:
        estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimators
            objects for the components of
            the mixture. All must be of the same class compatible with data type T.
        suff_stat (Optional[np.ndarray]): Mixture weights for components obtained from
            prev estimation or for
            regularization.
        pseudo_count (Optional[float]): Re-weight sufficient statistics, i.e. penalize
            sufficient statistics.
        keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for the weights
            and components.
        name (Optional[str]): Set name for object.

    """

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        suff_stat: Optional[np.ndarray] = None,
        pseudo_count: Optional[float] = None,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
        name: Optional[str] = None,
    ) -> None:
        """Initialize a semi-supervised mixture estimator.

        Args:
            estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimators
                objects for the components of
                the mixture. All must be of the same class compatible with data type T.
            suff_stat (Optional[np.ndarray]): Mixture weights for components obtained
                from prev estimation or for
                regularization.
            pseudo_count (Optional[float]): Re-weight sufficient statistics, i.e.
                penalize sufficient statistics.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for the
                weights and components.
            name (Optional[str]): Set name for object.

        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def accumulator_factory(self) -> "SemiSupervisedMixtureEstimatorAccumulatorFactory":
        """Return a factory sharing child statistics by component index."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return SemiSupervisedMixtureEstimatorAccumulatorFactory(
            est_factories, self.num_components, self.keys, self.name
        )

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, Tuple[SS0, ...]]
    ) -> "SemiSupervisedMixtureDistribution":
        """Estimate component distributions and mixture weights."""
        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [
            self.estimators[i].estimate(counts[i], comp_suff_stats[i])
            for i in range(num_components)
        ]

        if self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / num_components
            w = counts + p
            w /= w.sum()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            w = (counts + self.suff_stat * self.pseudo_count) / (
                counts.sum() + self.pseudo_count
            )
        else:

            nobs_loc = counts.sum()

            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        return SemiSupervisedMixtureDistribution(components, w)


class SemiSupervisedMixtureDataEncoder(DataSequenceEncoder):
    """Encode data and sparse optional label priors.

    For batch size ``B`` the result is ``(B, enc_data, prior_info, raw_data)``.
    ``prior_info`` is ``(prior_mat, prior_sum, has_prior)``. ``prior_mat`` contains
    four aligned one-dimensional arrays: observation indices, component indices,
    prior probabilities, and log probabilities. ``prior_sum`` and ``has_prior`` have
    shape ``(B,)``. ``raw_data`` is retained for scalar initialization.
    """

    def __init__(self, encoder: DataSequenceEncoder):
        """Initialize a semi-supervised mixture data encoder."""
        self.encoder = encoder

    def __str__(self) -> str:
        """Return a string representation of the encoder."""
        return "SemiSupervisedMixtureDataEncoder(encoder=" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether the child data encoders are equal."""
        if isinstance(other, SemiSupervisedMixtureDataEncoder):
            return self.encoder == other.encoder
        return False

    def seq_encode(
        self, x: Sequence[Tuple[T0, Optional[Sequence[Tuple[int, float]]]]]
    ) -> "SemiSupervisedMixtureEncodedDataSequence":
        """Encode a batch of data and optional sparse label priors."""
        prior_comp: List[int] = []
        prior_idx: List[int] = []
        prior_val: List[float] = []
        data: List[T0] = []

        for i, xi in enumerate(x):
            datum, prior = xi
            data.append(datum)
            if prior is not None:
                for prior_entry in prior:
                    prior_idx.append(i)
                    prior_comp.append(prior_entry[0])
                    prior_val.append(prior_entry[1])

        prior_comp_arr = np.asarray(prior_comp, dtype=int)
        prior_idx_arr = np.asarray(prior_idx, dtype=int)
        prior_val_arr = np.asarray(prior_val, dtype=float)

        prior_mat = (
            prior_idx_arr,
            prior_comp_arr,
            prior_val_arr,
            np.log(prior_val_arr),
        )

        prior_sum = np.bincount(prior_idx_arr, weights=prior_val_arr, minlength=len(x))
        has_prior = prior_sum != 0

        rv_enc = (
            len(x),
            self.encoder.seq_encode(data),
            (prior_mat, prior_sum, has_prior),
            x,
        )
        return SemiSupervisedMixtureEncodedDataSequence(data=rv_enc)


class SemiSupervisedMixtureEncodedDataSequence(EncodedDataSequence):
    """Store encoded data, sparse label-prior arrays, and original observations.

    Notes:
        E1 = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        E = Tuple[int, EncodedDataSequence, Tuple[E1, np.ndarray, np.ndarray], T]

    Attributes:
        data (E): Encoded sequence of semi-supervised mixture observations.


    """

    def __init__(self, data: E):
        """Initialize an encoded semi-supervised mixture batch.

        Args:
            data (E): Encoded sequence of semi-supervised mixture observations.


        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation of the encoded batch."""
        return f"SemiSupervisedMixtureEncodedDataSequence(data={self.data})"
