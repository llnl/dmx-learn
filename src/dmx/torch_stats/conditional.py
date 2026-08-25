"""Provide torch-backed distributions selected by a given observation value.

An observation is ``(given_value, dependent_value)``. ``dmap`` selects the
dependent child by ``given_value``; a default child handles unmapped values,
and a given distribution optionally models the selector. The encoder groups
``N`` observations by selector and stores one child encoding and integer index
tensor per group. Device movement propagates to every child. These contracts
match ``dmx.stats.conditional``; encoded scoring returns torch tensors.
"""

# pylint: disable=too-many-positional-arguments,duplicate-code

import math
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as tn
from torch import Generator

import dmx.torch_utils.vector as vec
from dmx.arithmetic import maxrandint
from dmx.torch_stats.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from dmx.torch_stats.pdist import (
    ConditionalSampler,
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)

T0 = TypeVar("T0")
T1 = TypeVar("T1")

E0 = TypeVar("E0")
E1 = TypeVar("E1")
E = Tuple[int, Tuple[T0, ...], Tuple[tn.Tensor, ...], Tuple[E0, ...], Optional[E1]]
SS0 = TypeVar("SS0")
SS1 = TypeVar("SS1")
SS2 = TypeVar("SS2")


class ConditionalDistribution(TorchProbabilityDistribution):
    """Model dependent values with children keyed by given values.

    A missing default gives unmapped selectors log mass ``-inf``. A null given
    distribution contributes zero log mass; otherwise the model represents
    the joint factorization ``p(dependent | given) p(given)``.
    """

    def __init__(
        self,
        dmap: Union[
            Dict[Any, TorchProbabilityDistribution], List[TorchProbabilityDistribution]
        ],
        default_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        given_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize mapped, default, and given child distributions.

        Args:
            dmap: Selector-to-child mapping. A list is keyed by consecutive integers.
            default_dist: Child used for selectors absent from ``dmap``.
            given_dist: Optional marginal distribution for selectors.
            keys: Optional key passed through the estimator contract.
            device: Device recorded by this wrapper.
        """
        super().__init__(device)
        if isinstance(dmap, list):
            dmap = dict(zip(range(len(dmap)), dmap))

        self.dmap = dmap
        self.default_dist = (
            default_dist if default_dist is not None else NullDistribution()
        )
        self.given_dist = given_dist if given_dist is not None else NullDistribution()

        self.has_default = not isinstance(self.default_dist, NullDistribution)
        self.has_given = not isinstance(self.given_dist, NullDistribution)
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-like representation of all child models."""
        s1 = repr(self.dmap)
        s2 = repr(self.default_dist)
        s3 = repr(self.given_dist)
        s4 = repr(self.keys)

        return (
            f"ConditionalDistribution({s1}, default_dist={s2}, "
            f"given_dist={s3}, keys={s4})"
        )

    def to(self, device: vec.DeviceLike) -> "ConditionalDistribution":
        """Move mapped, default, and given children to ``device`` in place."""
        target_device = self._resolve_device_arg(device)
        self._device = target_device
        for v in self.dmap.values():
            v.to(target_device)
        self.default_dist.to(target_device)
        self.given_dist.to(target_device)
        return self

    def density(self, x: Tuple[T0, T1]) -> float:
        """Evaluate the density of one ``(given, dependent)`` observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Tuple[T0, T1]) -> float:
        """Evaluate ``log p(dependent | given) + log p(given)``."""
        if self.has_default:
            rv = self.dmap.get(x[0], self.default_dist).log_density(x[1])
        else:
            if x[0] in self.dmap:
                rv = self.dmap[x[0]].log_density(x[1])
            else:
                return -np.inf

        rv += self.given_dist.log_density(x[0])

        return rv

    def seq_log_density(self, x: "ConditionalTorchEncodedSequence") -> tn.Tensor:
        """Scatter grouped child scores into a tensor of shape ``(N,)``."""
        if not isinstance(x, ConditionalTorchEncodedSequence):
            raise TypeError(
                "Requires ConditionalTorchEncodedSequence for `seq_` function calls."
            )

        sz, cond_vals, eobs_vals, idx_vals, given_enc = x.data
        rv = vec.zeros(sz, device=self._device)

        for i, cond_val in enumerate(cond_vals):
            idx = idx_vals[i].to(device=rv.device)
            if self.has_default:
                rv[idx] = self.dmap.get(cond_val, self.default_dist).seq_log_density(
                    eobs_vals[i]
                )
            elif cond_val in self.dmap:
                rv[idx] += self.dmap[cond_val].seq_log_density(eobs_vals[i])

        if self.has_given:
            rv += self.given_dist.seq_log_density(given_enc)

        return rv

    def sampler(self, seed: Optional[int] = None) -> "ConditionalDistributionSampler":
        """Create mapped, default, and given samplers from derived seeds."""
        return ConditionalDistributionSampler(self, seed=seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "ConditionalDistributionEstimator":
        """Create matching child estimators using ``pseudo_count``."""
        est_map = {k: v.estimator(pseudo_count) for k, v in self.dmap.items()}
        default_est = self.default_dist.estimator(pseudo_count)
        given_est = self.given_dist.estimator(pseudo_count)

        return ConditionalDistributionEstimator(
            estimator_map=est_map,
            default_estimator=default_est,
            given_estimator=given_est,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "ConditionalDistributionDataEncoder":
        """Create matching mapped, default, and given encoders."""
        encoder_map = {k: v.dist_to_encoder() for k, v in self.dmap.items()}
        default_encoder = (
            NullDataEncoder()
            if not self.has_default
            else self.default_dist.dist_to_encoder()
        )
        given_encoder = (
            NullDataEncoder()
            if not self.has_given
            else self.given_dist.dist_to_encoder()
        )

        return ConditionalDistributionDataEncoder(
            encoder_map=encoder_map,
            default_encoder=default_encoder,
            given_encoder=given_encoder,
        )


class ConditionalDistributionSampler(ConditionalSampler, DistributionSampler):
    """Sample complete pairs or dependent values conditional on a selector."""

    def __init__(
        self, dist: ConditionalDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize all child samplers with independently derived seeds."""
        self.dist = dist
        rng = np.random.RandomState(seed)

        loc_seed = rng.randint(0, maxrandint)

        self.has_default_sampler = dist.has_default
        self.default_sampler = dist.default_dist.sampler(loc_seed)

        loc_seed = rng.randint(0, maxrandint)
        self.given_sampler = dist.given_dist.sampler(loc_seed)
        self.has_given_sampler = isinstance(dist.given_dist, NullDistribution)

        self.samplers = {
            k: u.sampler(rng.randint(0, maxrandint)) for k, u in self.dist.dmap.items()
        }

    def single_sample(self) -> Tuple[Any, Any]:
        """Draw a selector from the given model and then its dependent value."""
        x0 = self.given_sampler.sample()
        if x0 in self.samplers:
            x1 = self.samplers[x0].sample()
        else:
            x1 = self.default_sampler.sample()
        return x0, x1

    def sample(
        self, size: Optional[int] = None
    ) -> Union[Tuple[Any, Any], List[Tuple[Any, Any]]]:
        """Draw one pair or a list of ``size`` independent pairs."""
        if size is None:
            return self.single_sample()
        return [self.single_sample() for _ in range(size)]

    def sample_given(self, x: T0) -> Any:
        """Draw a dependent value for selector ``x`` or from the default child."""
        if x in self.samplers:
            return self.samplers[x].sample()

        if self.has_default_sampler:
            return self.default_sampler.sample()

        raise RuntimeError("Conditional default distribution unspecified.")


class ConditionalDistributionAccumulator(TorchStatisticAccumulator):
    """Aggregate mapped, default, and given sufficient statistics.

    The statistic is ``(mapped_stats, default_stat, given_stat)``. Each mapped
    child receives only its selector group's weights; the given accumulator
    receives the complete weight tensor of shape ``(N,)``.
    """

    def __init__(
        self,
        accumulator_map: Dict[Any, TorchStatisticAccumulator],
        default_accumulator: Optional[TorchStatisticAccumulator] = NullAccumulator(),
        given_accumulator: Optional[TorchStatisticAccumulator] = NullAccumulator(),
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize mapped, default, and given child accumulators.

        Args:
            accumulator_map: Accumulators keyed by selector value.
            default_accumulator: Accumulator for unmapped selectors.
            given_accumulator: Accumulator for selector observations.
            keys: Metadata key retained by the wrapper; child keys control merges.
            device: Device metadata for encoded accumulation.
        """
        super().__init__(device)
        self.accumulator_map = accumulator_map
        self.default_accumulator = (
            default_accumulator
            if default_accumulator is not None
            else NullAccumulator()
        )
        self.given_accumulator = (
            given_accumulator if given_accumulator is not None else NullAccumulator()
        )

        self.has_default = not isinstance(default_accumulator, NullAccumulator)
        self.has_given = not isinstance(given_accumulator, NullAccumulator)
        self.key = keys

        #### seeds for intializers
        self._init_tng = False
        self._acc_tng: Optional[Dict[Any, Generator]] = None
        self._default_tng: Optional[Generator] = None
        self._given_tng: Optional[Generator] = None

    def _tng_initialize(self, tng: Generator) -> None:
        seed_rng = np.random.RandomState(int(tng.initial_seed()))
        seeds = seed_rng.randint(0, 2**31, size=(len(self.accumulator_map.keys()) + 2,))
        self._acc_tng = {}
        for i, acc_key in enumerate(self.accumulator_map.keys()):
            self._acc_tng[acc_key] = Generator().manual_seed(int(seeds[i + 2]))

        self._default_tng = Generator().manual_seed(int(seeds[0]))
        self._given_tng = Generator().manual_seed(int(seeds[1]))

    def seq_initialize(
        self, x: "ConditionalTorchEncodedSequence", weights: tn.Tensor, tng: Generator
    ) -> None:
        """Initialize grouped children with deterministic child generators."""
        _, cond_vals, eobs_vals, idx_vals, given_enc = x.data

        if not self._init_tng:
            self._tng_initialize(tng)
            self._init_tng = True

        assert self._acc_tng is not None
        assert self._default_tng is not None
        assert self._given_tng is not None

        for i, cond_val in enumerate(cond_vals):
            if cond_val in self.accumulator_map:
                self.accumulator_map[cond_val].seq_initialize(
                    eobs_vals[i], weights[idx_vals[i]], self._acc_tng[cond_val]
                )
            elif self.has_default:
                self.default_accumulator.seq_initialize(
                    eobs_vals[i], weights[idx_vals[i]], self._default_tng
                )

        if self.has_given:
            self.given_accumulator.seq_initialize(given_enc, weights, self._given_tng)

    def seq_update(
        self,
        x: "ConditionalTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: "ConditionalDistribution",
    ) -> None:
        """Update grouped children with sliced weights and matching estimates."""
        _, cond_vals, eobs_vals, idx_vals, given_enc = x.data

        for i, cond_val in enumerate(cond_vals):
            if cond_val in self.accumulator_map:
                self.accumulator_map[cond_val].seq_update(
                    eobs_vals[i], weights[idx_vals[i]], estimate.dmap[cond_val]
                )
            elif self.has_default:
                if estimate is None:
                    self.default_accumulator.seq_update(
                        eobs_vals[i], weights[idx_vals[i]], None
                    )
                else:
                    self.default_accumulator.seq_update(
                        eobs_vals[i], weights[idx_vals[i]], estimate.default_dist
                    )

        if self.has_given:
            if estimate is None:
                self.given_accumulator.seq_update(given_enc, weights, None)
            else:
                self.given_accumulator.seq_update(
                    given_enc, weights, estimate.given_dist
                )

    def combine(
        self, suff_stat: Tuple[Dict[Any, SS0], Optional[SS1], Optional[SS2]]
    ) -> "ConditionalDistributionAccumulator":
        """Merge mapped, default, and given sufficient statistics."""
        for k, v in suff_stat[0].items():
            if k in self.accumulator_map:
                self.accumulator_map[k].combine(v)
            else:
                self.accumulator_map[k].from_value(v)

        if self.has_default and suff_stat[1] is not None:
            self.default_accumulator.combine(suff_stat[1])

        if self.has_given and suff_stat[2] is not None:
            self.given_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[Dict[Any, Any], Optional[Any], Optional[Any]]:
        """Return ``(mapped_stats, default_stat, given_stat)``."""
        rv3 = self.given_accumulator.value()
        rv2 = self.default_accumulator.value()
        rv1 = {k: v.value() for k, v in self.accumulator_map.items()}

        return rv1, rv2, rv3

    def from_value(
        self, x: Tuple[Dict[Any, SS0], Optional[SS1], Optional[SS1]]
    ) -> "ConditionalDistributionAccumulator":
        """Replace mapped, default, and given statistics from a tuple."""
        for k, v in x[0].items():
            self.accumulator_map[k].from_value(v)

        if self.has_default and x[1] is not None:
            self.default_accumulator.from_value(x[1])

        if self.has_given and x[2] is not None:
            self.given_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Delegate keyed merges to mapped, default, and given children."""
        for v in self.accumulator_map.values():
            v.key_merge(stats_dict)

        if self.has_default:
            self.default_accumulator.key_merge(stats_dict)

        if self.has_given:
            self.given_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Delegate keyed replacement to mapped, default, and given children."""
        for v in self.accumulator_map.values():
            v.key_replace(stats_dict)

        if self.has_default:
            self.default_accumulator.key_replace(stats_dict)

        if self.has_given:
            self.given_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "ConditionalDistributionDataEncoder":
        """Create a grouped encoder from all child accumulators."""
        encoder_map = {k: v.acc_to_encoder() for k, v in self.accumulator_map.items()}
        default_encoder = self.default_accumulator.acc_to_encoder()
        given_encoder = self.given_accumulator.acc_to_encoder()

        return ConditionalDistributionDataEncoder(
            encoder_map=encoder_map,
            default_encoder=default_encoder,
            given_encoder=given_encoder,
        )


class ConditionalDistributionAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create mapped, default, and given child accumulators."""

    def __init__(
        self,
        factory_map: Dict[T0, TorchStatisticAccumulatorFactory],
        default_factory: TorchStatisticAccumulatorFactory = NullAccumulatorFactory(),
        given_factory: TorchStatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: Optional[str] = None,
    ) -> None:
        """Initialize child factories and optional wrapper key metadata."""
        self.factory_map = factory_map
        self.default_factory = default_factory
        self.given_factory = given_factory
        self.keys = keys

    def make(
        self, device: Optional[tn.device] = None
    ) -> "ConditionalDistributionAccumulator":
        """Create all child accumulators and associate the wrapper with ``device``."""
        acc = {k: v.make() for k, v in self.factory_map.items()}
        def_acc = self.default_factory.make()
        given_acc = self.given_factory.make()

        return ConditionalDistributionAccumulator(
            acc, def_acc, given_acc, self.keys, device=device
        )


class ConditionalDistributionEstimator(TorchParameterEstimator):
    """Estimate mapped, default, and given child distributions separately."""

    def __init__(
        self,
        estimator_map: Dict[Any, TorchParameterEstimator],
        default_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
        given_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
        keys: Optional[str] = None,
    ) -> None:
        """Initialize mapped, default, and given child estimators.

        Args:
            estimator_map: Estimators keyed by selector value.
            default_estimator: Estimator for the fallback dependent model.
            given_estimator: Estimator for the selector model.
            keys: Metadata propagated to the fitted wrapper; child keys merge stats.
        """
        self.estimator_map = estimator_map
        self.default_estimator = (
            default_estimator if default_estimator is not None else NullEstimator()
        )
        self.keys = keys
        self.given_estimator = (
            given_estimator if given_estimator is not None else NullEstimator()
        )

    def accumulator_factory(self) -> "ConditionalDistributionAccumulatorFactory":
        """Create corresponding mapped, default, and given factories."""
        emap_items = {k: v.accumulator_factory() for k, v in self.estimator_map.items()}
        def_factory = self.default_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()

        return ConditionalDistributionAccumulatorFactory(
            emap_items, def_factory, given_factory, self.keys
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[Dict[Any, SS0], Optional[SS1], Optional[SS2]],
        device: Optional[tn.device] = None,
    ) -> "ConditionalDistribution":
        """Estimate all children from the three-part statistic.

        Default and given estimates receive ``device``. Mapped estimators use
        their existing default device behavior; the returned wrapper records
        ``device``.
        """
        default_dist = self.default_estimator.estimate(
            None, suff_stat[1], device=device
        )
        given_dist = self.given_estimator.estimate(None, suff_stat[2], device=device)
        dist_map = {
            k: self.estimator_map[k].estimate(None, v) for k, v in suff_stat[0].items()
        }

        return ConditionalDistribution(
            dist_map,
            default_dist=default_dist,
            given_dist=given_dist,
            keys=self.keys,
            device=device,
        )


class ConditionalDistributionDataEncoder(TorchSequenceEncoder):
    """Group pairs by selector and encode dependent and given values."""

    def __init__(
        self,
        encoder_map: Dict[Any, TorchSequenceEncoder],
        default_encoder: TorchSequenceEncoder = NullDataEncoder(),
        given_encoder: TorchSequenceEncoder = NullDataEncoder(),
    ) -> None:
        """Initialize mapped, fallback, and selector encoders.

        Args:
            encoder_map: Dependent-value encoders keyed by selector.
            default_encoder: Fallback dependent-value encoder.
            given_encoder: Encoder for the complete selector sequence.
        """
        self.encoder_map = encoder_map
        self.default_encoder = default_encoder
        self.given_encoder = given_encoder

        self.null_default_encoder = isinstance(self.default_encoder, NullDataEncoder)
        self.null_given_encoder = isinstance(self.given_encoder, NullDataEncoder)

    def __str__(self) -> str:
        """Return a representation of mapped, default, and given encoders."""
        encoder_items = list(self.encoder_map.items())
        encoder_str = "ConditionalDataEncoder("
        for k, v in encoder_items[:-1]:
            encoder_str += str(k) + ":" + str(v) + ","
        encoder_str += str(encoder_items[-1][0]) + ":" + str(encoder_items[-1][1])

        if not self.null_default_encoder:
            encoder_str += ",default=" + str(self.default_encoder)
        else:
            encoder_str += ",default=None"

        if not self.null_given_encoder:
            encoder_str += ",given=" + str(self.given_encoder)
        else:
            encoder_str += ",given=None)"

        return encoder_str

    def __eq__(self, other: object) -> bool:
        """Return whether all mapped, default, and given encoders compare equal."""
        if not isinstance(other, ConditionalDistributionDataEncoder):
            return False

        if self.encoder_map != other.encoder_map:
            return False

        if self.default_encoder != other.default_encoder:
            return False

        if self.given_encoder != other.given_encoder:
            return False

        return True

    def seq_encode(
        self, x: List[Tuple[T0, T1]], device: Optional[tn.device] = None
    ) -> "ConditionalTorchEncodedSequence":
        """Encode ``N`` pairs into grouped dependent and selector sequences.

        The data tuple is ``(N, values, encoded_groups, indices, given)``.
        ``values`` identifies each selector group. Every ``indices[j]`` is an
        integer tensor of shape ``(N_j,)`` locating that group in the original
        order, and ``encoded_groups[j]`` encodes its ``N_j`` dependent values.
        ``given`` encodes all ``N`` selectors. All child encoders receive the
        same ``device`` argument.
        """
        cond_enc: Dict[Any, Tuple[List[Any], List[int]]] = {}
        given_vals: List[Any] = []

        for i, xx in enumerate(x):
            given_vals.append(xx[0])
            if xx[0] not in cond_enc:
                cond_enc[xx[0]] = ([xx[1]], [i])
            else:
                cond_enc_loc = cond_enc[xx[0]]
                cond_enc_loc[0].append(xx[1])
                cond_enc_loc[1].append(i)

        cond_enc_items = list(cond_enc.items())
        cond_vals = tuple(u[0] for u in cond_enc_items)

        eobs_vals: List[TorchEncodedSequence] = []
        idx_vals: List[tn.Tensor] = []

        for u in cond_enc_items:
            if self.null_default_encoder:
                if u[0] in self.encoder_map:
                    eobs_vals.append(
                        self.encoder_map[u[0]].seq_encode(u[1][0], device=device)
                    )
            else:
                eobs_vals.append(
                    self.encoder_map.get(u[0], self.default_encoder).seq_encode(
                        u[1][0], device=device
                    )
                )

            idx_vals.append(vec.int_tensor(u[1][1], device=device))

        given_enc = self.given_encoder.seq_encode(given_vals, device=device)

        return ConditionalTorchEncodedSequence(
            data=(len(x), cond_vals, tuple(eobs_vals), tuple(idx_vals), given_enc),
            device=device,
        )


class ConditionalTorchEncodedSequence(TorchEncodedSequence):
    """Store grouped conditional encodings for ``N`` original observations."""

    data: Tuple[
        int,
        Tuple[Any, ...],
        Tuple[TorchEncodedSequence, ...],
        Tuple[tn.Tensor, ...],
        TorchEncodedSequence,
    ]

    def __init__(
        self,
        data: Tuple[
            int,
            Tuple[Any, ...],
            Tuple[TorchEncodedSequence, ...],
            Tuple[tn.Tensor, ...],
            TorchEncodedSequence,
        ],
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize the five-part grouped encoding and associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"ConditionalTorchEncodedSequence(device={repr(self.device)})"
