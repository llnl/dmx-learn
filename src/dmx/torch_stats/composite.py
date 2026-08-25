"""Provide torch-backed products of positional component distributions.

A composite observation is a tuple ``(x_0, ..., x_{K-1})`` whose field ``k``
belongs to child distribution ``dists[k]``. Encoding transposes a sequence of
``N`` observation tuples into ``K`` child encoded sequences, each representing
the same ``N`` observations. Scoring sums child log densities and returns a
tensor of shape ``(N,)`` on the child/model device. Device movement, sampling,
accumulation, and estimation are delegated position by position. Unlike
``dmx.stats.composite``, this wrapper has no name and stores device state.
"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch as tn
from numpy.random import RandomState
from torch import Generator

import dmx.torch_utils.vector as vec
from dmx.arithmetic import maxrandint
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchDevice,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)

E = TypeVar("E")
SS = TypeVar("SS")


class CompositeDistribution(TorchProbabilityDistribution):
    """Model a tuple as independent draws from positional child distributions.

    The support is the Cartesian product of the child supports. Every
    observation must have the same number and order of fields as ``dists``.

    Attributes:
        dists (Sequence[TorchProbabilityDistribution]): Positional children.
        count (int): Number of tuple fields and child distributions.
    """

    def __init__(
        self,
        dists: Sequence[TorchProbabilityDistribution],
        device: Optional[TorchDevice] = None,
    ) -> None:
        """Initialize a positional product distribution.

        Args:
            dists: Child distributions in tuple-field order.
            device: Device recorded by the wrapper and propagated to children.
        """
        super().__init__(device)
        self.dists = dists
        self.count = len(dists)

    def to(self, device: vec.DeviceLike) -> "CompositeDistribution":
        """Move every child to ``device`` in place and return ``self``."""
        target_device = self._resolve_device_arg(device)
        self._device = target_device
        for comp in self.dists:
            comp.to(target_device)
        return self

    def __repr__(self) -> str:
        """Return a constructor-like representation of the child tuple."""
        s0 = ",".join(map(str, self.dists))
        return f"CompositeDistribution(({s0}))"

    def density(self, x: Tuple[Any, ...]) -> float:
        """Evaluate the scalar density for one positional observation tuple."""
        rv = 0.0

        for i in range(1, self.count):
            rv *= self.dists[i].density(x[i])

        return rv

    def log_density(self, x: Tuple[Any, ...]) -> float:
        """Evaluate the sum of child log densities for one tuple."""
        rv = self.dists[0].log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].log_density(x[i])

        return rv

    def seq_log_density(self, x: "CompositeTorchEncodedSequence") -> tn.Tensor:
        """Sum child scores into a tensor of shape ``(N,)``."""
        if not isinstance(x, CompositeTorchEncodedSequence):
            raise TypeError("Requires CompositeTorchEncodedSequence for `seq_` calls.")

        rv = self.dists[0].seq_log_density(x.data[0])

        for i in range(1, self.count):
            rv += self.dists[i].seq_log_density(x.data[i])

        return rv

    def sampler(self, seed: Optional[int] = None) -> "CompositeSampler":
        """Create independent child samplers from seeds derived from ``seed``."""
        return CompositeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "CompositeEstimator":
        """Create one child estimator per field using ``pseudo_count``."""
        return CompositeEstimator(
            [d.estimator(pseudo_count=pseudo_count) for d in self.dists]
        )

    def dist_to_encoder(self) -> "CompositeDataEncoder":
        """Create a positional encoder from the child encoders."""
        encoders = tuple(d.dist_to_encoder() for d in self.dists)

        return CompositeDataEncoder(encoders=encoders)


class CompositeSampler(DistributionSampler):
    """Draw tuples by sampling each positional child independently."""

    def __init__(
        self, dist: "CompositeDistribution", seed: Optional[int] = None
    ) -> None:
        """Initialize child samplers with independently derived seeds."""
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_samplers = [
            d.sampler(seed=self.rng.randint(maxrandint)) for d in dist.dists
        ]

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Tuple[Any, ...]], Tuple[Any, ...]]:
        """Draw one ``K``-tuple or a list of ``size`` such tuples."""
        if size is None:
            return tuple(d.sample(size=size) for d in self.dist_samplers)

        return list(zip(*[d.sample(size=size) for d in self.dist_samplers]))


class CompositeAccumulator(TorchStatisticAccumulator):
    """Aggregate a separate sufficient statistic for every tuple field.

    The same observation-weight tensor is passed to every child. The public
    statistic is a tuple in positional child order.
    """

    def __init__(
        self,
        accumulators: Sequence[TorchStatisticAccumulator],
        keys: Optional[str] = None,
        device: Optional[TorchDevice] = None,
    ) -> None:
        """Initialize positional child accumulators.

        Args:
            accumulators: Child accumulators in tuple-field order.
            keys: Optional key for sharing the whole positional statistic.
            device: Device metadata for encoded accumulation.
        """
        super().__init__(device)
        self.accumulators = accumulators
        self.count = len(accumulators)
        self.key = keys

    def seq_initialize(
        self, x: "CompositeTorchEncodedSequence", weights: tn.Tensor, tng: Generator
    ) -> None:
        """Initialize every child from its encoded field and shared weights."""
        for i in range(self.count):
            self.accumulators[i].seq_initialize(x.data[i], weights, tng)

    def seq_update(
        self,
        x: "CompositeTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: Optional["CompositeDistribution"],
    ) -> None:
        """Update every child from its encoded field and shared weights."""
        for i in range(self.count):
            self.accumulators[i].seq_update(
                x.data[i], weights, estimate.dists[i] if estimate is not None else None
            )

    def combine(self, suff_stat: Tuple[Any, ...]) -> "CompositeAccumulator":
        """Merge a tuple of child sufficient statistics position by position."""
        for i in range(self.count):
            self.accumulators[i].combine(suff_stat[i])

        return self

    def value(self) -> Tuple[Any, ...]:
        """Return the tuple of child sufficient statistics."""
        return tuple(x.value() for x in self.accumulators)

    def from_value(self, x: Tuple[Any, ...]) -> "CompositeAccumulator":
        """Replace child statistics from a positional tuple."""
        self.accumulators = [
            self.accumulators[i].from_value(x[i]) for i in range(len(x))
        ]
        self.count = len(x)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge the whole tuple and then recurse into child keys."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace the whole tuple and then recurse into child keys."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "CompositeDataEncoder":
        """Create a positional encoder from the child accumulators."""
        encoders = tuple(acc.acc_to_encoder() for acc in self.accumulators)

        return CompositeDataEncoder(encoders=encoders)


class CompositeAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create composite accumulators from positional child factories."""

    def __init__(
        self,
        factories: Sequence[TorchStatisticAccumulatorFactory],
        keys: Optional[str] = None,
    ) -> None:
        """Initialize positional factories and an optional whole-tuple key."""
        self.factories = factories
        self.keys = keys

    def make(self, device: Optional[TorchDevice] = None) -> "CompositeAccumulator":
        """Create child accumulators and associate the wrapper with ``device``."""
        return CompositeAccumulator(
            [u.make() for u in self.factories], keys=self.keys, device=device
        )


class CompositeEstimator(TorchParameterEstimator):
    """Estimate every positional child from its matching statistic."""

    def __init__(
        self, estimators: Sequence[TorchParameterEstimator], keys: Optional[str] = None
    ) -> None:
        """Initialize positional child estimators and a whole-tuple key."""
        self.estimators = estimators
        self.count = len(estimators)
        self.keys = keys

    def accumulator_factory(self) -> "CompositeAccumulatorFactory":
        """Create a composite factory from the child estimator factories."""
        return CompositeAccumulatorFactory(
            [u.accumulator_factory() for u in self.estimators], self.keys
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[Any, ...],
        device: Optional[TorchDevice] = None,
    ) -> "CompositeDistribution":
        """Estimate each child from its positional statistic on ``device``."""
        return CompositeDistribution(
            tuple(
                est.estimate(nobs, ss, device=device)
                for est, ss in zip(self.estimators, suff_stat)
            ),
            device=device,
        )


class CompositeDataEncoder(TorchSequenceEncoder):
    """Encode observation tuples using one positional child encoder per field."""

    def __init__(self, encoders: Sequence[TorchSequenceEncoder]) -> None:
        """Initialize child encoders in tuple-field order."""
        self.encoders = encoders

    def __eq__(self, other: object) -> bool:
        """Return whether positional child encoders compare equal."""
        if not isinstance(other, CompositeDataEncoder):
            return False

        for i, encoder in enumerate(self.encoders):
            if encoder != other.encoders[i]:
                return False

        return True

    def __str__(self) -> str:
        """Return a representation listing positional child encoders."""
        s = "CompositeDataEncoder(["

        for d in self.encoders[:-1]:
            s += str(d) + ","

        s += str(self.encoders[-1]) + "])"

        return s

    def seq_encode(
        self, x: Sequence[Tuple[Any, ...]], device: Optional[TorchDevice] = None
    ) -> "CompositeTorchEncodedSequence":
        """Transpose and encode ``N`` tuples into ``K`` child sequences.

        Each child encoder receives its field values in observation order and
        the same ``device`` argument.
        """
        enc_data: List[TorchEncodedSequence] = []

        for i, encoder in enumerate(self.encoders):
            enc_data.append(encoder.seq_encode([u[i] for u in x], device=device))

        return CompositeTorchEncodedSequence(data=tuple(enc_data), device=device)


class CompositeTorchEncodedSequence(TorchEncodedSequence):
    """Store a tuple of ``K`` child encodings for the same ``N`` observations."""

    data: Tuple[TorchEncodedSequence, ...]

    def __init__(
        self,
        data: Tuple[TorchEncodedSequence, ...],
        device: Optional[TorchDevice] = None,
    ) -> None:
        """Initialize the child-encoding tuple and associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"CompositeTorchEncodedSequence(device={repr(self.device)})"
