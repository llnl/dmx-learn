"""Model ordered variable-length sequences of independent observations.

``dist`` supplies item factors and optional ``len_dist`` supplies the mass of
the non-negative integer length. With ``len_normalized=True``, only the item
log-density sum is divided by sequence length. Encoding flattens ``N`` parent
sequences containing ``M`` total items and records their parent indices and
inverse lengths. Item and length operations remain separate throughout
sampling, accumulation, and estimation, matching ``dmx.stats.sequence``.
Child encoders and ``to`` calls receive the requested torch device.
"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch as tn
from numpy.random import RandomState

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
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)

T = TypeVar("T")  # Data type of Sequence distribution dist.
E1 = TypeVar("E1")  # Generic type of distribution encoding.
E2 = TypeVar("E2")  # Generic type of length encoding.
SS1 = TypeVar("SS1")  # Generic type for sufficient statistic of base dist.
SS2 = TypeVar("SS2")  # Generic type for sufficient statistics of length dist.

E = Tuple[tn.Tensor, tn.Tensor, tn.Tensor, E1, Optional[E2]]


class SequenceDistribution(TorchProbabilityDistribution):
    """Model a sequence using independent item and optional length factors."""

    def __init__(
        self,
        dist: TorchProbabilityDistribution,
        len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        len_normalized: Optional[bool] = False,
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize item and length distributions.

        Args:
            dist: Distribution for each sequence item.
            len_dist: Optional distribution for non-negative integer lengths.
            len_normalized: Whether to average the item log-density by length.
            device: Device recorded by the wrapper.
        """
        super().__init__(device)
        self.dist = dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.len_normalized = len_normalized
        self.null_len_dist = isinstance(self.len_dist, NullDistribution)

    def __str__(self) -> str:
        """Return an evaluable representation including the device type."""
        s1 = str(self.dist)
        s2 = str(self.len_dist)
        s3 = repr(self.len_normalized)
        s4 = repr(self.model_device().type)

        return (
            f"SequenceDistribution({s1}, len_dist={s2}, "
            f"len_normalized={s3}, device=tn.device({s4}))"
        )

    def to(self, device: vec.DeviceLike) -> "SequenceDistribution":
        """Move item and length children to ``device`` in place."""
        target_device = self._resolve_device_arg(device)
        self.dist.to(target_device)
        self.len_dist.to(target_device)
        self._device = target_device
        return self

    def density(self, x: Sequence[T]) -> float:
        """Evaluate the density of SequenceDistribution at observed sequence x."""
        rv = 1.0

        for x_i in x:
            rv *= self.dist.density(x_i)

        if not self.null_len_dist:
            rv *= self.len_dist.density(len(x))

        if self.len_normalized and len(x) > 0:
            rv = np.power(rv, 1.0 / len(x))

        return rv

    def log_density(self, x: Sequence[T]) -> float:
        """Evaluate the log-density of SequenceDistribution at observed sequence x."""
        rv = 0.0

        for x_i in x:
            rv += self.dist.log_density(x_i)

        if self.len_normalized and len(x) > 0:
            rv /= len(x)

        if not self.null_len_dist:
            rv += self.len_dist.log_density(len(x))

        return rv

    def seq_log_density(self, x: "SequenceTorchEncodedSequence") -> tn.Tensor:
        """Aggregate flat item scores into one log density per parent sequence.

        The result has shape ``(N,)``. For nonempty data it follows the item
        score device; the all-empty branch uses the vector helper's default
        device allocation.
        """
        if not isinstance(x, SequenceTorchEncodedSequence):
            raise TypeError(
                "SequenceTorchEncodedSequence required for `seq_` function calls."
            )

        idx, icnt, _, enc_seq, enc_nseq = x.data

        if tn.all(icnt == 0):
            ll_sum = vec.zeros(len(icnt))

        else:
            ll = self.dist.seq_log_density(enc_seq)
            ll_sum = tn.bincount(idx, weights=ll, minlength=len(icnt))

            if self.len_normalized:
                ll_sum = ll_sum * icnt

        if not self.null_len_dist and enc_nseq is not None:
            nll = self.len_dist.seq_log_density(enc_nseq)
            ll_sum += nll

        return ll_sum

    def sampler(self, seed: Optional[int] = None) -> "SequenceSampler":
        """Create item and length samplers, requiring a non-null length model."""
        if self.null_len_dist:
            raise RuntimeError(
                "Error: len_dist cannot be none for "
                "SequenceDistribution.sampler(seed:Optional[int]=None)."
            )
        return SequenceSampler(self.dist, self.len_dist, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "SequenceEstimator":
        """Create separate item and length estimators using ``pseudo_count``."""
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)

        return SequenceEstimator(
            self.dist.estimator(pseudo_count=pseudo_count),
            len_estimator=len_est,
            len_normalized=self.len_normalized,
        )

    def dist_to_encoder(self) -> "SequenceDataEncoder":
        """Create an encoder composed from the item and length encoders."""
        dist_encoder = self.dist.dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        encoders = (dist_encoder, len_encoder)

        return SequenceDataEncoder(encoders=encoders)


class SequenceSampler(DistributionSampler):
    """SequenceSampler object for sampling from an SequenceDistribution instance."""

    def __init__(
        self,
        dist: TorchProbabilityDistribution,
        len_dist: TorchProbabilityDistribution,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize item and length samplers with independent seeds."""
        self.dist = dist
        self.len_dist = len_dist
        self.rng = RandomState(seed)
        self.dist_sampler = self.dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> List[Any]:
        """Draw one variable-length sequence or ``size`` such sequences."""
        if size is None:
            n = self.len_sampler.sample()
            return [self.dist_sampler.sample() for _ in range(n)]
        return [self.sample() for _ in range(size)]


class SequenceAccumulator(TorchStatisticAccumulator):
    """Accumulate separate item and length sufficient statistics.

    Item observations are flattened in sequence order. When normalized, each
    item receives its parent weight divided by parent length; the length child
    always receives the original parent weight.
    """

    def __init__(
        self,
        accumulator: TorchStatisticAccumulator,
        len_accumulator: TorchStatisticAccumulator = NullAccumulator(),
        len_normalized: Optional[bool] = False,
        keys: Optional[str] = None,
        device: vec.DeviceLike = None,
    ) -> None:
        """Initialize item and length child accumulators.

        Args:
            accumulator: Accumulator for flattened sequence items.
            len_accumulator: Accumulator for one length per parent sequence.
            len_normalized: Whether item weights are divided by parent length.
            keys: Optional key for sharing the paired statistic.
            device: Device metadata for encoded accumulation.
        """
        super().__init__(device)
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator
        self.keys = keys
        self.len_normalized = len_normalized

        self.null_len_accumulator = isinstance(self.len_accumulator, NullAccumulator)

    def seq_initialize(
        self, x: "SequenceTorchEncodedSequence", weights: tn.Tensor, tng: tn.Generator
    ) -> None:
        """Initialize item and length children from ``N`` encoded sequences."""
        idx, icnt, _, enc_seq, enc_nseq = x.data

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]

        self.accumulator.seq_initialize(enc_seq, w, tng)

        if not self.null_len_accumulator:
            self.len_accumulator.seq_initialize(enc_nseq, weights, tng)

    def seq_update(
        self,
        x: "SequenceTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: Optional["SequenceDistribution"],
    ) -> None:
        """Update item and length children from ``N`` encoded sequences."""
        idx, icnt, _, enc_seq, enc_nseq = x.data

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]

        self.accumulator.seq_update(
            enc_seq, w, estimate.dist if estimate is not None else None
        )

        if not self.null_len_accumulator:
            self.len_accumulator.seq_update(
                enc_nseq, weights, estimate.len_dist if estimate is not None else None
            )

    def combine(self, suff_stat: Tuple[SS1, Optional[SS2]]) -> "SequenceAccumulator":
        """Merge the ``(item_stat, length_stat)`` pair."""
        self.accumulator.combine(suff_stat[0])

        if not self.null_len_accumulator:
            self.len_accumulator.combine(suff_stat[1])

        return self

    def value(self) -> Tuple[Any, Optional[Any]]:
        """Return the item and length sufficient-statistic pair."""
        return self.accumulator.value(), self.len_accumulator.value()

    def from_value(self, x: Tuple[SS1, Optional[SS2]]) -> "SequenceAccumulator":
        """Restore item and length statistics from a paired value."""
        self.accumulator.from_value(x[0])

        if not self.null_len_accumulator:
            self.len_accumulator.from_value(x[1])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge the pair by wrapper key, then recurse into child keys."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        self.accumulator.key_merge(stats_dict)

        if not self.null_len_accumulator:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace the pair by wrapper key, then recurse into child keys."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        self.accumulator.key_replace(stats_dict)

        if not self.null_len_accumulator:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "SequenceDataEncoder":
        """Create an encoder from the item and length accumulators."""
        encoder = self.accumulator.acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()
        encoders = (encoder, len_encoder)
        return SequenceDataEncoder(encoders=encoders)


class SequenceAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """SequenceAccumulatorFactory object for creating SequenceAccumulator objects."""

    def __init__(
        self,
        dist_factory: TorchStatisticAccumulatorFactory,
        len_factory: TorchStatisticAccumulatorFactory = NullAccumulatorFactory(),
        len_normalized: Optional[bool] = False,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize item and length factories with wrapper configuration."""
        self.dist_factory = dist_factory
        self.len_factory = len_factory
        self.len_normalized = len_normalized
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> "SequenceAccumulator":
        """Create both child accumulators on ``device``."""
        len_acc = self.len_factory.make(device=device)
        return SequenceAccumulator(
            self.dist_factory.make(device=device),
            len_acc,
            self.len_normalized,
            self.keys,
            device=device,
        )


class SequenceEstimator(TorchParameterEstimator):
    """Estimate item and length children from paired sufficient statistics."""

    def __init__(
        self,
        estimator: TorchParameterEstimator,
        len_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
        len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        len_normalized: Optional[bool] = False,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize item and length estimators and wrapper configuration."""
        self.estimator = estimator
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.keys = keys
        self.len_normalized = len_normalized

    def accumulator_factory(self) -> "SequenceAccumulatorFactory":
        """Create a factory from the item and length estimator factories."""
        len_factory = self.len_estimator.accumulator_factory()
        dist_factory = self.estimator.accumulator_factory()

        return SequenceAccumulatorFactory(
            dist_factory, len_factory, self.len_normalized, self.keys
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[Any, Optional[Any]],
        device: Optional[tn.device] = None,
    ) -> "SequenceDistribution":
        """Estimate item and length children from their paired statistics.

        Length estimates receive ``device``; item estimates use their existing
        default device behavior. The returned wrapper records ``device``.
        """
        if isinstance(self.len_estimator, NullEstimator):
            return SequenceDistribution(
                self.estimator.estimate(nobs, suff_stat[0]),
                len_dist=self.len_dist.to(device),
                len_normalized=self.len_normalized,
                device=device,
            )

        return SequenceDistribution(
            self.estimator.estimate(nobs, suff_stat[0]),
            len_dist=self.len_estimator.estimate(nobs, suff_stat[1], device),
            len_normalized=self.len_normalized,
            device=device,
        )


class SequenceDataEncoder(TorchSequenceEncoder):
    """Flatten parent sequences and encode their items and lengths separately."""

    def __init__(
        self, encoders: Tuple[TorchSequenceEncoder, TorchSequenceEncoder]
    ) -> None:
        """Initialize the item and length child encoders."""
        self.encoder = encoders[0]
        self.len_encoder = encoders[1]

        self.null_len_enc = isinstance(self.len_encoder, NullDataEncoder)

    def __str__(self) -> str:
        """Return a representation of the item and length encoders."""
        s = "SequenceDataEncoder("
        s += str(self.encoder) + ",len_encoder="
        s += str(self.len_encoder) + ")"

        return s

    def __eq__(self, other: object) -> bool:
        """Return whether both child encoders compare equal."""
        if not isinstance(other, SequenceDataEncoder):
            return False

        if self.encoder != other.encoder:
            return False

        if self.len_encoder != other.len_encoder:
            return False

        return True

    def seq_encode(
        self, x: Sequence[Sequence[T]], device: Optional[tn.device] = None
    ) -> "SequenceTorchEncodedSequence":
        """Encode ``N`` sequences containing ``M`` total items.

        The tuple is ``(indices, inverse_lengths, nonempty, items, lengths)``.
        ``indices`` has shape ``(M,)`` and maps flat items to parents;
        ``inverse_lengths`` and ``nonempty`` have shape ``(N,)``. The child
        item encoding represents ``M`` values and the length encoding
        represents ``N`` integers. Every tensor or child encoding receives
        ``device``; floating inverse lengths use the vector-helper dtype.
        """
        tx = []
        nx = []
        tidx = []

        for i, x_i in enumerate(x):
            nx.append(len(x_i))

            for x_ij in x_i:
                tidx.append(i)
                tx.append(x_ij)

        rv1 = vec.int_tensor(tidx, device=device)
        rv2 = vec.tensor(nx, device=device)
        rv3 = rv2 != 0

        if tn.any(rv3):
            rv2[rv3] = 1.0 / rv2[rv3]

        rv4 = self.encoder.seq_encode(tx, device=device)

        ### None if NullDataEncoder() for length
        rv5 = self.len_encoder.seq_encode(nx, device=device)

        return SequenceTorchEncodedSequence(
            data=(rv1, rv2, rv3, rv4, rv5), device=device
        )


class SequenceTorchEncodedSequence(TorchEncodedSequence):
    """Store flattened item, parent-index, and length encodings."""

    def __init__(
        self,
        data: Tuple[
            tn.Tensor, tn.Tensor, tn.Tensor, TorchEncodedSequence, TorchEncodedSequence
        ],
        device: Optional[tn.device] = None,
    ):
        """Initialize the five-part sequence encoding and associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"SequenceTorchEncodedSequence(device={repr(self.device)})"
