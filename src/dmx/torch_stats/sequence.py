"""
Create, estimate, and sample from a sequence of iid sequence of base distribution
'dist'.
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
    """
    SequenceDistribution object for sequence of iid observations from distribution of
    data.
    """

    def __init__(
        self,
        dist: TorchProbabilityDistribution,
        len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        len_normalized: Optional[bool] = False,
        device: Optional[tn.device] = None,
    ) -> None:
        """
        SequenceDistribution object for sequence of iid observations from distribution a
        of data.
        """
        super().__init__(device)
        self.dist = dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.len_normalized = len_normalized
        self.null_len_dist = isinstance(self.len_dist, NullDistribution)

    def __str__(self) -> str:
        s1 = str(self.dist)
        s2 = str(self.len_dist)
        s3 = repr(self.len_normalized)
        s4 = repr(self.model_device().type)

        return (
            f"SequenceDistribution({s1}, len_dist={s2}, "
            f"len_normalized={s3}, device=tn.device({s4}))"
        )

    def to(self, device: tn.device) -> None:
        self.dist.to(device)
        self.len_dist.to(device)
        self._device = device

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
        if self.null_len_dist:
            raise RuntimeError(
                "Error: len_dist cannot be none for "
                "SequenceDistribution.sampler(seed:Optional[int]=None)."
            )
        return SequenceSampler(self.dist, self.len_dist, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "SequenceEstimator":
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)

        return SequenceEstimator(
            self.dist.estimator(pseudo_count=pseudo_count),
            len_estimator=len_est,
            len_normalized=self.len_normalized,
        )

    def dist_to_encoder(self) -> "SequenceDataEncoder":
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
        """SequenceSampler object."""
        self.dist = dist
        self.len_dist = len_dist
        self.rng = RandomState(seed)
        self.dist_sampler = self.dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> List[Any]:
        """Generate iid samples from SequenceSampler object."""
        if size is None:
            n = self.len_sampler.sample()
            return [self.dist_sampler.sample() for _ in range(n)]
        return [self.sample() for _ in range(size)]


class SequenceAccumulator(TorchStatisticAccumulator):
    """SequenceAccumulator object for aggregating sufficient statistics of sequence."""

    def __init__(
        self,
        accumulator: TorchStatisticAccumulator,
        len_accumulator: TorchStatisticAccumulator = NullAccumulator(),
        len_normalized: Optional[bool] = False,
        keys: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """SequenceAccumulator object."""
        super().__init__(device)
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator
        self.keys = keys
        self.len_normalized = len_normalized

        self.null_len_accumulator = isinstance(self.len_accumulator, NullAccumulator)

    def seq_initialize(
        self, x: "SequenceTorchEncodedSequence", weights: tn.Tensor, tng: tn.Generator
    ) -> None:
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
        self.accumulator.combine(suff_stat[0])

        if not self.null_len_accumulator:
            self.len_accumulator.combine(suff_stat[1])

        return self

    def value(self) -> Tuple[Any, Optional[Any]]:
        return self.accumulator.value(), self.len_accumulator.value()

    def from_value(self, x: Tuple[SS1, Optional[SS2]]) -> "SequenceAccumulator":
        self.accumulator.from_value(x[0])

        if not self.null_len_accumulator:
            self.len_accumulator.from_value(x[1])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        self.accumulator.key_merge(stats_dict)

        if not self.null_len_accumulator:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        self.accumulator.key_replace(stats_dict)

        if not self.null_len_accumulator:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "SequenceDataEncoder":
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
        """SequenceAccumulatorFactory object."""
        self.dist_factory = dist_factory
        self.len_factory = len_factory
        self.len_normalized = len_normalized
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> "SequenceAccumulator":
        len_acc = self.len_factory.make(device=device)
        return SequenceAccumulator(
            self.dist_factory.make(device=device),
            len_acc,
            self.len_normalized,
            self.keys,
            device=device,
        )


class SequenceEstimator(TorchParameterEstimator):
    """
    SequenceEstimator object for estimating SequenceDistribution from aggregated
    sufficient.
    """

    def __init__(
        self,
        estimator: TorchParameterEstimator,
        len_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
        len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        len_normalized: Optional[bool] = False,
        keys: Optional[str] = None,
    ) -> None:
        """SequenceEstimator object."""
        self.estimator = estimator
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.keys = keys
        self.len_normalized = len_normalized

    def accumulator_factory(self) -> "SequenceAccumulatorFactory":
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
    """
    SequenceDataEncoder object for encoding sequences of iid observations from sequence.
    """

    def __init__(
        self, encoders: Tuple[TorchSequenceEncoder, TorchSequenceEncoder]
    ) -> None:
        """SequenceDataEncoder object."""
        self.encoder = encoders[0]
        self.len_encoder = encoders[1]

        self.null_len_enc = isinstance(self.len_encoder, NullDataEncoder)

    def __str__(self) -> str:
        s = "SequenceDataEncoder("
        s += str(self.encoder) + ",len_encoder="
        s += str(self.len_encoder) + ")"

        return s

    def __eq__(self, other: object) -> bool:
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

    def __init__(
        self,
        data: Tuple[
            tn.tensor, tn.tensor, tn.tensor, TorchEncodedSequence, TorchEncodedSequence
        ],
        device: Optional[tn.device] = None,
    ):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f"SequenceTorchEncodedSequence(device={repr(self.device)})"
