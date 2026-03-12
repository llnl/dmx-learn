"""Create, estimate, and sample from a sequence of iid sequence of base distribution 'dist' with data type T. A
length distribution for the lengths of the iid sequences can be specified as a discrete distribution compatible with
non-negative integer values.

Defines the SequenceDistribution, SequenceSampler, SequenceAccumulatorFactory, SequenceAccumulator,
SequenceEstimator, and the SequenceDataEncoder classes for use with pysparkplug.

Data type (T): Assume the sequence distribution has a base distribution 'dist' compatible with data type T and length
distribution compatible with positive integers len_dist with respective densities P_dist() and P_len(). The density
of the sequence distribution is given by

p_mat(x) = P_dist(x[0])*...*P_dist(x[n-1])*P_len(n),

for an observation x of data type Sequence[T] having length n.

"""

import torch as tn
from dmx.torch_stats.null_dist import *
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence

import dmx.torch_utils.vector as vec
from dmx.utils.arithmetic import maxrandint
from typing import List, Union, Tuple, Any, Optional, TypeVar, Sequence, Dict


T = TypeVar('T')  # Data type of Sequence distribution dist.
E1 = TypeVar('E1')  # Generic type of distribution encoding.
E2 = TypeVar('E2')  # Generic type of length encoding.
SS1 = TypeVar('SS1')  # Generic type for sufficient statistic of base dist.
SS2 = TypeVar('SS2')  # Generic type for sufficient statistics of length dist.

E = Tuple[tn.Tensor, tn.Tensor, tn.Tensor, E1, Optional[E2]]


class SequenceDistribution(TorchProbabilityDistribution):
    """SequenceDistribution object for sequence of iid observations from distribution of data type T.

    Attributes:
        dist (TorchProbabilityDistribution): Base distribution of sequence (compatible with T).
        len_dist (Optional[TorchProbabilityDistribution]): Length distribution for modeling lengths
            of sequences of observations (compatible with type int). Set to NullDistribution if None is passed.
        len_normalized (Optional[bool]): If True, take geometric mean density for any density evaluation.
        null_len_dist (bool): True if 'len_dist' is set to instance of NullDistribution.

    """

    def __init__(self, dist: TorchProbabilityDistribution,
                 len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
                 len_normalized: Optional[bool] = False,
                 device: Optional[tn.device] = None) -> None:
        """SequenceDistribution object for sequence of iid observations from distribution a of data type T.

        Args:
            dist (TorchProbabilityDistribution): Set base distribution of sequence (compatible with T).
            len_dist (Optional[TorchProbabilityDistribution]): Length distribution for modeling lengths
                of sequences of observations (compatible with type int).
            len_normalized (Optional[bool]): If True, take geometric mean density for any density evaluation.

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

        return 'SequenceDistribution(%s, len_dist=%s, len_normalized=%s, device=tn.device(%s))' % (s1, s2, s3, s4)

    def to(self, device: tn.device) -> None:
        self.dist.to(device)
        self.len_dist.to(device)
        self._device = device

    def density(self, x: Sequence[T]) -> float:
        """Evaluate the density of SequenceDistribution at observed sequence x.

        Notes:
            Assume x is a Sequence of data type T with length n > 0. Assume P_dist() is the density for the base
            distribution with data type T of SequenceDistribution, and P_len() is the length distribution with data type
            int. Then,

            P(x) = P_dist(x[0])*...*P_dist(x[n-1])*P_len(n), if len_normalize is False,

            or,

            P(x) = (P_dist(x[0])*...*P_dist(x[n-1])*P_len(n))^(1/n) if len_normalize is True.



        Args:
            x (Sequence[T]): Sequence of iid observations from base distribution of SequenceDistribution.

        Returns:
            float: Density evaluated at observation x.


        """
        rv = 1.0

        for i in range(len(x)):
            rv *= self.dist.density(x[i])

        if not self.null_len_dist:
            rv *= self.len_dist.density(len(x))

        if self.len_normalized and len(x) > 0:
            rv = np.power(rv, 1.0 / len(x))

        return rv

    def log_density(self, x: Sequence[T]) -> float:
        """Evaluate the log-density of SequenceDistribution at observed sequence x.

        Notes:
            See density() for details.

        Args:
            x (Sequence[T]): Sequence of iid observations from base distribution of SequenceDistribution.

        Returns:
            float: Log-density evaluated at observation x.

        """
        rv = 0.0

        for i in range(len(x)):
            rv += self.dist.log_density(x[i])

        if self.len_normalized and len(x) > 0:
            rv /= len(x)

        if not self.null_len_dist:
            rv += self.len_dist.log_density(len(x))

        return rv

    def seq_log_density(self, x: 'SequenceTorchEncodedSequence') -> tn.Tensor:

        if not isinstance(x, SequenceTorchEncodedSequence):
            raise Exception('SequenceTorchEncodedSequence required for `seq_` function calls.')

        idx, icnt, inz, enc_seq, enc_nseq = x.data

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

    def sampler(self, seed: Optional[int] = None) -> 'SequenceSampler':
        if self.null_len_dist:
            raise Exception('Error: len_dist cannot be none for SequenceDistribution.sampler(seed:Optional[int]=None).')
        else:
            return SequenceSampler(self.dist, self.len_dist, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SequenceEstimator':
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)

        return SequenceEstimator(self.dist.estimator(pseudo_count=pseudo_count), len_estimator=len_est,
                                 len_normalized=self.len_normalized)

    def dist_to_encoder(self) -> 'SequenceDataEncoder':
        dist_encoder = self.dist.dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        encoders = (dist_encoder, len_encoder)

        return SequenceDataEncoder(encoders=encoders)


class SequenceSampler(DistributionSampler):
    """SequenceSampler object for sampling from an SequenceDistribution instance.

    Attributes:
        dist (TorchProbabilityDistribution): The Base distribution for the sequences (data type T).
        len_dist (TorchProbabilityDistribution): Length distribution for the length of the
            sequences (support on positive integers).
        rng (RandomState): RandomState object for random sampling.
        dist_sampler (DistributionSampler): DistributionSampler instance from base distribution.
        len_sampler (DistributionSampler): DistributionSampler instance from length distribution.

    """

    def __init__(self,
                 dist: TorchProbabilityDistribution,
                 len_dist: TorchProbabilityDistribution,
                 seed: Optional[int] = None) -> None:
        """SequenceSampler object.

        Args:
            dist (TorchProbabilityDistribution): Set the base distribution for the sequences (data type T).
            len_dist (TorchProbabilityDistribution): Set the length distribution for the length of the
                sequences (support on positive integers).
            seed (Optional[int]): Set seed of random number generator for sampling.

        """
        self.dist = dist
        self.len_dist = len_dist
        self.rng = RandomState(seed)
        self.dist_sampler = self.dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> List[Any]:
        """Generate iid samples from SequenceSampler object.

        If size is None, the length 'n' of the iid sequence is sampled from len_sampler. Then 'n' iid samples are
        drawn from the base dist sampled 'dist_sampler'.

        If size > 0, above is repeated size times and a List of size List[T] is retured.

        Args:
            size (Optional[int]) Number of sequences to be sampled.

        Returns:
            List[T] or List[List[T]] with length(size).

        """
        if size is None:
            n = self.len_sampler.sample()
            return [self.dist_sampler.sample() for i in range(n)]
        else:
            return [self.sample() for i in range(size)]


class SequenceAccumulator(TorchStatisticAccumulator):
    """SequenceAccumulator object for aggregating sufficient statistics of sequence distribution from observed data.

    Attributes:
        accumulator (TorchStatisticAccumulator): TorchStatisticAccumulator object for
            accumulating sufficient statistics of base distribution compatible with data type T.
        len_accumulator (TorchStatisticAccumulator): TorchStatisticAccumulator object
            for accumulating sufficient statistics of length distribution compatible with non-negative integers.
        len_normalized (Optional[bool]): Geometric mean of density taken if set to True. Else ignored.
        keys (Optional[str]): Set keys for merging sufficient statistics of SequenceAccumulator objects with
            matching keys.
        null_len_accumulator (bool): True if len_accumulator is an instance of NullAccumulator object.

    """

    def __init__(self,
                 accumulator: TorchStatisticAccumulator,
                 len_accumulator: TorchStatisticAccumulator = NullAccumulator(),
                 len_normalized: Optional[bool] = False,
                 keys: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        """SequenceAccumulator object.

        Args:
            accumulator (TorchStatisticAccumulator): Set TorchStatisticAccumulator object for
                accumulating sufficient statistics of base distribution compatible with data type T.
            len_accumulator (TorchStatisticAccumulator): Set TorchStatisticAccumulator object
                for accumulating sufficient statistics of length distribution compatible with non-negative integers.
            len_normalized (Optional[bool]): Geometric mean of density taken if set to True. Else ignored.
            keys (Optional[str]): Set keys for merging sufficient statistics of SequenceAccumulator objects with
                matching keys.

        """
        super().__init__(device)
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator
        self.keys = keys
        self.len_normalized = len_normalized

        self.null_len_accumulator = isinstance(self.len_accumulator, NullAccumulator)

    def seq_initialize(self, x: 'SequenceTorchEncodedSequence', weights: tn.Tensor, tng: tn.Generator) -> None:
        idx, icnt, inz, enc_seq, enc_nseq = x.data

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]

        self.accumulator.seq_initialize(enc_seq, w, tng)

        if not self.null_len_accumulator:
            self.len_accumulator.seq_initialize(enc_nseq, weights, tng)

    def seq_update(self, x: 'SequenceTorchEncodedSequence', weights: tn.Tensor, estimate: Optional['SequenceDistribution']) -> None:

        idx, icnt, inz, enc_seq, enc_nseq = x.data

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]

        self.accumulator.seq_update(enc_seq, w, estimate.dist if estimate is not None else None)

        if not self.null_len_accumulator:
            self.len_accumulator.seq_update(enc_nseq, weights, estimate.len_dist if estimate is not None else None)

    def combine(self, suff_stat: Tuple[SS1, Optional[SS2]]) -> 'SequenceAccumulator':
        self.accumulator.combine(suff_stat[0])

        if not self.null_len_accumulator:
            self.len_accumulator.combine(suff_stat[1])

        return self

    def value(self) -> Tuple[Any, Optional[Any]]:
        return self.accumulator.value(), self.len_accumulator.value()

    def from_value(self, x: Tuple[SS1, Optional[SS2]]) -> 'SequenceAccumulator':
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

    def acc_to_encoder(self) -> 'SequenceDataEncoder':
        encoder = self.accumulator.acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()
        encoders = (encoder, len_encoder)
        return SequenceDataEncoder(encoders=encoders)


class SequenceAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """SequenceAccumulatorFactory object for creating SequenceAccumulator objects.

    Attributes:
        dist_factory (TorchStatisticAccumulatorFactory): TorchStatisticAccumulatorFactory for base distribution of sequence
            distribution.
        len_factory (TorchStatisticAccumulatorFactory): TorchStatisticAccumulatorFactory for length distribution of sequence
            distribution, set to NullAccumulatorFactory() if corresponding SequenceDistribution has no length
            distribution desired to be estimated.
        len_normalized (Optional[bool]): Standardize by length of sequence distribution.
        keys (Optional[str]): Key for merging/combining sufficient statistics of SequenceAccumulator.

    """

    def __init__(self,
                 dist_factory: TorchStatisticAccumulatorFactory,
                 len_factory: TorchStatisticAccumulatorFactory = NullAccumulatorFactory(),
                 len_normalized: Optional[bool] = False,
                 keys: Optional[str] = None) -> None:
        """SequenceAccumulatorFactory object.

        Args:
            dist_factory (TorchStatisticAccumulatorFactory): TorchStatisticAccumulatorFactory for base distribution of sequence
                distribution.
            len_factory (TorchStatisticAccumulatorFactory): TorchStatisticAccumulatorFactory for length distribution of sequence
                distribution.
            len_normalized (Optional[bool]): Standardize by length of sequence distribution.
            keys (Optional[str]): Set key for merging/combining sufficient statistics of SequenceAccumulator.

        """
        self.dist_factory = dist_factory
        self.len_factory = len_factory
        self.len_normalized = len_normalized
        self.keys = keys

    def make(self, device: Optional[tn.device] = None) -> 'SequenceAccumulator':
        len_acc = self.len_factory.make(device=device)
        return SequenceAccumulator(self.dist_factory.make(device=device), len_acc, self.len_normalized, self.keys, device=device)


class SequenceEstimator(TorchParameterEstimator):
    """SequenceEstimator object for estimating SequenceDistribution from aggregated sufficient statistics.

    Notes:
        Requires arg 'estimator' to be TorchParameterEstimator of data type T, compatible with the observed entry values
        of SequenceDistribution.

        If arg 'len_estimator' is passed, it must be a TorchParameterEstimator object compatible with non-negative
        integers.

        If len_estimator is NullEstimator() or None, len_dist is used as length distribution in estimation.

    Attributes:
        estimator (TorchParameterEstimator): TorchParameterEstimator for base distribution.
        len_estimator (Optional[TorchParameterEstimator]): TorchParameterEstimator for length distribution. If None,
            set to NullEstimator.
        len_dist (Optional[TorchProbabilityDistribution]): Set a fixed length distribution.
        len_normalized (Optional[bool]): Take geometric mean of density if True.
        keys (Optional[str]): Key for SequenceEstimator instance used in aggregating sufficient statistics.

    """

    def __init__(self,
                 estimator: TorchParameterEstimator,
                 len_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
                 len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
                 len_normalized: Optional[bool] = False,
                 keys: Optional[str] = None ) -> None:
        """SequenceEstimator object.

        Args:
            estimator (TorchParameterEstimator): Set TorchParameterEstimator for base distribution.
            len_estimator (Optional[TorchParameterEstimator]): Set TorchParameterEstimator for length distribution.
            len_dist (Optional[TorchProbabilityDistribution]): Set a fixed length distribution.
            len_normalized (Optional[bool]): Take geometric mean of density if True.
            keys (Optional[str]): Set key to SequenceEstimator instance for merging sufficient statistics.

        """
        self.estimator = estimator
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.keys = keys
        self.len_normalized = len_normalized

    def accumulator_factory(self) -> 'SequenceAccumulatorFactory':
        len_factory = self.len_estimator.accumulator_factory()
        dist_factory = self.estimator.accumulator_factory()

        return SequenceAccumulatorFactory(dist_factory, len_factory, self.len_normalized, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[Any, Optional[Any]], device: Optional[tn.device] = None) -> 'SequenceDistribution':
        if isinstance(self.len_estimator, NullEstimator):
            return SequenceDistribution(self.estimator.estimate(nobs, suff_stat[0]), len_dist=self.len_dist.to(device),
                                        len_normalized=self.len_normalized, device=device)

        else:
            return SequenceDistribution(self.estimator.estimate(nobs, suff_stat[0]),
                                        len_dist=self.len_estimator.estimate(nobs, suff_stat[1], device),
                                        len_normalized=self.len_normalized, device=device)


class SequenceDataEncoder(TorchSequenceEncoder):
    """SequenceDataEncoder object for encoding sequences of iid observations from sequence distributions.

    Notes:
        encoders[0] is a TorchSequenceEncoder for data type T, producing encoded sequences of type T1.
        encoders[1] is a TorchSequenceEncoder for data type int, production encoded sequences of type T2 or None.

    Attributes:
        encoder (TorchSequenceEncoder): TorchSequenceEncoder object for the distribution of sequence distribution.
        len_encoder (TorchSequenceEncoder): TorchSequenceEncoder object for the length distribution of sequence
            distribution. Generally NullDataEncoder() object is no intended length distribution.
        null_len_enc (bool): True if len_encoder is a NullDataEncoder(), else False.

    """

    def __init__(self, encoders: Tuple[TorchSequenceEncoder, TorchSequenceEncoder]) -> None:
        """SequenceDataEncoder object.

        Args:
            encoders (Tuple[TorchSequenceEncoder, TorchSequenceEncoder]): Tuple of TorchSequenceEncoder objects for
                distribution and length distribution of sequence distribution.

        """
        self.encoder = encoders[0]
        self.len_encoder = encoders[1]

        self.null_len_enc = isinstance(self.len_encoder, NullDataEncoder)

    def __str__(self) -> str:
        s = 'SequenceDataEncoder('
        s += str(self.encoder) + ',len_encoder='
        s += str(self.len_encoder) + ')'

        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceDataEncoder):
            return False

        else:
            if not self.encoder == other.encoder:
                return False

            if not self.len_encoder == other.len_encoder:
                return False

            return True

    def seq_encode(self, x: Sequence[Sequence[T]], device: Optional[tn.device] = None) -> 'SequenceTorchEncodedSequence':
        tx = []
        nx = []
        tidx = []

        for i in range(len(x)):
            nx.append(len(x[i]))

            for j in range(len(x[i])):
                tidx.append(i)
                tx.append(x[i][j])

        rv1 = vec.int_tensor(tidx, device=device)
        rv2 = vec.tensor(nx, device=device)
        rv3 = (rv2 != 0)

        if tn.any(rv3):
            rv2[rv3] = 1.0 / rv2[rv3]

        rv4 = self.encoder.seq_encode(tx, device=device)

        ### None if NullDataEncoder() for length
        rv5 = self.len_encoder.seq_encode(nx, device=device)

        return SequenceTorchEncodedSequence(data=(rv1, rv2, rv3, rv4, rv5), device=device)


class SequenceTorchEncodedSequence(TorchEncodedSequence):

    def __init__(self, data: Tuple[tn.tensor, tn.tensor, tn.tensor, TorchEncodedSequence, TorchEncodedSequence], device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'SequenceTorchEncodedSequence(device={repr(self.device)})'

