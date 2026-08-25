r"""Model paired observations with coupled component assignments.

Each observation is ``(x1, x2)``. Latent indices ``Z1`` and ``Z2`` select components
from separate banks for the two views. The joint density uses ``w1[i]`` and the
conditional matrix ``taus12[i, j]``; ``w2`` and ``taus21`` store the reverse marginal
and conditional representation. EM responsibilities therefore form an ``N`` by ``M``
table for every observation, not a single component vector.

Unlike :mod:`hmixture`, the two values are paired views rather than items in a
variable-length sequence. Unlike :mod:`heterogeneous_mixture`, each view has its own
component bank. Unlike :mod:`ss_mixture`, neither tuple entry is a label prior.
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

T0 = TypeVar("T0")
T1 = TypeVar("T1")
E0 = TypeVar("E0")
E1 = TypeVar("E1")
SS0 = TypeVar("SS0")
SS1 = TypeVar("SS1")


class JointMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """JointMixtureDistribution object for defining a joint mixture distribution.

    Notes:
        Data type is Tuple[T0, T1] where all components1 entries and component2 entries
        are compatible with
        T0 and T1 respectively.

    Attributes:
        components1(Sequence[SequenceEncodableProbabilityDistribution]): Mixture
            components for mixture of X1.
        components2 (Sequence[SequenceEncodableProbabilityDistribution]): Mixture
            components for mixture X2.
        w1 (np.ndarray): Probability of drawing X1 from component i.
        w2 (np.ndarray): Probability of drawing X2 from component j.
        num_components1 (int): Number of mixture components for X1.
        num_components2 (int): Number of mixture components for X2.
        taus12 (np.ndarray): 2-d Numpy array with probabilities of drawing X2 from comp
            j given X1 was drawn from
            comp i. Rows are component X1 state.
        taus21 (np.ndarray): 2-d Numpy array with probabilities of drawing X1 from comp
            i given X2 was drawn from
            comp j. Rows are component X1 state.
        log_w1 (np.ndarray): Log-probability of drawing X1 from component i.
        log_w2 (np.ndarray): Log-probability of drawing X2 from component j.
        log_taus12 (np.ndarray): 2-d Numpy array with log-probabilities of drawing X2
            from comp j given X1 was
            drawn from comp i. Rows are component X1 state.
        log_taus21 (np.ndarray): 2-d Numpy array with log-probabilities of drawing X1
            from comp i given X2 was
            drawn from comp j. Rows are component X1 state.
        keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys
            for weights, mixture
            components of X1, mixture components of X2.
        name (Optional[str]): Set name to object.

    """

    def __init__(
        self,
        components1: Sequence[SequenceEncodableProbabilityDistribution],
        components2: Sequence[SequenceEncodableProbabilityDistribution],
        w1: Union[Sequence[float], np.ndarray],
        w2: Union[Sequence[float], np.ndarray],
        taus12: Union[List[List[float]], np.ndarray],
        taus21: Union[List[List[float]], np.ndarray],
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        name: Optional[str] = None,
    ) -> None:
        """Initialize a joint mixture distribution.

        Args:
            components1(Sequence[SequenceEncodableProbabilityDistribution]): Mixture
                components for mixture of X1.
            components2 (Sequence[SequenceEncodableProbabilityDistribution]): Mixture
                components for mixture X2.
            w1 (np.ndarray): Probability of drawing X1 from component i.
            w2 (np.ndarray): Probability of drawing X2 from component j.
            taus12 (np.ndarray): 2-d Numpy array with probabilities of drawing X2 from
                comp j given X1 was drawn from
                comp i. Rows are component X1 state.
            taus21 (np.ndarray): 2-d Numpy array with probabilities of drawing X1 from
                comp i given X2 was drawn from
                comp j. Rows are component X1 state.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set
                keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        """
        super().__init__()
        with np.errstate(divide="ignore"):
            self.components1 = components1
            self.components2 = components2
            self.w1 = vec.make(w1)
            self.w2 = vec.make(w2)
            self.num_components1 = len(components1)
            self.num_components2 = len(components2)
            self.taus12 = np.reshape(
                taus12, (self.num_components1, self.num_components2)
            )
            self.taus21 = np.reshape(
                taus21, (self.num_components1, self.num_components2)
            )
            self.log_w1 = np.log(self.w1)
            self.log_w2 = np.log(self.w2)
            self.log_taus12 = np.log(self.taus12)
            self.log_taus21 = np.log(self.taus21)
            self.keys = keys if keys is not None else (None, None, None)
            self.name = name

    def __str__(self) -> str:
        """Return a string representation of the distribution."""
        s1 = ",".join([str(u) for u in self.components1])
        s2 = ",".join([str(u) for u in self.components2])
        s3 = ",".join(map(str, self.w1))
        s4 = ",".join(map(str, self.w2))
        s5 = ",".join(map(str, self.taus12.flatten()))
        s6 = ",".join(map(str, self.taus21.flatten()))
        s7 = repr(self.name)
        s8 = repr(self.keys)

        return (
            f"JointMixtureDistribution([{s1}], [{s2}], [{s3}], [{s4}], [{s5}], [{s6}], "
            f"name={s7}, keys={s8})"
        )

    def density(self, x: Tuple[T0, T1]) -> float:
        """Evaluate the density of one paired observation."""
        return float(exp(self.log_density(x)))

    def log_density(self, x: Tuple[T0, T1]) -> float:
        """Evaluate the log-density of one paired observation."""
        ll1 = np.zeros((1, self.num_components1))
        ll2 = np.zeros((1, self.num_components2))

        for i in range(self.num_components1):
            ll1[0, i] = self.components1[i].log_density(x[0]) + self.log_w1[i]
        for i in range(self.num_components2):
            ll2[0, i] += self.components2[i].log_density(x[1])

        max1 = ll1.max()
        ll1 -= max1
        np.exp(ll1, out=ll1)

        max2 = np.max(ll2)
        ll2 -= max2
        np.exp(ll2, out=ll2)

        ll12 = np.dot(ll1, self.taus12)
        ll2 *= ll12

        rv = np.log(ll2.sum()) + max1 + max2

        return float(rv)

    def seq_log_density(self, x: "JointMixtureEncodedDataSequence") -> np.ndarray:
        """Evaluate log-densities for encoded paired observations."""
        if not isinstance(x, JointMixtureEncodedDataSequence):
            raise TypeError(
                "JointMixtureEncodedDataSequence required for seq_log_density()."
            )

        sz, enc_data1, enc_data2 = x.data
        ll_mat1 = np.zeros((sz, self.num_components1))
        ll_mat2 = np.zeros((sz, self.num_components2))

        for i in range(self.num_components1):
            ll_mat1[:, i] = self.components1[i].seq_log_density(enc_data1)
            ll_mat1[:, i] += self.log_w1[i]

        for i in range(self.num_components2):
            ll_mat2[:, i] = self.components2[i].seq_log_density(enc_data2)

        ll_max1 = ll_mat1.max(axis=1, keepdims=True)
        ll_mat1 -= ll_max1
        np.exp(ll_mat1, out=ll_mat1)

        ll_max2 = ll_mat2.max(axis=1, keepdims=True)
        ll_mat2 -= ll_max2
        np.exp(ll_mat2, out=ll_mat2)

        ll_mat12 = np.dot(ll_mat1, self.taus12)
        ll_mat2 *= ll_mat12

        rv = np.log(ll_mat2.sum(axis=1)) + ll_max1[:, 0] + ll_max2[:, 0]

        return np.asarray(rv, dtype=float)

    def sampler(self, seed: Optional[int] = None) -> "JointMixtureSampler":
        """Return a sampler for paired observations."""
        return JointMixtureSampler(self, seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "JointMixtureEstimator":
        """Return an estimator for both component banks and their coupling."""
        estimators1 = [comp1.estimator() for comp1 in self.components1]
        estimators2 = [comp2.estimator() for comp2 in self.components2]

        pc = (
            None if pseudo_count is None else (pseudo_count, pseudo_count, pseudo_count)
        )
        return JointMixtureEstimator(
            estimators1=estimators1,
            estimators2=estimators2,
            pseudo_count=pc,
            keys=self.keys,
            name=self.name,
        )

    def dist_to_encoder(self) -> "DataSequenceEncoder":
        """Return an encoder that separates the two observed views."""
        encoder1 = self.components1[0].dist_to_encoder()
        encoder2 = self.components2[0].dist_to_encoder()
        return JointMixtureDataEncoder(encoder1=encoder1, encoder2=encoder2)


class JointMixtureSampler(DistributionSampler):
    """JointMixtureSampler object for sampling from a joint mixture distribution.

    Attributes:
        rng (RandomState): RandomState for seeding samples.
        dist (JointMixtureDistribution): Distribution to sample from.
        comp_sampler1 (DistributionSampler): Inner-mixture sampler.
        comp_sampler2 (DistributionSampler): Outer-mixture sampler.


    """

    def __init__(
        self, dist: JointMixtureDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize a joint-mixture sampler.

        Args:
            dist (JointMixtureDistribution): Distribution to sample from.
            seed (Optional[int]): Set seed for sampling.


        """
        super().__init__(dist, seed)
        self.comp_sampler1 = [
            d.sampler(seed=self.rng.randint(0, maxrandint))
            for d in self.dist.components1
        ]
        self.comp_sampler2 = [
            d.sampler(seed=self.rng.randint(0, maxrandint))
            for d in self.dist.components2
        ]

    def sample(
        self, size: Optional[int] = None
    ) -> Union[Tuple[Any, Any], Sequence[Tuple[Any, Any]]]:
        """Draw one paired observation or a collection of pairs."""
        if size is None:
            comp_state1 = self.rng.choice(
                range(0, self.dist.num_components1), replace=True, p=self.dist.w1
            )
            f1 = self.comp_sampler1[comp_state1].sample()
            comp_state2 = self.rng.choice(
                range(0, self.dist.num_components2),
                replace=True,
                p=self.dist.taus12[comp_state1, :],
            )
            f2 = self.comp_sampler2[comp_state2].sample()

            return f1, f2
        rv: List[Tuple[Any, Any]] = []
        for _ in range(size):
            sample = self.sample()
            assert isinstance(sample, tuple)
            rv.append(sample)
        return rv


class JointMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Aggregate sufficient statistics for both views and their latent pairing.

    ``comp_counts1`` and ``comp_counts2`` are marginal responsibility vectors with
    shapes ``(N,)`` and ``(M,)``. ``joint_counts`` has shape ``(N, M)`` and drives
    both conditional matrices. Child accumulators receive the corresponding marginal
    responsibilities for their view.

    Attributes:
        accumulators1 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators
            for the mixture components
            of X1.
        accumulators2 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators
            for the mixture components
            of X2.
        keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys
            for weights, mixture
            components of X1, mixture components of X2.
        num_components1 (int): Number of X1 mixture components.
        num_components2 (int): Number of X2 mixture components.
        comp_counts1 (np.ndarray): Weighted observation counts for states of mixture on
            X1.
        comp_counts2 (np.ndarray): Weighted observation counts for states of mixture on
            X2.
        joint_counts (np.ndarray): 2-d Numpy array for counts of state-given-state
            weights. Row indexed by states
            of X1, cols indexed by states of X2.
        name (Optional[str]): Set name to object.

        _rng_init (bool): Set to True once _rng_ members have been set.
        _idx1_rng (Optional[RandomState]): RandomState for generating states for X1 in
            initializer.
        _idx2_rng (Optional[RandomState]): RandomState for generating states for X2 in
            initializer.
        _acc1_rng (Optional[List[RandomState]]): List of RandomStates for initializing
            each accumulator for
            mixture components of X1.
        _acc2_rng (Optional[List[RandomState]]): List of RandomStates for initializing
            each accumulator for
            mixture components of X2.


    """

    def __init__(
        self,
        accumulators1: Sequence[SequenceEncodableStatisticAccumulator],
        accumulators2: Sequence[SequenceEncodableStatisticAccumulator],
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        name: Optional[str] = None,
    ) -> None:
        """Initialize a joint-mixture accumulator.

        Args:
            accumulators1 (Sequence[SequenceEncodableStatisticAccumulator]):
                Accumulators for the mixture components
                of X1.
            accumulators2 (Sequence[SequenceEncodableStatisticAccumulator]):
                Accumulators for the mixture components
                of X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set
                keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        """
        self.accumulators1 = accumulators1
        self.accumulators2 = accumulators2
        self.keys = keys if keys is not None else (None, None, None)
        self.num_components1 = len(accumulators1)
        self.num_components2 = len(accumulators2)
        self.comp_counts1 = vec.zeros(self.num_components1)
        self.comp_counts2 = vec.zeros(self.num_components2)
        self.joint_counts = vec.zeros((self.num_components1, self.num_components2))
        self.name = name

        self._rng_init = False
        self._idx1_rng: Optional[RandomState] = None
        self._idx2_rng: Optional[RandomState] = None
        self._acc1_rng: Optional[List[RandomState]] = None
        self._acc2_rng: Optional[List[RandomState]] = None

    def update(
        self, x: Tuple[T0, T1], weight: float, estimate: JointMixtureDistribution
    ) -> None:
        """Leave scalar posterior updating unsupported; use :meth:`seq_update`."""
        pass

    def _rng_initialize(self, rng: RandomState) -> None:
        self._idx1_rng = RandomState(seed=rng.randint(0, maxrandint))
        self._idx2_rng = RandomState(seed=rng.randint(0, maxrandint))
        self._acc1_rng = [
            RandomState(seed=rng.randint(0, maxrandint))
            for i in range(self.num_components1)
        ]
        self._acc2_rng = [
            RandomState(seed=rng.randint(0, maxrandint))
            for i in range(self.num_components2)
        ]
        self._rng_init = True

    def initialize(self, x: Tuple[T0, T1], weight: float, rng: RandomState) -> None:
        """Initialize one pair with a random joint latent assignment."""
        if not self._rng_init:
            self._rng_initialize(rng)

        assert self._idx1_rng is not None
        assert self._acc1_rng is not None
        assert self._acc2_rng is not None

        # idx1 = self._idx1_rng.choice(self.num_components1)
        # idx2 = self._idx2_rng.choice(self.num_components2)

        idx = self._idx1_rng.choice(self.num_components1 * self.num_components2)
        idx1, idx2 = idx // self.num_components2, idx % self.num_components2

        self.joint_counts[idx1, idx2] += 1.0

        for i in range(self.num_components1):
            w = 1.0 if i == idx1 else 0.0
            self.accumulators1[i].initialize(x[0], w, self._acc1_rng[i])
            self.comp_counts1[i] += w
        for i in range(self.num_components2):
            w = 1.0 if i == idx2 else 0.0
            self.accumulators2[i].initialize(x[1], w, self._acc2_rng[i])
            self.comp_counts2[i] += w

    def seq_initialize(
        self,
        x: "JointMixtureEncodedDataSequence",
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Initialize encoded pairs with random joint latent assignments."""
        sz, enc1, enc2 = x.data

        if not self._rng_init:
            self._rng_initialize(rng)

        assert self._idx1_rng is not None
        assert self._acc1_rng is not None
        assert self._acc2_rng is not None

        # idx1 = self._idx1_rng.choice(self.num_components1, size=sz)
        # idx2 = self._idx2_rng.choice(self.num_components2, size=sz)
        # temp = np.bincount(idx1*self.num_components1 + idx2,
        # minlength=self.num_components1*self.num_components2)

        idx = self._idx1_rng.choice(
            self.num_components1 * self.num_components2, size=sz
        )
        temp = np.bincount(idx, minlength=self.num_components1 * self.num_components2)
        idx1, idx2 = np.divmod(idx, self.num_components2)
        self.joint_counts += np.reshape(
            temp, (self.num_components1, self.num_components2)
        )

        for i in range(self.num_components1):
            w = np.zeros(sz)
            w[idx1 == i] = 1.0
            self.accumulators1[i].seq_initialize(enc1, w, self._acc1_rng[i])
            self.comp_counts1[i] += np.sum(w)

        for i in range(self.num_components2):
            w = np.zeros(sz)
            w[idx2 == i] = 1.0
            self.accumulators2[i].seq_initialize(enc2, w, self._acc2_rng[i])
            self.comp_counts2[i] += np.sum(w)

    def seq_update(
        self,
        x: "JointMixtureEncodedDataSequence",
        weights: np.ndarray,
        estimate: JointMixtureDistribution,
    ) -> None:
        """Accumulate posterior-weighted joint and marginal statistics."""
        sz, enc_data1, enc_data2 = x.data
        ll_mat1 = np.zeros((sz, self.num_components1, 1))
        ll_mat2 = np.zeros((sz, 1, self.num_components2))
        log_w = estimate.log_w1

        for i in range(estimate.num_components1):
            ll_mat1[:, i, 0] = estimate.components1[i].seq_log_density(enc_data1)
            ll_mat1[:, i, 0] += log_w[i]

        ll_max1 = ll_mat1.max(axis=1, keepdims=True)
        ll_mat1 -= ll_max1
        np.exp(ll_mat1, out=ll_mat1)

        for i in range(estimate.num_components2):
            ll_mat2[:, 0, i] = estimate.components2[i].seq_log_density(enc_data2)

        ll_max2 = ll_mat2.max(axis=2, keepdims=True)
        ll_mat2 -= ll_max2
        np.exp(ll_mat2, out=ll_mat2)

        ll_joint = ll_mat1 * ll_mat2
        ll_joint *= estimate.taus12

        gamma_2 = np.sum(ll_joint, axis=1, keepdims=True)
        sf = np.sum(gamma_2, axis=2, keepdims=True)
        ww = np.reshape(weights, [-1, 1, 1])

        gamma_1 = np.sum(ll_joint, axis=2, keepdims=True)
        gamma_1 *= ww / sf
        gamma_2 *= ww / sf

        ll_joint *= ww / sf

        self.comp_counts1 += np.sum(gamma_1, axis=0).flatten()
        self.comp_counts2 += np.sum(gamma_2, axis=0).flatten()
        self.joint_counts += ll_joint.sum(axis=0)

        for i in range(self.num_components1):
            self.accumulators1[i].seq_update(
                enc_data1, gamma_1[:, i, 0], estimate.components1[i]
            )

        for i in range(self.num_components2):
            self.accumulators2[i].seq_update(
                enc_data2, gamma_2[:, 0, i], estimate.components2[i]
            )

    def combine(
        self,
        suff_stat: Tuple[
            np.ndarray, np.ndarray, np.ndarray, Tuple[E0, ...], Tuple[E1, ...]
        ],
    ) -> "JointMixtureEstimatorAccumulator":
        """Merge another joint-mixture sufficient statistic."""
        cc1, cc2, jc, s1, s2 = suff_stat

        self.joint_counts += jc
        self.comp_counts1 += cc1
        for i in range(self.num_components1):
            self.accumulators1[i].combine(s1[i])
        self.comp_counts2 += cc2
        for i in range(self.num_components2):
            self.accumulators2[i].combine(s2[i])

        return self

    def value(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[Any, ...], Tuple[Any, ...]]:
        """Return marginal counts, joint counts, and child statistics."""
        return (
            self.comp_counts1,
            self.comp_counts2,
            self.joint_counts,
            tuple(u.value() for u in self.accumulators1),
            tuple(u.value() for u in self.accumulators2),
        )

    def from_value(
        self,
        x: Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[E0, ...], Tuple[E1, ...]],
    ) -> "JointMixtureEstimatorAccumulator":
        """Replace accumulated statistics from a serialized value."""
        cc1, cc2, jc, s1, s2 = x

        self.comp_counts1 = cc1
        self.comp_counts2 = cc2
        self.joint_counts = jc

        for i in range(self.num_components1):
            self.accumulators1[i].from_value(s1[i])
        for i in range(self.num_components2):
            self.accumulators2[i].from_value(s2[i])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge keyed latent-count and component statistics."""
        weight_key, acc1_key, acc2_key = self.keys

        if weight_key is not None:
            if weight_key in stats_dict:
                x1, x2, x3 = stats_dict[weight_key]
                self.comp_counts1 += x1
                self.comp_counts2 += x2
                self.joint_counts += x3

        if acc1_key is not None:
            if acc1_key in stats_dict:
                for i, u in enumerate(stats_dict[acc1_key]):
                    self.accumulators1[i].combine(u)
            else:
                stats_dict[acc1_key] = tuple(acc.value() for acc in self.accumulators1)

        if acc2_key is not None:
            if acc2_key in stats_dict:
                for i, u in enumerate(stats_dict[acc2_key]):
                    self.accumulators2[i].combine(u)
            else:
                stats_dict[acc2_key] = tuple(acc.value() for acc in self.accumulators2)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace statistics from matching keyed values."""
        weight_key, acc1_key, acc2_key = self.keys
        if weight_key is not None:
            if weight_key in stats_dict:
                x1, x2, x3 = stats_dict[weight_key]
                self.comp_counts1 = x1
                self.comp_counts2 = x2
                self.joint_counts = x3

        if acc1_key is not None:
            if acc1_key in stats_dict:
                for i, u in enumerate(stats_dict[acc1_key]):
                    self.accumulators1[i].from_value(u)

        if acc2_key is not None:
            if acc2_key in stats_dict:
                for i, u in enumerate(stats_dict[acc2_key]):
                    self.accumulators2[i].from_value(u)

    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Return an encoder for both child accumulator banks."""
        encoder1 = self.accumulators1[0].acc_to_encoder()
        encoder2 = self.accumulators2[0].acc_to_encoder()
        return JointMixtureDataEncoder(encoder1=encoder1, encoder2=encoder2)


class JointMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Create joint-mixture accumulators.

    Attributes:
        factories1 (Sequence[StatisticAccumulatorFactory]): List of mixture component
            factories for X1.
        factories2 (Sequence[StatisticAccumulatorFactory]): List of mixture component
            factories for X2.
        keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys
            for weights, mixture
            components of X1, mixture components of X2.
        name (Optional[str]): Set name to object.

    """

    def __init__(
        self,
        factories1: Sequence[StatisticAccumulatorFactory],
        factories2: Sequence[StatisticAccumulatorFactory],
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        name: Optional[str] = None,
    ) -> None:
        """Initialize a joint-mixture accumulator factory.

        Args:
            factories1 (Sequence[StatisticAccumulatorFactory]): List of mixture
                component factories for X1.
            factories2 (Sequence[StatisticAccumulatorFactory]): List of mixture
                component factories for X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set
                keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        """
        self.factories1 = factories1
        self.factories2 = factories2
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def make(self) -> "JointMixtureEstimatorAccumulator":
        """Create a joint-mixture accumulator."""
        f1 = [self.factories1[i].make() for i in range(len(self.factories1))]
        f2 = [self.factories2[i].make() for i in range(len(self.factories2))]
        return JointMixtureEstimatorAccumulator(f1, f2, name=self.name, keys=self.keys)


class JointMixtureEstimator(ParameterEstimator):
    """Estimate a joint mixture from aggregated sufficient statistics.

    Notes:
        ``keys`` identifies the statistic blocks that may be shared with other
        JointMixture estimators using the same key values.

        - ``keys[0]`` corresponds to the joint latent-count block used for the
          marginal weights and cross-view transition matrices.
        - ``keys[1]`` shares the ``X1`` component sufficient statistics by
          component index.
        - ``keys[2]`` shares the ``X2`` component sufficient statistics by
          component index.

        Tuple positions allow partial sharing; they do not tie the whole joint
        mixture unless all relevant positions use matching keys. Component-bank keys
        assume component index ``i`` has the same meaning in every estimator that
        shares the key. Treat keyed sharing of the joint latent-count block carefully
        and verify it in the fitted model before relying on it.

    Attributes:
        estimators1 (Sequence[ParameterEstimator]): Estimators for mixture component of
            X1.
        estimators2 (Sequence[ParameterEstimator]): Estimators for mixture component of
            X2.
        suff_stat (Optional[Tuple[Any, ...]]): Optional reference statistics.
        pseudo_count (Optional[Tuple[float, float, float]]): Used to re-weight the state
            counts in estimation.
        keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Keys for
            joint latent counts, ``X1`` component statistics, and ``X2`` component
            statistics.
        name (Optional[str]): Set name to object.

    """

    def __init__(
        self,
        estimators1: Sequence[ParameterEstimator],
        estimators2: Sequence[ParameterEstimator],
        suff_stat: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[E0, ...], Tuple[E1, ...]]
        ] = None,
        pseudo_count: Optional[Tuple[float, float, float]] = None,
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        name: Optional[str] = None,
    ) -> None:
        """Initialize a joint-mixture estimator.

        Args:
            estimators1 (Sequence[ParameterEstimator]): Estimators for mixture component
                of X1.
            estimators2 (Sequence[ParameterEstimator]): Estimators for mixture component
                of X2.
            suff_stat (Optional[Tuple[Any, ...]]): Optional reference statistics.
            pseudo_count (Optional[Tuple[float, float, float]]): Used to re-weight the
                state counts in estimation.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Keys
                that control which sufficient-statistic blocks are shared. ``keys[0]``
                refers to the joint latent-count block used for weights and
                cross-view transition matrices. ``keys[1]`` and ``keys[2]`` share
                the ``X1`` and ``X2`` component accumulators positionally. Shared
                component keys require aligned component ordering across estimators.
            name (Optional[str]): Set name to object.

        """
        self.num_components1 = len(estimators1)
        self.num_components2 = len(estimators2)
        self.estimators1 = estimators1
        self.estimators2 = estimators2
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def accumulator_factory(self) -> "JointMixtureEstimatorAccumulatorFactory":
        """Return a factory for the two component-estimator banks."""
        est_factories1 = [u.accumulator_factory() for u in self.estimators1]
        est_factories2 = [u.accumulator_factory() for u in self.estimators2]
        return JointMixtureEstimatorAccumulatorFactory(
            est_factories1, est_factories2, name=self.name, keys=self.keys
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[
            np.ndarray, np.ndarray, np.ndarray, Tuple[E0, ...], Tuple[E1, ...]
        ],
    ) -> "JointMixtureDistribution":
        """Estimate component banks, marginals, and conditional matrices."""
        num_components1 = self.num_components1
        num_components2 = self.num_components2
        counts1, counts2, joint_counts, comp_suff_stats1, comp_suff_stats2 = suff_stat

        components1 = [
            self.estimators1[i].estimate(counts1[i], comp_suff_stats1[i])
            for i in range(num_components1)
        ]
        components2 = [
            self.estimators2[i].estimate(counts2[i], comp_suff_stats2[i])
            for i in range(num_components2)
        ]

        if self.pseudo_count is not None and self.suff_stat is None:
            p1 = self.pseudo_count[0] / float(self.num_components1)
            p2 = self.pseudo_count[1] / float(self.num_components2)
            p3 = self.pseudo_count[2] / float(
                self.num_components2 * self.num_components1
            )

            w1 = (counts1 + p1) / (counts1.sum() + p1)
            w2 = (counts2 + p2) / (counts2.sum() + p2)
            taus = joint_counts + p3

            taus12_sum = np.sum(taus, axis=1, keepdims=True)
            taus12_sum[taus12_sum == 0] = 1.0
            taus12 = taus / taus12_sum

            taus21_sum = np.sum(taus, axis=0, keepdims=True)
            taus21_sum[taus21_sum == 0] = 1.0
            taus21 = taus / taus21_sum

        else:
            w1 = counts1 / counts1.sum()
            w2 = counts2 / counts2.sum()
            taus = joint_counts

            taus12_sum = np.sum(taus, axis=1, keepdims=True)
            taus12_sum[taus12_sum == 0] = 1.0
            taus12 = taus / taus12_sum

            taus21_sum = np.sum(taus, axis=0, keepdims=True)
            taus21_sum[taus21_sum == 0] = 1.0
            taus21 = taus / taus21_sum

        return JointMixtureDistribution(
            components1, components2, w1, w2, taus12, taus21, name=self.name
        )


class JointMixtureDataEncoder(DataSequenceEncoder):
    """Encode batches of paired joint-mixture observations.

    A batch of size ``B`` becomes ``(B, enc_data1, enc_data2)``. Each child encoding
    contains all ``B`` values from its view and retains observation order.

    Attributes:
        encoder1 (DataSequenceEncoder): DataSequenceEncoder for the components of X1.
        encoder2 (DataSequenceEncoder): DataSequenceEncoder for the components of X2.

    """

    def __init__(
        self, encoder1: DataSequenceEncoder, encoder2: DataSequenceEncoder
    ) -> None:
        """Initialize a joint-mixture data encoder.

        Args:
            encoder1 (DataSequenceEncoder): DataSequenceEncoder for the components of
                X1.
            encoder2 (DataSequenceEncoder): DataSequenceEncoder for the components of
                X2.

        """
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def __str__(self) -> str:
        """Return a string representation of the encoder."""
        return (
            "JointMixtureDataEncoder(encoder0="
            + str(self.encoder1)
            + ",encoder1="
            + str(self.encoder2)
            + ")"
        )

    def __eq__(self, other: object) -> bool:
        """Return whether both view encoders are equal."""
        if isinstance(other, JointMixtureDataEncoder):
            return self.encoder2 == other.encoder2 and self.encoder1 == other.encoder1
        return False

    def seq_encode(
        self, x: Sequence[Tuple[T0, T1]]
    ) -> "JointMixtureEncodedDataSequence":
        """Split and encode a batch of paired observations."""
        rv0 = len(x)
        rv1 = self.encoder1.seq_encode([u[0] for u in x])
        rv2 = self.encoder2.seq_encode([u[1] for u in x])

        return JointMixtureEncodedDataSequence(data=(rv0, rv1, rv2))


class JointMixtureEncodedDataSequence(EncodedDataSequence):
    """Store batch size and aligned encodings for the two observed views."""

    def __init__(self, data: Tuple[int, EncodedDataSequence, EncodedDataSequence]):
        """Initialize an encoded joint-mixture batch."""
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation of the encoded batch."""
        return f"JointMixtureEncodedDataSequence(data={self.data})"
