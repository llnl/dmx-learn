r"""Provide integer-valued Chow--Liu tree distributions.

An observation is a fixed-length vector :math:`x=(x_0,\ldots,x_{F-1})` of
nonnegative categories. ``dependency_list`` gives a rooted feature tree as
``(feature, parent)`` pairs, with one ``None`` parent. The model is

.. math::

   p(x) = p(x_r)\prod_{(j,p)\in\mathcal{T},\,p\ne\mathrm{None}}
          p(x_j\mid x_p).

The accumulator collects feature marginals and pairwise category tables. The
estimator uses pairwise mutual information to find a maximum-dependence
spanning tree, roots it at feature zero, then estimates the conditional PMFs.
This is the Chow--Liu tree approximation (Chow and Liu, 1968).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class ICLTreeDistribution(SequenceEncodableProbabilityDistribution):
    """Represent a directed Chow--Liu tree over integer feature vectors.

    Attributes:
        feature_order (Sequence[int]): Ordering of features. If None, ordering is
            assumed as entered.
        dependency_list (List[ Tuple[int, Tuple[int, Optional[int]]]]): List of Tuples
            containing features
            order id and Tuple of feature and feature dep.
        conditional_log_densities (Union[Sequence[float], np.ndarray]): Conditional log
            densities for each features'
            dependency split.
        conditional_densities (np.ndarray): Conditional densities as numpy array.
        num_features (int): Total number of features.
        name (Optional[str]): Name for object instance.
        keys (Optional[str]): Keys for parameters of model.

    """

    def __init__(
        self,
        dependency_list: Sequence[Tuple[int, Optional[int]]],
        conditional_log_densities: Sequence[Any],
        feature_order: Optional[Sequence[int]] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """Initialize a directed integer Chow--Liu tree.

        Args:
            dependency_list (List[Tuple[int, Optional[int]]]): List of Tuples containing
                node id and parent dependence
                if any dependence is present.
            conditional_log_densities (Union[Sequence[float], np.ndarray]): Conditional
                log densities for each features'
                dependency split.
            feature_order (Optional[Sequence[int]]): Ordering of features. If None,
                ordering is assumed as entered.
            name (Optional[str]): Set name to object.
            keys (Optional[str]): Keys for parameters of model.

        """
        super().__init__()
        self.feature_order = list(
            range(len(dependency_list)) if feature_order is None else feature_order
        )
        self.dependency_list = (
            list(dependency_list)
            if feature_order is None
            else [dependency_list[i] for i in self.feature_order]
        )
        self.conditional_log_densities = [
            np.asarray(u, dtype=float) for u in conditional_log_densities
        ]
        self.conditional_densities = [np.exp(u) for u in self.conditional_log_densities]
        self.num_features = len(dependency_list)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return an evaluable distribution representation."""
        f1 = ",".join([str(u[1]) for u in self.dependency_list])
        f3 = ",".join([str(u[0]) for u in self.dependency_list])
        f2 = [
            "[" + ",".join(map(str, u.flatten())) + "]"
            for u in self.conditional_log_densities
        ]
        f4 = repr(self.name)
        f5 = repr(self.keys)
        return (
            f"ICLTreeDistribution([{f1}], [{f2}], feature_order=[{f3}], name={f4}, "
            f"keys={f5})"
        )

    def density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        """Return the probability of one integer feature vector."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        """Return the log probability of one integer feature vector."""
        x_arr = np.asarray(x, dtype=int)
        rv = 0.0
        for i, (j, k) in enumerate(self.dependency_list):
            if k is None:
                rv += float(self.conditional_log_densities[i][x_arr[j]])
            else:
                rv += float(self.conditional_log_densities[i][x_arr[k], x_arr[j]])

        return rv

    def seq_log_density(self, x: "ICLTreeEncodedDataSequence") -> np.ndarray:
        """Return log probabilities for an encoded batch of feature vectors."""
        if not isinstance(x, ICLTreeEncodedDataSequence):
            raise TypeError("Requires ICLTreeEncodedDataSequence.")

        rv = np.zeros(x.data.shape[0])
        for i, (j, k) in enumerate(self.dependency_list):
            if k is None:
                rv += self.conditional_log_densities[i][x.data[:, j]]
            else:
                rv += self.conditional_log_densities[i][x.data[:, k], x.data[:, j]]

        return rv

    def sampler(self, seed: Optional[int] = None) -> "ICLTreeSampler":
        """Create a sampler that draws features in tree order."""
        return ICLTreeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "ICLTreeEstimator":
        """Create an estimator; this convenience method ignores pseudo-counts."""
        return ICLTreeEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "ICLTreeDataEncoder":
        """Return the encoder for fixed-length integer vectors."""
        return ICLTreeDataEncoder()


class ICLTreeSampler(DistributionSampler):
    """Sample independent integer vectors from an ICL tree.

    Attributes:
          rng (RandomState): RandomState for setting sampling seed.
          dist (ICLTreeDistribution): ICL Tree distribution to sample from.

    """

    def __init__(self, dist: ICLTreeDistribution, seed: Optional[int] = None) -> None:
        """Initialize an ICL tree sampler.

        Args:
              dist (ICLTreeDistribution): ICL Tree distribution to sample from.
              seed (Optional[int]): Seed passed to random number generator.

        """
        super().__init__(dist, seed)

    def sample(
        self, size: Optional[int] = None
    ) -> Union[List[Optional[int]], Sequence[List[Optional[int]]]]:
        """Draw one vector, or ``size`` independently drawn vectors."""
        if size is None:
            rv: List[Optional[int]] = [None] * self.dist.num_features
            for i, (j, k) in enumerate(self.dist.dependency_list):
                if k is None:
                    pmat = self.dist.conditional_densities[i]
                else:
                    parent = rv[k]
                    assert parent is not None
                    pmat = self.dist.conditional_densities[i][parent, :]

                rv[j] = int(self.rng.choice(len(pmat), p=pmat))

            return rv
        samples: List[List[Optional[int]]] = []
        for _ in range(size):
            sample = self.sample()
            assert isinstance(sample, list)
            samples.append(sample)
        return samples


class ICLTreeAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate marginal and pairwise counts for Chow--Liu fitting."""

    def __init__(
        self,
        num_features: Optional[int],
        num_states: Optional[int],
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize optional feature and category count dimensions."""
        self.num_states = num_states
        self.num_features = num_features
        self.counts: Optional[np.ndarray]
        self.marginal_counts: Optional[np.ndarray]

        if num_states is not None and num_features is not None:
            self.counts = np.zeros((num_features, num_features, num_states, num_states))
            self.marginal_counts = np.zeros((num_features, num_states))
        else:
            self.counts = None
            self.marginal_counts = None

        self.key = keys
        self.name = name

    def _expand_states(self, num_states: int, num_features: int) -> None:
        """Allocate or expand category-count arrays to observed dimensions."""
        if (
            (self.counts is None)
            and (num_states is not None)
            and (num_features is not None)
        ):
            self.num_features = num_features
            self.num_states = num_states
            self.counts = np.zeros((num_features, num_features, num_states, num_states))
            self.marginal_counts = np.zeros((num_features, num_states))

        elif (
            (self.counts is not None)
            and (num_states is not None)
            and (num_features is not None)
        ):
            old_num_states = self.num_states
            new_counts = np.zeros((num_features, num_features, num_states, num_states))
            new_marginal = np.zeros((num_features, num_states))
            new_counts[:, :, :old_num_states, :old_num_states] = self.counts
            new_marginal[:, :old_num_states] = self.marginal_counts
            self.num_features = num_features
            self.num_states = num_states
            self.counts = new_counts
            self.marginal_counts = new_marginal

    def update(
        self,
        x: Union[Sequence[int], np.ndarray],
        weight: float,
        estimate: Optional[ICLTreeDistribution],
    ) -> None:
        """Accumulate weighted counts from one feature vector."""
        if (
            (self.counts is None)
            or (self.num_states is None)
            or (self.num_states <= np.max(x))
        ):
            self._expand_states(max(x) + 1, len(x))

        assert self.counts is not None
        assert self.marginal_counts is not None
        assert self.num_features is not None
        xx = np.asarray(x)
        ff = np.arange(self.num_features)

        self.marginal_counts[ff, xx] += weight
        for i in range(self.num_features):
            self.counts[i, ff, xx[i], xx] += weight

    def seq_update(
        self,
        x: "ICLTreeEncodedDataSequence",
        weights: np.ndarray,
        estimate: Optional[ICLTreeDistribution],
    ) -> None:
        """Accumulate weighted counts from an encoded batch."""
        max_x = int(np.max(x.data))

        if (
            (self.counts is None)
            or (self.num_states is None)
            or (self.num_states <= max_x)
        ):
            self._expand_states(max_x + 1, x.data.shape[1])

        num_states = self.num_states
        assert num_states is not None
        assert self.num_features is not None
        assert self.counts is not None
        assert self.marginal_counts is not None

        for i in range(self.num_features):
            self.marginal_counts[i, :] += np.bincount(
                x.data[:, i], weights=weights, minlength=num_states
            )

            for j in range(i + 1, self.num_features):
                joint_idx = x.data[:, i] * num_states + x.data[:, j]
                joint_cnt = np.bincount(
                    joint_idx, weights=weights, minlength=(num_states * num_states)
                )
                joint_cnt = np.reshape(joint_cnt, (num_states, num_states))

                self.counts[i, j, :, :] += joint_cnt

    def initialize(
        self,
        x: Union[Sequence[int], np.ndarray],
        weight: float,
        rng: Optional[RandomState],
    ) -> None:
        """Initialize counts from one observation without randomization."""
        del rng
        self.update(x, weight, None)

    def seq_initialize(
        self,
        x: "ICLTreeEncodedDataSequence",
        weights: np.ndarray,
        rng: Optional[RandomState],
    ) -> None:
        """Initialize counts from an encoded batch without randomization."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: Tuple[int, int, np.ndarray, np.ndarray]
    ) -> "ICLTreeAccumulator":
        """Combine another marginal-and-pairwise count tuple."""
        num_features, num_states, counts, marginal_counts = suff_stat

        if self.counts is None and counts is None:
            return self

        if (self.counts is None) and (counts is not None):
            self.counts = counts
            self.marginal_counts = marginal_counts
            self.num_features = num_features
            self.num_states = num_states

        elif self.counts is not None and counts is None:
            pass

        else:
            assert self.num_states is not None
            assert self.counts is not None
            assert self.marginal_counts is not None
            if self.num_states < num_states:
                self._expand_states(num_states, num_features)
                assert self.counts is not None
                assert self.marginal_counts is not None
                self.counts += counts
                self.marginal_counts += marginal_counts

            elif self.num_states > num_states:
                self.counts[:, :, :num_states, :num_states] += counts
                self.marginal_counts[:, :num_states] += marginal_counts

            else:
                self.counts += counts
                self.marginal_counts += marginal_counts

        return self

    def value(self) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """Return feature count, category count, and count arrays."""
        assert self.num_features is not None
        assert self.num_states is not None
        assert self.counts is not None
        assert self.marginal_counts is not None
        return self.num_features, self.num_states, self.counts, self.marginal_counts

    def from_value(
        self, x: Tuple[int, int, np.ndarray, np.ndarray]
    ) -> "ICLTreeAccumulator":
        """Restore count arrays from a sufficient-statistic tuple."""
        self.num_features = x[0]
        self.num_states = x[1]
        self.counts = x[2]
        self.marginal_counts = x[3]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Leave keyed merging unimplemented for this accumulator."""

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Leave keyed replacement unimplemented for this accumulator."""

    def acc_to_encoder(self) -> "ICLTreeDataEncoder":
        """Return the encoder for accumulated integer vectors."""
        return ICLTreeDataEncoder()


class ICLTreeAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for integer Chow--Liu sufficient statistics."""

    def __init__(
        self,
        num_features: Optional[int] = None,
        num_states: Optional[int] = None,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize optional dimensions and estimator metadata."""
        self.num_features = num_features
        self.num_states = num_states
        self.keys = keys
        self.name = name

    def make(self) -> "ICLTreeAccumulator":
        """Create a fresh ICL tree accumulator."""
        return ICLTreeAccumulator(self.num_features, self.num_states, self.keys)


class ICLTreeEstimator(ParameterEstimator):
    """Fit a mutual-information spanning-tree approximation to integer data."""

    def __init__(
        self,
        num_features: Optional[int] = None,
        num_states: Optional[int] = None,
        pseudo_count: Optional[float] = None,
        suff_stat: Optional[Any] = None,
        keys: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the accumulator factory dimensions."""
        self.num_features = num_features
        self.num_states = num_states
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "ICLTreeAccumulatorFactory":
        """Return a factory for the estimator's count arrays."""
        return ICLTreeAccumulatorFactory(self.num_features, self.num_states, self.keys)

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[int, int, np.ndarray, np.ndarray]
    ) -> "ICLTreeDistribution":
        """Estimate a rooted Chow--Liu tree and its conditional PMFs."""
        num_features, num_states, counts, marginal_counts = suff_stat

        mi_mat = np.zeros((num_features, num_features))

        pseudo_count = self.pseudo_count if self.pseudo_count is not None else 0.0
        pseudo_count_adj0 = pseudo_count / num_states
        pseudo_count_adj1 = pseudo_count / (num_states * num_states)

        for i in range(num_features - 1):
            for j in range(i + 1, num_features):
                joint_ij = counts[i, j, :, :] + pseudo_count_adj1
                indep_ij = (
                    np.outer(marginal_counts[i, :], marginal_counts[j, :])
                    + pseudo_count_adj1
                )

                joint_ij_sum = joint_ij.sum()
                indep_ij_sum = indep_ij.sum()

                if joint_ij_sum > 0:
                    joint_ij /= joint_ij_sum
                if indep_ij_sum > 0:
                    indep_ij /= indep_ij_sum

                good = np.bitwise_and(joint_ij > 0, indep_ij > 0)

                if good.sum() > 0:
                    mi_val = (
                        joint_ij[good]
                        * (np.log(joint_ij[good]) - np.log(indep_ij[good]))
                    ).sum()
                    mi_mat[i, j] = 1.0 + mi_val

                else:
                    mi_mat[i, j] = 1.0

        cost_mat = np.abs((mi_mat.max() - mi_mat))
        cost_mat[mi_mat > 0] += 1.0
        cost_mat[mi_mat == 0] = 0

        span_tree = minimum_spanning_tree(cost_mat)

        root_node = 0
        feature_order, deps = breadth_first_order(
            span_tree, root_node, directed=False, return_predecessors=True
        )

        deps_list: List[Optional[int]] = [int(deps[i]) for i in feature_order]
        tmats: List[np.ndarray] = [
            np.empty(0, dtype=float) for _ in range(num_features)
        ]

        with np.errstate(divide="ignore"):

            root_marginal = marginal_counts[root_node, :] + pseudo_count_adj0
            tmats[0] = np.log(root_marginal / (root_marginal.sum()))
            deps_list[0] = None

            for i in range(1, num_features):
                n = feature_order[i]
                p = deps_list[i]
                assert p is not None

                if p < n:
                    tmat = counts[p, n, :, :]
                else:
                    tmat = counts[n, p, :, :].T

                tmat = tmat + pseudo_count_adj1
                tmat_sum = np.sum(tmat, axis=1, keepdims=True)
                tmat_sum[tmat_sum == 0] = 1.0
                tmat /= tmat_sum

                tmats[i] = np.log(tmat)

        dependency_list = [
            (int(feature_order[i]), deps_list[i]) for i in range(num_features)
        ]
        return ICLTreeDistribution(dependency_list, tmats)


class ICLTreeDataEncoder(DataSequenceEncoder):
    """ICLTreeDataEncoder object for encoding sequences of iid ICL observations."""

    def __str__(self) -> str:
        """Return an evaluable encoder representation."""
        return "ICLTreeDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is an ICL tree encoder."""
        return isinstance(other, ICLTreeDataEncoder)

    def seq_encode(
        self, x: Union[List[int], np.ndarray]
    ) -> "ICLTreeEncodedDataSequence":
        """Encode vectors as an integer array with shape ``(N, F)``."""
        return ICLTreeEncodedDataSequence(data=np.asarray(x, dtype=int))


class ICLTreeEncodedDataSequence(EncodedDataSequence):
    """Hold encoded integer vectors with batch-first shape ``(N, F)``.

    Attributes:
        data (np.ndarray): Numpy array of observations.

    """

    def __init__(self, data: np.ndarray):
        """Initialize encoded integer feature vectors.

        Args:
            data (np.ndarray): Numpy array of observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing encoded vector data."""
        return f"ICLTreeEncodedDataSequence(data={self.data})"
