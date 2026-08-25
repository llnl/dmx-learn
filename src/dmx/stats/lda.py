r"""Provide latent Dirichlet allocation for sparse counted documents.

A document is a sparse sequence ``[(x_0, c_0), ..., (x_m, c_m)]`` whose values
are accepted by each topic distribution and whose nonnegative counts give token
multiplicity. For :math:`K` topics, the model draws document proportions
:math:`\theta_d\sim\operatorname{Dirichlet}(\alpha)`, then assigns each token a
topic :math:`z_{dn}\sim\operatorname{Categorical}(\theta_d)` and draws its value
from the corresponding topic distribution. A separate length distribution supplies
the number of tokens when sampling.

The distribution's ``log_density`` methods return the variational evidence lower
bound (ELBO) computed by the document-posterior iteration, not the exact marginal
log-likelihood. Accumulator updates use the resulting token-topic responsibilities.
Random initialization draws document-topic proportions from the previous ``alpha``
(or an all-ones vector) and randomly allocates topic statistics.
"""

import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import maxrandint
from dmx.stats.dirichlet import DirichletDistribution
from dmx.stats.null_dist import NullDistribution
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.special import digamma, digammainv, gammaln
from dmx.utils.vector import row_choice

E0 = TypeVar("E0")
SS0 = TypeVar("SS0")

# import dmx.c_ext


class LDADistribution(SequenceEncodableProbabilityDistribution):
    """Represent a latent Dirichlet allocation model.

    Model definition:
        ``topics[k]`` is the conditional value distribution for latent topic ``k``;
        ``alpha[k]`` is the positive Dirichlet concentration for its document-level
        proportion. ``gamma_threshold`` controls only variational inference and is
        not a model parameter.

    Attributes:
        topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions
            for the LDA.
        alpha (np.ndarray): Parameter to the prior Dirichlet for which topics are drawn.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution for length of
            documents.
            Must be set to non-negative support distribution for sampling. Default to
            NullDistribution.
        gamma_threshold (float): For numerical stability in estimation.

    """

    def __init__(
        self,
        topics: Sequence[SequenceEncodableProbabilityDistribution],
        alpha: Union[Sequence[float], np.ndarray],
        len_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        gamma_threshold: float = 1.0e-8,
        keys: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None,
    ) -> None:
        """Initialize an LDA distribution.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic
                distributions for the LDA.
            alpha (Union[Sequence[float], np.ndarray]): Parameter to the prior Dirichlet
                for which topics are drawn.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution
                for length of documents.
                Must be set to non-negative support distribution for sampling.
            gamma_threshold (float): For numerical stability in estimation.
            keys: Keys for sharing alpha and topic sufficient statistics.
            name: Optional model name.

        """
        super().__init__()
        self.topics = topics
        self.n_topics = len(topics)
        self.alpha = np.asarray(alpha)
        self.len_dist = len_dist
        self.gamma_threshold = gamma_threshold
        self.keys = keys
        self.name = name

    def __str__(self) -> str:
        """Return a constructor-like representation of the distribution."""
        s0 = ",".join([str(u) for u in self.topics])
        s1 = ",".join(map(str, self.alpha))
        s2 = repr(self.len_dist)
        s3 = repr(self.gamma_threshold)
        s4 = repr(self.keys)
        s5 = repr(self.name)

        rv = (s0, s1, s2, s3, s4, s5)

        return (
            f"LDADistribution(topics=[{rv[0]}], alpha=[{rv[1]}], len_dist={rv[2]}, "
            f"gamma_threshold={rv[3]}, keys={rv[4]}, name={rv[5]})"
        )

    def density(self, x: Sequence[Tuple[int, float]]) -> float:
        """Exponentiate the variational lower bound for one document."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[Tuple[int, float]]) -> float:
        """Return the variational evidence lower bound for one document."""
        enc_x = self.dist_to_encoder().seq_encode([x])
        return float(self.seq_log_density(enc_x)[0])

    def seq_log_density(self, x: "LDAEncodedDataSequence") -> np.ndarray:
        """Return the variational evidence lower bound for encoded documents.

        The returned vector has one entry per document. Despite the protocol method
        name, these values are ELBOs obtained after iterative variational posterior
        updates, rather than exact marginal log-likelihoods.
        """
        if not isinstance(x, LDAEncodedDataSequence):
            raise TypeError("Requires LDAEncodedDataSequence for `seq` function calls.")

        num_topics = self.n_topics
        alpha = self.alpha
        _num_documents, idx, _counts, _, _enc_data = x.data

        idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
        idx_full *= num_topics
        idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

        log_density_gamma, document_gammas, per_topic_log_densities = seq_posterior(
            self, x.data
        )

        # This block keeps the gammas positive
        log_density_gamma[
            np.bitwise_or(np.isnan(log_density_gamma), np.isinf(log_density_gamma))
        ] = sys.float_info.min
        log_density_gamma[log_density_gamma <= 0] = sys.float_info.min
        document_gammas[
            np.bitwise_or(np.isnan(document_gammas), np.isinf(document_gammas))
        ] = sys.float_info.min

        elob0 = digamma(document_gammas) - digamma(
            np.sum(document_gammas, axis=1, keepdims=True)
        )
        elob1 = elob0[idx, :]
        elob2 = log_density_gamma * (
            elob1 + per_topic_log_densities - np.log(log_density_gamma)
        )
        elob3 = np.sum(elob0 * ((alpha - 1.0) - (document_gammas - 1.0)), axis=1)
        elob4 = np.bincount(idx_full.flat, weights=elob2.flat)
        elob5 = np.sum(np.reshape(elob4, (-1, num_topics)), axis=1)
        elob6 = np.sum(gammaln(document_gammas), axis=1) - gammaln(
            document_gammas.sum(axis=1)
        )
        elob7 = gammaln(alpha.sum()) - gammaln(alpha).sum()

        elob = elob3 + elob5 + elob6 + elob7

        return np.asarray(elob, dtype=np.float64)

    def seq_component_log_density(self, x: "LDAEncodedDataSequence") -> np.ndarray:
        """Return count-weighted value log-densities for each document and topic."""
        if not isinstance(x, LDAEncodedDataSequence):
            raise TypeError("Requires LDAEncodedDataSequence for `seq` function calls.")

        num_topics = self.n_topics
        num_documents, idx, counts, _, enc_data = x.data

        ll_mat = np.zeros((len(idx), self.n_topics))
        ll_mat.fill(-np.inf)

        rv = np.zeros((num_documents, self.n_topics))
        rv.fill(-np.inf)

        for i in range(num_topics):
            ll_mat[:, i] = self.topics[i].seq_log_density(enc_data)
            rv[:, i] = np.bincount(
                idx, weights=ll_mat[:, i] * counts, minlength=num_documents
            )

        return rv

    def seq_posterior(self, x: "LDAEncodedDataSequence") -> np.ndarray:
        """Return normalized variational document-topic parameters."""
        if not isinstance(x, LDAEncodedDataSequence):
            raise TypeError("Requires LDAEncodedDataSequence for `seq` function calls.")

        _log_density_gamma, document_gammas, _per_topic_log_densities = seq_posterior(
            self, x.data
        )

        document_gammas /= document_gammas.sum(axis=1, keepdims=True)

        return np.asarray(document_gammas, dtype=np.float64)

    def sampler(self, seed: Optional[int] = None) -> "LDASampler":
        """Create a sampler for this LDA distribution."""
        return LDASampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "LDAEstimator":
        """Create an estimator for the topic models and Dirichlet parameter."""
        if pseudo_count is None:
            return LDAEstimator(
                estimators=[d.estimator() for d in self.topics],
                name=self.name,
                keys=self.keys,
            )
        return LDAEstimator(
            estimators=[d.estimator() for d in self.topics],
            pseudo_count=(pseudo_count, pseudo_count),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "LDADataEncoder":
        """Create a document encoder using the first topic's value encoder."""
        return LDADataEncoder(encoder=self.topics[0].dist_to_encoder())


class LDASampler(DistributionSampler):
    """Sample sparse documents from an LDA distribution."""

    def __init__(self, dist: LDADistribution, seed: Optional[int] = None) -> None:
        """Initialize an LDA sampler from ``dist`` and an optional seed."""
        super().__init__(dist, seed)
        self.dist = dist
        self.n_topics = dist.n_topics
        self.comp_samplers = [
            self.dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint))
            for i in range(dist.n_topics)
        ]
        self.dirichlet_sampler = DirichletDistribution(dist.alpha).sampler(
            self.rng.randint(0, maxrandint)
        )
        self.len_dist = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def _sample_single(self) -> List[Tuple[Any, int]]:
        n = int(self.len_dist.sample())
        weights = np.asarray(self.dirichlet_sampler.sample(), dtype=np.float64)
        topic_counts = self.rng.multinomial(n, pvals=weights)
        rv: List[Any] = []
        for i in np.flatnonzero(topic_counts):
            samples = self.comp_samplers[int(i)].sample(size=int(topic_counts[i]))
            rv.extend(cast(Sequence[Any], samples))
        return cast(List[Tuple[Any, int]], rv)

    def sample(
        self, size: Optional[int] = None
    ) -> Union[Sequence[List[Tuple[Any, int]]], List[Tuple[Any, int]]]:
        """Draw one document or a sequence of documents."""
        if size is None:
            return self._sample_single()

        return [self._sample_single() for i in range(size)]


class LDAEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate variational LDA sufficient statistics.

    The stored values are the previous Dirichlet parameter, sums of expected
    log document-topic proportions, weighted document mass, topic responsibility
    mass, and the sufficient statistics of each topic model.
    """

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
        prev_alpha: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize an LDA sufficient-statistic accumulator.

        Args:
            accumulators: One value-distribution accumulator per topic.
            name: Optional accumulator name.
            keys: Keys for sharing alpha-update and topic statistics.
            prev_alpha: Dirichlet parameter used for random initialization.
        """
        self.accumulators = accumulators
        self.num_topics = len(accumulators)
        self.sum_of_logs = np.zeros(self.num_topics)
        self.doc_counts = 0.0
        self.topic_counts = np.zeros(self.num_topics)
        self.prev_alpha = prev_alpha
        alpha_key, topics_key = keys if keys is not None else (None, None)
        self.alpha_key = alpha_key
        self.topics_key = topics_key

        self._init_rng = False
        self._rng_theta: Optional[RandomState] = None
        self._rng_idx: Optional[RandomState] = None
        self._rng_w: Optional[RandomState] = None
        self._rng_topics: Optional[List[RandomState]] = None

        self.name = name

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        """Leave scalar observations unchanged; LDA updates require encoding."""
        pass

    def _rng_initialize(self, rng: RandomState) -> None:
        if not self._init_rng:
            seeds = rng.randint(maxrandint, size=3 + self.num_topics)
            self._rng_theta = RandomState(seed=seeds[0])
            self._rng_idx = RandomState(seed=seeds[1])
            self._rng_w = RandomState(seed=seeds[2])
            self._rng_topics = [
                RandomState(seed=seeds[3 + j]) for j in range(self.num_topics)
            ]
            self._init_rng = True

    def seq_initialize(
        self,
        x: "LDAEncodedDataSequence",
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Randomly initialize sufficient statistics for encoded documents.

        Document proportions are drawn from ``prev_alpha``, defaulting to an
        all-ones vector, and token-topic weights are randomized before being passed
        to each topic accumulator.
        """
        num_documents, idx, counts, _old_gammas, enc_data = x.data

        if not self._init_rng:
            self._rng_initialize(rng)
        assert self._rng_theta is not None
        assert self._rng_idx is not None
        assert self._rng_w is not None
        assert self._rng_topics is not None

        if self.prev_alpha is None:
            self.prev_alpha = np.ones(self.num_topics)

        theta = self._rng_theta.dirichlet(self.prev_alpha, size=num_documents)
        theta_rep = theta[idx, :]

        idx_list = row_choice(
            p_mat=np.reshape(theta_rep, (-1, self.num_topics)), rng=self._rng_idx
        )

        self.sum_of_logs += np.sum(np.log(theta), axis=0, keepdims=False)
        self.doc_counts += np.sum(weights)

        ww_v = -np.log(self._rng_w.rand(self.num_topics * len(idx)))
        ww_v[idx_list + np.arange(0, len(ww_v), self.num_topics)] += 1
        ww_v = np.reshape(ww_v, (-1, self.num_topics))
        ww_v /= ww_v.sum(axis=1, keepdims=True)

        temp = np.reshape(weights[idx] * counts, (len(idx), 1))
        ww_v *= temp

        for j in range(self.num_topics):
            w = ww_v[:, j]
            self.topic_counts[j] += np.sum(w)
            self.accumulators[j].seq_initialize(enc_data, w, self._rng_topics[j])

    def initialize(
        self, x: Sequence[Tuple[Any, float]], weight: float, rng: np.random.RandomState
    ) -> None:
        """Randomly initialize sufficient statistics for one sparse document."""
        if self.prev_alpha is None:
            self.prev_alpha = np.ones(self.num_topics)

        if not self._init_rng:
            self._rng_initialize(rng)
        assert self._rng_theta is not None
        assert self._rng_idx is not None
        assert self._rng_w is not None
        assert self._rng_topics is not None

        counts = np.reshape([x[i][1] for i in range(len(x))], (len(x), 1))

        theta = self._rng_theta.dirichlet(self.prev_alpha)
        print(theta)

        theta_rep = theta[np.arange(0, self.num_topics * len(x)) % self.num_topics]
        idx_list = row_choice(
            p_mat=np.reshape(theta_rep, (-1, self.num_topics)), rng=self._rng_idx
        )
        print("\n")
        print(idx_list)
        self.sum_of_logs += np.log(theta)
        self.doc_counts += weight

        ww_v = -np.log(self._rng_w.rand(self.num_topics * len(x)))
        ww_v[idx_list + np.arange(0, self.num_topics * len(x), self.num_topics)] += 1
        ww_v = np.reshape(ww_v, (-1, self.num_topics))
        ww_v /= np.sum(ww_v, axis=1, keepdims=True)
        ww_v *= counts * weight

        for j in range(self.num_topics):
            w = ww_v[:, j]
            for i, x_i in enumerate(x):
                self.accumulators[j].initialize(x_i[0], w[i], self._rng_topics[j])
                self.topic_counts[j] += w[i]

    def seq_update(
        self,
        x: "LDAEncodedDataSequence",
        weights: np.ndarray,
        estimate: LDADistribution,
    ) -> None:
        """Update statistics from variational token-topic responsibilities."""
        _num_documents, idx, counts, _old_gammas, enc_data = x.data
        log_density_gamma, final_gammas, _per_topic_log_densities = seq_posterior(
            estimate, x.data
        )

        for i in range(self.num_topics):
            self.accumulators[i].seq_update(
                enc_data,
                log_density_gamma[:, i] * weights[idx] * counts,
                estimate.topics[i],
            )

        mlpf = digamma(final_gammas) - digamma(
            np.sum(final_gammas, axis=1, keepdims=True)
        )

        self.sum_of_logs += np.dot(weights, mlpf)
        self.doc_counts += weights.sum()
        self.topic_counts += np.sum(log_density_gamma, axis=0)
        self.prev_alpha = estimate.alpha

    # return num_documents, idx, counts, final_gammas, enc_data

    def combine(
        self,
        suff_stat: Tuple[
            Optional[np.ndarray], np.ndarray, float, np.ndarray, Sequence[SS0]
        ],
    ) -> "LDAEstimatorAccumulator":
        """Add another accumulator's sufficient statistics in place."""
        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = suff_stat

        if self.prev_alpha is None:
            self.prev_alpha = prev_alpha

        self.sum_of_logs += sum_of_logs
        self.doc_counts += doc_counts
        self.topic_counts += topic_counts

        for i in range(self.num_topics):
            self.accumulators[i].combine(topic_suff_stats[i])

        return self

    def value(
        self,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, float, np.ndarray, Sequence[Any]]:
        """Return alpha-update, document, and per-topic sufficient statistics."""
        return (
            self.prev_alpha,
            self.sum_of_logs,
            self.doc_counts,
            self.topic_counts,
            [u.value() for u in self.accumulators],
        )

    def from_value(
        self,
        x: Tuple[Optional[np.ndarray], np.ndarray, float, np.ndarray, Sequence[SS0]],
    ) -> "LDAEstimatorAccumulator":
        """Replace the sufficient statistics and return this accumulator."""
        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = x

        self.prev_alpha = prev_alpha
        self.sum_of_logs = sum_of_logs
        self.doc_counts = doc_counts
        self.topic_counts = topic_counts
        self.accumulators = [
            self.accumulators[i].from_value(topic_suff_stats[i])
            for i in range(self.num_topics)
        ]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge alpha-update and topic statistics under configured keys."""
        if self.alpha_key is not None:
            if self.alpha_key in stats_dict:

                p_sol, p_doc, p_pa = stats_dict[self.alpha_key]

                prev_alpha = self.prev_alpha if self.prev_alpha is not None else p_pa
                stats_dict[self.alpha_key] = (
                    self.sum_of_logs + p_sol,
                    self.doc_counts + p_doc,
                    prev_alpha,
                )

            else:
                stats_dict[self.alpha_key] = (
                    self.sum_of_logs,
                    self.doc_counts,
                    self.prev_alpha,
                )

        if self.topics_key is not None:
            if self.topics_key in stats_dict:
                acc = stats_dict[self.topics_key]
                for i, acc_i in enumerate(acc):
                    acc_i = acc_i.combine(self.accumulators[i].value())
            else:
                stats_dict[self.topics_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace alpha-update and topic statistics from configured keys."""
        if self.alpha_key is not None:
            if self.alpha_key in stats_dict:
                p_sol, p_doc, p_pa = stats_dict[self.alpha_key]
                self.prev_alpha = p_pa
                self.sum_of_logs = p_sol
                self.doc_counts = p_doc

        if self.topics_key is not None:
            if self.topics_key in stats_dict:
                acc = stats_dict[self.topics_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> "LDADataEncoder":
        """Create an LDA encoder from the first topic accumulator."""
        return LDADataEncoder(encoder=self.accumulators[0].acc_to_encoder())


class LDAEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Create LDA sufficient-statistic accumulators."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        dim: int,
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
        prev_alpha: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize an LDA accumulator factory.

        Args:
            factories: One value-distribution accumulator factory per topic.
            dim: Number of topics and factories to use.
            name: Optional accumulator name.
            keys: Keys for sharing alpha-update and topic statistics.
            prev_alpha: Dirichlet parameter used during random initialization.
        """
        self.factories = factories
        self.dim = dim
        self.keys = keys if keys is not None else (None, None)
        self.name = name
        self.prev_alpha = prev_alpha

    def make(self) -> "LDAEstimatorAccumulator":
        """Create an LDA sufficient-statistic accumulator."""
        return LDAEstimatorAccumulator(
            [self.factories[i].make() for i in range(self.dim)],
            name=self.name,
            keys=self.keys,
            prev_alpha=self.prev_alpha,
        )


class LDAEstimator(ParameterEstimator):
    """Estimate an LDADistribution from aggregated document and topic statistics.

    Notes:
        ``keys`` controls two different sharing decisions for LDA estimators.

        - ``keys[0]`` shares the statistics used to update ``alpha``:
          accumulated expected log-topic proportions, document counts, and the
          previous alpha iterate.
        - ``keys[1]`` shares topic sufficient statistics by topic index.

        This means ``keys=(None, "shared_topics")`` ties the topic-word models while
        allowing each LDA estimator to keep its own document-topic prior. In contrast,
        ``keys=("shared_alpha", None)`` shares only the alpha update.

        Shared topic keys assume topic index ``i`` represents the same topic role
        across the models being fit. If topic order is not aligned, keyed sharing will
        pool the wrong topic statistics. When ``fixed_alpha`` is set, alpha-sharing
        keys do not change the final alpha because the estimator uses the fixed value.
    """

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        suff_stat: Optional[Any] = None,
        pseudo_count: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
        fixed_alpha: Optional[np.ndarray] = None,
        gamma_threshold: float = 1.0e-8,
        alpha_threshold: float = 1.0e-8,
    ) -> None:
        """Initialize LDAEstimator.

        Args:
            estimators (Sequence[ParameterEstimator]): Estimators for the topic-word
                distributions.
            suff_stat (Optional[Any]): Optional prior sufficient statistics used during
                estimation.
            pseudo_count (Optional[Tuple[float, float]]): Optional pseudo-counts for
                regularization.
            name (Optional[str]): Name for the estimator.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys that control
                sharing of sufficient statistics across LDA estimators with matching
                key values. ``keys[0]`` shares the alpha-update statistics, while
                ``keys[1]`` shares topic sufficient statistics by topic index. Use
                ``keys=(None, "shared_topics")`` to tie topic-word distributions but
                keep alpha separate. Shared topic keys assume aligned topic ordering.
            fixed_alpha (Optional[np.ndarray]): Fixed alpha value. When provided, the
                estimator does not update alpha from shared statistics.
            gamma_threshold (float): Threshold used in document-posterior updates.
            alpha_threshold (float): Threshold used in alpha updates.

        """
        self.num_topics = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None)
        self.gamma_threshold = gamma_threshold
        self.alpha_threshold = alpha_threshold
        self.fixed_alpha = fixed_alpha
        self.name = name

    def accumulator_factory(self) -> "LDAEstimatorAccumulatorFactory":
        """Create a factory for compatible LDA accumulators."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return LDAEstimatorAccumulatorFactory(
            factories=est_factories,
            dim=self.num_topics,
            keys=self.keys,
            name=self.name,
            prev_alpha=self.fixed_alpha,
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[
            Optional[np.ndarray], np.ndarray, float, np.ndarray, Sequence[Any]
        ],
    ) -> LDADistribution:
        """Estimate topic distributions and the Dirichlet concentration.

        Args:
            nobs: Ignored; weighted document mass is included in ``suff_stat``.
            suff_stat: Previous alpha, summed expected log topic proportions,
                weighted document mass, topic responsibility totals, and one set of
                sufficient statistics per topic estimator.

        Returns:
            An LDA distribution with updated topics and either an updated or fixed
            Dirichlet concentration.
        """
        del nobs

        prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = suff_stat

        num_topics = self.num_topics
        prev_alpha_arr = (
            np.ones(num_topics, dtype=np.float64)
            if prev_alpha is None
            else np.asarray(prev_alpha, dtype=np.float64)
        )
        topics = [
            self.estimators[i].estimate(topic_counts[i], topic_suff_stats[i])
            for i in range(num_topics)
        ]

        if doc_counts == 0:
            sys.stderr.write("Warning: LDA Estimation performed with zero documents.\n")
            return LDADistribution(
                topics, prev_alpha_arr, gamma_threshold=self.gamma_threshold
            )

        if self.fixed_alpha is None:

            # new_alpha, _ = find_alpha(prev_alpha, sum_of_logs/doc_counts,
            # gamma_threshold*np.sqrt(float(doc_counts)))
            new_alpha, _ = update_alpha(
                prev_alpha_arr, sum_of_logs / doc_counts, self.alpha_threshold
            )
        else:
            new_alpha = np.asarray(self.fixed_alpha).copy()

        return LDADistribution(topics, new_alpha, gamma_threshold=self.gamma_threshold)


class LDADataEncoder(DataSequenceEncoder):
    """Encode sparse counted documents for vectorized LDA operations."""

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        """Initialize the encoder used for flattened token values."""
        self.encoder = encoder

    def __str__(self) -> str:
        """Return a constructor-like representation of the encoder."""
        return "LDADataEncoder(encoder=" + str(self.encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether two LDA encoders wrap equal value encoders."""
        if isinstance(other, LDADataEncoder):
            return self.encoder == other.encoder
        return False

    def seq_encode(
        self, x: Sequence[Sequence[Tuple[int, float]]]
    ) -> "LDAEncodedDataSequence":
        """Encode sparse documents for vectorized LDA operations.

        The encoded payload is ``(num_documents, idx, counts, gammas, enc_data)``.
        ``idx`` maps each flattened distinct token to its document, ``counts`` stores
        its multiplicity, ``gammas`` is initially ``None``, and ``enc_data`` is the
        wrapped encoder's representation of the flattened token values.

        Args:
            x (Sequence[Sequence[Tuple[int, float]]]): Sequence of LDA documents.

        Returns:
            The encoded document sequence.

        """
        num_documents = len(x)

        tx = []
        ctx = []
        nx = []
        tidx = []
        for i, x_i in enumerate(x):
            nx.append(len(x_i))
            for _j, x_i_j in enumerate(x_i):
                tidx.append(i)
                tx.append(x_i_j[0])
                ctx.append(x_i_j[1])

        idx = np.asarray(tidx)
        counts = np.asarray(ctx)
        gammas = None
        enc_data = self.encoder.seq_encode(tx)

        return LDAEncodedDataSequence(
            data=(num_documents, idx, counts, gammas, enc_data)
        )


class LDAEncodedDataSequence(EncodedDataSequence):
    """Store flattened documents and optional variational initialization values."""

    def __init__(
        self,
        data: Tuple[
            int, np.ndarray, np.ndarray, Optional[np.ndarray], EncodedDataSequence
        ],
    ):
        """Initialize an encoded LDA sequence from its five-part payload."""
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded data."""
        return f"LDAEncodedDataSequence(data={self.data})"


def update_alpha(
    alpha_curr: np.ndarray, mean_log_p: np.ndarray, alpha_threshold: float
) -> Tuple[np.ndarray, int]:
    """Solve the Dirichlet fixed-point update for mean log proportions.

    Args:
        alpha_curr: Positive starting concentration vector.
        mean_log_p: Mean expected log topic proportions.
        alpha_threshold: Relative L1 convergence threshold.

    Returns:
        The updated concentration vector and iteration count.
    """
    alpha = np.asarray(alpha_curr.copy(), dtype=np.float64)
    asum = float(alpha.sum())
    res = np.inf
    its_cnt = 0
    while res > alpha_threshold:
        dasum = digamma(asum)
        alpha_old = alpha
        alpha = np.asarray(digammainv(mean_log_p + dasum), dtype=np.float64)
        asum = float(alpha.sum())
        res = float(np.abs(alpha - alpha_old).sum() / asum)
        its_cnt += 1

    return alpha, its_cnt


def mpe_update(
    x_mat: Optional[np.ndarray], y: np.ndarray, min_size: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Append a fixed-point iterate and compute a minimal-polynomial extrapolation."""
    if x_mat is None:
        x_mat = np.reshape(y, (1, -1))
        return x_mat, y
    if x_mat.shape[0] < min_size:
        x_mat = np.concatenate((x_mat, np.reshape(y, (1, -1))), axis=0)
        return x_mat, y

    dy = y - x_mat[-1, :]
    u_mat = (x_mat[1:, :] - x_mat[:-1, :]).T
    x2_mat = x_mat[1:, :].T
    c = np.dot(np.linalg.pinv(u_mat), dy)
    c *= -1
    s = (np.dot(x2_mat, c) + y) / (c.sum() + 1)

    x_mat = np.concatenate((x_mat, np.reshape(y, (1, -1))), axis=0)

    return x_mat, s


def mpe(
    x0: np.ndarray, f: Callable[[np.ndarray], np.ndarray], eps: float
) -> Tuple[np.ndarray, int]:
    """Iterate a fixed-point map with minimal-polynomial extrapolation."""
    x1 = f(x0)
    x2 = f(x1)
    x3 = f(x2)
    x_mat = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 2

    while res > eps:
        y = f(x_mat[-1, :])
        dy = y - x_mat[-1, :]
        u_mat = (x_mat[1:, :] - x_mat[:-1, :]).T
        x2_mat = x_mat[1:, :].T
        c = np.dot(np.linalg.pinv(u_mat), dy)
        c *= -1
        s = (np.dot(x2_mat, c) + y) / (c.sum() + 1)

        res = np.abs(s - s0).sum()
        s0 = s
        x_mat = np.concatenate((x_mat, np.reshape(y, (1, -1))), axis=0)
        its_cnt += 1

    return s, its_cnt


def alpha_seq_lambda(mean_log_p: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Create the fixed-point map for a Dirichlet concentration update."""

    def next_alpha(alpha_current: np.ndarray) -> np.ndarray:
        return np.asarray(
            digammainv(mean_log_p + digamma(float(alpha_current.sum()))),
            dtype=np.float64,
        )

    return next_alpha


def find_alpha(
    current_alpha: np.ndarray, mlp: np.ndarray, thresh: float
) -> Tuple[np.ndarray, int]:
    """Estimate a Dirichlet concentration using extrapolated fixed-point updates."""
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh)


def seq_posterior2(
    estimate: LDADistribution, x: Tuple[int, np.ndarray, np.ndarray, Optional[Any], E0]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delegate to :func:`seq_posterior` for backward compatibility."""
    return seq_posterior(estimate, x)


def seq_posterior(
    estimate: LDADistribution, x: Tuple[int, np.ndarray, np.ndarray, Optional[Any], E0]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the mean-field variational posterior for encoded documents.

    Args:
        estimate: LDA model supplying topics, ``alpha``, and convergence threshold.
        x: Encoded payload ``(num_documents, idx, counts, gammas, enc_data)``.

    Returns:
        Token-topic responsibilities of shape ``(M, K)``, document Dirichlet
        parameters of shape ``(D, K)``, and topic log-densities of shape ``(M, K)``,
        where ``M`` is the number of distinct encoded document-token pairs.
    """
    alpha = estimate.alpha
    topics = estimate.topics
    gamma_threshold = estimate.gamma_threshold

    num_documents, idx, counts, gammas, enc_data = x

    num_topics = len(topics)
    num_samples = len(idx)

    per_topic_log_densities = np.asarray(
        [topics[i].seq_log_density(enc_data) for i in range(num_topics)]
    ).transpose()
    per_topic_log_densities2 = per_topic_log_densities.copy()
    per_topic_log_densities2 -= np.max(per_topic_log_densities2, axis=1, keepdims=True)
    np.exp(per_topic_log_densities2, out=per_topic_log_densities2)
    per_topic_log_densities3 = per_topic_log_densities2.copy()

    idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
    idx_full *= num_topics
    idx_full += np.reshape(np.arange(num_topics), (1, num_topics))
    alpha_loc = np.reshape(alpha, (1, num_topics))

    if gammas is None:
        document_gammas = alpha_loc + np.reshape(
            np.bincount(idx_full.flat), (num_documents, num_topics)
        ) / float(num_topics)
    else:
        document_gammas = gammas.copy()

    document_gammas2 = np.zeros((num_documents, num_topics), dtype=float)
    document_gammas3 = np.zeros((num_documents, num_topics), dtype=float)

    gamma_sum = np.zeros((num_documents, 1), dtype=float)
    gamma_asum = np.zeros((num_documents, 1), dtype=float)

    posterior_sum_ll = np.zeros((num_samples, 1), dtype=float)

    log_density_gamma = np.zeros(per_topic_log_densities.shape, dtype=float)
    document_gamma_diff_loc = np.zeros((num_documents, num_topics), dtype=float)
    log_density_gamma_loc = log_density_gamma.view()
    posterior_sum_ll_loc = posterior_sum_ll.view()
    gamma_asum_loc = gamma_asum.view()
    gamma_sum_loc = gamma_sum.view()

    ndoc = num_documents

    rel_idx = idx.copy()
    rel_counts = counts.copy()
    rel_counts = np.reshape(rel_counts, (-1, 1))

    rem_gammas_idx = np.arange(num_documents, dtype=int)
    final_gammas = np.zeros((num_documents, num_topics), dtype=float)
    final_gammas_idx = np.zeros(num_documents, dtype=int)
    finished_count = 0
    itr_cnt = 0
    gamma_itr_cnt = np.zeros(num_documents, dtype=int)

    #

    digamma(document_gammas, out=document_gammas2)
    temp = np.max(document_gammas2, axis=1, keepdims=True)
    np.exp(document_gammas2 - temp, out=document_gammas3)

    np.multiply(
        per_topic_log_densities2,
        document_gammas3[rel_idx, :],
        out=log_density_gamma_loc,
    )
    np.sum(log_density_gamma_loc, axis=1, keepdims=True, out=posterior_sum_ll_loc)
    log_density_gamma_loc /= posterior_sum_ll_loc

    while ndoc > 0:

        itr_cnt += 1

        digamma(document_gammas, out=document_gammas2)
        temp = np.max(document_gammas2, axis=1, keepdims=True)
        document_gammas2 -= temp
        np.exp(document_gammas2, out=document_gammas3)

        np.multiply(
            per_topic_log_densities2,
            document_gammas3[rel_idx, :],
            out=log_density_gamma_loc,
        )
        np.sum(log_density_gamma_loc, axis=1, keepdims=True, out=posterior_sum_ll_loc)
        posterior_sum_ll_loc /= rel_counts
        log_density_gamma_loc /= posterior_sum_ll_loc

        gamma_updates = np.asarray(
            np.bincount(idx_full.flat, weights=log_density_gamma_loc.flat),
            dtype=np.float64,
        )
        gamma_updates = np.reshape(gamma_updates, (-1, num_topics))
        gamma_updates += alpha_loc

        np.subtract(document_gammas, gamma_updates, out=document_gamma_diff_loc)
        np.abs(document_gamma_diff_loc, out=document_gamma_diff_loc)
        np.sum(document_gamma_diff_loc, axis=1, keepdims=True, out=gamma_asum_loc)
        np.sum(gamma_updates, axis=1, keepdims=True, out=gamma_sum_loc)
        gamma_asum_loc /= gamma_sum_loc

        document_gammas = gamma_updates

        has_finished = np.nonzero(gamma_asum_loc.flat <= gamma_threshold)[0]

        if has_finished.size != 0:
            final_gammas[finished_count : (finished_count + len(has_finished)), :] = (
                document_gammas[has_finished, :]
            )
            final_gammas_idx[finished_count : (finished_count + len(has_finished))] = (
                rem_gammas_idx[has_finished]
            )
            gamma_itr_cnt[finished_count : (finished_count + len(has_finished))] = (
                itr_cnt
            )

            is_rem_bool = gamma_asum_loc.flat > gamma_threshold

            is_rem_idx = np.nonzero(is_rem_bool)[0]
            rem_gammas_idx = rem_gammas_idx[is_rem_bool]
            finished_count += has_finished.size

            temp = np.zeros(ndoc, dtype=bool)
            temp[is_rem_bool] = True
            temp2 = np.arange(ndoc, dtype=int)
            temp2[temp] = np.arange(is_rem_idx.size, dtype=int)

            keep = temp[rel_idx]
            rel_idx = temp2[rel_idx[temp[rel_idx]]]

            idx_full = np.repeat(np.reshape(rel_idx, (-1, 1)), num_topics, axis=1)
            idx_full *= num_topics
            idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

            per_topic_log_densities2 = per_topic_log_densities2[keep, :]
            rel_counts = rel_counts[keep]
            nrec = per_topic_log_densities2.shape[0]
            ndoc = is_rem_idx.size

            log_density_gamma_loc = log_density_gamma[:nrec, :]
            posterior_sum_ll_loc = posterior_sum_ll[:nrec, :]
            gamma_sum_loc = gamma_sum[:ndoc, :]
            gamma_asum_loc = gamma_asum[:ndoc, :]
            document_gamma_diff_loc = document_gamma_diff_loc[:ndoc, :]

            document_gammas = document_gammas[is_rem_idx, :]
            document_gammas2 = document_gammas2[:ndoc, :]
            document_gammas3 = document_gammas3[:ndoc, :]

    #
    # Accumulate per-bag-sample
    #

    sidx = np.argsort(final_gammas_idx)
    final_gammas = final_gammas[sidx, :]
    gamma_itr_cnt = gamma_itr_cnt[sidx]

    digamma_gammas = digamma(final_gammas)
    temp2 = np.max(digamma_gammas, axis=1, keepdims=True)
    temp3 = np.exp(digamma_gammas - temp2)

    # per_topic_log_densities2  = per_topic_log_densities.copy()
    # per_topic_log_densities2 -= np.max(per_topic_log_densities2, axis=1,
    # keepdims=True)
    # np.exp(per_topic_log_densities2, out=per_topic_log_densities2)

    np.multiply(per_topic_log_densities3, temp3[idx, :], out=log_density_gamma)
    np.sum(log_density_gamma, axis=1, keepdims=True, out=posterior_sum_ll)
    posterior_sum_ll /= np.reshape(counts, (-1, 1))
    log_density_gamma /= posterior_sum_ll

    idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
    idx_full *= num_topics
    idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

    gamma_updates = np.asarray(
        np.bincount(idx_full.flat, weights=log_density_gamma.flat), dtype=np.float64
    )
    gamma_updates = np.reshape(gamma_updates, (-1, num_topics))
    gamma_updates += alpha_loc
    final_gammas = gamma_updates

    return log_density_gamma, final_gammas, per_topic_log_densities
