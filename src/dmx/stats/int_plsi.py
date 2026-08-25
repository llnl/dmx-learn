r"""Provide probabilistic latent semantic indexing for integer tokens.

An observation is a document identifier and a sparse bag of token counts,
``(d, [(v_0, c_0), ..., (v_m, c_m)])``. Document and token identifiers must be
zero-based integers. For topic :math:`z`, the model parameters are the token
probabilities :math:`\phi_{vz}=P(v\mid z)`, document-topic probabilities
:math:`\theta_{dz}=P(z\mid d)`, and document probabilities :math:`\pi_d=P(d)`.
With :math:`n=\sum_j c_j`, the implemented document mass is

.. math::

   P(d, x)=P_{\mathrm{len}}(n)\,\pi_d
   \prod_j\left(\sum_z \phi_{v_jz}\theta_{dz}\right)^{c_j}.

The module supplies the distribution, sampler, sufficient-statistic accumulator,
estimator, and vectorized data encoder. Estimation performs responsibility-weighted
updates of the three probability tables; initialization instead assigns token counts
to topics randomly from a symmetric Dirichlet draw.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numba
import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import maxrandint
from dmx.stats.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from dmx.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EncodedDataSequence,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from dmx.utils.optsutil import count_by_value

T1 = TypeVar("T1")  ## type for encoded sequence of lengths.
SS1 = TypeVar("SS1")  ### type for value of length dist sufficient statistics.


class IntegerPLSIDistribution(SequenceEncodableProbabilityDistribution):
    """Represent an integer-token PLSI distribution.

    Attributes:
       prob_mat (np.ndarray): Token-topic probabilities ``P(v | z)`` with shape
           ``(V, S)``; each topic column sums to one.
       state_mat (np.ndarray): Document-topic probabilities ``P(z | d)`` with shape
           ``(D, S)``; each document row sums to one.
       doc_vec (np.ndarray): Document probabilities ``P(d)`` with shape ``(D,)``.
       log_doc_vec (np.ndarray): 1-d numpy array of the log(p_mat(doc=d)).
       num_vals (int): Number of total words in corpus. (Number of rows in prob_mat).
       num_states (int): Number of word topics (mixture components). (Number of columns
           in prob_mat/state_mat).
       num_docs (int): Total number of document ids in corpus. (Number of rows in
           state_mat).
       name (Optional[str]): Optional name for object instance.
       len_dist (SequenceEncodableProbabilityDistribution): Distribution object for the
           number of words per
           document. Defaults to the NullDistribution if None is passed.
       keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for word_probs,
           state_probs, and doc_probs.

    """

    def __init__(
        self,
        state_word_mat: Union[List[List[float]], np.ndarray],
        doc_state_mat: Union[List[List[float]], np.ndarray],
        doc_vec: Union[List[float], np.ndarray],
        len_dist: Optional[
            SequenceEncodableProbabilityDistribution
        ] = NullDistribution(),
        name: Optional[str] = None,
        keys: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
    ) -> None:
        """Initialize an integer-token PLSI distribution.

        Args:
            state_word_mat (Union[List[List[float]], np.ndarray]): Token-topic
                probability matrix ``P(v | z)`` of shape ``(V, S)``. Columns should
                sum to one.
            doc_state_mat (Union[List[List[float]], np.ndarray]): Document-topic
                probability matrix ``P(z | d)`` of shape ``(D, S)``. Rows should sum
                to one.
            doc_vec (Union[List[float], np.ndarray]): Array-like containing prior for
                documents p_mat(d) for each
                document id in corpus of documents. Should sum to 1.0
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                distribution for the length of
                each document (i.e. word count in an observed document). Should have
                support on positive integers.
            name (Optional[str]): Set name to object instance.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for
                word_probs, state_probs, and doc_probs.

        """
        super().__init__()
        self.prob_mat = np.asarray(state_word_mat, dtype=np.float64)
        self.state_mat = np.asarray(doc_state_mat, dtype=np.float64)
        self.doc_vec = np.asarray(doc_vec, dtype=np.float64)
        self.log_doc_vec = np.log(self.doc_vec)
        self.num_vals = self.prob_mat.shape[0]
        self.num_states = self.prob_mat.shape[1]
        self.num_docs = self.state_mat.shape[0]
        self.name = name
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-like representation of the distribution."""
        s1 = ",".join(
            [
                "[" + ",".join(map(str, self.prob_mat[i, :])) + "]"
                for i in range(len(self.prob_mat))
            ]
        )
        s2 = ",".join(
            [
                "[" + ",".join(map(str, self.state_mat[i, :])) + "]"
                for i in range(len(self.state_mat))
            ]
        )
        s3 = ",".join(map(str, self.doc_vec))
        s4 = repr(self.name)
        s5 = str(self.len_dist)
        s6 = repr(self.keys)
        return (
            f"IntegerPLSIDistribution([{s1}], [{s2}], [{s3}], "
            f"name={s4}, len_dist={s5}, keys={s6})"
        )

    def density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        """Evaluate the density of PLSI model for an observation x.

        See log_density() for details on the density evaluation.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): Single observation of integer
                PLSI.

        Returns:
            float: Density evaluated at observed value x.

        """
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        r"""Evaluate the document log-likelihood.

        For ``x = (d, [(v_j, c_j), ...])``, this returns

        .. math::

           \log \pi_d + \log P_{\mathrm{len}}(\textstyle\sum_j c_j)
           + \sum_j c_j\log\left(\sum_z\phi_{v_jz}\theta_{dz}\right).

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): (doc_id, [(value_id,
                count_for_value)]). See above for details.

        Returns:
            float: Log-density evaluated at a single observation x.

        """
        d_id = x[0]
        xv = np.asarray([u[0] for u in x[1]], dtype=int)
        xc = np.asarray([u[1] for u in x[1]], dtype=float)

        rv = 0.0
        rv += np.dot(np.log(np.dot(self.prob_mat[xv, :], self.state_mat[d_id, :])), xc)
        rv += np.log(self.doc_vec[d_id])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(np.sum(xc))

        return float(rv)

    def component_log_density(
        self, x: Tuple[int, Sequence[Tuple[int, float]]]
    ) -> np.ndarray:
        """Evaluate the log-density for each state in the PLSI.

        Returns count*log(p_mat(W|S)) for each word-count pair in the document. Returned
        value is S by 1 where S is the
        number of components in the model.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): Single PLSI observation of form
                (doc_id, [(value_id, count_for_value)]).

        Returns:
            np.ndarray: Numpy array of length S (num_states).

        """
        xv = np.asarray([u[0] for u in x[1]], dtype=int)
        xc = np.asarray([u[1] for u in x[1]], dtype=float)

        return np.asarray(np.dot(np.log(self.prob_mat[xv, :]).T, xc), dtype=float)

    def seq_log_density(self, x: "IntegerPLSIEncodedDataSequence") -> np.ndarray:
        """Evaluate log-likelihoods for an encoded sequence of documents."""
        if not isinstance(x, IntegerPLSIEncodedDataSequence):
            raise TypeError(
                "IntegerPLSIEncodedDataSequence required for seq_log_density()."
            )

        nn, (xv, xc, xd, xi, xn, xm) = x.data
        cnt = len(xn)

        w = np.zeros(len(xv), dtype=np.float64)
        index_dot(self.prob_mat, xv, self.state_mat, xd, w)
        w = np.log(w, out=w)
        w *= xc

        rv = np.zeros(cnt, dtype=np.float64)
        bincount(xi, w, rv)
        rv += self.log_doc_vec[xm]

        if self.len_dist is not None:
            rv += self.len_dist.seq_log_density(nn)

        return np.asarray(rv, dtype=float)

    def seq_component_log_density(
        self, x: "IntegerPLSIEncodedDataSequence"
    ) -> np.ndarray:
        """Evaluate per-topic token log-likelihoods for encoded documents.

        See component_log_density() function for details on component log-likelihood
        evaluation.

        Args:
            x (IntegerPLSIEncodedDataSequence): EncodedDataSequence of PLSI
                observations.

        Returns:
            np.ndarray: 2-d numpy array containing N rows of num_state sized arrays.

        """
        if not isinstance(x, IntegerPLSIEncodedDataSequence):
            raise TypeError(
                "IntegerPLSIEncodedDataSequence required for "
                "seq_component_log_density()."
            )

        _nn, (xv, xc, xd, xi, _xn, xm) = x.data
        rv = np.zeros((xi[-1] + 1, self.num_states), dtype=np.float64)
        w_mat = self.prob_mat
        fast_seq_component_log_density(xv, xc, xd, xi, xm, w_mat, rv)
        return rv

    def sampler(self, seed: Optional[int] = None) -> "IntegerPLSISampler":
        """Create a sampler for this distribution."""
        return IntegerPLSISampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "IntegerPLSIEstimator":
        """Create an estimator, optionally regularized toward these parameters."""
        if pseudo_count is None:
            return IntegerPLSIEstimator(
                num_vals=self.num_vals,
                num_states=self.num_states,
                num_docs=self.num_docs,
                len_estimator=self.len_dist.estimator(),
                name=self.name,
                keys=self.keys,
            )
        pseudo_count_tuple = (pseudo_count, pseudo_count, pseudo_count)
        return IntegerPLSIEstimator(
            num_vals=self.num_vals,
            num_states=self.num_states,
            num_docs=self.num_docs,
            pseudo_count=pseudo_count_tuple,
            suff_stat=(self.prob_mat.T, self.state_mat, self.doc_vec),
            len_estimator=self.len_dist.estimator(),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerPLSIDataEncoder":
        """Create a data encoder compatible with this distribution."""
        return IntegerPLSIDataEncoder(len_encoder=self.len_dist.dist_to_encoder())


class IntegerPLSISampler(DistributionSampler):
    """IntegerPLSISampler object for sampling from IntegerPLSIDistribution.

    Attributes:
        rng (RandomState): RandomState object with seed set if passed.
        dist (IntegerPLSIDistribution): IntegerPLSIDistribution instance to sampler
            from.
        size_rng (RandomState): RandomState object for sampling the length of documents.

    """

    def __init__(
        self, dist: IntegerPLSIDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize a PLSI sampler.

        Args:
            dist (IntegerPLSIDistribution): IntegerPLSIDistribution instance to sampler
                from.
            seed (Optional[int]): Set seed for random number generator used in sampling.

        """
        super().__init__(dist, seed)
        self.size_rng = self.dist.len_dist.sampler(self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> Union[
        Tuple[int, Sequence[Tuple[int, float]]],
        Sequence[Tuple[int, Sequence[Tuple[int, float]]]],
    ]:
        """Generate iid samples from PLSI model.

        Args:
            size (Optional[int]): Number of samples to generate. Defaults to 0 if size
                is None.

        Returns:
            Sequence of iid PLSI samples if size is not None, else a single sample from
            PLSI model.

        """
        if size is None:
            d_id = int(self.rng.choice(self.dist.num_docs, p=self.dist.doc_vec))
            cnt = int(self.size_rng.sample())
            z = self.rng.multinomial(cnt, pvals=self.dist.state_mat[d_id, :])
            rv: List[int] = []
            for i, n in enumerate(z):
                if n > 0:
                    rv.extend(
                        int(v)
                        for v in self.rng.choice(
                            self.dist.num_vals,
                            p=self.dist.prob_mat[:, i],
                            replace=True,
                            size=n,
                        )
                    )

            return d_id, list(count_by_value(rv).items())

        samples: List[Tuple[int, Sequence[Tuple[int, float]]]] = []
        for _ in range(size):
            sample = self.sample()
            assert isinstance(sample, tuple)
            samples.append(sample)
        return samples


class IntegerPLSIAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate responsibility-weighted PLSI sufficient statistics.

    Note: Keys in order, words/values, states, documents.

    Attributes:
        num_vals (int): Number of words in the corpus.
        num_states (int): Number of word-topics or mixture components.
        num_docs (int): Number of authors (doc_ids) in the corpus.
        word_count (ndarray): Numpy array of shape num_states by num_vals for
            aggregating state/word counts.
        comp_count (ndarray): Numpy array (num_docs by num_states) for aggregating
            doc/state counts.
        doc_count (ndarray): Numpy array for aggregating counts of authors (prior on
            doc_ids).
        name (Optional[str]): Name of object instance.
        wc_key (Optional[str]): Key for merging 'word_count' with objects containing
            matching keys.
        sc_key (Optional[str]): Key for merging 'comp_count' with objects containing
            matching keys.
        dc_key (Optional[str]): Key for merging 'doc_count' with objects containing
            matching keys.
        len_acc (SequenceEncodableStatisticAccumulator): Accumulator object for the
            lengths of documents (total
            word counts). Defaults to the NullAccumulator if None is passed.

        _init_rng (bool): True if RandomState objects for accumulator have been
            initialized.
        _acc_rng (Optional[RandomState]): RandomState object for initializing the PLSI
            model.
        _len_rng (Optional[RandomState]): RandomState object for initializing the length
            accumulator.

    """

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
    ) -> None:
        """Initialize a PLSI sufficient-statistic accumulator.

        Note: Keys in order, words/values, states, documents.

        Args:
            num_vals (int): Number of words in the corpus.
            num_states (int): Number of word-topics.
            num_docs (int): Number of authors (doc_ids) in the corpus.
            len_acc (Optional[SequenceEncodableStatisticAccumulator]): Optional
                accumulator for the length of documents.
                Should have support on non-negative integer values.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]):
                Optional keys for words, states, and
                authors (doc_ids).

        """
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs
        self.word_count = np.zeros((num_states, num_vals), dtype=np.float64)
        self.comp_count = np.zeros((num_docs, num_states), dtype=np.float64)
        self.doc_count = np.zeros(num_docs, dtype=np.float64)
        self.name = name
        self.wc_key, self.sc_key, self.dc_key = (
            keys if keys is not None else (None, None, None)
        )
        self.len_acc = len_acc if len_acc is not None else NullAccumulator()

        # Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: Optional[RandomState] = None
        self._len_rng: Optional[RandomState] = None

    def update(
        self,
        x: Tuple[int, Sequence[Tuple[int, float]]],
        weight: float,
        estimate: IntegerPLSIDistribution,
    ) -> None:
        """Accumulate expected topic counts under a current PLSI estimate."""
        d_id = x[0]
        xv = np.asarray([u[0] for u in x[1]])
        xc = np.asarray([u[1] for u in x[1]])

        update = (estimate.prob_mat[xv, :] * estimate.state_mat[d_id, :]).T
        update *= xc * weight / np.sum(update, axis=0)
        self.comp_count[d_id, :] += np.sum(update, axis=1)
        self.word_count[:, xv] += update
        self.doc_count[d_id] += weight

        self.len_acc.update(np.sum(xc), weight, estimate.len_dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        seeds = rng.randint(maxrandint, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._len_rng = RandomState(seed=seeds[1])
        self._init_rng = True

    def initialize(
        self,
        x: Tuple[int, Sequence[Tuple[int, float]]],
        weight: float,
        rng: RandomState,
    ) -> None:
        """Initialize one document with random token-topic allocations.

        A symmetric Dirichlet draw independently divides each distinct token count
        among topics. The supplied observation weight scales all resulting counts.
        """
        if not self._init_rng:
            self._rng_initialize(rng)

        assert self._acc_rng is not None
        assert self._len_rng is not None

        d_id = x[0]
        xv = np.asarray([u[0] for u in x[1]])
        xc = np.asarray([u[1] for u in x[1]])

        update = self._acc_rng.dirichlet(
            np.ones(self.num_states) / self.num_states, size=len(xc)
        ).T
        update *= xc * weight
        self.word_count[:, xv] += update
        self.comp_count[d_id, :] += np.sum(update, axis=1)
        self.doc_count[d_id] += weight

        self.len_acc.update(np.sum(xc), weight, self._len_rng)

    def seq_initialize(
        self, x: "IntegerPLSIEncodedDataSequence", weights: np.ndarray, rng: RandomState
    ) -> None:
        """Initialize encoded documents with random token-topic allocations."""
        nn, (xv, xc, xd, xi, _xn, xm) = x.data

        if not self._init_rng:
            self._rng_initialize(rng)

        assert self._acc_rng is not None
        assert self._len_rng is not None

        update = self._acc_rng.dirichlet(
            np.ones(self.num_states) / self.num_states, size=len(xv)
        ).T
        update *= xc * weights[xi]
        self.word_count += vec_bincount3(
            xv, update, out=np.zeros_like(self.word_count, dtype=np.float64)
        )
        self.doc_count += np.bincount(xm, weights, minlength=self.num_docs)
        self.comp_count += vec_bincount4(
            xd, update, out=np.zeros_like(self.comp_count, dtype=np.float64)
        )

        self.len_acc.seq_initialize(nn, weights, self._len_rng)

    def seq_update(
        self,
        x: "IntegerPLSIEncodedDataSequence",
        weights: np.ndarray,
        estimate: IntegerPLSIDistribution,
    ) -> None:
        """Accumulate expected topic counts for encoded documents."""
        nn, (xv, xc, xd, xi, _xn, xm) = x.data
        fast_seq_update(
            xv,
            xc,
            xd,
            xi,
            xm,
            weights,
            estimate.prob_mat,
            estimate.state_mat,
            self.word_count,
            self.comp_count,
            self.doc_count,
        )

        self.len_acc.seq_update(nn, weights, estimate.len_dist)

    def combine(
        self, suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]
    ) -> "IntegerPLSIAccumulator":
        """Add another accumulator's sufficient statistics in place."""
        self.word_count += suff_stat[0]
        self.comp_count += suff_stat[1]
        self.doc_count += suff_stat[2]

        self.len_acc.combine(suff_stat[3])

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
        """Return token-topic, document-topic, document, and length statistics."""
        return self.word_count, self.comp_count, self.doc_count, self.len_acc.value()

    def from_value(
        self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]
    ) -> "IntegerPLSIAccumulator":
        """Replace the accumulated sufficient statistics and return this object."""
        self.word_count = x[0]
        self.comp_count = x[1]
        self.doc_count = x[2]
        self.len_acc.from_value(x[3])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics into a keyed statistics dictionary."""
        if self.wc_key is not None:
            if self.wc_key in stats_dict:
                stats_dict[self.wc_key] += self.word_count
            else:
                stats_dict[self.wc_key] = self.word_count

        if self.sc_key is not None:
            if self.sc_key in stats_dict:
                stats_dict[self.sc_key] += self.comp_count
            else:
                stats_dict[self.sc_key] = self.comp_count

        if self.dc_key is not None:
            if self.dc_key in stats_dict:
                stats_dict[self.dc_key] += self.doc_count
            else:
                stats_dict[self.dc_key] = self.doc_count

        self.len_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace sufficient statistics from matching dictionary keys."""
        if self.wc_key is not None:
            if self.wc_key in stats_dict:
                self.word_count = stats_dict[self.wc_key]
        if self.sc_key is not None:
            if self.sc_key in stats_dict:
                self.comp_count = stats_dict[self.sc_key]
        if self.dc_key is not None:
            if self.dc_key in stats_dict:
                self.doc_count = stats_dict[self.dc_key]

        self.len_acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerPLSIDataEncoder":
        """Return an IntegerPLSIDataEncoder object."""
        len_encoder = self.len_acc.acc_to_encoder()
        return IntegerPLSIDataEncoder(len_encoder=len_encoder)


class IntegerPLSIAccumulatorFactory(StatisticAccumulatorFactory):
    """IntegerPLSIAccumulatorFactory object for creating IntegerPLSIAccumulator objects.

    Attributes:
        num_vals (int): Number of words/values in PLSI.
        num_states (int): Number of states in PLSI.
        num_docs (int): Number of doc_ids (authors) in PLSI.
        len_factory (StatisticsAccumulatorFactory): Accumulator factory object for
            length distribution. Defaults
            to the NullAccumulatorFactory(). Should have support on non-negative
            integers.
        keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for merging
            word, state, and doc
            sufficient statistics with matching keys.
        name (Optional[str]): Set name for object.

    """

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        name: Optional[str] = None,
    ) -> None:
        """Initialize a PLSI accumulator factory.

        Args:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_factory (Optional[StatisticsAccumulatorFactory]): Accumulator factory
                object for length distribution.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set
                keys for merging
                word, state, and doc sufficient statistics with matching keys.
            name (Optional[str]): Set name for object.

        """
        self.len_factory = (
            len_factory if len_factory is not None else NullAccumulatorFactory()
        )
        self.keys = keys if keys is not None else (None, None, None)
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs
        self.name = name

    def make(self) -> "IntegerPLSIAccumulator":
        """Create a PLSI sufficient-statistic accumulator."""
        return IntegerPLSIAccumulator(
            self.num_vals,
            self.num_states,
            self.num_docs,
            len_acc=self.len_factory.make(),
            keys=self.keys,
            name=self.name,
        )


class IntegerPLSIEstimator(ParameterEstimator):
    """Estimate integer PLSI parameters from sufficient statistics.

    Attributes:
        num_vals (int): Number of words/values in PLSI.
        num_states (int): Number of states in PLSI.
        num_docs (int): Number of doc_ids (authors) in PLSI.
        len_estimator (ParameterEstimator): Optional ParameterEstimator object for the
            length of documents. Should
            have support on non-negative integers. Defaults to NullEstimator() if None
            is passed.
        pseudo_count (Tuple[Optional[float], Optional[float], Optional[float]]):
            Optional re-weight sufficient
            statistics in 'estimate()' function. Defaults to (None, None, None) if None
            is passed.
        suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray],
            Optional[np.ndarray]]): Optional
            Tuple of numpy arrays containing 'word_counts' (num_states by num_vals),
            'state_counts' (num_docs by
            num_states), and doc_counts (length num_docs). Defaults to (None, None,
            None) if None is passed.
        name (Optional[str]): Name of object instance.
        keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for merging
            word, state, and doc
            sufficient statistics with matching keys.
    """

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_estimator: Optional[ParameterEstimator] = NullEstimator(),
        pseudo_count: Optional[
            Tuple[Optional[float], Optional[float], Optional[float]]
        ] = (None, None, None),
        suff_stat: Optional[
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        ] = (None, None, None),
        name: Optional[str] = None,
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
    ) -> None:
        """Initialize an integer PLSI estimator.

        Args:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator
                object for the length of
                documents. Should have support on non-negative integers if not None.
            pseudo_count: Optional weights for token-topic, document-topic, and
                document pseudo-counts in ``estimate``.
            suff_stat: Optional reference token-topic, document-topic, and document
                probability arrays toward which pseudo-counts regularize.
            name (Optional[str]): Set name to object instance.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for
                merging word, state, and doc
                sufficient statistics with matching keys.

        """
        self.suff_stat = suff_stat if suff_stat is not None else (None, None, None)
        self.pseudo_count = (
            pseudo_count if pseudo_count is not None else (None, None, None)
        )
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs
        self.len_estimator = (
            len_estimator if len_estimator is not None else NullEstimator()
        )
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def accumulator_factory(self) -> "IntegerPLSIAccumulatorFactory":
        """Create a factory configured for this estimator's model dimensions."""
        len_fac = self.len_estimator.accumulator_factory()
        return IntegerPLSIAccumulatorFactory(
            num_vals=self.num_vals,
            num_states=self.num_states,
            num_docs=self.num_docs,
            len_factory=len_fac,
            keys=self.keys,
            name=self.name,
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]],
    ) -> "IntegerPLSIDistribution":
        """Normalize sufficient statistics into a PLSI distribution.

        Args:
            nobs: Ignored; document mass is obtained from ``doc_count``.
            suff_stat: Token-topic counts of shape ``(S, V)``, document-topic counts
                of shape ``(D, S)``, document counts of shape ``(D,)``, and length
                sufficient statistics.

        Returns:
            A distribution whose probability tables are the normalized counts,
            including configured pseudo-count regularization.
        """
        word_count, comp_count, doc_count, len_suff_stats = suff_stat

        if self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt * self.suff_stat[0].T
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)

        elif self.pseudo_count[0] is not None and self.suff_stat[0] is None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)

        else:
            word_prob_mat = word_count.T / np.sum(word_count, axis=1)

        if self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt * self.suff_stat[1]
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)

        elif self.pseudo_count[1] is not None and self.suff_stat[1] is None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)

        else:
            state_prob_mat = comp_count / np.sum(comp_count, axis=1, keepdims=True)

        if self.pseudo_count[2] is not None and self.suff_stat[2] is not None:
            adj_cnt_doc = self.pseudo_count[2] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt_doc * self.suff_stat[2]
            doc_prob_vec /= np.sum(doc_prob_vec)

        elif self.pseudo_count[2] is not None and self.suff_stat[2] is None:
            adj_cnt_doc = self.pseudo_count[2] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt_doc
            doc_prob_vec /= np.sum(doc_prob_vec)

        else:
            doc_prob_vec = doc_count / np.sum(doc_count)

        len_dist = self.len_estimator.estimate(None, len_suff_stats)

        return IntegerPLSIDistribution(
            word_prob_mat,
            state_prob_mat,
            doc_prob_vec,
            name=self.name,
            len_dist=len_dist,
        )


class IntegerPLSIDataEncoder(DataSequenceEncoder):
    """Encode sparse integer-token documents for vectorized PLSI operations.

    Attributes:
        len_encoder (DataSequenceEncoder): DataSequenceEncoder for the total number of
            words in each document,
            defaulting to NullDataEncoder if None is passed.

    """

    def __init__(
        self, len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder()
    ) -> None:
        """Initialize a PLSI data encoder.

        Args:
            len_encoder (Optional[DataSequenceEncoder]): Optional DataSequenceEncoder
                for the total number of words
                in each document.

        """
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Return a constructor-like representation of the encoder."""
        return "IntegerPLSIDataEncoder(len_dist=" + str(self.len_encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether two encoders use equal length encoders."""
        if isinstance(other, IntegerPLSIDataEncoder):
            return other.len_encoder == self.len_encoder
        return False

    def seq_encode(
        self, x: Sequence[Tuple[int, Sequence[Tuple[int, float]]]]
    ) -> "IntegerPLSIEncodedDataSequence":
        """Encode a sequence of iid PLSI observations for use with vectorized functions.

        Input arg 'x' is a sequence of iid PLSI observations having form

        x = [ (doc_id, [(value, count),...]),... ].

        The return value is a Tuple length 2. The first component contains data type
        Optional[T1] corresponding to the
        sequence encoding of the lengths. The second component is a Tuple of length 6
        containing
            xv (ndarray[int]): Numpy array of flattened word values.
            xc (ndarray[float]): Numpy array of flattened counts for word values above.
            xd (ndarray[int]): Document d_id for each word-count pair in the arrays
                above.
            xi (ndarray[int]): Observed sequence index for each word-count pair in the
                arrays above.
            xn (ndarray[float]): Numpy array of the total number of words in each
                document.
            xm (ndarray[float]): Flattened array of document d_id's for the lengths
                above (len = len(x)).

        Args:
            x (Sequence[Tuple[int, Sequence[Tuple[int, float]]]]): See above for
                details.

        Returns:
            IntegerPLSIEncodedDataSequence

        """
        xv: List[int] = []
        xc: List[float] = []
        xd: List[int] = []
        xi: List[int] = []
        xn: List[float] = []
        xm: List[int] = []

        for i, (d_id, xx) in enumerate(x):

            v = [u[0] for u in xx]
            c = [u[1] for u in xx]

            xv.extend(v)
            xc.extend(c)
            xd.extend([d_id] * len(v))
            xi.extend([i] * len(v))
            xn.append(np.sum(c))
            xm.append(d_id)

        xv_arr = np.asarray(xv, dtype=np.int32)
        xc_arr = np.asarray(xc, dtype=np.float64)
        xd_arr = np.asarray(xd, dtype=np.int32)
        xi_arr = np.asarray(xi, dtype=np.int32)
        xn_arr = np.asarray(xn, dtype=np.float64)
        xm_arr = np.asarray(xm, dtype=np.int32)

        nn = self.len_encoder.seq_encode(xn_arr)

        return IntegerPLSIEncodedDataSequence(
            data=(nn, (xv_arr, xc_arr, xd_arr, xi_arr, xn_arr, xm_arr))
        )


class IntegerPLSIEncodedDataSequence(EncodedDataSequence):
    """IntegerPLSIEncodedDataSequence object for vectorized function calls.

    Notes:
        E = Tuple[EncodedDataSequence, Tuple[np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray]]

    Attributes:
        data (E): Encoded sequence of PLSI observations.

    """

    def __init__(
        self,
        data: Tuple[
            EncodedDataSequence,
            Tuple[
                np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
            ],
        ],
    ):
        """Initialize an encoded PLSI sequence.

        Args:
            data (E): Encoded sequence of PLSI observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """Return a representation containing the encoded data."""
        return f"IntegerPLSIEncodedDataSequence(data={self.data})"


@numba.njit(
    "void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:,:], "
    "float64[:,:], float64[:], "
    "float64[:])",
    fastmath=True,
)
def fast_seq_log_density(
    xv: Any,
    xc: Any,
    xd: Any,
    xi: Any,
    xm: Any,
    wmat: Any,
    smat: Any,
    dvec: Any,
    out: Any,
) -> None:
    """Accumulate PLSI log-likelihoods into ``out`` using Numba."""
    n = len(xv)
    m = len(xm)
    k = smat.shape[1]
    for i in range(n):
        ll = 0.0
        cc = xc[i]
        i1 = xv[i]
        i2 = xd[i]
        i3 = xi[i]
        for j in range(k):
            ll += wmat[i1, j] * smat[i2, j]
        out[i3] += cc * np.log(ll)
    for i in range(m):
        out[i] += dvec[xm[i]]


@numba.njit(
    "void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:,:], "
    "float64[:,:])",
    fastmath=True,
)
def fast_seq_component_log_density(  # pylint: disable=unused-argument
    xv: Any, xc: Any, xd: Any, xi: Any, xm: Any, wmat: Any, out: Any
) -> None:
    """Accumulate per-topic token log-likelihoods using Numba."""
    n = len(xv)
    k = wmat.shape[1]
    for i in range(n):
        cc = xc[i]
        i1 = xv[i]
        i3 = xi[i]
        for j in range(k):
            out[i3, j] += np.log(wmat[i1, j]) * cc


@numba.njit(
    "void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:], "
    "float64[:,:], "
    "float64[:,:], "
    "float64[:,:], float64[:,:], float64[:])",
    fastmath=True,
)
def fast_seq_update(  # pylint: disable=too-many-positional-arguments
    xv: Any,
    xc: Any,
    xd: Any,
    xi: Any,
    xm: Any,
    weights: Any,
    wmat: Any,
    smat: Any,
    wcnt: Any,
    scnt: Any,
    dcnt: Any,
) -> None:
    """Accumulate responsibility-weighted PLSI counts using Numba."""
    n = len(xv)
    m = len(xm)
    k = smat.shape[1]
    posterior = np.zeros(k, dtype=np.float64)
    for i in range(n):
        norm_const = 0.0
        cc = xc[i]
        i1 = xv[i]
        i2 = xd[i]
        ww = weights[xi[i]]
        for j in range(k):
            temp = wmat[i1, j] * smat[i2, j]
            posterior[j] = temp
            norm_const += temp
        norm_const = ww * cc / norm_const
        for j in range(k):
            temp = posterior[j] * norm_const
            wcnt[j, i1] += temp
            scnt[i2, j] += temp
    for i in range(m):
        dcnt[xm[i]] += weights[i]


@numba.njit("float64[:](float64[:,:], int32[:], float64[:,:], int32[:], float64[:])")
def index_dot(x: Any, xi: Any, y: Any, yi: Any, out: Any) -> Any:
    """Store indexed row-wise dot products in ``out``."""
    n = x.shape[1]
    for i, xi_i in enumerate(xi):
        i1 = xi_i
        i2 = yi[i]
        for j in range(n):
            out[i] += x[i1, j] * y[i2, j]
    return out


@numba.njit("float64[:](int32[:], float64[:], float64[:])")
def bincount(x: Any, w: Any, out: Any) -> Any:
    """Add scalar weights to indexed entries of ``out``."""
    for i, x_i in enumerate(x):
        out[x_i] += w[i]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])")
def vec_bincount1(x: Any, w: Any, out: Any) -> Any:
    """Add matrix rows to indexed rows of ``out``."""
    n = w.shape[1]
    for i, x_i in enumerate(x):
        for j in range(n):
            out[x_i, j] += w[i, j]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], int32[:], float64[:,:])")
def vec_bincount2(x: Any, w: Any, y: Any, out: Any) -> Any:
    """Add selected matrix rows to indexed rows of ``out``."""
    for i, x_i in enumerate(x):
        out[x_i, :] += w[y[i], :]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])")
def vec_bincount3(x: Any, w: Any, out: Any) -> Any:
    """Numba bincount on the rows of matrix w for groups x.

    Used to update comp counts for word/state probabilities.

    N = len(x)
    S = number of states.
    U = unique values in x can take on (unique words in corpus).

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): S by U matrix.

    Returns:
        Numpy 2-d array.

    """
    for j, x_j in enumerate(x):
        out[:, x_j] += w[:, j]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])")
def vec_bincount4(x: Any, w: Any, out: Any) -> Any:
    """Numba bincount on the rows of matrix w for groups x.

    Used to initialize doc/state counts.

    N = len(x)
    S = number of states.
    U = unique values in x can take on. (Unique number of authors).

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): U by S matrix.

    Returns:
        Numpy 2-d array.

    """
    for j, x_j in enumerate(x):
        out[x_j, :] += w[:, j]
    return out
