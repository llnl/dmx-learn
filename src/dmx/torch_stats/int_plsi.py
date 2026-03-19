"""Create, estimate, and sample from an integer PLSI model.

Defines the IntegerPLSIDistribution, IntegerPLSISampler, IntegerPLSIAccumulatorFactory, IntegerPLSIAccumulator,
IntegerPLSIEstimator, and the IntegerPLSIDataEncoder classes for use with pysparkplug.

Consider an Integer PLSI model for a corpus of documents with S states, V word values, and D authors (doc_ids).

Let x (Tuple[int, Sequence[Tuple[int, float]]]) be an observation from a PLSI model, consisting of

    x = (d, [(v_0, c_0), (v_1, c_1), ..., (v_{k-1}, c_{k-1})]),

where the 'd' is some author (doc_id) in the corpus and each tuple (v_i, c_i) corresponds to a value-count couple
for some value 'v_i' in dictionary of words used in the corpus. Let w denote the distinct words {v_i} in the document
represented by x. The density for the PLSI model is given by

    p_mat(w, d) = P_len(nn)*p_mat(d) sum_{j=0}^{k-1} sum_{s=0}^{S-1} ( p_mat(v_j | s )p_mat(s | d) )^(c_j),

where P_len(nn) is the density of the length distribution for 'nn' representing the total number of words in
the document (i.e. nn = sum_i c_i), p_mat(d) is the probability of observing a document from author 'd', p_mat(v_j|s) is the
probability of observing word (integer-valued) given word-topic 's', and p_mat(s|d) are the weights for the word-topic for
author 'd'.

Note: To use this distribution, convert your words and authors of the corpus to unique integer keys.

"""

from typing import List, Optional, Sequence, Tuple, Union, Any, TypeVar, Dict

import numpy as np
import torch as tn
from torch import Generator
from dmx.utils.optsutil import count_by_value

import dmx.torch_utils.vector as vec
from dmx.torch_stats.null_dist import NullDistribution, NullEstimator, NullDataEncoder, NullAccumulator, \
    NullAccumulatorFactory
from dmx.torch_stats.pdist import TorchProbabilityDistribution, TorchParameterEstimator, TorchSequenceEncoder, \
    TorchStatisticAccumulator, TorchStatisticAccumulatorFactory, DistributionSampler, TorchEncodedSequence

T1 = TypeVar('T1')  # type for encoded sequence of lengths.
SS1 = TypeVar('SS1')  # type for value of length dist sufficient statistics.


class IntegerPLSIDistribution(TorchProbabilityDistribution):

    def __init__(self, state_word_mat: Union[List[List[float]], np.ndarray],
                 doc_state_mat: Union[List[List[float]], np.ndarray], doc_vec: Union[List[float], np.ndarray],
                 len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
                 device: Optional[tn.device] = None) -> None:
        """IntegerPLSIDistribution object defining an Integer PLSI distribution.

        Args:
            state_word_mat (Union[List[List[float]], tn.Tensor]): Array-like of floats that contains a
                p_mat(word | states) for each word in corpus of documents. Cols should sum to 1.0
            doc_state_mat (Union[List[List[float]], tn.Tensor]): Array-like of floats that contains a p_mat(doc | states)
                for each document id in corpus of documents. Rows should sum to 1.0
            doc_vec (Union[List[float], tn.Tensor]): Array-like containing prior for documents p_mat(d) for each
                document id in corpus of documents. Should sum to 1.0
            len_dist (Optional[TorchProbabilityDistribution]): Optional distribution for the length of
                each document (i.e. word count in an observed document). Should have support on positive integers.
            device (Optional[str]): Set the device type for object. 

        Attributes:
            prob_mat (tn.Tensor): 2-d numpy array of floats containing p_mat(word | states) in each row. Dimension is
                given by number of words times number of states.
            state_mat (tn.Tensor): 2-d numpy array of floats containing p_mat(doc | states) in each row. Dimension is
                given by number of documents times number of states.
            doc_vec (tn.Tensor): 1-d numpy array of floats containing p_mat(doc=d) for each entry. Length is equal to
                number of document ids.
            log_doc_vec (tn.Tensor): 1-d numpy array of the log(p_mat(doc=d)).
            num_vals (int): Number of total words in corpus. (Number of rows in prob_mat).
            num_states (int): Number of word topics (mixture components). (Number of columns in prob_mat/state_mat).
            num_docs (int): Total number of document ids in corpus. (Number of rows in state_mat).
            name (Optional[str]): Optional name for object instance.
            len_dist (TorchProbabilityDistribution): Distribution object for the number of words per
                document. Defaults to the NullDistribution if None is passed.

        """
        super().__init__(device)
        self.prob_mat = vec.tensor(state_word_mat, device=self._device)
        self.state_mat = vec.tensor(doc_state_mat, device=device)
        self.doc_vec = vec.tensor(doc_vec, device=device)

        self.log_doc_vec = tn.log(self.doc_vec)
        self.num_vals = self.prob_mat.shape[0]
        self.num_states = self.prob_mat.shape[1]
        self.num_docs = self.state_mat.shape[0]
        self.len_dist = len_dist if len_dist is not None else NullDistribution()

    def to(self, device: tn.device) -> None:
        self._device = device
        self.prob_mat.to(device)
        self.state_mat.to(device)
        self.doc_vec.to(device)
        self.log_doc_vec = tn.log(self.doc_vec)
        self.len_dist.to(device)

    def __repr__(self) -> str:
        """Return string representation of object instance."""
        pmat = self.prob_mat.data.cpu().numpy()
        smat = self.state_mat.data.cpu().numpy()

        s1 = ','.join(['[' + ','.join(map(str, pmat[i, :])) + ']' for i in range(len(self.prob_mat))])
        s2 = ','.join(['[' + ','.join(map(str, smat[i, :])) + ']' for i in range(len(self.state_mat))])
        s3 = ','.join(map(str, self.doc_vec.data.cpu().numpy()))
        s4 = str(self.len_dist)

        return 'IntegerPLSIDistribution([%s], [%s], [%s], len_dist=%s)' % (s1, s2, s3, s4)

    def density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        """Evaluate the density of PLSI model for an observation x.

        See log_density() for details on the density evaluation.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): Single observation of integer PLSI.

        Returns:
            Density evaluated at observed value x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        """Evaluate the log-density of PLSI model for an observation of x.

        Consider an Integer PLSI model for a corpus of documents with S states, V word values, and D documents ids
        (authors).

        Let x (Tuple[int, Sequence[Tuple[int, float]]]) be an observation from a PLSI model, consisting of
        x = (d, [(v_0, c_0), (v_1, c_1), ..., (v_{k-1}, c_{k-1})]), where the 'd' is some document d_id in the corpus and
        each tuple (v_i, c_i) corresponds to a value-count couple in the corpus. The log-likelihood is given by

        log(p_mat(x)) = log(p_mat(d)) + sum_{j=0}^{k-1} c_k*log( sum_{s=0}^{S-1} p_mat(d|s)p_mat(s|v_k) ) + log(P_len(nn)),

        where P_len(nn) is the density of the length distribution for 'nn' representing the total number of words in
        the document.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): (doc_id, [(value_id, count_for_value)]). See above for details.

        Returns:
            Log-density evaluated at a single observation x.

        """

        d_id = x[0]
        xv = vec.int_tensor([u[0] for u in x[1]], device=self._device)
        xc = vec.tensor([u[1] for u in x[1]], device=self._device)

        rv = 0.0
        rv += tn.matmul(tn.log(tn.matmul(self.prob_mat[xv, :], self.state_mat[d_id, :])), xc)
        rv += tn.log(self.doc_vec[d_id])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(int(tn.sum(xc)))

        return float(rv)

    def component_log_density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> tn.Tensor:
        """Evaluate the log-density for each state in the PLSI.

        Returns count*log(p_mat(W|S)) for each word-count pair in the document. Returned value is S by 1 where S is the
        number of components in the model.

        Args:
            x (Tuple[int, Sequence[Tuple[int, float]]]): Single PLSI observation of form
                (doc_id, [(value_id, count_for_value)]).

        Returns:
            Tensor of length S (num_states).

        """
        xv = vec.int_tensor([u[0] for u in x[1]], device=self._device)
        xc = vec.tensor([u[1] for u in x[1]], device=self._device)

        return tn.matmul(tn.log(self.prob_mat[xv, :]).T, xc)

    def seq_log_density(self, x: 'IntegerPLSITorchSequence') -> tn.Tensor:

        if not isinstance(x, IntegerPLSITorchSequence):
            raise Exception('IntegerPLSITorchSequence required for `seq_` calls')

        if x.device != self.model_device():
            raise Exception('IntegerPLSITorchSequence must be on same device as model.')
        
        nn, (xv, xc, xd, xi, xn, xm) = x.data
        cnt = len(xn)
        rv = vec.zeros(cnt)

        w = self.prob_mat[xv, :] * self.state_mat[xd, :]
        w = tn.sum(w, dim=1, keepdim=False)
        tn.log(w, out=w)
        w *= xc
        rv += tn.bincount(xi, w)
        rv += self.log_doc_vec[xm]

        if self.len_dist is not None:
            rv += self.len_dist.seq_log_density(nn)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'IntegerPLSISampler':
        """Return an IntegerPLSISampler object from IntegerPLSIDistribution instance."""
        return IntegerPLSISampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerPLSIEstimator':
        """Create an IntegerPLSIEstimator object from IntegerPLSIDistribution instance.

        Args:
            pseudo_count (Optional[float]): Re-weight object instance sufficient statistics when passed to estimator.

        Returns:
            IntegerPLSIEstimator object.

        """
        if pseudo_count is None:
            return IntegerPLSIEstimator(num_vals=self.num_vals, num_states=self.num_states,num_docs=self.num_docs,
                                        len_estimator=self.len_dist.estimator())
        else:
            pseudo_count = (pseudo_count, pseudo_count, pseudo_count)
            return IntegerPLSIEstimator(num_vals=self.num_vals, num_states=self.num_states, num_docs=self.num_docs,
                                        pseudo_count=pseudo_count,
                                        suff_stat=(self.prob_mat.T, self.state_mat, self.doc_vec),
                                        len_estimator=self.len_dist.estimator())

    def dist_to_encoder(self) -> 'IntegerPLSIDataEncoder':
        """Returns IntegerPLSIDataEncoder object."""
        return IntegerPLSIDataEncoder(len_encoder=self.len_dist.dist_to_encoder())


class IntegerPLSISampler(DistributionSampler):

    def __init__(self, dist: IntegerPLSIDistribution, seed: Optional[int] = None) -> None:
        """IntegerPLSISampler object for sampling from IntegerPLSIDistribution.

        Args:
            dist (IntegerPLSIDistribution): IntegerPLSIDistribution instance to sampler from.
            seed (Optional[int]): Set seed for random number generator used in sampling.

        Attributes:
            tng (Generator): RandomState object with seed set if passed.
            dist (IntegerPLSIDistribution): IntegerPLSIDistribution instance to sampler from.
            size_tng (Generator): RandomState object for sampling the length of documents.

        """
        self.rng = np.random.RandomState(seed)
        self.doc_vec = dist.doc_vec.data.cpu().numpy()
        self.state_mat = dist.state_mat.data.cpu().numpy()
        self.prob_mat = dist.prob_mat.data.cpu().numpy()
        self.num_vals = dist.num_vals
        self.num_docs = dist.num_docs

        self.size_rng = dist.len_dist.sampler(self.rng.randint(2**31))

    def sample(self, size: Optional[int] = None) \
            -> Union[Tuple[int, Sequence[Tuple[int, float]]], Sequence[Tuple[int, Sequence[Tuple[int, float]]]]]:
        """Generate iid samples from PLSI model.

        Args:
            size (Optional[int]): Number of samples to generate. Defaults to 0 if size is None.

        Returns:
            Sequence of iid PLSI samples if size is not None, else a single sample from PLSI model.

        """
        if size is None:
            d_id = self.rng.choice(self.num_docs, p=self.doc_vec)
            cnt = self.size_rng.sample()
            z = self.rng.multinomial(cnt, pvals=self.state_mat[d_id, :])
            rv = []
            for i, n in enumerate(z):
                if n > 0:
                    rv.extend(self.rng.choice(self.num_vals, p=self.prob_mat[:, i], replace=True, size=n))

            return d_id, list(count_by_value(rv).items())

        else:
            return [self.sample() for i in range(size)]


class IntegerPLSIAccumulator(TorchStatisticAccumulator):

    def __init__(self, num_vals: int, num_states: int, num_docs: int,
                 len_acc: Optional[TorchStatisticAccumulator] = NullAccumulator(),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 device: Optional[str] = None) -> None:
        """IntegerPLSIAccumulator object for aggregating sufficient statistics from observed data.

        Note: Keys in order, words/values, states, documents.

        Args:
            num_vals (int): Number of words in the corpus.
            num_states (int): Number of word-topics.
            num_docs (int): Number of authors (doc_ids) in the corpus.
            len_acc (Optional[TorchStatisticAccumulator]): Optional accumulator for the length of documents.
                Should have support on non-negative integer values.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Optional keys for words, states, and
                authors (doc_ids).
            device (Optional[str]): Set device type for object.

        Attributes:
            num_vals (int): Number of words in the corpus.
            num_states (int): Number of word-topics or mixture components.
            num_docs (int): Number of authors (doc_ids) in the corpus.
            word_count (tn.Tensor): Numpy array of shape num_states by num_vals for aggregating state/word counts.
            comp_count (tn.Tensor): Numpy array (num_docs by num_states) for aggregating doc/state counts.
            doc_count (tn.Tensor): Numpy array for aggregating counts of authors (prior on doc_ids).
            name (Optional[str]): Name of object instance.
            wc_key (Optional[str]): Key for merging 'word_count' with objects containing matching keys.
            sc_key (Optional[str]): Key for merging 'comp_count' with objects containing matching keys.
            dc_key (Optional[str]): Key for merging 'doc_count' with objects containing matching keys.
            len_acc (TorchStatisticAccumulator): Accumulator object for the lengths of documents (total
                word counts). Defaults to the NullAccumulator if None is passed.

            _init_tng (bool): True if Generator objects for accumulator have been initialized.
            _acc_tng (Optional[Generator]): Generator object for initializing the PLSI model.
            _len_tng (Optional[Generator]): Generator object for initializing the length accumulator.

        """
        super().__init__(device)
        self.num_vals   = num_vals
        self.num_states = num_states
        self.num_docs   = num_docs
        self.word_count = np.zeros((num_states, num_vals), dtype=np.float64)
        self.comp_count = np.zeros((num_docs, num_states), dtype=np.float64)
        self.doc_count  = np.zeros(num_docs, dtype=np.float64)

        self.wc_key, self.sc_key, self.dc_key = keys if keys is not None else (None, None, None)
        self.len_acc = len_acc if len_acc is not None else NullAccumulator()

    def seq_initialize(self, x: 'IntegerPLSITorchSequence', weights: tn.Tensor, tng: Generator) -> None:
        nn, (xv, xc, xd, xi, xn, xm) = x.data

        # update = vec.mixture_weights(k=self.num_states, alpha=1.0/self.num_states, size=len(xv), tng=tng).T
        update = vec.sample_dirichlet(alpha=vec.ones(self.num_states) / self.num_states, size=len(xv), tng=tng).T
        update *= xc * weights[xi]

        for i in range(self.num_states):
            self.word_count[i, :] += tn.bincount(xv, weights=update[i, :], minlength=self.num_vals).cpu().detach().numpy()
            self.comp_count[:, i] += tn.bincount(xd, weights=update[i, :], minlength=self.num_docs).cpu().detach().numpy()

        self.doc_count += tn.bincount(xm, weights=weights, minlength=self.num_docs).data.cpu().numpy()

        self.len_acc.seq_initialize(nn, weights, tng)

    def seq_update(self, x: 'IntegerPLSITorchSequence', weights: tn.Tensor, estimate: IntegerPLSIDistribution) -> None:

        nn, (xv, xc, xd, xi, xn, xm) = x.data

        temp = xc * weights[xi]
        update = estimate.prob_mat[xv, :] * estimate.state_mat[xd, :]
        temp /= tn.sum(update, dim=1)
        update *= temp[:, None]

        for i in range(self.num_states):
            self.word_count[i, :] += tn.bincount(xv, weights=update[:, i], minlength=self.num_vals).cpu().detach().numpy()
            self.comp_count[:, i] += tn.bincount(xd, weights=update[:, i], minlength=self.num_docs).cpu().detach().numpy()

        self.doc_count += tn.bincount(xm, weights=weights, minlength=self.num_docs).data.cpu().numpy()

        self.len_acc.seq_update(nn, weights, estimate.len_dist)

    def combine(self, suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]) -> 'IntegerPLSIAccumulator':
        """Combine the sufficient statistics in arg 'suff_stat' with object instance.

        Arg 'suff_stat' is Tuple[tn.Tensor, tn.Tensor, tn.Tensor, Optional[SS1]] containing:
            suff_stat[0] (np.ndarray): State/word counts with matching dimension of num_states by num_vals.
            suff_stat[1] (np.ndarray): Doc/state counts with matching dimension of num_docs by num_states.
            suff_stat[2] (np.ndarray): Author counts with length (num_docs).
            suff_stat[3] (Optional[SS1]): Sufficient statistics for the length of document distribution having type SS1.

        Args:
            suff_stat: See above for details.

        Returns:
            IntegerPLSIAccumulator object.

        """
        self.word_count += suff_stat[0]
        self.comp_count += suff_stat[1]
        self.doc_count += suff_stat[2]

        self.len_acc.combine(suff_stat[3])

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
        return self.word_count, self.comp_count, self.doc_count, self.len_acc.value()

    def from_value(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]) -> 'IntegerPLSIAccumulator':
        self.word_count = x[0]
        self.comp_count = x[1]
        self.doc_count = x[2]
        self.len_acc.from_value(x[3])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge the sufficient statistics of object instance with matching keys.

        If wc_key is set, merge the state/word count variable.
        If sc_key is set, merge the doc/state count variable.
        If dc_key is set, merge the author count variable.

        Call key_merge() of accumulator for the length. Note nothing is done for this if default NullAccumulator is set.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to sufficient statistics.

        Returns:
            None.

        """
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
        """Set the sufficient statistics of object instance to matching key values in arg 'stats_dict'.

        If wc_key is set, set the state/word count variable to matching key in stats_dict.
        If sc_key is set, set the doc/state count variable to matching key in stats_dict.
        If dc_key is set, set the author count variable to matching key in stats_dict.

        Call key_replace() of accumulator for the length. Note nothing is done for this if default NullAccumulator is
        set.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to sufficient statistics.

        Returns:
            None.

        """
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

    def acc_to_encoder(self) -> 'IntegerPLSIDataEncoder':
        """Return an IntegerPLSIDataEncoder object."""
        len_encoder = self.len_acc.acc_to_encoder()
        return IntegerPLSIDataEncoder(len_encoder=len_encoder)


class IntegerPLSIAccumulatorFactory(TorchStatisticAccumulatorFactory):

    def __init__(self, num_vals: int, num_states: int, num_docs: int,
                 len_factory: Optional[TorchStatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 device: Optional[tn.device] = None) -> None:
        """IntegerPLSIAccumulatorFactory object for creating IntegerPLSIAccumulator objects.

        Args:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_factory (Optional[StatisticsAccumulatorFactory]): Accumulator factory object for length distribution.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for merging
                word, state, and doc sufficient statistics with matching keys.
            device (Optional[str]): Set device type for object

        Attributes:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_factory (StatisticsAccumulatorFactory): Accumulator factory object for length distribution. Defaults
                to the NullAccumulatorFactory(). Should have support on non-negative integers.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for merging word, state, and doc
                sufficient statistics with matching keys.

        """
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys if keys is not None else (None, None, None)
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs

    def make(self, device: Optional[tn.device] = None) -> 'IntegerPLSIAccumulator':
        """Returns IntegerPLSIAccumulator object."""
        return IntegerPLSIAccumulator(self.num_vals, self.num_states, self.num_docs,
                                      len_acc=self.len_factory.make(device=device),
                                      keys=self.keys, device=device)

class IntegerPLSIEstimator(TorchParameterEstimator):

    def __init__(self, num_vals: int, num_states: int, num_docs: int,
                 len_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
                 pseudo_count: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = (None, None, None),
                 suff_stat: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                           Optional[tn.Tensor]]] = (None, None, None),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None)) -> None:
        """IntegerPLSIEstimator for estimating integer PLSI distributions from aggregated sufficient statistics.

        Args:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_estimator (Optional[TorchParameterEstimator]): Optional ParameterEstimator object for the length of
                documents. Should have support on non-negative integers if not None.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float], Optional[float]]]): Optional re-weight
                sufficient statistics in 'estimate()' function.
            suff_stat (Optional[Tuple[Optional[tn.Tensor], Optional[tn.Tensor], Optional[tn.Tensor]]]): Optional
                Tuple of numpy arrays containing 'word_counts' (num_states by num_vals), 'state_counts' (num_docs by
                num_states), and doc_counts (length num_docs).
            name (Optional[str]): Set name to object instance.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for merging word, state, and doc
                sufficient statistics with matching keys.

        Attributes:
            num_vals (int): Number of words/values in PLSI.
            num_states (int): Number of states in PLSI.
            num_docs (int): Number of doc_ids (authors) in PLSI.
            len_estimator (TorchParameterEstimator): Optional ParameterEstimator object for the length of documents.
                Should have support on non-negative integers. Defaults to NullEstimator() if None is passed.
            pseudo_count (Tuple[Optional[float], Optional[float], Optional[float]]): Optional re-weight sufficient
                statistics in 'estimate()' function. Defaults to (None, None, None) if None is passed.
            suff_stat (Tuple[Optional[tn.Tensor], Optional[tn.Tensor], Optional[tn.Tensor]]): Optional
                Tuple of numpy arrays containing 'word_counts' (num_states by num_vals), 'state_counts' (num_docs by
                num_states), and doc_counts (length num_docs). Defaults to (None, None, None) if None is passed.
            name (Optional[str]): Name of object instance.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for merging word, state, and doc
                sufficient statistics with matching keys.
        """
        self.suff_stat     = suff_stat if suff_stat is not None else (None, None, None)
        self.pseudo_count  = pseudo_count if pseudo_count is not None else (None, None, None)
        self.num_vals      = num_vals
        self.num_states    = num_states
        self.num_docs      = num_docs
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.keys          = keys if keys is not None else (None, None, None)

    def accumulator_factory(self) -> 'IntegerPLSIAccumulatorFactory':
        """Returns IntegerPLSIAccumulatorFactory object."""
        len_est = self.len_estimator.accumulator_factory()
        return IntegerPLSIAccumulatorFactory(self.num_vals, self.num_states, self.num_docs, len_est, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]], device: Optional[tn.device] = None)\
            -> 'IntegerPLSIDistribution':
        """Estimate IntegerPLSIDistribution from aggregated sufficient statistics in arg 'suff_stat'.

        Args:
            nobs (Optional[float]): Optional number of observations used to accumulate 'suff_stat'.
            suff_stat: See above for details.

        Returns:
            IntegerPLSIDistribution object.

        """
        word_count, comp_count, doc_count, len_suff_stats = suff_stat

        if self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt*self.suff_stat[0].T
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)

        elif self.pseudo_count[0] is not None and self.suff_stat[0] is None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)

        else:
            wsum = np.sum(word_count, axis=1)
            wsum = np.where(wsum > 0.0, wsum, 1.0)
            word_prob_mat = word_count.T / wsum

        if self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt * self.suff_stat[1]
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)

        elif self.pseudo_count[1] is not None and self.suff_stat[1] is None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)

        else:
            ssum = np.sum(comp_count, axis=1, keepdims=False)
            ssum = np.where(ssum > 0.0, ssum, 1.0)[:, None]
            state_prob_mat = comp_count / ssum

        if self.pseudo_count[2] is not None and self.suff_stat[2] is not None:
            adj_cnt = self.pseudo_count[1] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt*self.suff_stat[2]
            doc_prob_vec /= np.sum(doc_prob_vec)

        elif self.pseudo_count[2] is not None and self.suff_stat[2] is None:
            adj_cnt = self.pseudo_count[1] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt
            doc_prob_vec /= np.sum(doc_prob_vec)

        else:
            doc_prob_vec = doc_count / np.sum(doc_count)

        len_dist = self.len_estimator.estimate(None, len_suff_stats, device=device)

        return IntegerPLSIDistribution(word_prob_mat, state_prob_mat, doc_prob_vec, len_dist=len_dist, device=device)


class IntegerPLSIDataEncoder(TorchSequenceEncoder):

    def __init__(self, len_encoder: Optional[TorchSequenceEncoder] = NullDataEncoder(), device: Optional[str] = None):
        """IntegerPLSIDataEncoder object for encoding sequences of iid observations from a PLSI model.

        Args:
            len_encoder (Optional[TorchSequenceEncoder]): Optional TorchSequenceEncoder for the total number of words
                in each document.

        Attributes:
            len_encoder (TorchSequenceEncoder): TorchSequenceEncoder for the total number of words in each document,
                defaulting to NullDataEncoder if None is passed.

        """
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Returns a string representation of object instance."""
        return 'IntegerPLSIDataEncoder(len_dist=%s)' % (repr(self.len_encoder))

    def __eq__(self, other: object) -> bool:
        """Check if object is equivalent to instance of IntegerPLSIDataEncoder.

        Args:
            other (object): Other object to compare to instance.

        Returns:
            True if object is an instance of IntegerPLSIDataEncoder with matching 'len_encoder' attribute.

        """
        if isinstance(other, IntegerPLSIDataEncoder):
            return other.len_encoder == self.len_encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Tuple[int, Sequence[Tuple[int, float]]]], device: Optional[tn.device] = None)\
            -> 'IntegerPLSITorchSequence':
        """Encode a sequence of iid PLSI observations for use with vectorized functions.

        Input arg 'x' is a sequence of iid PLSI observations having form

        x = [ (doc_id, [(value, count),...]),... ].

        The return value is a Tuple length 2. The first component contains data type Optional[T1] corresponding to the
        sequence encoding of the lengths. The second component is a Tuple of length 6 containing
            xv (tn.Tensor[int]): Numpy array of flattened word values.
            xc (tn.Tensor[float]): Numpy array of flattened counts for word values above.
            xd (tn.Tensor[int]): Document d_id for each word-count pair in the arrays above.
            xi (tn.Tensor[int]): Observed sequence index for each word-count pair in the arrays above.
            xn (tn.Tensor[float]): Numpy array of the total number of words in each document.
            xm (tn.Tensor[float]): Flattened array of document d_id's for the lengths above (len = len(x)).

        Args:
            x (Sequence[Tuple[int, Sequence[Tuple[int, float]]]]): See above for details.

        Returns:
            See above for details.

        """
        xv = []
        xc = []
        xd = []
        xi = []
        xn = []
        xm = []

        for i, (d_id, xx) in enumerate(x):

            v = [u[0] for u in xx]
            c = [u[1] for u in xx]

            xv.extend(v)
            xc.extend(c)
            xd.extend([d_id]*len(v))
            xi.extend([i]*len(v))
            xn.append(np.sum(c))
            xm.append(d_id)

        xv = vec.int_tensor(xv, device=device)
        xc = vec.tensor(xc, device=device)
        xd = vec.int_tensor(xd, device=device)
        xi = vec.int_tensor(xi, device=device)
        xn = vec.tensor(xn, device=device)
        xm = vec.int_tensor(xm, device=device)

        nn = self.len_encoder.seq_encode(xn, device=device)

        return IntegerPLSITorchSequence(data=(nn, (xv, xc, xd, xi, xn, xm)), device=device)


class IntegerPLSITorchSequence(TorchEncodedSequence):

    def __init__(self, data: Tuple[TorchEncodedSequence, Tuple[tn.tensor, tn.tensor, tn.tensor, tn.tensor, tn.tensor, tn.tensor]], device: Optional[tn.device] = None):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f'IntegerPLSITorchSequence(device={repr(self.device)})'
