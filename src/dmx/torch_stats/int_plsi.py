"""Provide torch-backed integer probabilistic latent semantic indexing models.

An observation is a document identifier and sparse ``(term, count)`` pairs.
Each token has a latent topic; document-topic and topic-term probabilities
produce its marginal term probability. Encoded batches flatten ``M`` nonzero
term entries from ``N`` documents, retaining document and observation indices.
Model tensors move with ``to`` and use the vector-helper floating dtype,
normally float64 and float32 on MPS; sufficient statistics and sampling are
CPU NumPy based. This mirrors ``dmx.stats.int_plsi``.
"""

# pylint: disable=too-many-positional-arguments,duplicate-code

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch as tn
from torch import Generator

import dmx.torch_utils.vector as vec
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
from dmx.utils.optsutil import count_by_value

T1 = TypeVar("T1")  # type for encoded sequence of lengths.
SS1 = TypeVar("SS1")  # type for value of length dist sufficient statistics.


class IntegerPLSIDistribution(TorchProbabilityDistribution):
    """Represent a document-topic-term model with an optional length child."""

    def __init__(
        self,
        state_word_mat: Union[List[List[float]], np.ndarray],
        doc_state_mat: Union[List[List[float]], np.ndarray],
        doc_vec: Union[List[float], np.ndarray],
        len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize term-topic, document-topic, and document probabilities."""
        super().__init__(device)
        self.prob_mat = vec.tensor(state_word_mat, device=self._device)
        self.state_mat = vec.tensor(doc_state_mat, device=device)
        self.doc_vec = vec.tensor(doc_vec, device=device)

        self.log_doc_vec = tn.log(self.doc_vec)
        self.num_vals = self.prob_mat.shape[0]
        self.num_states = self.prob_mat.shape[1]
        self.num_docs = self.state_mat.shape[0]
        self.len_dist = len_dist if len_dist is not None else NullDistribution()

    def to(self, device: vec.DeviceLike) -> "IntegerPLSIDistribution":
        """Move model and length-child tensors to ``device`` in place."""
        target_device = self._resolve_device_arg(device)
        self._device = target_device
        self.prob_mat = self.prob_mat.to(target_device)
        self.state_mat = self.state_mat.to(target_device)
        self.doc_vec = self.doc_vec.to(target_device)
        self.log_doc_vec = tn.log(self.doc_vec)
        self.len_dist.to(target_device)
        return self

    def __repr__(self) -> str:
        """Return a constructor-like representation using CPU parameters."""
        pmat = self.prob_mat.data.cpu().numpy()
        smat = self.state_mat.data.cpu().numpy()

        s1 = ",".join(
            [
                "[" + ",".join(map(str, pmat[i, :])) + "]"
                for i in range(len(self.prob_mat))
            ]
        )
        s2 = ",".join(
            [
                "[" + ",".join(map(str, smat[i, :])) + "]"
                for i in range(len(self.state_mat))
            ]
        )
        s3 = ",".join(map(str, self.doc_vec.data.cpu().numpy()))
        s4 = str(self.len_dist)

        return f"IntegerPLSIDistribution([{s1}], [{s2}], [{s3}], len_dist={s4})"

    def density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        """Evaluate the marginal density of one sparse document observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        """Marginalize latent topics for one sparse document observation."""
        d_id = x[0]
        xv = vec.int_tensor([u[0] for u in x[1]], device=self._device)
        xc = vec.tensor([u[1] for u in x[1]], device=self._device)

        rv = tn.matmul(
            tn.log(tn.matmul(self.prob_mat[xv, :], self.state_mat[d_id, :])), xc
        )
        rv += tn.log(self.doc_vec[d_id])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(int(tn.sum(xc)))

        return float(rv)

    def component_log_density(
        self, x: Tuple[int, Sequence[Tuple[int, float]]]
    ) -> tn.Tensor:
        """Return per-topic token log densities with shape ``(K,)``."""
        xv = vec.int_tensor([u[0] for u in x[1]], device=self._device)
        xc = vec.tensor([u[1] for u in x[1]], device=self._device)

        return tn.matmul(tn.log(self.prob_mat[xv, :]).T, xc)

    def seq_log_density(self, x: "IntegerPLSITorchSequence") -> tn.Tensor:
        """Return marginal document log densities with shape ``(N,)``."""
        if not isinstance(x, IntegerPLSITorchSequence):
            raise TypeError("IntegerPLSITorchSequence required for `seq_` calls")

        nn, (xv, xc, xd, xi, xn, xm) = x.data
        cnt = len(xn)
        rv = vec.zeros(cnt, device=self._device)

        xv_dev = xv.to(device=self.prob_mat.device)
        xd_dev = xd.to(device=self.state_mat.device)
        xi_dev = xi.to(device=self._device)
        xm_dev = xm.to(device=self._device)
        w = self.prob_mat[xv_dev, :] * self.state_mat[xd_dev, :]
        w = tn.sum(w, dim=1, keepdim=False)
        tn.log(w, out=w)
        w *= xc.to(device=w.device, dtype=w.dtype)
        rv += tn.bincount(xi_dev, w.to(device=self._device), minlength=cnt)
        rv += self.log_doc_vec[xm_dev]

        if self.len_dist is not None:
            rv += self.len_dist.seq_log_density(nn)

        return rv

    def sampler(self, seed: Optional[int] = None) -> "IntegerPLSISampler":
        """Create a CPU sampler for documents, topics, terms, and lengths."""
        return IntegerPLSISampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "IntegerPLSIEstimator":
        """Create an estimator with optional smoothing of all probability tables."""
        if pseudo_count is None:
            return IntegerPLSIEstimator(
                num_vals=self.num_vals,
                num_states=self.num_states,
                num_docs=self.num_docs,
                len_estimator=self.len_dist.estimator(),
            )
        pseudo_counts = (pseudo_count, pseudo_count, pseudo_count)
        return IntegerPLSIEstimator(
            num_vals=self.num_vals,
            num_states=self.num_states,
            num_docs=self.num_docs,
            pseudo_count=pseudo_counts,
            suff_stat=(
                self.prob_mat.T.detach().cpu().numpy(),
                self.state_mat.detach().cpu().numpy(),
                self.doc_vec.detach().cpu().numpy(),
            ),
            len_estimator=self.len_dist.estimator(),
        )

    def dist_to_encoder(self) -> "IntegerPLSIDataEncoder":
        """Create a sparse document encoder with the length child encoder."""
        return IntegerPLSIDataEncoder(len_encoder=self.len_dist.dist_to_encoder())


class IntegerPLSISampler(DistributionSampler):
    """Sample document identifiers, token topics, terms, and total lengths."""

    def __init__(
        self, dist: IntegerPLSIDistribution, seed: Optional[int] = None
    ) -> None:
        """Initialize CPU NumPy copies of PLSI parameters and length sampler."""
        self.rng = np.random.RandomState(seed)
        self.doc_vec = dist.doc_vec.data.cpu().numpy()
        self.state_mat = dist.state_mat.data.cpu().numpy()
        self.prob_mat = dist.prob_mat.data.cpu().numpy()
        self.num_vals = dist.num_vals
        self.num_docs = dist.num_docs

        self.size_rng = dist.len_dist.sampler(self.rng.randint(2**31))

    def sample(self, size: Optional[int] = None) -> Union[
        Tuple[int, Sequence[Tuple[int, float]]],
        Sequence[Tuple[int, Sequence[Tuple[int, float]]]],
    ]:
        """Draw one sparse document observation or ``size`` observations."""
        if size is None:
            d_id = self.rng.choice(self.num_docs, p=self.doc_vec)
            cnt = self.size_rng.sample()
            z = self.rng.multinomial(cnt, pvals=self.state_mat[d_id, :])
            rv = []
            for i, n in enumerate(z):
                if n > 0:
                    rv.extend(
                        self.rng.choice(
                            self.num_vals, p=self.prob_mat[:, i], replace=True, size=n
                        )
                    )

            return d_id, list(count_by_value(rv).items())

        samples: List[Tuple[int, Sequence[Tuple[int, float]]]] = []
        for _ in range(size):
            sample = self.sample()
            assert isinstance(sample, tuple)
            samples.append(sample)
        return samples


class IntegerPLSIAccumulator(TorchStatisticAccumulator):
    """Accumulate topic-term, document-topic, document, and length statistics."""

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_acc: Optional[TorchStatisticAccumulator] = NullAccumulator(),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        device: vec.DeviceLike = None,
    ) -> None:
        """Initialize CPU count arrays for a fixed document/topic/term shape."""
        super().__init__(device)
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs
        self.word_count = np.zeros((num_states, num_vals), dtype=np.float64)
        self.comp_count = np.zeros((num_docs, num_states), dtype=np.float64)
        self.doc_count = np.zeros(num_docs, dtype=np.float64)

        self.wc_key, self.sc_key, self.dc_key = (
            keys if keys is not None else (None, None, None)
        )
        self.len_acc = len_acc if len_acc is not None else NullAccumulator()

    def seq_initialize(
        self, x: "IntegerPLSITorchSequence", weights: tn.Tensor, tng: Generator
    ) -> None:
        """Randomly allocate each flat term count across topics for initialization."""
        nn, (xv, xc, xd, xi, _, xm) = x.data

        # Equivalent to mixture-weights initialization, but sampled directly.
        update = vec.sample_dirichlet(
            alpha=vec.ones(self.num_states) / self.num_states, size=len(xv), tng=tng
        ).T
        update *= xc * weights[xi]

        for i in range(self.num_states):
            self.word_count[i, :] += (
                tn.bincount(xv, weights=update[i, :], minlength=self.num_vals)
                .cpu()
                .detach()
                .numpy()
            )
            self.comp_count[:, i] += (
                tn.bincount(xd, weights=update[i, :], minlength=self.num_docs)
                .cpu()
                .detach()
                .numpy()
            )

        self.doc_count += (
            tn.bincount(xm, weights=weights, minlength=self.num_docs).data.cpu().numpy()
        )

        self.len_acc.seq_initialize(nn, weights, tng)

    def seq_update(
        self,
        x: "IntegerPLSITorchSequence",
        weights: tn.Tensor,
        estimate: IntegerPLSIDistribution,
    ) -> None:
        """Update counts from posterior token-topic responsibilities."""
        nn, (xv, xc, xd, xi, _, xm) = x.data

        temp = xc * weights[xi]
        update = estimate.prob_mat[xv, :] * estimate.state_mat[xd, :]
        temp /= tn.sum(update, dim=1)
        update *= temp[:, None]

        for i in range(self.num_states):
            self.word_count[i, :] += (
                tn.bincount(xv, weights=update[:, i], minlength=self.num_vals)
                .cpu()
                .detach()
                .numpy()
            )
            self.comp_count[:, i] += (
                tn.bincount(xd, weights=update[:, i], minlength=self.num_docs)
                .cpu()
                .detach()
                .numpy()
            )

        self.doc_count += (
            tn.bincount(xm, weights=weights, minlength=self.num_docs).data.cpu().numpy()
        )

        self.len_acc.seq_update(nn, weights, estimate.len_dist)

    def combine(
        self, suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]
    ) -> "IntegerPLSIAccumulator":
        """Merge topic-term, document-topic, document, and length statistics."""
        self.word_count += suff_stat[0]
        self.comp_count += suff_stat[1]
        self.doc_count += suff_stat[2]

        self.len_acc.combine(suff_stat[3])

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
        """Return CPU statistics ``(topic_terms, doc_topics, docs, length)``."""
        return self.word_count, self.comp_count, self.doc_count, self.len_acc.value()

    def from_value(
        self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]
    ) -> "IntegerPLSIAccumulator":
        """Replace all PLSI sufficient statistics from a tuple."""
        self.word_count = x[0]
        self.comp_count = x[1]
        self.doc_count = x[2]
        self.len_acc.from_value(x[3])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge separately keyed topic-term, document-topic, and document counts."""
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
        """Replace separately keyed statistics and recurse into the length child."""
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
        """Create a sparse document encoder from the length accumulator."""
        len_encoder = self.len_acc.acc_to_encoder()
        return IntegerPLSIDataEncoder(len_encoder=len_encoder)


class IntegerPLSIAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """Create PLSI accumulators with a fixed document/topic/term shape."""

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_factory: Optional[
            TorchStatisticAccumulatorFactory
        ] = NullAccumulatorFactory(),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
        _device: Optional[tn.device] = None,
    ) -> None:
        """Initialize dimensions, length factory, and sufficient-statistic keys."""
        self.len_factory = (
            len_factory if len_factory is not None else NullAccumulatorFactory()
        )
        self.keys = keys if keys is not None else (None, None, None)
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs

    def make(self, device: Optional[tn.device] = None) -> "IntegerPLSIAccumulator":
        """Create a PLSI accumulator and length child associated with ``device``."""
        return IntegerPLSIAccumulator(
            self.num_vals,
            self.num_states,
            self.num_docs,
            len_acc=self.len_factory.make(device=device),
            keys=self.keys,
            device=device,
        )


class IntegerPLSIEstimator(TorchParameterEstimator):
    """Estimate term-topic, document-topic, and document distributions."""

    def __init__(
        self,
        num_vals: int,
        num_states: int,
        num_docs: int,
        len_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
        pseudo_count: Optional[
            Tuple[Optional[float], Optional[float], Optional[float]]
        ] = (None, None, None),
        suff_stat: Optional[
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        ] = (None, None, None),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
    ) -> None:
        """Initialize dimensions, priors, smoothing, keys, and length estimator."""
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

    def accumulator_factory(self) -> "IntegerPLSIAccumulatorFactory":
        """Create an accumulator factory retaining dimensions and keys."""
        len_est = self.len_estimator.accumulator_factory()
        return IntegerPLSIAccumulatorFactory(
            self.num_vals, self.num_states, self.num_docs, len_est, self.keys
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]],
        device: Optional[tn.device] = None,
    ) -> "IntegerPLSIDistribution":
        """Normalize aggregated counts into PLSI probabilities on ``device``."""
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
            doc_pc = self.pseudo_count[2]
            assert doc_pc is not None
            doc_adj_cnt = doc_pc / len(doc_count)
            doc_prob_vec = doc_count + doc_adj_cnt * self.suff_stat[2]
            doc_prob_vec /= np.sum(doc_prob_vec)

        elif self.pseudo_count[2] is not None and self.suff_stat[2] is None:
            doc_pc = self.pseudo_count[2]
            assert doc_pc is not None
            doc_adj_cnt = doc_pc / len(doc_count)
            doc_prob_vec = doc_count + doc_adj_cnt
            doc_prob_vec /= np.sum(doc_prob_vec)

        else:
            doc_prob_vec = doc_count / np.sum(doc_count)

        len_dist = self.len_estimator.estimate(None, len_suff_stats, device=device)

        return IntegerPLSIDistribution(
            word_prob_mat,
            state_prob_mat,
            doc_prob_vec,
            len_dist=len_dist,
            device=device,
        )


class IntegerPLSIDataEncoder(TorchSequenceEncoder):
    """Flatten sparse document term counts and encode document lengths."""

    def __init__(
        self,
        len_encoder: Optional[TorchSequenceEncoder] = NullDataEncoder(),
        _device: Optional[str] = None,
    ) -> None:
        """Initialize the optional sequence-length encoder."""
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Return a representation of the length encoder."""
        return f"IntegerPLSIDataEncoder(len_dist={self.len_encoder!r})"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder has the same length encoder."""
        if isinstance(other, IntegerPLSIDataEncoder):
            return other.len_encoder == self.len_encoder
        return False

    def seq_encode(
        self,
        x: Sequence[Tuple[int, Sequence[Tuple[int, float]]]],
        device: Optional[tn.device] = None,
    ) -> "IntegerPLSITorchSequence":
        """Encode ``N`` sparse documents with ``M`` total nonzero entries.

        The flat tensors are ``(terms, counts, document_ids, observation_ids,
        lengths, document_ids_per_observation)``. Terms, document IDs, and
        observation IDs are integer tensors of shape ``(M,)``; counts also have
        shape ``(M,)``. Lengths and document IDs per observation have shape
        ``(N,)``. All tensors and the length encoding are created on ``device``.
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

        xv_tensor = vec.int_tensor(xv, device=device)
        xc_tensor = vec.tensor(xc, device=device)
        xd_tensor = vec.int_tensor(xd, device=device)
        xi_tensor = vec.int_tensor(xi, device=device)
        xn_tensor = vec.tensor(xn, device=device)
        xm_tensor = vec.int_tensor(xm, device=device)

        nn = self.len_encoder.seq_encode(xn, device=device)

        return IntegerPLSITorchSequence(
            data=(
                nn,
                (xv_tensor, xc_tensor, xd_tensor, xi_tensor, xn_tensor, xm_tensor),
            ),
            device=device,
        )


class IntegerPLSITorchSequence(TorchEncodedSequence):
    """Store a length encoding and flattened sparse document-count tensors."""

    data: Tuple[
        TorchEncodedSequence,
        Tuple[tn.Tensor, tn.Tensor, tn.Tensor, tn.Tensor, tn.Tensor, tn.Tensor],
    ]

    def __init__(
        self,
        data: Tuple[
            TorchEncodedSequence,
            Tuple[tn.Tensor, tn.Tensor, tn.Tensor, tn.Tensor, tn.Tensor, tn.Tensor],
        ],
        device: Optional[tn.device] = None,
    ) -> None:
        """Initialize the sparse document encoding and associated device."""
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        """Return a representation containing the encoded device."""
        return f"IntegerPLSITorchSequence(device={repr(self.device)})"
