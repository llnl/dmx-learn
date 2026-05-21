"""Create, estimate, and sample from an integer PLSI model."""

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

    def __init__(
        self,
        state_word_mat: Union[List[List[float]], np.ndarray],
        doc_state_mat: Union[List[List[float]], np.ndarray],
        doc_vec: Union[List[float], np.ndarray],
        len_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        device: Optional[tn.device] = None,
    ) -> None:
        """IntegerPLSIDistribution object defining an Integer PLSI distribution."""
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
        self.prob_mat = self.prob_mat.to(device)
        self.state_mat = self.state_mat.to(device)
        self.doc_vec = self.doc_vec.to(device)
        self.log_doc_vec = tn.log(self.doc_vec)
        self.len_dist.to(device)

    def __repr__(self) -> str:
        """Return string representation of object instance."""
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
        """Evaluate the density of PLSI model for an observation x."""
        return np.exp(self.log_density(x))

    def log_density(self, x: Tuple[int, Sequence[Tuple[int, float]]]) -> float:
        """Evaluate the log-density of PLSI model for an observation of x."""

        d_id = x[0]
        xv = vec.int_tensor([u[0] for u in x[1]], device=self._device)
        xc = vec.tensor([u[1] for u in x[1]], device=self._device)

        rv = 0.0
        rv += tn.matmul(
            tn.log(tn.matmul(self.prob_mat[xv, :], self.state_mat[d_id, :])), xc
        )
        rv += tn.log(self.doc_vec[d_id])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(int(tn.sum(xc)))

        return float(rv)

    def component_log_density(
        self, x: Tuple[int, Sequence[Tuple[int, float]]]
    ) -> tn.Tensor:
        """Evaluate the log-density for each state in the PLSI."""
        xv = vec.int_tensor([u[0] for u in x[1]], device=self._device)
        xc = vec.tensor([u[1] for u in x[1]], device=self._device)

        return tn.matmul(tn.log(self.prob_mat[xv, :]).T, xc)

    def seq_log_density(self, x: "IntegerPLSITorchSequence") -> tn.Tensor:

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
        """Return an IntegerPLSISampler object from IntegerPLSIDistribution instance."""
        return IntegerPLSISampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "IntegerPLSIEstimator":
        """
        Create an IntegerPLSIEstimator object from IntegerPLSIDistribution instance.
        """
        if pseudo_count is None:
            return IntegerPLSIEstimator(
                num_vals=self.num_vals,
                num_states=self.num_states,
                num_docs=self.num_docs,
                len_estimator=self.len_dist.estimator(),
            )
        pseudo_count = (pseudo_count, pseudo_count, pseudo_count)
        return IntegerPLSIEstimator(
            num_vals=self.num_vals,
            num_states=self.num_states,
            num_docs=self.num_docs,
            pseudo_count=pseudo_count,
            suff_stat=(self.prob_mat.T, self.state_mat, self.doc_vec),
            len_estimator=self.len_dist.estimator(),
        )

    def dist_to_encoder(self) -> "IntegerPLSIDataEncoder":
        """Returns IntegerPLSIDataEncoder object."""
        return IntegerPLSIDataEncoder(len_encoder=self.len_dist.dist_to_encoder())


class IntegerPLSISampler(DistributionSampler):

    def __init__(
        self, dist: IntegerPLSIDistribution, seed: Optional[int] = None
    ) -> None:
        """IntegerPLSISampler object for sampling from IntegerPLSIDistribution."""
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
        """Generate iid samples from PLSI model."""
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

        return [self.sample() for _ in range(size)]


class IntegerPLSIAccumulator(TorchStatisticAccumulator):

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
        device: Optional[str] = None,
    ) -> None:
        """
        IntegerPLSIAccumulator object for aggregating sufficient statistics from
        observed data.
        """
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
        """Combine the sufficient statistics in arg 'suff_stat' with object instance."""
        self.word_count += suff_stat[0]
        self.comp_count += suff_stat[1]
        self.doc_count += suff_stat[2]

        self.len_acc.combine(suff_stat[3])

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
        return self.word_count, self.comp_count, self.doc_count, self.len_acc.value()

    def from_value(
        self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1]]
    ) -> "IntegerPLSIAccumulator":
        self.word_count = x[0]
        self.comp_count = x[1]
        self.doc_count = x[2]
        self.len_acc.from_value(x[3])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge the sufficient statistics of object instance with matching keys."""
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
        """
        Set the sufficient statistics of object instance to matching key values in arg.
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

    def acc_to_encoder(self) -> "IntegerPLSIDataEncoder":
        """Return an IntegerPLSIDataEncoder object."""
        len_encoder = self.len_acc.acc_to_encoder()
        return IntegerPLSIDataEncoder(len_encoder=len_encoder)


class IntegerPLSIAccumulatorFactory(TorchStatisticAccumulatorFactory):

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
        """
        IntegerPLSIAccumulatorFactory object for creating IntegerPLSIAccumulator
        objects.
        """
        self.len_factory = (
            len_factory if len_factory is not None else NullAccumulatorFactory()
        )
        self.keys = keys if keys is not None else (None, None, None)
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs

    def make(self, device: Optional[tn.device] = None) -> "IntegerPLSIAccumulator":
        """Returns IntegerPLSIAccumulator object."""
        return IntegerPLSIAccumulator(
            self.num_vals,
            self.num_states,
            self.num_docs,
            len_acc=self.len_factory.make(device=device),
            keys=self.keys,
            device=device,
        )


class IntegerPLSIEstimator(TorchParameterEstimator):

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
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[tn.Tensor]]
        ] = (None, None, None),
        keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (
            None,
            None,
            None,
        ),
    ) -> None:
        """
        IntegerPLSIEstimator for estimating integer PLSI distributions from aggregated.
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

    def accumulator_factory(self) -> "IntegerPLSIAccumulatorFactory":
        """Returns IntegerPLSIAccumulatorFactory object."""
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
        """
        Estimate IntegerPLSIDistribution from aggregated sufficient statistics in arg.
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
            doc_prob_vec = doc_count + adj_cnt * self.suff_stat[2]
            doc_prob_vec /= np.sum(doc_prob_vec)

        elif self.pseudo_count[2] is not None and self.suff_stat[2] is None:
            adj_cnt = self.pseudo_count[1] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt
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

    def __init__(
        self,
        len_encoder: Optional[TorchSequenceEncoder] = NullDataEncoder(),
        _device: Optional[str] = None,
    ):
        """
        IntegerPLSIDataEncoder object for encoding sequences of iid observations from a
        PLSI.
        """
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Returns a string representation of object instance."""
        return f"IntegerPLSIDataEncoder(len_dist={self.len_encoder!r})"

    def __eq__(self, other: object) -> bool:
        """Check if object is equivalent to instance of IntegerPLSIDataEncoder."""
        if isinstance(other, IntegerPLSIDataEncoder):
            return other.len_encoder == self.len_encoder
        return False

    def seq_encode(
        self,
        x: Sequence[Tuple[int, Sequence[Tuple[int, float]]]],
        device: Optional[tn.device] = None,
    ) -> "IntegerPLSITorchSequence":
        """
        Encode a sequence of iid PLSI observations for use with vectorized functions.
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
            xd.extend([d_id] * len(v))
            xi.extend([i] * len(v))
            xn.append(np.sum(c))
            xm.append(d_id)

        xv = vec.int_tensor(xv, device=device)
        xc = vec.tensor(xc, device=device)
        xd = vec.int_tensor(xd, device=device)
        xi = vec.int_tensor(xi, device=device)
        xn = vec.tensor(xn, device=device)
        xm = vec.int_tensor(xm, device=device)

        nn = self.len_encoder.seq_encode(xn, device=device)

        return IntegerPLSITorchSequence(
            data=(nn, (xv, xc, xd, xi, xn, xm)), device=device
        )


class IntegerPLSITorchSequence(TorchEncodedSequence):

    def __init__(
        self,
        data: Tuple[
            TorchEncodedSequence,
            Tuple[tn.tensor, tn.tensor, tn.tensor, tn.tensor, tn.tensor, tn.tensor],
        ],
        device: Optional[tn.device] = None,
    ):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:
        return f"IntegerPLSITorchSequence(device={repr(self.device)})"
