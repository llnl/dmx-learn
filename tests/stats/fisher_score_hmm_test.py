"""Fisher scores of hidden Markov models.

Per-sequence forward-backward statistics are checked against brute-force
enumeration of hidden state paths, and the raw score against its formula.
The whitened score is checked for model-only dependence, unsupported parts,
empty sequences, and nesting in a composite.
"""

import itertools
import math
import warnings
from typing import Any, List, Tuple

import numpy as np
import pytest

from dmx.stats import (
    CompositeDistribution,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    IntegerCategoricalDistribution,
    IntegerHiddenMarkovModelDistribution,
    PoissonDistribution,
)
from dmx.stats._fisher_latent import HmmPosteriors, hmm_raw_score
from dmx.stats.hidden_markov import hmm_forward_backward
from dmx.stats.pdist import ProbabilityDistribution
from dmx.utils.fisher import FisherScoreOptions, has_fisher_score

W = np.asarray([0.6, 0.4])
A = np.asarray([[0.8, 0.2], [0.3, 0.7]])
SEQUENCES = [[0.2], [2.5, -0.3], [1.0, 3.1, 2.2]]


class _UnscoredGaussian(GaussianDistribution):
    """Gaussian emission with the base-class Fisher methods restored."""

    seq_fisher_score = ProbabilityDistribution.seq_fisher_score
    fisher_dim = ProbabilityDistribution.fisher_dim


def _hmm(use_numba: bool = False, second: Any = None) -> HiddenMarkovModelDistribution:
    """Return a two-state Gaussian HMM with lengths 1..3."""
    second = GaussianDistribution(2.0, 1.0) if second is None else second
    return HiddenMarkovModelDistribution(
        [GaussianDistribution(0.0, 1.0), second],
        W,
        A,
        len_dist=IntegerCategoricalDistribution(1, [0.4, 0.3, 0.3]),
        use_numba=use_numba,
    )


def _encode(dist: Any, data: Any) -> Any:
    """Encode ``data`` for ``dist``."""
    return dist.dist_to_encoder().seq_encode(data)


def _brute_force(
    log_obs: np.ndarray, w: np.ndarray, a: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact posterior expectations for one sequence by enumerating paths."""
    length, k = log_obs.shape
    initial, xi, gamma = np.zeros(k), np.zeros((k, k)), np.zeros((length, k))
    total = 0.0
    for path in itertools.product(range(k), repeat=length):
        log_p = math.log(w[path[0]]) + log_obs[0, path[0]]
        for t in range(1, length):
            log_p += math.log(a[path[t - 1], path[t]]) + log_obs[t, path[t]]
        p = math.exp(log_p)
        total += p
        initial[path[0]] += p
        for t in range(length):
            gamma[t, path[t]] += p
        for t in range(1, length):
            xi[path[t - 1], path[t]] += p
    return initial / total, xi / total, gamma / total


def _gaussian_log_obs(sequence: List[float]) -> np.ndarray:
    """Return per-state Gaussian log-likelihoods of a sequence."""
    x = np.asarray(sequence)
    return np.column_stack(
        [-0.5 * (x - mu) ** 2 - 0.5 * math.log(2.0 * math.pi) for mu in (0.0, 2.0)]
    )


@pytest.mark.parametrize("use_numba", [False, True], ids=["banded", "numba"])
def test_posteriors_match_brute_force(use_numba: bool) -> None:
    dist = _hmm(use_numba)
    # pylint: disable-next=protected-access
    posteriors, _, _ = dist._fisher_posteriors(_encode(dist, SEQUENCES))
    np.testing.assert_array_equal(posteriors.lengths, [1, 2, 3])
    for n, sequence in enumerate(SEQUENCES):
        initial, xi, gamma = _brute_force(_gaussian_log_obs(sequence), W, A)
        np.testing.assert_allclose(posteriors.initial[n], initial, rtol=1e-9)
        np.testing.assert_allclose(posteriors.transitions[n], xi, rtol=1e-9, atol=1e-12)
        rows = np.flatnonzero(posteriors.owner == n)
        np.testing.assert_allclose(posteriors.gamma[rows], gamma, rtol=1e-9)


def test_raw_score_matches_formula() -> None:
    dist = _hmm()
    options = FisherScoreOptions()
    encoded = _encode(dist, SEQUENCES)
    # pylint: disable-next=protected-access
    posteriors, enc_data, len_enc = dist._fisher_posteriors(encoded)
    emissions = [topic.seq_fisher_score(enc_data) for topic in dist.topics]
    length_block = dist.len_dist.seq_fisher_score(len_enc)
    raw = hmm_raw_score(posteriors, W, A, emissions, length_block, options)
    assert raw.shape == (3, dist.fisher_dim())
    for n, sequence in enumerate(SEQUENCES):
        initial, xi, gamma = _brute_force(_gaussian_log_obs(sequence), W, A)
        length = len(sequence)
        trans = (xi - xi.sum(axis=1, keepdims=True) * A).ravel()
        if length > 1:
            trans = trans / math.sqrt(length - 1)
        blocks = [initial - W, trans]
        for state, topic in enumerate(dist.topics):
            scores = topic.seq_fisher_score(_encode(topic, sequence))
            blocks.append(gamma[:, state] @ scores / math.sqrt(length))
        blocks.append(length_block[n])
        np.testing.assert_allclose(
            raw[n], np.concatenate(blocks), rtol=1e-8, atol=1e-12
        )


def test_marginal_scaling_leaves_blocks_unscaled() -> None:
    dist = _hmm()
    # pylint: disable-next=protected-access
    posteriors, enc_data, _ = dist._fisher_posteriors(_encode(dist, SEQUENCES))
    emissions = [topic.seq_fisher_score(enc_data) for topic in dist.topics]
    conditional = hmm_raw_score(posteriors, W, A, emissions, None, FisherScoreOptions())
    marginal = hmm_raw_score(
        posteriors,
        W,
        A,
        emissions,
        None,
        FisherScoreOptions(sequence_scaling="marginal"),
    )
    lengths = np.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(marginal[:, :2], conditional[:, :2])
    np.testing.assert_allclose(
        marginal[:, 2:6],
        conditional[:, 2:6] * np.sqrt(np.maximum(lengths - 1, 1))[:, None],
    )
    np.testing.assert_allclose(
        marginal[:, 6:], conditional[:, 6:] * np.sqrt(lengths)[:, None]
    )


def test_empty_sequences_score_zero_parts() -> None:
    posteriors = HmmPosteriors(
        *hmm_forward_backward(np.zeros((0, 2)), np.asarray([0, 0]), W, A),
        np.zeros(0, dtype=int),
        np.asarray([0, 0]),
    )
    raw = hmm_raw_score(
        posteriors,
        W,
        A,
        [np.zeros((0, 2)), np.zeros((0, 2))],
        None,
        FisherScoreOptions(),
    )
    np.testing.assert_array_equal(raw, np.zeros((2, 2 + 4 + 4)))


@pytest.mark.parametrize("use_numba", [False, True], ids=["banded", "numba"])
def test_whitened_score_depends_only_on_the_model(use_numba: bool) -> None:
    dist = _hmm(use_numba)
    data: Any = dist.sampler(seed=4).sample(200)
    full = dist.seq_fisher_score(_encode(dist, data))
    part = dist.seq_fisher_score(_encode(dist, data[:5]))
    np.testing.assert_allclose(full[:5], part, rtol=1e-10, atol=1e-12)


def test_both_encodings_give_the_same_score() -> None:
    banded = _hmm(use_numba=False)
    numba = _hmm(use_numba=True)
    np.testing.assert_allclose(
        banded.seq_fisher_score(_encode(banded, SEQUENCES)),
        numba.seq_fisher_score(_encode(numba, SEQUENCES)),
        rtol=1e-9,
        atol=1e-12,
    )


def test_field_slices_and_composite_nesting() -> None:
    dist = _hmm()
    assert dist.fisher_field_slices() == {
        "initial": slice(0, 2),
        "transitions": slice(2, 6),
        "emission_0": slice(6, 8),
        "emission_1": slice(8, 10),
        "length": slice(10, 13),
    }
    composite = CompositeDistribution((PoissonDistribution(1.0, name="n"), dist))
    slices = composite.fisher_field_slices()
    assert slices["n"] == slice(0, 1)
    assert slices["field_1.transitions"] == slice(3, 7)


def test_topic_mixture_hmm_has_no_score() -> None:
    dist = HiddenMarkovModelDistribution(
        [GaussianDistribution(0.0, 1.0), GaussianDistribution(2.0, 1.0)],
        W,
        A,
        taus=[[0.5, 0.5], [0.5, 0.5]],
    )
    assert not has_fisher_score(dist)
    with pytest.raises(NotImplementedError, match="taus"):
        dist.fisher_dim()


def test_unsupported_emission_raises_or_is_skipped() -> None:
    dist = _hmm(second=_UnscoredGaussian(2.0, 1.0))
    assert not has_fisher_score(dist)
    options = FisherScoreOptions(on_missing="posterior_only")
    assert dist.fisher_dim(options) == 2 + 4 + 2 + 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = dist.seq_fisher_score(_encode(dist, SEQUENCES), options)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == 1 and "_UnscoredGaussian" in messages[0]
    assert scores.shape == (3, 11)


def test_integer_hmm_posteriors_match_brute_force() -> None:
    pmat = np.asarray([[0.6, 0.1], [0.3, 0.2], [0.1, 0.7]])
    dist = IntegerHiddenMarkovModelDistribution(
        pmat, W, A, len_dist=IntegerCategoricalDistribution(1, [0.4, 0.3, 0.3])
    )
    sequences = [[0, 2, 2], [1], [2, 0]]
    words = np.concatenate(sequences)
    initial, xi, gamma = hmm_forward_backward(
        np.log(pmat[words, :]), np.asarray([3, 1, 2]), W, A
    )
    offset = 0
    for n, sequence in enumerate(sequences):
        exact = _brute_force(np.log(pmat[sequence, :]), W, A)
        np.testing.assert_allclose(initial[n], exact[0], rtol=1e-9)
        np.testing.assert_allclose(xi[n], exact[1], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            gamma[offset : offset + len(sequence)], exact[2], rtol=1e-9
        )
        offset += len(sequence)
    assert dist.fisher_dim() == 2 + 4 + 2 * 3 + 3
    scores = dist.seq_fisher_score(_encode(dist, sequences))
    assert scores.shape == (3, 15)
    assert np.all(np.isfinite(scores))


def test_hmm_without_length_distribution_cannot_be_whitened() -> None:
    # Whitening draws sequences from the model, which needs a length
    # distribution (or terminal values).
    dist = HiddenMarkovModelDistribution(
        [GaussianDistribution(0.0, 1.0), GaussianDistribution(2.0, 1.0)], W, A
    )
    with pytest.raises(RuntimeError, match="whitened with draws from the model"):
        dist.seq_fisher_score(_encode(dist, SEQUENCES))


@pytest.mark.parametrize(
    "dist",
    [_hmm(), IntegerHiddenMarkovModelDistribution(np.eye(2) * 0.5 + 0.25, W, A)],
    ids=lambda v: type(v).__name__,
)
def test_wrong_encoding_raises(dist: Any) -> None:
    encoded: Any = PoissonDistribution(1.0).dist_to_encoder().seq_encode([1])
    with pytest.raises(TypeError, match="seq_fisher_score"):
        dist.seq_fisher_score(encoded)
