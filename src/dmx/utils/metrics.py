"""Functions for classification evaluation.

Create ROC curves and search depth rankings.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dmx.stats.pdist import SequenceEncodableProbabilityDistribution


def classify(
    data: Sequence[Tuple[Any, Any]],
    model: SequenceEncodableProbabilityDistribution,
    labels: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Any, np.ndarray]]:
    """Rank true labels using conditional model log densities.

    For every observation value, the model scores a copy paired with each
    candidate label. Scores are normalized with a softmax. Labels must be
    sortable, and an explicit ``labels`` sequence must include every observed
    true label.

    Args:
        data: ``(true_label, value)`` pairs for independent observations.
        model: Joint or conditional distribution that accepts the same pairs.
        labels: Candidate labels. By default, use the sorted unique true
            labels.

    Returns:
        Four objects: zero-based true-label ranks with shape ``(n_samples,)``;
        true-label probabilities with shape ``(n_samples,)``; true labels with
        shape ``(n_samples,)``; and a mapping from each candidate label to its
        probability vector of shape ``(n_samples,)``.
    """
    cnt = len(data)
    data_labels = [label for label, _ in data]

    encoder = model.dist_to_encoder()

    if labels is None:
        label_values = sorted(set(data_labels))
    else:
        label_values = list(labels)

    class_probs = np.zeros((len(data), len(label_values)))
    u_labels, true_labels = np.unique(data_labels, return_inverse=True)

    other_labs = sorted(set(label_values).difference(list(u_labels)))
    u_label_map = dict(
        zip(list(u_labels) + other_labs, range(len(u_labels) + len(other_labs)))
    )

    for label in label_values:
        idx = u_label_map[label]
        loc_data = [(label, value) for _, value in data]
        class_probs[:, idx] = model.seq_log_density(encoder.seq_encode(loc_data))

    max_ll = np.max(class_probs, axis=1, keepdims=True)
    class_probs -= max_ll
    np.exp(class_probs, out=class_probs)
    class_probs /= class_probs.sum(axis=1, keepdims=True)

    class_prob = class_probs[np.arange(cnt), true_labels]
    class_diff = class_probs - class_prob[:, None]
    class_rank = (class_diff >= 0).sum(axis=1) - 1
    data_labels_arr = np.asarray(data_labels)
    class_prob_by_label = {
        label: class_probs[:, u_label_map[label]] for label in label_values
    }

    return class_rank, class_prob, data_labels_arr, class_prob_by_label


def roc_curve(
    pos_x: Union[List[float], np.ndarray], neg_x: Union[List[float], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an empirical ROC curve from positive and negative scores.

    Scores are sorted in descending order; each returned position corresponds
    to accepting one additional sample at that threshold.

    Args:
        pos_x: One-dimensional scores for positive examples.
        neg_x: One-dimensional scores for negative examples.

    Returns:
        Cumulative true-positive and false-positive rates, each with shape
        ``(len(pos_x) + len(neg_x),)``.
    """
    res = np.zeros((len(pos_x) + len(neg_x), 2))
    res[: len(pos_x), 0] = np.asarray(pos_x)
    res[: len(pos_x), 1] = 1
    res[len(pos_x) :, 0] = np.asarray(neg_x)
    res[len(pos_x) :, 1] = 0

    sidx = np.argsort(-res[:, 0])
    res = res[sidx, :]

    pd = np.cumsum(res[:, 1])
    fa = np.cumsum(1 - res[:, 1])

    pd /= float(len(pos_x))
    fa /= float(len(neg_x))

    return pd, fa


def roc_percentiles(
    pos_x: Union[List[float], np.ndarray],
    neg_x: Union[List[float], np.ndarray],
    perc_points: Union[List[float], np.ndarray],
) -> np.ndarray:
    """Sample an empirical ROC curve at target true-positive rates.

    For each target, the greatest attainable true-positive rate not exceeding
    that target is selected. Targets below the first attainable rate are
    omitted rather than represented by a placeholder row.

    Args:
        pos_x: One-dimensional scores for positive examples.
        neg_x: One-dimensional scores for negative examples.
        perc_points: Target true-positive rates, normally in ``[0, 1]``.

    Returns:
        Array with up to ``len(perc_points)`` rows and two columns containing
        ``[false_positive_rate, true_positive_rate]``. If no target is
        attainable, the current implementation returns an empty array with
        shape ``(0,)``.
    """
    pd, fa = roc_curve(pos_x, neg_x)
    rv: List[List[float]] = []

    for _, perc_point in enumerate(perc_points):

        points = pd <= perc_point

        if np.sum(points) == 0:
            continue

        y = np.max(pd[points])
        x = np.max(fa[pd == y])
        rv.append([x, y])

    return np.asarray(rv)


def ranking_depth(
    x: List[Tuple[Any, List[Tuple[Any, float]]]],
    k: Optional[int] = None,
    comp_func: Callable[[Any, Any], bool] = lambda a, b: a == b,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Find zero-based ranks of matching candidates in scored lists.

    Candidates for each target are sorted by descending score before matching.

    Args:
        x: Pairs of a target value and a list of ``(candidate, score)`` pairs.
        k: Number of matching ranks to retain. ``None`` retains all matches.
        comp_func: Predicate applied as ``comp_func(target, candidate)``.

    Returns:
        When ``k`` is given, a float array of shape ``(len(x), k)`` padded
        with NaN. Otherwise, a list of variable-length integer rank arrays.
    """
    all_ranks: List[np.ndarray] = []
    for entry in x:

        scores = np.asarray([u[1] for u in entry[1]])
        matches = np.asarray([comp_func(entry[0], u[0]) for u in entry[1]])

        sidx = np.argsort(-scores)

        matches = matches[sidx]
        scores = scores[sidx]

        ranks = np.arange(len(sidx))[matches]

        all_ranks.append(ranks)

    if k is None:
        return all_ranks

    retval = np.zeros((len(x), k))
    retval.fill(np.nan)
    for idx, ranks in enumerate(all_ranks):
        sz = min(k, len(ranks))
        retval[idx, :sz] = ranks[:sz]
    return retval
