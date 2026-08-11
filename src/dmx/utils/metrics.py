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
    """Classifies iid observations from model predictions.

    Returns
    Args:
        data (Sequence[T]): Sequence of iid observations for classification.
        model (SequenceEncodableProbabilityDistribution): Distribution for
            classification.
        labels (Optional[List[T]]): List of labels for the data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

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
    """Create ROC curve.

    Args:
        pos_x (Union[List[float], np.ndarray]): Probs for positive classifications.
        neg_x (Union[List[float], np.ndarray]): Probs for negative classifications.

    Returns:
        (total positive rate, 1 - false positive rate)

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
    """
    Computes the ROC (Receiver Operating Characteristic) curve percentiles.

    This function calculates the false alarm rate (FA) and detection probability (PD)
    at specified percentile points based on the ROC curve.

    Args:
        pos_x (Union[List[float], np.ndarray]): Scores or probabilities for
            positive samples.
        neg_x (Union[List[float], np.ndarray]): Scores or probabilities for
            negative samples.
        perc_points (Union[List[float], np.ndarray]): Percentile points at
            which to compute the ROC values.

    Returns:
        np.ndarray: A 2D array where each row contains [FA, PD] values
            corresponding to the given percentiles.
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
    """
    Computes the ranking depth for a set of entries based on a comparison function.

    This function calculates the ranks of matching entries in a list of scored items
    for each input entry. If `k` is specified, only the top `k` ranks are returned.

    Args:
        x (List[Tuple[Any, List[Tuple[Any, float]]]]): A list of entries where
            each entry is a tuple. The first element of the tuple is the
            target value, and the second element is a list of tuples
            containing candidate values and their associated scores.
        k (Optional[int], optional): The number of top ranks to return for
            each entry. If `None`, all ranks are returned. Defaults to `None`.
        comp_func (Callable[[Any, Any], bool], optional): A comparison
            function that determines whether a candidate value matches the
            target value. Defaults to `lambda a, b: a == b`.

    Returns:
        Union[np.ndarray, List[np.ndarray]]:
            - If `k` is specified, returns a 2D NumPy array of shape
              `(len(x), k)` where each row contains the ranks of the top `k`
              matching entries for the corresponding input entry.
            - If `k` is `None`, returns a list of 1D NumPy arrays, where each
              array contains all ranks of matching entries for the
              corresponding input entry.
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
