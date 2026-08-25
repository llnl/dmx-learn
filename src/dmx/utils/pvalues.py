"""P-value utilities for approximating composite binomial ranks."""

import itertools
from importlib import import_module
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

SPECIAL = import_module("scipy.special")


def binomial_rank(
    log_p_vec: Union[Sequence[float], np.ndarray],
    log_p1_vec: Optional[Union[Sequence[float], np.ndarray]] = None,
    count_vec: Optional[Union[Sequence[float], np.ndarray]] = None,
    ll_eps: float = 1.0e-4,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Approximate a composite-binomial log-density histogram.

    Each input position defines a binomial count and success probability in
    log space. Individual discrete log-density histograms are placed on a
    shared grid and convolved. Entries with zero counts or deterministic
    probabilities are skipped. At least one nondegenerate entry is required.

    Args:
        log_p_vec: One-dimensional log success probabilities of shape ``(m,)``.
        log_p1_vec: Optional log failure probabilities with shape ``(m,)``.
            When absent, they are computed with ``log1p(-exp(log_p_vec))``.
        count_vec: Optional nonnegative, integer-valued binomial counts with
            shape ``(m,)``; defaults to one per entry.
        ll_eps: Maximum grid-alignment remainder used while halving automatic
            bin spacing.
        max_len: Optional approximate upper bound used to choose bin spacing.

    Returns:
        Grid log densities and their normalized probabilities, both of shape
        ``(n_bins,)``, plus ``(grid_origin, grid_spacing, total_count)``.

    Raises:
        ValueError: If no nondegenerate binomial entry remains or array shapes
            cannot be combined.
    """
    entries: List[Tuple[np.ndarray, np.ndarray, float]] = []
    log_p_arr = np.asarray(log_p_vec, dtype=float)

    if log_p1_vec is None:
        log_p1_arr = np.log1p(-np.exp(log_p_arr))
    else:
        log_p1_arr = np.asarray(log_p1_vec, dtype=float)

    if count_vec is None:
        count_arr = np.ones(len(log_p_arr), dtype=float)
    else:
        count_arr = np.asarray(count_vec, dtype=float)

    # Compute binomial log-densities and probabilities
    for log_p, log_p1, n in zip(log_p_arr, log_p1_arr, count_arr):
        if n == 0 or log_p == -np.inf or log_p1 == -np.inf:
            continue
        nn = np.arange(0, n + 1)
        llv = log_p * nn + log_p1 * (n - nn)
        ell = (
            SPECIAL.gammaln(n + 1)
            - SPECIAL.gammaln(nn + 1)
            - SPECIAL.gammaln(n - nn + 1)
        )
        ell = np.exp(ell - ell.max())
        ell /= np.sum(ell)
        llv = llv[ell > 0]
        ell = ell[ell > 0]

        entries.append((llv, ell, float(n)))

    # Find parameters for a common fixed-space grid [ll0, ll0 + dll, ll0 + 2*dll, ...]
    min_vec = np.asarray([entry[0].min() for entry in entries])
    llv_vec = np.concatenate([entry[0] - entry[0].min() for entry in entries])
    llv_vec = np.sort(np.unique(llv_vec))

    if max_len is not None:
        mll = np.sum([entry[0].max() - entry[0].min() for entry in entries])
        dll = mll / max_len
    else:
        dll = np.diff(llv_vec).min()
        while np.abs(llv_vec - np.floor(llv_vec / dll) * dll).max() > ll_eps:
            dll /= 2

    # Adjust log-density histograms to a common grid and convolve
    temp_idx = np.floor((entries[0][0] - entries[0][0].min()) / dll).astype(int)
    acc_prob = np.asarray(np.bincount(temp_idx, weights=entries[0][1]), dtype=float)
    acc_count = entries[0][2]

    for next_llv, next_ell, next_count in entries[1:]:
        next_idx = np.floor((next_llv - next_llv.min()) / dll).astype(int)

        next_prob = np.asarray(np.bincount(next_idx, weights=next_ell), dtype=float)
        max_count = max(next_count, acc_count)
        acc_weight = np.power(2.0, acc_count - max_count)
        next_weight = np.power(2.0, next_count - max_count)

        acc_prob = np.convolve(acc_prob * acc_weight, next_prob * next_weight)
        acc_prob /= np.sum(acc_prob)
        acc_count += next_count

    ll0 = min_vec.sum()
    acc_ll = ll0 + np.arange(len(acc_prob)) * dll
    return acc_ll, acc_prob, (ll0, dll, acc_count)


if __name__ == "__main__":

    pvec = np.asarray([0.3, 0.8, 0.4])
    pvec = np.log(pvec)
    nvec = np.log1p(-np.exp(pvec))
    cvec = np.asarray([2, 3, 3])

    pvec_long = np.concatenate([[u] * n for u, n in zip(pvec, cvec)])
    nvec_long = np.concatenate([[u] * n for u, n in zip(nvec, cvec)])

    test = np.asarray([1, 0, 1, 1, 0, 1, 0, 1])
    ll = np.where(test == 1, pvec_long, nvec_long).sum()

    rank_ll, rank_prob, (rank_ll0, rank_dll, rank_count) = binomial_rank(
        pvec, count_vec=cvec, max_len=100000
    )
    left = rank_prob[(int((ll - rank_ll0) / rank_dll) - 1) :].sum() * np.power(
        2, rank_count
    )
    mid = rank_prob[int((ll - rank_ll0) / rank_dll) :].sum() * np.power(2, rank_count)
    right = rank_prob[(int((ll - rank_ll0) / rank_dll) + 1) :].sum() * np.power(
        2, rank_count
    )
    print(f"Approximate rank: {mid:f} ( Somewhere in [{right:f}, {left:f}] )")

    # Verify this
    temp = np.asarray(
        [
            np.where([u == 1 for u in x], pvec_long, nvec_long).sum()
            for x in itertools.product([0, 1], repeat=len(pvec_long))
        ]
    )
    print("True rank:" + str((temp >= ll).sum()))
