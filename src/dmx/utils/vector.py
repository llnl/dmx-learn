"""Vector utilities for estimation and evaluation in dmx-learn."""

from typing import (
    Iterable,
    List,
    Optional,
    Sequence,
    SupportsIndex,
    Tuple,
    Union,
    overload,
)

import numpy as np
import scipy.linalg
import scipy.special


@overload
def gammaln(x: np.ndarray) -> np.ndarray: ...


@overload
def gammaln(x: float) -> float: ...


def gammaln(x: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
    """Return the logarithm of the gamma function.

    Returns `log(abs(Gamma(x)))`.

    Args:
        x (Union[np.ndarray, float, int]): Scalar or array-like numeric input.

    Returns:
        float | np.ndarray: Scalar output for scalar input, otherwise a NumPy
        array.

    """
    if isinstance(x, float):
        return float(scipy.special.gammaln(x))

    return np.asarray(scipy.special.gammaln(x))


def sorted_merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merge two sorted arrays into one sorted array.

    Args:
        a (ndarray): Sorted numpy array.
        b (ndarray): Sorted numpy array.

    Returns:
        Sorted NumPy array containing the merged contents of `a` and `b`.

    """
    if len(a) < len(b):
        b, a = a, b
    c = np.empty(len(a) + len(b), dtype=a.dtype)
    b_indices = np.arange(len(b)) + np.searchsorted(a, b)
    a_indices = np.ones(len(c), dtype=bool)
    a_indices[b_indices] = False
    c[b_indices] = b
    c[a_indices] = a

    return c


def sorted_dict_merge_add(
    k_vec1: np.ndarray, c_vec1: np.ndarray, k_vec2: np.ndarray, c_vec2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge two sorted key/count arrays and add counts for shared keys.

    Returns the merge sorted keys and corresponding counts.

    Args:
        k_vec1 (ndarray): Numpy array of sorted dictionary keys.
        c_vec1 (ndarray): Numpy array of counts for keys in vector k_vec1.
        k_vec2 (ndarray): Numpy array of sorted dictionary keys.
        c_vec2 (ndarray): Numpy array of counts for keys in vector k_vec2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Merged sorted keys and their
        corresponding counts.
    """
    if len(k_vec2) == 0:
        return k_vec1, c_vec1

    if len(k_vec1) == 0:
        return k_vec2, c_vec2

    if len(k_vec1) < len(k_vec2):
        return sorted_dict_merge_add(
            k_vec1=k_vec2,
            c_vec1=c_vec2,
            k_vec2=k_vec1,
            c_vec2=c_vec1,
        )

    _, idx1, idx2 = np.intersect1d(
        k_vec1, k_vec2, assume_unique=True, return_indices=True
    )

    adj_cnt = c_vec1[idx1] + c_vec2[idx2]
    new_vals = np.delete(k_vec2, idx2)
    new_cnts = np.delete(c_vec2, idx2)
    new_idx = np.searchsorted(k_vec1, new_vals)
    rv_vals = np.insert(k_vec1, new_idx, new_vals)
    rv_cnts = np.insert(adj_cnt, new_idx, new_cnts)

    return rv_vals, rv_cnts


def make(
    x: Union[np.ndarray, Sequence[Union[int, float, str]], List[np.ndarray]]
) -> np.ndarray:
    """Convert the array x into a numpy array.

    Args:
        x (Union[np.ndarray, Sequence[Union[int, float, str]]]): Array-like
            object that can be converted to a NumPy array.

    Returns:
        Numpy array conversion of x.

    """
    return np.asarray(x)


def make_pdf(x: Union[np.ndarray, Sequence[float], List[np.ndarray]]) -> np.ndarray:
    """Normalize log-density values on the log scale.

    The returned array `rv` satisfies `np.exp(rv).sum() == 1.0`.

    Args:
        x (Union[np.ndarray, Sequence[float], List[np.ndarray]]): Array-like
            object with float-like values.

    Returns:
        np.ndarray: Log-scale normalized density values.
    """
    rv = np.asarray(x)
    n = len(rv)
    rv_max = rv.max()

    if rv_max == -np.inf:
        rv = zeros(n) - np.log(n)
    else:
        rv_sum = np.log(np.sum(np.exp(rv - rv_max))) + np.log(rv_max)
        rv /= rv_sum

    return rv


def zeros(n: Union[int, Iterable, Tuple[int]]) -> np.ndarray:
    """Return numpy array of shape n, with default dtype=float64.

    Args:
        n (Union[int, Iterable, Tuple[int]]): Shape tuple of ints, Iterable, or int.

    Returns:
        Return numpy array of shape n, with default dtype=float64.

    """
    return np.zeros(n)  # type: ignore[arg-type]


def mat_inv(
    x: Union[List[List[Union[float, int]]], List[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Computes the inverse of a square matrix x.

    Arg x data type is
    `Union[List[List[Union[float, int]]], List[np.ndarray], np.ndarray]`.

    Args:
        x (See above): List of lists, list of NumPy arrays, or a 2D square
            NumPy array.

    Returns:
        Inverse of x as 2-d numpy array.

    """
    return np.linalg.inv(x)


def dot(  # type: ignore[no-redef]
    x: Union[np.ndarray, Iterable, int, float],
    y: Union[np.ndarray, Iterable, int, float],
) -> Union[np.ndarray, float]:
    """Compute the dot product of `x` and `y`.

    Args:
        x: Numpy array, array-like, or scalar.
        y: Numpy array, array-like, or scalar.

    Returns:
        float | np.ndarray: Scalar if both inputs are 1D vectors, a 1D array
        if exactly one input is scalar-like, and a matrix otherwise.

    """
    return np.dot(x, y)  # type: ignore[arg-type,no-any-return]


def outer(
    x: Union[np.ndarray, Iterable, int, float],
    y: Union[np.ndarray, Iterable, int, float],
) -> np.ndarray:
    """Compute the outer product of two vectors.

    Args:
        x:  (M,) array_like
        y:  (N,) array_like

    Returns: (M, N) ndarray.

    """
    return np.outer(x, y)  # type: ignore[arg-type]


def diag(x: np.ndarray) -> np.ndarray:
    """Extract a diagonal or construct a diagonal array.

    Note:
        If `x` is 2D, return its diagonal. If `x` is 1D, return a 2D
        diagonal matrix with `x` on the diagonal.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Args:
        x: 2-D array, or 1-D array.

    Returns:
        The extracted diagonal or constructed diagonal array.

    """
    return np.diag(x)


def reshape(
    x: np.ndarray, sz: Union[SupportsIndex, Sequence[SupportsIndex]]
) -> np.ndarray:
    """Gives a new shape to an array without changing its data.

    Args:
        x (np.ndarray): Array to be reshaped.
        sz (Tuple[int,...]): Shape compatible with size of array x.

    Return:
        Reshaped array containing elements of x with shape = sz.

    """
    return np.reshape(x, sz)


def cholesky(x_mat: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
    """Compute the Cholesky decomposition of a matrix, to use in cho_solve.

    Returns a matrix containing the Cholesky decomposition of a Hermitian
    positive-definite matrix. The return value can be used directly as the
    first parameter to `cho_solve`.

    Args:
        x_mat (np.ndarray): Square np.ndarray of matrix to be decomposed.

    Returns:
        Optional[Tuple[np.ndarray, bool]]: Cholesky factorization when it can
        be computed, otherwise `None`.
    """
    try:
        rv = scipy.linalg.cho_factor(x_mat)
    except np.linalg.LinAlgError:
        rv = None

    return rv  # type: ignore[no-any-return]


def cho_solve(a_mat: Tuple[np.ndarray, bool], b: np.ndarray) -> np.ndarray:
    """Solve `a_mat x = b` using a Cholesky factorization.

    Args:
        a_mat (Tuple[np.ndarray, bool]): Cholesky factorization of `a`, as
            returned by `cho_factor`.
        b (np.ndarray): Right-hand side np.ndarray in a_mat*x = b.

    Returns:
        The solution to the system a_mat*x = b.

    """
    return scipy.linalg.cho_solve(a_mat, b)  # type: ignore[no-any-return]


def maximum(
    x: Union[float, int, Iterable, np.ndarray],
    y: Union[float, int, Iterable, np.ndarray],
    output: Optional[Union[float, int, np.ndarray]] = None,
) -> Union[float, int, np.ndarray]:
    """Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Args:
        x (array-like): Values to compare. If `x.shape != y.shape`, the
            inputs must be broadcastable to a common shape.
        y (array-like): Values to compare. If `x.shape != y.shape`, the
            inputs must be broadcastable to a common shape.
        output: Optional np.ndarray of float to output results to.

    Returns:
        ndarray or scalar: The element-wise maximum of `x` and `y`. This is a
        scalar if both inputs are scalars.
    """
    return np.maximum(x, y, output=output)  # type: ignore[arg-type,no-any-return]


def log_sum(x: np.ndarray) -> float:
    """Compute `log(sum(exp(x)))` for a 1D NumPy array.

    Args:
        x (ndarray): Numpy array on log-scale. E.g. x_i = log(y_i).

    Returns:
        Float value log(sum(exp(x)), or -np.inf if max(x) is -np.inf.
    """
    max_val = np.max(x)

    if max_val == -np.inf:
        return -np.inf

    rv = x - max_val
    np.exp(rv, out=rv)
    return np.log(rv.sum()) + max_val  # type: ignore[no-any-return]


def weighted_log_sum(x: np.ndarray, w: np.ndarray) -> float:
    """Compute a numerically stable weighted log-sum-of-exponentials.

    This uses `weights = exp(w)` on observation values `y = exp(x)`,
    returning `log(sum(exp(x) * exp(w)))`.

    Note: The weights are on the log-scale.

    Args:
        x (ndarray): Numpy array on log-scale. E.g. x_i = log(y_i).
        w (ndarray): Numpy array of log-weights for `y_i = exp(x_i)`.

    Returns:
        Float value log(sum(exp(x)*exp(w)), or -np.inf if any x or w are -np.inf.

    """
    y = x + w
    y[np.bitwise_or(np.isinf(x), np.isinf(w))] = -np.inf

    return log_sum(y)


def log_posterior(x: np.ndarray) -> np.ndarray:
    """Compute the log posterior for component log-likelihoods.

    If `x[j] = log(p(obs_i | theta_j))`, the returned array contains
    `log(p(theta_j | obs_i))` values up to normalization.

    Args:
        x (np.ndarray): Log-density values for each component or parameter
            value.

    Returns:
        np.ndarray: Log-posterior values for each component. Returns a
        uniform log distribution if `x` contains `nan` or `inf`.
    """
    max_val = x.max()

    if np.isinf(max_val) or np.isnan(max_val):
        return zeros(len(x)) - np.log(len(x))  # type: ignore[no-any-return]

    mass = np.log(np.exp(x - max_val).sum()) + max_val
    return x - mass  # type: ignore[no-any-return]


def posterior(  # pylint: disable=redefined-outer-name
    log_x: np.ndarray, out: Optional[np.ndarray] = None, log_sum: Optional[bool] = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Compute posterior probabilities from component log-likelihoods.

    If `log_x[j] = log(p(obs_i | theta_j))`, the returned array contains the
    normalized posterior probabilities over components.

    Args:
        log_x (ndarray): Log-density values for each component or parameter
            value.
        out (Optional[ndarray]): Optional numpy array to store returned value.
        log_sum (Optional[bool]): If true, also return the log normalizing
            constant.

    Returns:
        np.ndarray | Tuple[np.ndarray, float]: Posterior probabilities, and
        optionally the log normalizing constant.
    """
    if out is None:
        rv = np.zeros(len(log_x))
    else:
        rv = out

    max_val = log_x.max()
    rv_sum = 0.0

    if np.isinf(max_val) or np.isnan(max_val):
        rv.fill(1.0 / float(len(log_x)))

    else:
        np.subtract(log_x, max_val, out=rv)
        np.exp(rv, out=rv)
        rv_sum = rv.sum()
        rv /= rv_sum
        rv_sum = np.log(rv_sum) + max_val

    if log_sum:
        return rv, rv_sum

    return rv


def log_posterior_sum(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute log posterior values and their log normalizing constant.

    This returns the log posterior vector together with the log normalizing
    constant.

    Args:
        x (np.ndarray): Log-density values for each component or parameter
            value.

    Returns:
        Tuple[np.ndarray, float]: Log-posterior values and the log
        normalizing constant. Returns a uniform log distribution if `x`
        contains `nan` or `-np.inf`.

    """
    max_val = x.max()
    if np.isinf(max_val) or np.isnan(max_val):
        return zeros(len(x)) - np.log(len(x)), -np.inf

    mass = np.log(np.exp(x - max_val).sum()) + max_val
    return x - mass, mass


def weighted_log_posterior(x: np.ndarray, w: np.ndarray) -> List[float]:
    """Compute weighted log posterior values.

    The weights are assumed to be on the log scale.

    Args:
        x (ndarray): Numpy array of log-density values for each component or
            parameter value.
        w (ndarray): Numpy array of log weights for each parameter value.

    Returns:
        List[float]: Weighted log-posterior value for each component.

    """
    max_val = -np.inf

    rv = [0.0] * len(x)

    for i, x_i in enumerate(x):
        r = w[i] + x_i
        max_val = max(max_val, r)
        rv[i] = r

    e_sum = 0.0
    for i in range(len(x)):
        e_sum += np.exp(rv[i] - max_val)

    mass = np.log(e_sum) + max_val

    for i in range(len(x)):
        rv[i] -= mass

    return rv


def weighted_log_posterior_sum(
    x: np.ndarray, w: np.ndarray
) -> Tuple[List[float], float]:
    """Compute weighted log posterior values and their normalizer.

    The weights are assumed to be on the log scale.

    Args:
        x (np.ndarray): Log-density values for each component or parameter
            value.
        w (np.ndarray): List[float] or NumPy array of log weights for each
            parameter value.

    Returns:
        Tuple[List[float], float]: Weighted log-posterior values and the log
        normalizing constant.

    """
    max_val = -np.inf

    rv = [0.0] * len(x)

    for i, x_i in enumerate(x):
        r = w[i] + x_i
        max_val = max(max_val, r)
        rv[i] = r

    e_sum = 0.0
    for i in range(len(x)):
        e_sum += np.exp(rv[i] - max_val)

    mass = np.log(e_sum) + max_val

    for i in range(len(x)):
        rv[i] -= mass

    return rv, mass


def matrix_log_posteriors(
    x: np.ndarray, u_mat: np.ndarray, u: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute row posteriors, outer posteriors, and log-likelihood.

    This function calculates posterior probabilities for rows and columns
    based on input matrices and vectors. It also computes the log-likelihood
    of the data given the model.

    Args:
        x (np.ndarray): A 2D array with shape `(m, z)`, where `m` is the number
            of rows and `z` is the number of columns.
        u_mat (np.ndarray): A 2D array with shape `(h, w)` representing the matrix
            of prior probabilities or weights.
        u (np.ndarray): A 1D array with shape `(h,)` representing additional
            prior probabilities for each row.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - `row_posteriors` (np.ndarray): A 3D array with shape `(h, w, z)`
              containing posterior probabilities for each row and column.
            - `outer_posterior` (np.ndarray): A 1D array with shape `(h,)` containing
              normalized posterior probabilities for each row.
            - `ll` (float): The log-likelihood of the data given the model.

    Raises:
        ValueError: If the shapes of `x`, `u_mat`, or `u` are incompatible.

    Examples:
        >>> x = np.array([[1, 2], [3, 4]])
        >>> u_mat = np.array([[0.5, 0.2], [0.1, 0.7]])
        >>> u = np.array([0.3, 0.4])
        >>> row_posteriors, outer_posterior, ll = matrix_log_posteriors(x, u_mat, u)
        >>> row_posteriors.shape
        (2, 2, 2)
        >>> outer_posterior.shape
        (2,)
        >>> print(ll)
        -1.234
    """
    # Extract dimensions
    h = u_mat.shape[0]
    w = u_mat.shape[1]
    z = x.shape[1]

    # Initialize arrays
    row_posteriors = np.zeros((h, w, z))
    outer_posterior = np.zeros(h)
    outer_max = -np.inf

    # Iterate over rows
    for i in range(h):
        row_sum = 0  # Initialize row sum

        # Iterate over columns
        for j in range(z):
            temp = u_mat[i, :] + x[:, j]
            inner_max = temp.max()  # Find max value for numerical stability
            temp = np.exp(temp - inner_max)  # Normalize with stability
            inner_sum = temp.sum()  # Sum normalized values

            # Compute posterior probabilities for the current row and column
            row_posteriors[i, :, j] = temp / inner_sum
            row_sum += np.log(inner_sum) + inner_max

        # Add prior probability for the row
        row_sum += u[i]
        outer_max = max(outer_max, row_sum)
        outer_posterior[i] = row_sum

    # Normalize outer posterior probabilities
    outer_posterior = np.exp(outer_posterior - outer_max)
    outer_sum = outer_posterior.sum()
    outer_posterior /= outer_sum

    # Compute log-likelihood
    ll = np.log(outer_sum) + outer_max

    return row_posteriors, outer_posterior, ll


def row_choice(p_mat: np.ndarray, rng: Optional[np.random.RandomState]) -> np.ndarray:
    """Vectorized choice using row-wise sampling weights.

    Choice is called on the range `[0, S)`, where each row of `p_mat`
    contains sampling weights.

    Args:
        p_mat (np.ndarray): N x S numpy array.
        rng (Optional[np.random.RandomState]): RandomState for sampling.

    Returns:
        N dim numpy array of ints.

    """
    N, m = p_mat.shape
    u = rng.rand(N) if rng is not None else np.random.rand(N)
    rv = np.zeros(N, dtype=int)

    bins = np.hstack((np.zeros((N, 1)), np.cumsum(p_mat, axis=1)))
    idx = np.arange(0, N)

    l = np.zeros(N, dtype=int)
    r = np.zeros(N, dtype=int)
    r.fill(m)

    mid = (r - l) // 2

    l_cond = u >= bins[idx, mid]
    r_cond = u < bins[idx, mid + 1]

    bin_cond = np.bitwise_and(l_cond, r_cond)
    in_bin = np.flatnonzero(bin_cond)

    if np.any(bin_cond):
        rv[idx[in_bin]] = mid[in_bin]
        idx = np.delete(idx, in_bin)
        l = l[idx]
        r = r[idx]
        r_cond = r_cond[idx]
        l_cond = l_cond[idx]
        mid = mid[idx]

    if np.any(r_cond):
        r[r_cond] = mid[r_cond]
        l[r_cond] = 0

    if np.any(~r_cond):
        l[~r_cond] = mid[~r_cond]
        r[~r_cond] = m

    iterate_cond = len(idx) > 0

    while iterate_cond:

        mid = (r - l) // 2 + l

        l_cond = u[idx] >= bins[idx, mid]
        r_cond = u[idx] < bins[idx, mid + 1]

        in_bin = np.bitwise_and(l_cond, r_cond)
        in_bin_idx = np.flatnonzero(in_bin)

        if np.any(in_bin):
            rv[idx[in_bin]] = mid[in_bin_idx]
            idx = np.delete(idx, in_bin_idx)

            if len(idx) > 0:
                not_in_bin = ~in_bin
                not_in_bin = np.flatnonzero(~in_bin)
                l = l[not_in_bin]
                r = r[not_in_bin]
                r_cond = r_cond[not_in_bin]
                l_cond = l_cond[not_in_bin]
                mid = mid[not_in_bin]

        if np.any(r_cond):
            r[r_cond] = mid[r_cond]
            l[r_cond] = 0

        if np.any(~r_cond):
            r[~r_cond] = mid[~r_cond]
            l[~r_cond] = m

        if len(idx) == 0 or np.all(l >= r):
            iterate_cond = False

    return rv
