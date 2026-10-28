"""Score fitted models with held-out likelihood and information criteria.

These helpers compare candidate models fitted to the same data. They accept
:mod:`dmx.stats` and :mod:`dmx.bstats` distributions, encode local data with the
model's own encoder, and return frozen dataclasses. Nothing is printed.

Information criteria use the deviance scale, so lower values are better:

- ``bic = -2 * LL + p * ln(n)``
- ``aic = -2 * LL + 2 * p``
- ``icl = bic + 2 * entropy``

Here ``LL`` is the total log-likelihood of the data under the model, ``p`` is
``model.num_params()``, ``n`` is the number of observations, and ``entropy`` is
the posterior entropy of the latent assignments,
``-sum_i sum_k tau_ik * ln(tau_ik)``, from ``model.seq_posterior``. Bayesian
(:mod:`dmx.bstats`) models are scored on their plug-in log-likelihood.

Pass ``num_params`` to override the parameter count, for example
``dpm.effective_num_params(min_weight)`` for a truncated Dirichlet-process
mixture.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.random import RandomState

from dmx.arithmetic import maxrandint
from dmx.bstats.bestimation import optimize as bstats_optimize
from dmx.bstats.pdist import ParameterEstimator as BayesianParameterEstimator
from dmx.bstats.pdist import ProbabilityDistribution as BayesianDistribution
from dmx.utils.estimation import k_fold_split_index
from dmx.utils.estimation import optimize as stats_optimize

__all__ = [
    "LogLikelihoodScore",
    "KFoldScore",
    "InformationCriterion",
    "heldout_log_likelihood",
    "kfold_log_likelihood",
    "bic",
    "aic",
    "icl",
]


@dataclass(frozen=True)
class LogLikelihoodScore:
    """Total log-likelihood of a model on a data set.

    Attributes:
        log_likelihood: Sum of per-observation log densities.
        num_obs: Number of scored observations.
    """

    log_likelihood: float
    num_obs: int

    @property
    def mean_log_likelihood(self) -> float:
        """Return the log-likelihood per observation."""
        return self.log_likelihood / self.num_obs


@dataclass(frozen=True)
class KFoldScore:
    """Held-out log-likelihood from k-fold cross-validation.

    Each fold is scored by a model fitted on the other ``k - 1`` folds.

    Attributes:
        fold_log_likelihoods: Held-out log-likelihood of each fold.
        fold_sizes: Number of observations in each fold.
        seed: Seed that determined the folds and the per-fold fits.
    """

    fold_log_likelihoods: tuple[float, ...]
    fold_sizes: tuple[int, ...]
    seed: int

    @property
    def k(self) -> int:
        """Return the number of folds."""
        return len(self.fold_sizes)

    @property
    def log_likelihood(self) -> float:
        """Return the held-out log-likelihood summed over folds."""
        return float(sum(self.fold_log_likelihoods))

    @property
    def num_obs(self) -> int:
        """Return the number of held-out observations across folds."""
        return int(sum(self.fold_sizes))

    @property
    def mean_log_likelihood(self) -> float:
        """Return the held-out log-likelihood per observation."""
        return self.log_likelihood / self.num_obs


@dataclass(frozen=True)
class InformationCriterion:
    """Value and ingredients of an information criterion.

    Attributes:
        name: ``"bic"``, ``"aic"``, or ``"icl"``.
        value: Criterion on the deviance scale; lower is better.
        log_likelihood: Total log-likelihood of the data under the model.
        num_params: Free-parameter count used in the penalty.
        num_obs: Number of observations.
        penalty: Total added to ``-2 * log_likelihood``; for ICL this includes
            ``2 * entropy``.
        entropy: Posterior assignment entropy (nonzero only for ICL).
    """

    name: str
    value: float
    log_likelihood: float
    num_params: int
    num_obs: int
    penalty: float
    entropy: float = 0.0


def _encode(model: Any, data: Sequence[Any]) -> Any:
    """Encode local observations with the model's own sequence encoder."""
    if isinstance(model, BayesianDistribution):
        return model.seq_encode(data)
    return model.dist_to_encoder().seq_encode(data)


def _check_data(data: Sequence[Any]) -> int:
    """Return the number of observations, rejecting empty data."""
    n = len(data)
    if n == 0:
        raise ValueError("Scoring requires at least one observation.")
    return n


def _log_likelihood(model: Any, enc_data: Any) -> float:
    """Return the total log-likelihood of encoded data."""
    return float(np.sum(model.seq_log_density(enc_data)))


def _num_params(model: Any, num_params: Optional[int]) -> int:
    """Return the override when given, else ``model.num_params()``."""
    if num_params is None:
        return int(model.num_params())
    if num_params < 0:
        raise ValueError("num_params must be nonnegative.")
    return int(num_params)


def heldout_log_likelihood(model: Any, data: Sequence[Any]) -> LogLikelihoodScore:
    """Score a fitted model on observations it was not fitted to.

    Args:
        model: Fitted :mod:`dmx.stats` or :mod:`dmx.bstats` distribution.
        data: Held-out observations of the model's data type.

    Returns:
        Total log-likelihood of ``data`` and the number of observations.

    Raises:
        ValueError: If ``data`` is empty.
    """
    n = _check_data(data)
    return LogLikelihoodScore(_log_likelihood(model, _encode(model, data)), n)


def kfold_log_likelihood(
    data: Sequence[Any],
    estimator: Any,
    k: int = 5,
    seed: int = 0,
    max_its: int = 100,
    delta: Optional[float] = 1.0e-9,
    init_p: float = 0.1,
) -> KFoldScore:
    """Estimate out-of-sample log-likelihood with k-fold cross-validation.

    Observations are assigned to ``k`` balanced random folds with
    :func:`dmx.utils.estimation.k_fold_split_index`. For each fold, a model is
    fitted on the remaining folds with ``optimize`` (from
    :mod:`dmx.utils.estimation`, or :mod:`dmx.bstats.bestimation` for a
    Bayesian estimator) and scored on the held-out fold. Optimizer progress is
    discarded. The same ``seed`` reproduces the folds and every fit.

    Args:
        data: Observations of the estimator's data type.
        estimator: :mod:`dmx.stats` or :mod:`dmx.bstats` parameter estimator.
        k: Number of folds, between 2 and ``len(data)``.
        seed: Seed for the fold assignment and the per-fold fits.
        max_its: Maximum optimizer iterations per fold.
        delta: Optimizer stopping threshold on log-likelihood improvement.
        init_p: Fraction of training observations used for initialization.

    Returns:
        Per-fold held-out log-likelihoods and fold sizes.

    Raises:
        ValueError: If ``k`` is outside ``[2, len(data)]``.
    """
    n = _check_data(data)
    if not 2 <= k <= n:
        raise ValueError(f"k must be between 2 and the number of observations ({n}).")

    rng = RandomState(seed)
    folds = k_fold_split_index(n, k, rng)
    fold_lls: list[float] = []
    fold_sizes: list[int] = []
    for fold in range(k):
        train = [data[i] for i in np.flatnonzero(folds != fold)]
        test = [data[i] for i in np.flatnonzero(folds == fold)]
        fit_rng = RandomState(rng.randint(maxrandint))
        model = _fit(train, estimator, max_its, delta, init_p, fit_rng)
        fold_lls.append(heldout_log_likelihood(model, test).log_likelihood)
        fold_sizes.append(len(test))

    return KFoldScore(tuple(fold_lls), tuple(fold_sizes), seed)


def _fit(
    data: Sequence[Any],
    estimator: Any,
    max_its: int,
    delta: Optional[float],
    init_p: float,
    rng: RandomState,
) -> Any:
    """Fit a model silently with the optimizer matching the estimator."""
    optimizer = (
        bstats_optimize
        if isinstance(estimator, BayesianParameterEstimator)
        else stats_optimize
    )
    return optimizer(
        data,
        estimator,
        max_its=max_its,
        delta=delta,
        init_p=init_p,
        rng=rng,
        out=io.StringIO(),
    )


def bic(
    model: Any, data: Sequence[Any], num_params: Optional[int] = None
) -> InformationCriterion:
    """Compute the Bayesian information criterion ``-2 * LL + p * ln(n)``.

    Args:
        model: Fitted :mod:`dmx.stats` or :mod:`dmx.bstats` distribution.
        data: Observations the model was fitted to.
        num_params: Parameter count override. Defaults to
            ``model.num_params()``.

    Returns:
        BIC value (lower is better) and its ingredients.

    Raises:
        ValueError: If ``data`` is empty or ``num_params`` is negative.
    """
    n = _check_data(data)
    ll = _log_likelihood(model, _encode(model, data))
    p = _num_params(model, num_params)
    penalty = p * math.log(n)
    return InformationCriterion("bic", -2.0 * ll + penalty, ll, p, n, penalty)


def aic(
    model: Any, data: Sequence[Any], num_params: Optional[int] = None
) -> InformationCriterion:
    """Compute the Akaike information criterion ``-2 * LL + 2 * p``.

    Args:
        model: Fitted :mod:`dmx.stats` or :mod:`dmx.bstats` distribution.
        data: Observations the model was fitted to.
        num_params: Parameter count override. Defaults to
            ``model.num_params()``.

    Returns:
        AIC value (lower is better) and its ingredients.

    Raises:
        ValueError: If ``data`` is empty or ``num_params`` is negative.
    """
    n = _check_data(data)
    ll = _log_likelihood(model, _encode(model, data))
    p = _num_params(model, num_params)
    penalty = 2.0 * p
    return InformationCriterion("aic", -2.0 * ll + penalty, ll, p, n, penalty)


def icl(
    model: Any, data: Sequence[Any], num_params: Optional[int] = None
) -> InformationCriterion:
    """Compute the integrated completed likelihood ``bic + 2 * entropy``.

    ICL adds the entropy of the posterior component assignments to BIC, so it
    favors well-separated clusters. The model must provide ``seq_posterior``
    returning an ``(n, K)`` responsibility matrix, as the mixture
    distributions do.

    Args:
        model: Fitted mixture-type distribution with ``seq_posterior``.
        data: Observations the model was fitted to.
        num_params: Parameter count override. Defaults to
            ``model.num_params()``.

    Returns:
        ICL value (lower is better) and its ingredients.

    Raises:
        TypeError: If the model has no ``seq_posterior`` returning an
            ``(n, K)`` array.
        ValueError: If ``data`` is empty or ``num_params`` is negative.
    """
    n = _check_data(data)
    if not callable(getattr(model, "seq_posterior", None)):
        raise TypeError(
            f"icl() requires seq_posterior; {type(model).__name__} has none."
        )
    enc_data = _encode(model, data)
    tau = np.asarray(model.seq_posterior(enc_data), dtype=np.float64)
    if tau.ndim != 2 or tau.shape[0] != n:
        raise TypeError(
            "icl() requires seq_posterior to return an (n, K) responsibility "
            f"matrix; {type(model).__name__} returned shape {tau.shape}."
        )
    positive = tau > 0.0
    entropy = float(-np.sum(tau[positive] * np.log(tau[positive])))
    ll = _log_likelihood(model, enc_data)
    p = _num_params(model, num_params)
    penalty = p * math.log(n) + 2.0 * entropy
    return InformationCriterion("icl", -2.0 * ll + penalty, ll, p, n, penalty, entropy)
