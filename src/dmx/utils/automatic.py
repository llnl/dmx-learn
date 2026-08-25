"""Infer estimators from heterogeneous data and fit mixture models.

The helpers inspect nested observations, select matching ``dmx.stats`` or
``dmx.bstats`` estimators, and support the mixture-model preprocessing used by
the heterogeneous embedding utilities.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from importlib import import_module
from typing import Any, DefaultDict, List, Optional, Sequence, Union, cast

import numpy as np

from dmx.bstats import CategoricalEstimator as BstatsCategoricalEstimator
from dmx.bstats import CompositeEstimator as BstatsCompositeEstimator
from dmx.bstats import GaussianEstimator as BstatsGaussianEstimator
from dmx.bstats import IgnoredEstimator as BstatsIgnoredEstimator
from dmx.bstats import MixtureDistribution as BstatsMixtureDistribution
from dmx.bstats import OptionalEstimator as BstatsOptionalEstimator
from dmx.bstats import ParameterEstimator as BstatsParameterEstimator
from dmx.bstats import PoissonEstimator as BstatsPoissonEstimator
from dmx.bstats import SequenceEstimator as BstatsSequenceEstimator
from dmx.bstats.bestimation import optimize as optimize_bstats
from dmx.bstats.dpm import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
)
from dmx.stats import ParameterEstimator
from dmx.stats.mixture import MixtureDistribution as StatsMixtureDistribution

MixtureModel = Union[StatsMixtureDistribution, BstatsMixtureDistribution]


def _get_class(
    stats_module: str,
    bstats_class: type[Any],
    class_name: str,
    use_bstats: bool,
) -> Any:
    """Select a ``stats`` class or its eagerly imported ``bstats`` counterpart."""
    if use_bstats:
        return bstats_class
    return getattr(import_module(stats_module), class_name)


def encode_mixture_data(
    data: Sequence[Any],
    mix_model: MixtureModel,
) -> Any:
    """Encode observations using the API implemented by a mixture model.

    Args:
        data: Observations accepted by the model's encoder.
        mix_model: A ``dmx.stats`` or ``dmx.bstats`` mixture distribution.

    Returns:
        The model-specific encoded sequence. Its structure depends on the
        component distribution.

    Raises:
        TypeError: If ``mix_model`` is not a supported mixture distribution.
    """
    if isinstance(mix_model, StatsMixtureDistribution):
        return mix_model.dist_to_encoder().seq_encode(data)
    if isinstance(mix_model, BstatsMixtureDistribution):
        return mix_model.seq_encode(data)
    raise TypeError(f"Unsupported mixture model type: {type(mix_model)!r}")


# Keep the current helper call signature stable for now.
# pylint: disable-next=too-many-positional-arguments
def prepare_mixture_model(
    data: Sequence[Any],
    rng: np.random.RandomState,
    max_components: int = 30,
    mix_threshold_count: float = 0.5,
    max_its: int = 1000,
    print_iter: int = 100,
    comp_estimator: Optional[Any] = None,
    mix_model: Optional[MixtureModel] = None,
) -> tuple[MixtureModel, Any, np.ndarray]:
    """Fit or reuse a mixture model and compute component posteriors.

    When ``mix_model`` is absent, a Dirichlet-process mixture is fitted using
    ``rng`` and the supplied optimization controls. A supplied model is used
    unchanged. In either case, the observations are encoded with that model.

    Args:
        data: Observations to encode and, when needed, fit.
        rng: Random state used to initialize mixture fitting.
        max_components: Maximum number of mixture components; must be an
            integer greater than one even when a model is supplied.
        mix_threshold_count: Minimum effective component count retained after
            fitting.
        max_its: Maximum mixture-fitting iterations.
        print_iter: Interval for mixture-fitting progress output.
        comp_estimator: Optional estimator for a single mixture component.
        mix_model: Optional pre-fitted mixture model.

    Returns:
        The mixture model, its model-specific encoded data, and an array of
        component posterior probabilities with shape ``(n_observations,
        n_components)``.

    Raises:
        ValueError: If ``max_components`` is not an integer greater than one.
        RuntimeError: If the resulting mixture has no components.
        TypeError: If the resulting model type cannot encode ``data``.
    """
    if max_components <= 1 or not isinstance(max_components, (int, np.integer)):
        raise ValueError("max_components must be an integer greater than 1.")

    if mix_model is None:
        mix_model = get_dpm_mixture(
            data=data,
            estimator=comp_estimator,
            max_comp=max_components,
            rng=rng,
            max_its=max_its,
            print_iter=print_iter,
            mix_threshold_count=mix_threshold_count,
        )

    if mix_model.num_components == 0:
        raise RuntimeError("Something is broken. Mixture model has zero components.")

    enc_data = encode_mixture_data(data, mix_model)
    return mix_model, enc_data, mix_model.seq_posterior(enc_data)


def get_optional_estimator(
    est: Union[ParameterEstimator, BstatsParameterEstimator],
    missing_value: Optional[Any],
    use_bstats: bool = False,
) -> Any:
    """Wrap an estimator so a designated missing value is modeled separately.

    Args:
        est: Base estimator for non-missing values.
        missing_value: Value treated as missing, including ``None`` or NaN.
        use_bstats: Select the Bayesian ``dmx.bstats`` implementation.

    Returns:
        An optional estimator from the selected package.
    """
    OptionalEstimator = _get_class(
        "dmx.stats.optional",
        BstatsOptionalEstimator,
        "OptionalEstimator",
        use_bstats,
    )
    return OptionalEstimator(est, missing_value=missing_value)


def get_sequence_estimator(
    est: Union[ParameterEstimator, BstatsParameterEstimator],
    use_bstats: bool = False,
) -> Any:
    """Wrap an estimator to model variable-length sequences.

    Args:
        est: Estimator for individual sequence elements.
        use_bstats: Select the Bayesian ``dmx.bstats`` implementation.

    Returns:
        A sequence estimator from the selected package.
    """
    SequenceEstimator = _get_class(
        "dmx.stats.sequence",
        BstatsSequenceEstimator,
        "SequenceEstimator",
        use_bstats,
    )
    return SequenceEstimator(est)


def get_ignored_estimator(use_bstats: bool = False) -> Any:
    """Create an estimator that ignores its input field.

    Args:
        use_bstats: Select the Bayesian ``dmx.bstats`` implementation.

    Returns:
        An ignored estimator from the selected package.
    """
    IgnoredEstimator = _get_class(
        "dmx.stats.ignored", BstatsIgnoredEstimator, "IgnoredEstimator", use_bstats
    )
    return IgnoredEstimator()


def get_composite_estimator(ests: Sequence[Any], use_bstats: bool = False) -> Any:
    """Combine field estimators into a fixed-width composite estimator.

    Args:
        ests: Estimators in the same order as fields in each observation.
        use_bstats: Select the Bayesian ``dmx.bstats`` implementation.

    Returns:
        A composite estimator from the selected package.
    """
    CompositeEstimator = _get_class(
        "dmx.stats.composite",
        BstatsCompositeEstimator,
        "CompositeEstimator",
        use_bstats,
    )
    return CompositeEstimator(ests)


def get_categorical_estimator(
    vdict: Mapping[Any, float],
    pseudo_count: Optional[float] = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> Any:
    """Create a categorical estimator from weighted observed values.

    For ``dmx.stats``, empirical sufficient statistics are normalized from
    ``vdict`` when requested. The ``dmx.bstats`` estimator is returned with
    its defaults and does not consume ``pseudo_count`` or ``emp_suff_stat``.

    Args:
        vdict: Mapping from category values to nonnegative observation weights.
        pseudo_count: Smoothing mass for the ``dmx.stats`` estimator.
        emp_suff_stat: Whether to initialize ``dmx.stats`` probabilities from
            normalized observed weights.
        use_bstats: Select the Bayesian ``dmx.bstats`` implementation.

    Returns:
        A categorical estimator from the selected package.
    """
    if not use_bstats:
        CategoricalEstimator = _get_class(
            "dmx.stats.categorical",
            BstatsCategoricalEstimator,
            "CategoricalEstimator",
            use_bstats,
        )

        if emp_suff_stat:
            cnt = sum(vdict.values())
            suff_stat = {k: v / cnt for k, v in vdict.items()}
        else:
            suff_stat = None
        return CategoricalEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat)
    CategoricalEstimator = _get_class(
        "dmx.stats.categorical",
        BstatsCategoricalEstimator,
        "CategoricalEstimator",
        use_bstats,
    )

    return CategoricalEstimator()


def get_poisson_estimator(
    vdict: Mapping[Any, float],
    pseudo_count: Optional[float] = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> Any:
    """Create a Poisson estimator from weighted observed values.

    The empirical sufficient statistic is the weighted mean of finite keys.
    Bayesian estimators are returned with their defaults.

    Args:
        vdict: Mapping from numeric values to observation weights.
        pseudo_count: Smoothing mass for the ``dmx.stats`` estimator.
        emp_suff_stat: Whether to initialize from the weighted finite values.
        use_bstats: Select the Bayesian ``dmx.bstats`` implementation.

    Returns:
        A Poisson estimator from the selected package.
    """
    if use_bstats:
        PoissonEstimator = _get_class(
            "dmx.stats.poisson",
            BstatsPoissonEstimator,
            "PoissonEstimator",
            use_bstats,
        )

        return PoissonEstimator()
    PoissonEstimator = _get_class(
        "dmx.stats.poisson", BstatsPoissonEstimator, "PoissonEstimator", use_bstats
    )

    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0
        for k, v in vdict.items():
            if math.isfinite(k):
                ss_0 += v
                ss_1 += k * v
        ss_1 = ss_1 / ss_0
    elif pseudo_count is not None:
        ss_1 = 1.0
    else:
        ss_1 = None
    return PoissonEstimator(pseudo_count=pseudo_count, suff_stat=ss_1)


def get_gaussian_estimator(
    vdict: Mapping[Any, float],
    pseudo_count: Optional[float] = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> Any:
    """Create a Gaussian estimator from weighted observed values.

    The empirical sufficient statistics are the weighted mean and population
    variance of finite keys. Bayesian estimators are returned with defaults.

    Args:
        vdict: Mapping from numeric values to observation weights.
        pseudo_count: Smoothing mass for both Gaussian sufficient statistics.
        emp_suff_stat: Whether to initialize from the weighted finite values.
        use_bstats: Select the Bayesian ``dmx.bstats`` implementation.

    Returns:
        A Gaussian estimator from the selected package.
    """
    if use_bstats:
        GaussianEstimator = _get_class(
            "dmx.stats.gaussian",
            BstatsGaussianEstimator,
            "GaussianEstimator",
            use_bstats,
        )

        return GaussianEstimator()
    GaussianEstimator = _get_class(
        "dmx.stats.gaussian", BstatsGaussianEstimator, "GaussianEstimator", use_bstats
    )

    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0
        ss_2 = 0.0
        for k, v in vdict.items():
            if math.isfinite(k):
                ss_0 += v
                ss_1 += k * v
                ss_2 += k * k * v
        ss_1 = ss_1 / ss_0
        ss_2 = (ss_2 / ss_0) - ss_1 * ss_1
    elif pseudo_count is not None:
        ss_1 = 1.0e-6
        ss_2 = 1.0e-6
    else:
        ss_1 = None
        ss_2 = None
    return GaussianEstimator(
        pseudo_count=(pseudo_count, pseudo_count), suff_stat=(ss_1, ss_2)
    )


class DatumNode:
    """Summarize values at one position in nested observations.

    Tuples, lists, and non-string iterables create positional child nodes.
    Scalar values are counted by value and classified for automatic estimator
    selection.

    Attributes:
        children: Positional summaries for nested values.
        parent: Parent summary, or ``None`` at the root.
        vdict: Counts of scalar values.
        count: Total values added at this node.
        none_count: Number of ``None`` values.
        nan_count: Number of floating-point NaN values.
        inf_count: Number of infinite values.
        str_count: Number of string values.
        float_count: Number of finite, non-integral floating-point values.
        int_count: Number of integer-valued numeric values.
        bool_count: Number of boolean values classified as such.
        obj_count: Number of values not covered by another scalar category.
        neg_count: Number of negative finite floating-point values.
        zero_count: Number of floating-point zeros.
    """

    def __init__(
        self, parent: Optional["DatumNode"] = None, data: Optional[Sequence[Any]] = None
    ) -> None:
        """Initialize a node and optionally summarize data.

        Args:
            parent: Parent node.
            data: Optional initial observations.
        """
        self.children: List[DatumNode] = []
        self.parent = parent
        self.vdict: DefaultDict[Any, int] = defaultdict(int)
        self.count = 0
        self.none_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.str_count = 0
        self.float_count = 0
        self.int_count = 0
        self.bool_count = 0
        self.obj_count = 0
        self.neg_count = 0
        self.zero_count = 0
        if data is not None:
            self.add_data(data)

    def add_data(self, x: Iterable[Any]) -> None:
        """Add each value from an iterable to the summary.

        Args:
            x: Values to summarize.
        """
        for xx in x:
            self.add_datum(xx)

    def add_datum(self, x: Any) -> None:
        """Add one scalar or nested value to the summary.

        Args:
            x: Value to summarize. Non-string iterables are expanded into
                positional child nodes.
        """
        self.count += 1
        if isinstance(x, (tuple, list)):
            for i, xx in enumerate(x):
                self._get_child_node(i).add_datum(xx)
        elif isinstance(x, (Iterable,)) and not isinstance(x, (str,)):
            for i, xx in enumerate(x):
                self._get_child_node(i).add_datum(xx)
        elif x is None:
            self.none_count += 1
        else:
            self.vdict[x] += 1
            self._analyze_type(x)

    def copy(self) -> "DatumNode":
        """Copy the node's children and scalar value counts.

        Returns:
            A new node with recursively copied children and ``vdict``.
        """
        rv = DatumNode(self.parent)
        rv.children = [u.copy() for u in self.children]
        rv.vdict = self.vdict.copy()
        return rv

    def merge(self, x: "DatumNode") -> "DatumNode":
        """Merge another summary into this node in place.

        Args:
            x: Node whose counts and children are added.

        Returns:
            This mutated node.
        """
        self.count += x.count
        self.none_count += x.none_count
        self.nan_count += x.nan_count

        for i, child in enumerate(x.children):
            temp = self._get_child_node(i).merge(child)
            self.children[i] = temp
        for k, v in x.vdict.items():
            self.vdict[k] += v

        self.neg_count += x.neg_count
        self.inf_count += x.inf_count
        self.int_count += x.int_count
        self.float_count += x.float_count
        self.obj_count += x.obj_count
        self.str_count += x.str_count
        self.bool_count += x.bool_count
        return self

    def _analyze_type(self, x: Any, v: int = 1) -> None:
        """Add a scalar value's type characteristics to the counters.

        Args:
            x: Scalar value to classify.
            v: Amount by which matching counters are incremented.
        """
        if isinstance(x, (float, np.floating)):
            if math.isnan(x):
                self.nan_count += v
            elif math.isinf(x):
                self.inf_count += v
            elif math.floor(x) == x:
                self.int_count += v
            else:
                self.float_count += v
            if x == 0:
                self.zero_count += v
            if math.isfinite(x) and x < 0:
                self.neg_count += v

        elif isinstance(x, (int, np.integer)):
            self.int_count += v
        elif isinstance(x, bool):
            self.bool_count += v
        elif isinstance(x, str):
            self.str_count += v
        else:
            self.obj_count += v

    def get_estimator(
        self,
        pseudo_count: float = 1.0,
        emp_suff_stat: bool = True,
        use_bstats: bool = False,
    ) -> Any:
        """Infer an estimator from the summarized observation structure.

        Scalar strings and nonnegative integers become categorical fields;
        non-integral floats become Gaussian fields. Fixed-width nested values
        become composites, variable-width values become sequences, unsupported
        values are ignored, and observed ``None`` or NaN values add optional
        wrappers.

        Args:
            pseudo_count: Smoothing mass for inferred ``dmx.stats`` estimators.
            emp_suff_stat: Whether to initialize supported estimators from
                empirical counts.
            use_bstats: Select Bayesian estimators from ``dmx.bstats``.

        Returns:
            An estimator matching the inferred scalar or nested structure.
        """
        rv = get_ignored_estimator(use_bstats)

        if len(self.children) == 0 and len(self.vdict) > 0:
            if self.obj_count > 0:
                rv = get_ignored_estimator(use_bstats)
            elif self.str_count > 0:
                rv = get_categorical_estimator(
                    self.vdict, pseudo_count, emp_suff_stat, use_bstats
                )
            elif self.float_count > 0:
                rv = get_gaussian_estimator(
                    self.vdict, pseudo_count, emp_suff_stat, use_bstats
                )
            elif self.int_count > 0:
                if self.neg_count > 0:
                    rv = get_categorical_estimator(
                        self.vdict, pseudo_count, emp_suff_stat, use_bstats
                    )
                else:
                    rv = get_categorical_estimator(
                        self.vdict, pseudo_count, emp_suff_stat, use_bstats
                    )
                    # More checking before we use this
                    # rv = get_poisson_estimator(
                    #     self.vdict, pseudo_count, emp_suff_stat, use_bstats
                    # )
            else:
                rv = get_ignored_estimator(use_bstats)

        # Lists of Same Size
        elif (
            len(self.children) > 0
            and len({u.count for u in self.children}) == 1
            and all(u.count == self.count for u in self.children)
        ):
            rv = get_composite_estimator(
                [
                    u.get_estimator(pseudo_count, emp_suff_stat, use_bstats)
                    for u in self.children
                ],
                use_bstats,
            )

        # Lists of Different Size
        elif len(self.children) > 0 and len({u.count for u in self.children}) > 1:
            child = self.children[0].copy()
            for u in self.children[1:]:
                child = child.merge(u)
            rv = get_sequence_estimator(
                child.get_estimator(pseudo_count, emp_suff_stat, use_bstats), use_bstats
            )

        if self.none_count > 0:
            rv = get_optional_estimator(rv, None, use_bstats)

        if self.nan_count > 0:
            rv = get_optional_estimator(rv, math.nan, use_bstats)

        return rv

    def _get_child_node(self, idx: int) -> "DatumNode":
        """Return a positional child, creating missing children as needed.

        Args:
            idx: Zero-based child index.

        Returns:
            The child node at ``idx``.
        """
        while len(self.children) <= idx:
            self.children.append(DatumNode(self))
        return self.children[idx]


def get_estimator(
    data: Sequence[Any],
    pseudo_count: float = 1.0,
    emp_suff_stat: bool = True,
    use_bstats: bool = True,
) -> Any:
    """Infer an estimator from a sequence of heterogeneous observations.

    Args:
        data: Observations whose scalar or nested structure is inspected.
        pseudo_count: Smoothing mass for inferred ``dmx.stats`` estimators.
        emp_suff_stat: Whether to initialize supported estimators from
            empirical counts.
        use_bstats: Select Bayesian estimators from ``dmx.bstats``.

    Returns:
        An estimator matching the inferred data structure.
    """
    return DatumNode(data=data).get_estimator(pseudo_count, emp_suff_stat, use_bstats)


# Keep the current public call signature stable for now.
# pylint: disable=too-many-positional-arguments
def get_dpm_mixture(
    data: Sequence[Any],
    estimator: Optional[Any] = None,
    max_comp: int = 20,
    rng: Optional[np.random.RandomState] = None,
    max_its: int = 1000,
    print_iter: int = 100,
    mix_threshold_count: float = 0.5,
) -> BstatsMixtureDistribution:
    """Fit and prune a finite Dirichlet-process mixture model.

    The Bayesian optimizer initializes and updates at most ``max_comp``
    copies of the component estimator. Components whose fitted weight is less
    than ``mix_threshold_count / len(data)`` are removed. The optimizer and
    this function write progress and retained weights to standard output.

    Args:
        data: Nonempty observations to model.
        estimator: Component estimator. When absent, one is inferred from
            ``data``.
        max_comp: Maximum number of components used during fitting.
        rng: Random state used by mixture initialization. When absent, the
            Bayesian optimizer supplies its default random state.
        max_its: Maximum number of optimizer iterations.
        print_iter: Interval for optimizer progress output.
        mix_threshold_count: Effective-count threshold for retaining a
            component.

    Returns:
        A Bayesian mixture containing the retained components and their
        weights.

    Raises:
        ZeroDivisionError: If ``data`` is empty.
    """
    if estimator is None:
        est = get_estimator(data, use_bstats=True)
    else:
        est = estimator

    est = DirichletProcessMixtureEstimator([est] * max_comp)

    mix_model = cast(
        DirichletProcessMixtureDistribution,
        optimize_bstats(data, est, max_its=max_its, rng=rng, print_iter=print_iter),
    )

    thresh = mix_threshold_count / len(data)
    mix_comps = [mix_model.components[i] for i in np.flatnonzero(mix_model.w >= thresh)]
    mix_weights = mix_model.w[mix_model.w >= thresh]

    print(str(mix_weights))
    print(f"# Components = {len(mix_comps)}")

    return BstatsMixtureDistribution(mix_comps, mix_weights)
