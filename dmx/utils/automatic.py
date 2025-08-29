"""Automatic estimations for input data files. Use in auto-estimation step of htsne."""
from typing import Optional, Any, Tuple, Sequence, Dict
from collections import defaultdict
from collections.abc import Iterable
import math
import numpy as np

from dmx.bstats.mixture import MixtureDistribution
from dmx.stats import ParameterEstimator


def get_optional_estimator(est: ParameterEstimator, missing_value: Optional[Any], use_bstats: bool = False):
    """Gets an optional estimator that handles missing values.

    Args:
        est (ParameterEstimator): The base estimator to use.
        missing_value (Optional[Any]): The value to treat as missing.
        use_bstats (bool): Whether to use the `dml.bstats` module.

    Returns:
        OptionalEstimator: An estimator that handles missing values.
    """
    if use_bstats:
        from dmx.bstats.optional import OptionalEstimator
        return OptionalEstimator(est, missing_value=missing_value)
    else:
        from dmx.stats.optional import OptionalEstimator
        return OptionalEstimator(est, missing_value=missing_value)


def get_sequence_estimator(est: ParameterEstimator, use_bstats=False):
    """Gets a sequence estimator.

    Args:
        est (ParameterEstimator): The base estimator to use.
        use_bstats (bool): Whether to use the `dml.bstats` module.

    Returns:
        SequenceEstimator: An estimator for sequences.
    """
    if use_bstats:
        from dmx.bstats.sequence import SequenceEstimator
        return SequenceEstimator(est)
    else:
        from dmx.stats.sequence import SequenceEstimator
        return SequenceEstimator(est)


def get_ignored_estimator(use_bstats: bool = False):
    """Gets an ignored estimator.

    Args:
        use_bstats (bool): Whether to use the `dml.bstats` module.

    Returns:
        IgnoredEstimator: An estimator that ignores input data.
    """
    if use_bstats:
        from dmx.bstats.ignored import IgnoredEstimator
        return IgnoredEstimator()
    else:
        from dmx.stats.ignored import IgnoredEstimator
        return IgnoredEstimator()


def get_composite_estimator(ests: Sequence[ParameterEstimator], use_bstats: bool = False):
    """Gets a composite estimator.

    Args:
        ests (Sequence[ParameterEstimator]): A list of estimators to combine.
        use_bstats (bool): Whether to use the `dml.bstats` module.

    Returns:
        CompositeEstimator: An estimator that combines multiple estimators.
    """
    if use_bstats:
        from dmx.bstats.composite import CompositeEstimator
        return CompositeEstimator(ests)
    else:
        from dmx.stats.composite import CompositeEstimator
        return CompositeEstimator(ests)


def get_categorical_estimator(vdict: Dict[Any, float], pseudo_count: Optional[float] = None, emp_suff_stat: bool = True, use_bstats: bool = False):
    """Gets a categorical estimator. 

    Args:
        vdict (dict): A dictionary of values and their counts.
        pseudo_count (Optional[float]): A pseudo-count to use for smoothing.
        emp_suff_stat (bool): Whether to use empirical sufficient statistics.
        use_bstats (bool): Whether to use the `dml.bstats` module.

    Returns:
        CategoricalEstimator: An estimator for categorical data.
    """
    if not use_bstats:
        from dmx.stats.categorical import CategoricalEstimator
        if emp_suff_stat:
            cnt = sum(vdict.values())
            suff_stat = {k: v / cnt for k, v in vdict.items()}
        else:
            suff_stat = None
        return CategoricalEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat)
    else:
        from dmx.bstats.categorical import CategoricalEstimator
        return CategoricalEstimator()


def get_poisson_estimator(vdict: Dict[Any, float], pseudo_count: Optional[float] = None, emp_suff_stat: bool = True, use_bstats: bool = False):
    """Gets a Poisson estimator.

    Args:
        vdict (Dict[Any, float]): A dictionary of values and their counts.
        pseudo_count (Optional[float]): A pseudo-count to use for smoothing.
        emp_suff_stat (bool): Whether to use empirical sufficient statistics.
        use_bstats (bool): Whether to use the `dml.bstats` module.

    Returns:
        PoissonEstimator: An estimator for Poisson-distributed data.
    """
    if use_bstats:
        from dmx.bstats.poisson import PoissonEstimator
        return PoissonEstimator()
    else:
        from dmx.stats.poisson import PoissonEstimator
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


def get_gaussian_estimator(vdict: Dict[Any, float], pseudo_count: Optional[float] = None, emp_suff_stat: bool = True, use_bstats: bool = False):
    """Gets a Gaussian estimator.

    Args:
        vdict (Dict[Any, float]): A dictionary of values and their counts.
        pseudo_count (Optional[float]): A pseudo-count to use for smoothing.
        emp_suff_stat (bool): Whether to use empirical sufficient statistics.
        use_bstats (bool): Whether to use the `dml.bstats` module.

    Returns:
        GaussianEstimator: An estimator for Gaussian-distributed data.
    """
    if use_bstats:
        from dmx.bstats.gaussian import GaussianEstimator
        return GaussianEstimator()
    else:
        from dmx.stats.gaussian import GaussianEstimator
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
        return GaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=(ss_1, ss_2))

class DatumNode(object):
    """Represents a node for processing data.

    Attributes:
        children (Sequence[DatumNode]): List of child nodes.
        parent (DatumNode): Parent node.
        vdict (defaultdict): Dictionary of value counts.
        count (int): Total count of data points.
        none_count (int): Count of None values.
        nan_count (int): Count of NaN values.
        inf_count (int): Count of infinite values.
        str_count (int): Count of string values.
        float_count (int): Count of float values.
        int_count (int): Count of integer values.
        bool_count (int): Count of boolean values.
        obj_count (int): Count of object values.
        neg_count (int): Count of negative values.
        zero_count (int): Count of zero values.
    """

    def __init__(self, parent: Optional['DatumNode'] = None, data: Sequence[Any] = None):
        """Initializes a DatumNode.

        Args:
            parent (Optional[DatumNode]): Parent node.
            data (Sequence[Any]): Data to add to the node.
        """
        self.children = []
        self.parent = parent
        self.vdict = defaultdict(int)
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

    def add_data(self, x):
        """Adds multiple data points to the node.

        Args:
            x (iterable): Data points to add.
        """
        for xx in x:
            self.add_datum(xx)

    def add_datum(self, x):
        """Adds a single data point to the node.

        Args:
            x: The data point to add.
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
        """Creates a copy of the node.

        Returns:
            DatumNode: A copy of the current node.
        """
        rv = DatumNode(self.parent)
        rv.children = [u.copy() for u in self.children]
        rv.vdict = self.vdict.copy()
        return rv


    def merge(self, x: 'DatumNode') -> 'DatumNode':
        """Merges another node into this one.

        Args:
            x (DatumNode): The node to merge.

        Returns:
            DatumNode: The merged node.
        """

        self.count += x.count
        self.none_count += x.none_count
        self.nan_count += x.nan_count

        for i in range(len(x.children)):
            temp = self._get_child_node(i).merge(x.children[i])
            self.children[i] = temp
        for k,v in x.vdict.items():
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
        """Analyzes the type of a data point.

        Args:
            x (Any): The data point to analyze.
            v (int): The count to increment.
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

    def get_estimator(self, pseudo_count: float = 1.0, emp_suff_stat: bool = True, use_bstats: bool = False):
        """Gets an estimator based on the node's data.

        Args:
            pseudo_count (float): A pseudo-count to use for smoothing.
            emp_suff_stat (bool): Whether to use empirical sufficient statistics.
            use_bstats (bool): Whether to use the `dml.bstats` module.

        Returns:
            Estimator: An appropriate estimator for the node's data.
        """
        rv = get_ignored_estimator(use_bstats)

        if len(self.children) == 0 and len(self.vdict) > 0:
            if self.obj_count > 0:
                rv = get_ignored_estimator(use_bstats)
            elif self.str_count > 0:
                rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats)
            elif self.float_count > 0:
                rv = get_gaussian_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats)
            elif self.int_count > 0:
                if self.neg_count > 0:
                    rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats)
                else:
                    rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats)
                    # More checking before we use this
                    #rv = get_poisson_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats)
            else:
                rv = get_ignored_estimator(use_bstats)

        # Lists of Same Size
        elif len(self.children) > 0 and len(set([u.count for u in self.children])) == 1 and all([u.count==self.count for u in self.children]):
            rv = get_composite_estimator([u.get_estimator(pseudo_count, emp_suff_stat, use_bstats) for u in self.children], use_bstats)

        # Lists of Different Size
        elif len(self.children) > 0 and len(set([u.count for u in self.children])) > 1:
            child = self.children[0].copy()
            for u in self.children[1:]:
                child = child.merge(u)
            rv = get_sequence_estimator(child.get_estimator(pseudo_count, emp_suff_stat, use_bstats), use_bstats)

        if self.none_count > 0:
            rv = get_optional_estimator(rv, None, use_bstats)

        if self.nan_count > 0:
            rv = get_optional_estimator(rv, math.nan, use_bstats)

        return rv

    def _get_child_node(self, idx: int) -> 'DatumNode':
        """Gets the child node at a specific index.

        Args:
            idx (int): The index of the child node.

        Returns:
            DatumNode: The child node.
        """
        while len(self.children) <= idx:
            self.children.append(DatumNode(self))
        return self.children[idx]




def get_estimator(data: Sequence[Any], pseudo_count: float = 1.0, emp_suff_stat: bool = True, use_bstats: bool = True):
    """Gets an estimator for the given data.

    Args:
        data (Sequence[Any]): The data to estimate.
        pseudo_count (float): A pseudo-count to use for smoothing.
        emp_suff_stat (bool): Whether to use empirical sufficient statistics.
        use_bstats (bool): Whether to use the `dml.bstats` module.

    Returns:
        Estimator: An appropriate estimator for the data.
    """
    return DatumNode(data=data).get_estimator(pseudo_count, emp_suff_stat, use_bstats)


def get_dpm_mixture(data: Sequence[Any], estimator: Optional[ParameterEstimator] = None, max_comp: int = 20, rng: Optional[np.random.RandomState] = None, max_its: int = 1000, print_iter: int = 100, mix_threshold_count: int = 0.5) -> MixtureDistribution:
    """Gets a Dirichlet Process Mixture model for the data.

    Args:
        data (Sequence[Any]): The data to model.
        estimator (Optional[ParameterEstimator]): The base estimator to use.
        max_comp (int): Maximum number of components in the mixture.
        rng (Optinal[numpy.random.RandomState]): Random number generator.
        max_its (int): Maximum number of iterations for optimization.
        print_iter (int): Frequency of printing iteration progress.
        mix_threshold_count (float): Threshold for component weights.

    Returns:
        MixtureDistribution: A mixture distribution model.
    """
    from dmx.bstats.dpm import DirichletProcessMixtureEstimator
    from dmx.bstats.mixture import MixtureDistribution
    from dmx.bstats.bestimation import optimize

    if estimator is None:
        est = get_estimator(data, use_bstats=True)
    else:
        est = estimator

    est = DirichletProcessMixtureEstimator([est]*max_comp)

    mix_model = optimize(data, est, max_its=max_its, rng=rng, print_iter=print_iter)

    thresh = mix_threshold_count/len(data)
    mix_comps = [mix_model.components[i] for i in np.flatnonzero(mix_model.w >= thresh)]
    mix_weights = mix_model.w[mix_model.w >= thresh]

    print(str(mix_weights))
    print('# Components = %d' % (len(mix_comps)))

    return MixtureDistribution(mix_comps, mix_weights)
