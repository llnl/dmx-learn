# pylint: disable=line-too-long
"""Create, estimate, and sample from a Conditional distribution.

Defines the ConditionalDistribution, ConditionalDistributionSampler, ConditionalDistributionAccumulatorFactory,
ConditionalDistributionAccumulator, ConditionalDistributionEstimator, and the ConditionalDistributionDataEncoder
classes for use with pysparkplug.

Data type: (Tuple[T0, T1]): The ConditionalDistribution if given by density,
    P(X0,X1) = P_cond(X1|X0)*P_given(X0).

The ConditionalDistribution allows for user defined conditional distributions P_cond(X1|X0), and given distributions
P_given(X0).

"""

# pylint: disable=line-too-long,too-many-positional-arguments,duplicate-code
# pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin
# pylint: disable=broad-exception-raised,consider-using-f-string,no-else-return
# pylint: disable=no-else-raise,consider-using-enumerate,consider-using-generator
# pylint: disable=use-dict-literal,super-with-arguments,unnecessary-comprehension
# pylint: disable=simplifiable-if-statement,nested-min-max

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as tn
from torch import Generator

import dmx.torch_utils.vector as vec
from dmx.arithmetic import maxrandint
from dmx.torch_stats.null_dist import *
from dmx.torch_stats.pdist import (
    DistributionSampler,
    TorchEncodedSequence,
    TorchParameterEstimator,
    TorchProbabilityDistribution,
    TorchSequenceEncoder,
    TorchStatisticAccumulator,
    TorchStatisticAccumulatorFactory,
)

T0 = TypeVar("T0")
T1 = TypeVar("T1")

E0 = TypeVar("E0")
E1 = TypeVar("E1")
E = Tuple[int, Tuple[T0, ...], Tuple[tn.Tensor, ...], Tuple[E0, ...], Optional[E1]]
SS0 = TypeVar("SS0")
SS1 = TypeVar("SS1")
SS2 = TypeVar("SS2")


class ConditionalDistribution(TorchProbabilityDistribution):
    """ConditionalDistribution object for data types x=Tuple[T0, T1].

    Notes:
        P(x) = P_cond(x[1] | x[0])*P_given(x[0]), where

        p_cond(x[1] | x[0]) is a conditional distribution defined through dictionary dmap, with keys over data type T0,
        and values containing the ArkoudaProbabilityDistribution objects compatible with data type T1.

        P_given(x[0]) is defined as the given distribution. If None is provided, it is assumed that P_given(x[0]) = 1
        for all x[0].

        default_dist defines the distribution for the case where x[0] is not a key in dmap. That is, x[0] is not in the
        support of P_cond(X_1 | X_0). If None is provided we assume that P_cond(X1 | X0) = 0, for all X0 not in dmap.

    Attributes:
        dmap (Dict[T0, TorchProbabilityDistribution]): T0 is integer if dmap arg was list, else T0 is
            data type of the "given" or conditional.
        default_dist (TorchProbabilityDistribution): Set to NullDistribution if None is passed as arg.
        given_dist (TorchProbabilityDistribution): Set to NullDistribution if None is passed as arg.
        has_default (bool): True if default distribution is not NullDistribution, else False.
        has_given (bool): True if given_dist is not NullDistribution, else False.
        keys (Optional[str]): All ConditionalDistribution objects with same keys value are the same distribution.

    """

    def __init__(
        self,
        dmap: Union[
            Dict[Any, TorchProbabilityDistribution], List[TorchProbabilityDistribution]
        ],
        default_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        given_dist: Optional[TorchProbabilityDistribution] = NullDistribution(),
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """ConditionalDistribution object.
        Args:
            dmap Union[Dict[Any, TorchProbabilityDistribution],
                List[TorchProbabilityDistribution]]): Used to create dictionary of
                TorchProbabilityDistribution objects. Type T0 is inferred to be type of dmap keys if dict,
                else the T0 is inferred to integer.
            default_dist (Optional[TorchProbabilityDistribution]): Defines the distribution for the case
                where x[0] is not a key in dmap
            given_dist (Optional[TorchProbabilityDistribution]): p_mat(x[0]) is defined as the given
                distribution.
            keys (Optional[str]): All ConditionalDistribution objects with same keys value are the same distribution.
            device (Optional[str]): Set the device of the object.

        """
        super().__init__(device)
        if isinstance(dmap, list):
            dmap = dict(zip(range(len(dmap)), dmap))

        self.dmap = dmap
        self.default_dist = (
            default_dist if default_dist is not None else NullDistribution()
        )
        self.given_dist = given_dist if given_dist is not None else NullDistribution()

        self.has_default = not isinstance(self.default_dist, NullDistribution)
        self.has_given = not isinstance(self.given_dist, NullDistribution)
        self.keys = keys

    def __str__(self) -> str:
        s1 = repr(self.dmap)
        s2 = repr(self.default_dist)
        s3 = repr(self.given_dist)
        s4 = repr(self.keys)

        return (
            "ConditionalDistribution(%s, default_dist=%s, given_dist=%s, keys=%s)"
            % (s1, s2, s3, s4)
        )

    def to(self, device: tn.device) -> None:
        self._device = device
        for v in self.dmap.values():
            v.to(device)
        self.default_dist.to(device)
        self.given_dist.to(device)

    def density(self, x: Tuple[T0, T1]) -> float:
        """Evaluates density of ConditionalDistribution at Tuple x.

        Notes:
            Calls log_density() and returns the exponentiated result. See log_density() for details.

        Args:
            x (Tuple[T0, T1]): T0 data type much match keys of dmap, T1 much match value of dmap distribution for key
                value.

        Returns:
            float: Density of ConditionalDistribution at Tuple x

        """
        return math.exp(self.log_density(x))

    def log_density(self, x: Tuple[T0, T1]) -> float:
        """Evaluate log-density of ConditionalDistribution at Tuple x.

        Log-density:
            log(P(x)) = log(P_cond(x[1] | x[0])) + log(P_given(x[0])), where
            log(P_cond(x[1] | x[0])) is defined from dmap, and log(P_given(x[0])) is defined from given_dist.

        Args:
            x (Tuple[T0, T1]): T0 data type much match keys of dmap, T1 much match value of dmap distribution for key
                value.

        Returns:
            float: Log-density of ConditionalDistribution at Tuple x.

        """
        if self.has_default:
            rv = self.dmap.get(x[0], self.default_dist).log_density(x[1])
        else:
            if x[0] in self.dmap:
                rv = self.dmap[x[0]].log_density(x[1])
            else:
                return -np.inf

        rv += self.given_dist.log_density(x[0])

        return rv

    def seq_log_density(self, x: "ConditionalTorchEncodedSequence") -> tn.Tensor:

        if not isinstance(x, ConditionalTorchEncodedSequence):
            raise Exception(
                "Requires ConditionalTorchEncodedSequence for `seq_` function calls."
            )

        sz, cond_vals, eobs_vals, idx_vals, given_enc = x.data
        rv = vec.zeros(sz, device=self._device)

        for i in range(len(cond_vals)):
            idx = idx_vals[i].to(device=rv.device)
            if self.has_default:
                rv[idx] = self.dmap.get(
                    cond_vals[i], self.default_dist
                ).seq_log_density(eobs_vals[i])
            else:
                if cond_vals[i] in self.dmap:
                    rv[idx] += self.dmap[cond_vals[i]].seq_log_density(eobs_vals[i])

        if self.has_given:
            rv += self.given_dist.seq_log_density(given_enc)

        return rv

    def sampler(self, seed: Optional[int] = None) -> "ConditionalDistributionSampler":
        return ConditionalDistributionSampler(self, seed=seed)

    def estimator(
        self, pseudo_count: Optional[float] = None
    ) -> "ConditionalDistributionEstimator":
        est_map = {k: v.estimator(pseudo_count) for k, v in self.dmap.items()}
        default_est = self.default_dist.estimator(pseudo_count)
        given_est = self.given_dist.estimator(pseudo_count)

        return ConditionalDistributionEstimator(
            estimator_map=est_map,
            default_estimator=default_est,
            given_estimator=given_est,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "ConditionalDistributionDataEncoder":

        encoder_map = {k: v.dist_to_encoder() for k, v in self.dmap.items()}
        default_encoder = (
            NullDataEncoder()
            if not self.has_default
            else self.default_dist.dist_to_encoder()
        )
        given_encoder = (
            NullDataEncoder()
            if not self.has_given
            else self.given_dist.dist_to_encoder()
        )

        return ConditionalDistributionDataEncoder(
            encoder_map=encoder_map,
            default_encoder=default_encoder,
            given_encoder=given_encoder,
        )


class ConditionalDistributionSampler(ConditionalSampler, DistributionSampler):
    """ConditionalDistributionSampler object samples from ConditionalDistribution either directly or conditionally.

    Attributes:
        dist (ConditionalDistribution): ConditionalDistribution object to draw samples from.
        default_sampler (DistributionSampler): DistributionSampler object for sampling from default_dist of
            ConditionalDistribution.
        has_default_sampler (bool): True if default sampler is not NullDistribution, else False.
        given_sampler (DistributionSampler): DistributionSampler object for sampling from given_dist of
            ConditionalDistribution.
        has_given_sampler (bool): True if given sampler is not NullDistribution, else False.
        samplers (Dict[T0,DistributionSampler]): Dictionary of samplers for sampling from ConditionalDistribution,
            given a key of data type T0. Note returns List[T1] or T1.

    """

    def __init__(
        self, dist: ConditionalDistribution, seed: Optional[int] = None
    ) -> None:
        """ConditionalDistributionSampler object.

        Args:
            dist (ConditionalDistribution): ConditionalDistribution object to draw samples from.
            seed (Optional[int]): Used to set the seed of random number generator used in sampling.

        """
        self.dist = dist
        rng = np.random.RandomState(seed)

        loc_seed = rng.randint(0, maxrandint)

        self.has_default_sampler = dist.has_default
        self.default_sampler = dist.default_dist.sampler(loc_seed)

        loc_seed = rng.randint(0, maxrandint)
        self.given_sampler = dist.given_dist.sampler(loc_seed)
        self.has_given_sampler = isinstance(dist.given_dist, NullDistribution)

        self.samplers = {
            k: u.sampler(rng.randint(0, maxrandint)) for k, u in self.dist.dmap.items()
        }

    def single_sample(self) -> Tuple[Any, Any]:
        """Generates a simple sample from the ConditionalDistribution.

        Returns Tuple of T0 and T1, where T1 is the data type of the conditional distribution, and T0 is the type of
        the given distribution.

        Returns:
            Tuple[T0, T1] as defined from dmap and given_distribution types in dist (ConditionalDistribution instance).

        """
        x0 = self.given_sampler.sample()
        if x0 in self.samplers:
            x1 = self.samplers[x0].sample()
        else:
            x1 = self.default_sampler.sample()
        return x0, x1

    def sample(
        self, size: Optional[int] = None
    ) -> Union[Tuple[Any, Any], List[Tuple[Any, Any]]]:
        """Sample 'size' independent samples from ConditionalDistribution.

        Sequence of 'size' calls to single_sample(). If size is None, size is taken to be 1.

        Data type returned is a Tuple[T0, T1], where T0 and T1 are the respective data types of the given_dist and
        dmap defined in the CompositeDistribution instance 'dist'.

        Args:
            size (Optional[int]): Number of independent samples to draw from ConditionalDistribution.

        Returns:
            A list of 'size' tuples of Tuple[T0, T1], or a single Tuple[T0, T1].

        """

        if size is None:
            return self.single_sample()
        else:
            return [self.single_sample() for i in range(size)]

    def sample_given(self, x: T0) -> Any:
        """Sample from conditional distribution of ConditionalDistribution object with given value x.

        Return data type T1 as defined for dictionary of ConditionalDistribution instance.

        Args:
            x (T0): Value of given/conditional value for ConditionalDistribution.

        Returns:
            Single sample from ConditionalDistribution object 'dist.dmap' given x.

        """
        if x in self.samplers:
            return self.samplers[x].sample()

        elif self.has_default_sampler:
            return self.default_sampler.sample()

        else:
            raise Exception("Conditional default distribution unspecified.")


class ConditionalDistributionAccumulator(TorchStatisticAccumulator):
    """ConditionalDistributionAccumulator used for aggregating sufficient statistics of ConditionalDistribution.

    Attributes:
        accumulator_map (Dict[T0, TorchStatisticAccumulator]): Stores sufficient statistics of each
            conditional distribution for a given key value of data type T0.
        default_accumulator (Optional[TorchStatisticAccumulator]): Stores sufficient statistics of
            distribution for case where key not in accumulator_map.
        given_accumulator (Optional[TorchStatisticAccumulator]): Stores sufficient statistics of
            given distribution if provided.
        has_default (bool): True if default_accumulator is not NullAccumulator.
        has_given (bool): True if given_accumulator is not NullAccumulator.
        key (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

        _init_tng (bool): False unless a single call to initialize or seq_initialize has been made.
        _acc_tng (Optional[Dict[T0, Generator]]): Used to seed Generator calls of accumulator_map.
        _default_tng (Optional[Generator]): Used to seed Generator calls of defualt_accumulator initialize.
        _given_tng (Optional[Generator]): Used to seed Generator calls of given_accumulator initialize.

    """

    def __init__(
        self,
        accumulator_map: Dict[T0, TorchStatisticAccumulator],
        default_accumulator: Optional[TorchStatisticAccumulator] = NullAccumulator(),
        given_accumulator: Optional[TorchStatisticAccumulator] = NullAccumulator(),
        keys: Optional[str] = None,
        device: Optional[tn.device] = None,
    ) -> None:
        """ConditionalDistributionAccumulator object.

        Args:
            accumulator_map (Dict[T0, TorchStatisticAccumulator]): Stores sufficient statistics of each
                conditional distribution for a given key value of data type T0.
            default_accumulator (Optional[TorchStatisticAccumulator]): Stores sufficient statistics of
                distribution for case where key not in accumulator_map.
            given_accumulator (Optional[TorchStatisticAccumulator]): Stores sufficient statistics of
                given distribution if provided.
            keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.
            device (Optional[str]): Set object device.

        """
        super().__init__(device)
        self.accumulator_map = accumulator_map
        self.default_accumulator = (
            default_accumulator
            if default_accumulator is not None
            else NullAccumulator()
        )
        self.given_accumulator = (
            given_accumulator if given_accumulator is not None else NullAccumulator()
        )

        self.has_default = not isinstance(default_accumulator, NullAccumulator)
        self.has_given = not isinstance(given_accumulator, NullAccumulator)
        self.key = keys

        #### seeds for intializers
        self._init_tng = False
        self._acc_tng: Optional[Dict[T0, Generator]] = None
        self._default_tng: Optional[Generator] = None
        self._given_tng: Optional[Generator] = None

    def _tng_initialize(self, tng: Generator) -> None:
        seed_rng = np.random.RandomState(int(tng.initial_seed()))
        seeds = seed_rng.randint(0, 2**31, size=(len(self.accumulator_map.keys()) + 2,))
        self._acc_tng = dict()
        for i, acc_key in enumerate(self.accumulator_map.keys()):
            self._acc_tng[acc_key] = Generator().manual_seed(int(seeds[i + 2]))

        self._default_tng = Generator().manual_seed(int(seeds[0]))
        self._given_tng = Generator().manual_seed(int(seeds[1]))

    def seq_initialize(
        self, x: "ConditionalTorchEncodedSequence", weights: tn.Tensor, tng: Generator
    ) -> None:

        _, cond_vals, eobs_vals, idx_vals, given_enc = x.data

        if not self._init_tng:
            self._tng_initialize(tng)

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_initialize(
                    eobs_vals[i], weights[idx_vals[i]], self._acc_tng[cond_vals[i]]
                )
            else:
                if self.has_default:
                    self.default_accumulator.seq_initialize(
                        eobs_vals[i], weights[idx_vals[i]], self._default_tng
                    )

        if self.has_given:
            self.given_accumulator.seq_initialize(given_enc, weights, self._given_tng)

    def seq_update(
        self,
        x: "ConditionalTorchEncodedSequence",
        weights: tn.Tensor,
        estimate: "ConditionalDistribution",
    ) -> None:

        _, cond_vals, eobs_vals, idx_vals, given_enc = x.data

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_update(
                    eobs_vals[i], weights[idx_vals[i]], estimate.dmap[cond_vals[i]]
                )
            else:
                if self.has_default:
                    if estimate is None:
                        self.default_accumulator.seq_update(
                            eobs_vals[i], weights[idx_vals[i]], None
                        )
                    else:
                        self.default_accumulator.seq_update(
                            eobs_vals[i], weights[idx_vals[i]], estimate.default_dist
                        )

        if self.has_given:
            if estimate is None:
                self.given_accumulator.seq_update(given_enc, weights, None)
            else:
                self.given_accumulator.seq_update(
                    given_enc, weights, estimate.given_dist
                )

    def combine(
        self, suff_stat: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS2]]
    ) -> "ConditionalDistributionAccumulator":

        for k, v in suff_stat[0].items():
            if k in self.accumulator_map:
                self.accumulator_map[k].combine(v)
            else:
                self.accumulator_map[k].from_value(v)

        if self.has_default and suff_stat[1] is not None:
            self.default_accumulator.combine(suff_stat[1])

        if self.has_given and suff_stat[2] is not None:
            self.given_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[Dict[Any, Any], Optional[Any], Optional[Any]]:
        rv3 = self.given_accumulator.value()
        rv2 = self.default_accumulator.value()
        rv1 = {k: v.value() for k, v in self.accumulator_map.items()}

        return rv1, rv2, rv3

    def from_value(
        self, x: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS1]]
    ) -> "ConditionalDistributionAccumulator":

        for k, v in x[0].items():
            self.accumulator_map[k].from_value(v)

        if self.has_default and x[1] is not None:
            self.default_accumulator.from_value(x[1])

        if self.has_given and x[2] is not None:
            self.given_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:

        for v in self.accumulator_map.values():
            v.key_merge(stats_dict)

        if self.has_default:
            self.default_accumulator.key_merge(stats_dict)

        if self.has_given:
            self.given_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        for v in self.accumulator_map.values():
            v.key_replace(stats_dict)

        if self.has_default:
            self.default_accumulator.key_replace(stats_dict)

        if self.has_given:
            self.given_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "ConditionalDistributionDataEncoder":

        encoder_map = {k: v.acc_to_encoder() for k, v in self.accumulator_map.items()}
        default_encoder = self.default_accumulator.acc_to_encoder()
        given_encoder = self.given_accumulator.acc_to_encoder()

        return ConditionalDistributionDataEncoder(
            encoder_map=encoder_map,
            default_encoder=default_encoder,
            given_encoder=given_encoder,
        )


class ConditionalDistributionAccumulatorFactory(TorchStatisticAccumulatorFactory):
    """ConditionalDistributionAccumulatorFactory creates ConditionalDistributionAccumulator objects.

    Attributes:
        factory_map (Dict[T0, TorchStatisticAccumulatorFactory]): Dictionary of TorchStatisticAccumulatorFactory objects for
            creating TorchStatisticAccumulator objects in ConditionalDistributionAccumulator
        default_factory (TorchStatisticAccumulatorFactory): Used to create TorchStatisticAccumulator for
            defualt_accumulator in ConditionalDistributionAccumulator.
        given_factory (TorchStatisticAccumulatorFactory): Used to create TorchStatisticAccumulator for
            given_accumulator in ConditionalDistributionAccumulator.
        keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

    """

    def __init__(
        self,
        factory_map: Dict[T0, TorchStatisticAccumulatorFactory],
        default_factory: TorchStatisticAccumulatorFactory = NullAccumulatorFactory(),
        given_factory: TorchStatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: Optional[str] = None,
    ) -> None:
        """ConditionalDistributionAccumulatorFactory object.

        Args:
            factory_map (Dict[T0, TorchStatisticAccumulatorFactory]): Dictionary of TorchStatisticAccumulatorFactory objects for
                creating TorchStatisticAccumulator objects in ConditionalDistributionAccumulator
            default_factory (TorchStatisticAccumulatorFactory): Used to create TorchStatisticAccumulator for
                defualt_accumulator in ConditionalDistributionAccumulator.
            given_factory (TorchStatisticAccumulatorFactory): Used to create TorchStatisticAccumulator for
                given_accumulator in ConditionalDistributionAccumulator.
            keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

        """
        self.factory_map = factory_map
        self.default_factory = default_factory
        self.given_factory = given_factory
        self.keys = keys

    def make(
        self, device: Optional[tn.device] = None
    ) -> "ConditionalDistributionAccumulator":

        acc = {k: v.make() for k, v in self.factory_map.items()}
        def_acc = self.default_factory.make()
        given_acc = self.given_factory.make()

        return ConditionalDistributionAccumulator(
            acc, def_acc, given_acc, self.keys, device=device
        )


class ConditionalDistributionEstimator(TorchParameterEstimator):
    """ConditionalDistributionEstimator object used to estimate ConditionalDistribution from aggregated data.

    Notes:
        If None is passed for default_estimator, default_estimator is set to NullEstimator().
        If None is passed for given_estimator, given_estimator is set to NullEstimator().

    Attributes:
        estimator_map (Dict[T0, TorchParameterEstimator]):
        default_estimator (TorchParameterEstimator): TorchParameterEstimator for default_distribution set to NullEstimator,
            if None is passed as arg.
        given_estimator (TorchParameterEstimator): TorchParameterEstimator for given_distribution set to NullEstimator
            if None is passed as arg.
        keys (Optional[str]): ConditionalDistributionEstimator with matching 'keys' will be aggregated.

    """

    def __init__(
        self,
        estimator_map: Dict[T0, TorchParameterEstimator],
        default_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
        given_estimator: Optional[TorchParameterEstimator] = NullEstimator(),
        keys: Optional[str] = None,
    ) -> None:
        """ConditionalDistributionEstimator object.

        Args:
            estimator_map (Dict[T0, TorchParameterEstimator]):
            default_estimator (Optional[TorchParameterEstimator]): TorchParameterEstimator for default_distribution, can be None.
            given_estimator (Optional[TorchParameterEstimator]): TorchParameterEstimator for given_distribution, can be None.
            keys (Optional[str]): ConditionalDistributionEstimator with matching 'keys' will be aggregated.

        """
        self.estimator_map = estimator_map
        self.default_estimator = (
            default_estimator if default_estimator is not None else NullEstimator()
        )
        self.keys = keys
        self.given_estimator = (
            given_estimator if given_estimator is not None else NullEstimator()
        )

    def accumulator_factory(self) -> "ConditionalDistributionAccumulatorFactory":
        emap_items = {k: v.accumulator_factory() for k, v in self.estimator_map.items()}
        def_factory = self.default_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()

        return ConditionalDistributionAccumulatorFactory(
            emap_items, def_factory, given_factory, self.keys
        )

    def estimate(
        self,
        nobs: Optional[float],
        suff_stat: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS2]],
        device: Optional[tn.device] = None,
    ) -> "ConditionalDistribution":
        """Estimate a ConditionalDistribution from aggregated data.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency.
            suff_stat: See description above.
            device (Optional[tn.device]): Device to declare new estimated model on.

        Returns:
            ConditionalDistribution

        """
        default_dist = self.default_estimator.estimate(
            None, suff_stat[1], device=device
        )
        given_dist = self.given_estimator.estimate(None, suff_stat[2], device=device)
        dist_map = {
            k: self.estimator_map[k].estimate(None, v) for k, v in suff_stat[0].items()
        }

        return ConditionalDistribution(
            dist_map,
            default_dist=default_dist,
            given_dist=given_dist,
            keys=self.keys,
            device=device,
        )


class ConditionalDistributionDataEncoder(TorchSequenceEncoder):
    """ConditionalDistributionDataEncoder used to encode sequence of data.

    Notes:
        Data type should be Tuple[T0, T1] where T0 is the type of the conditional value in ConditionalDistribution.
        I.e.,
        p_mat(X1|X0), should have x_mat as type T0, and Y as type T1.

    Attributes:
        encoder_map (Dict[T0, TorchSequenceEncoder]): Dictionary of TorchSequenceEncoder objects for each conditional
            value of data type T0. Data types of the encoders must be of type T1.
        default_encoder (TorchSequenceEncoder): TorchSequenceEncoder compatible with data type T1.
        given_encoder (TorchSequenceEncoder): TorchSequenceEncoder compatible with data type T0.
        null_default_encoder (bool): True if default_encoder is instance of NullDataEncoder, else false.
        null_given_encoder (bool): True if default_encoder is instance of NullDataEncoder, else false.

    """

    def __init__(
        self,
        encoder_map: Dict[T0, TorchSequenceEncoder],
        default_encoder: TorchSequenceEncoder = NullDataEncoder(),
        given_encoder: TorchSequenceEncoder = NullDataEncoder(),
    ) -> None:
        """ConditionalDistributionDataEncoder object.

        Args:
            encoder_map (Dict[T0, TorchSequenceEncoder]): Dictionary of TorchSequenceEncoder objects for each conditional
                value of data type T0. Data types of the encoders must be of type T1.
            default_encoder (TorchSequenceEncoder): TorchSequenceEncoder compatible with data type T1.
            given_encoder ((TorchSequenceEncoder): TorchSequenceEncoder compatible with data type T0.

        """
        self.encoder_map = encoder_map
        self.default_encoder = default_encoder
        self.given_encoder = given_encoder

        self.null_default_encoder = isinstance(self.default_encoder, NullDataEncoder)
        self.null_given_encoder = isinstance(self.given_encoder, NullDataEncoder)

    def __str__(self) -> str:

        encoder_items = list(self.encoder_map.items())
        encoder_str = "ConditionalDataEncoder("
        for k, v in encoder_items[:-1]:
            encoder_str += str(k) + ":" + str(v) + ","
        encoder_str += str(encoder_items[-1][0]) + ":" + str(encoder_items[-1][1])

        if not self.null_default_encoder:
            encoder_str += ",default=" + str(self.default_encoder)
        else:
            encoder_str += ",default=None"

        if not self.null_given_encoder:
            encoder_str += ",given=" + str(self.given_encoder)
        else:
            encoder_str += ",given=None)"

        return encoder_str

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, ConditionalDistributionDataEncoder):
            return False
        else:
            if not self.encoder_map == other.encoder_map:
                return False

            if not self.default_encoder == other.default_encoder:
                return False

            if not self.given_encoder == other.given_encoder:
                return False

        return True

    def seq_encode(
        self, x: List[Tuple[T0, T1]], device: Optional[tn.device] = None
    ) -> "ConditionalTorchEncodedSequence":
        """Encode sequence of iid observations from ConditionalDistribution for vectorized "seq_" function calls.

        Notes:
            Data must be a List of Tuple of two types, T0 and T1. T0 is the data type compatible with the conditional
            values of the ConditionalDistribution. T1 must be consistent with the data type of the conditional
            distributions.

            E Tuple of length 5:
                E[0] (int): length of x (i.e. total observations).
                E[1] (Tuple[T0]): Unique conditional values in data.
                E[2] (Tuple[Encoded[T1]): Tuple of sequence encoded data of type T1 encoded by
                    encoder_map[key] or default_encoder if key not in default_encoder and default_encoder is not
                    the NullDataEncoder.
                E[3] (Tuple[tn.Tensor,...]): Tuple of length equal to the number of unique conditional
                    values encountered in the data. Each entry contains a numpy array for the indices of x that correspond
                    to a unique conditional value.
                E[4] (Optional[Encoded[T0]]): If the given_encoder is not the NullDataEncoder, the
                    observed conditional values of data type T0 are sequence encoded by given_encoder. Else return None.

        Args:
            x (List[Tuple[T0, T1]]): List of data observations.
            device (Optional[tn.device]): Device to write tensors to.

        Returns:
            ConditionalTorchEncodedSequence

        """
        cond_enc = dict()
        given_vals = []

        for i in range(len(x)):
            xx = x[i]
            given_vals.append(xx[0])
            if xx[0] not in cond_enc:
                cond_enc[xx[0]] = [[xx[1]], [i]]
            else:
                cond_enc_loc = cond_enc[xx[0]]
                cond_enc_loc[0].append(xx[1])
                cond_enc_loc[1].append(i)

        cond_enc_items = list(cond_enc.items())
        cond_vals = tuple([u[0] for u in cond_enc_items])

        eobs_vals = []
        idx_vals = []

        for u in cond_enc_items:
            if self.null_default_encoder:
                if u[0] in self.encoder_map:
                    eobs_vals.append(
                        self.encoder_map[u[0]].seq_encode(u[1][0], device=device)
                    )
            else:
                eobs_vals.append(
                    self.encoder_map.get(u[0], self.default_encoder).seq_encode(
                        u[1][0], device=device
                    )
                )

            idx_vals.append(vec.int_tensor(u[1][1], device=device))

        given_enc = self.given_encoder.seq_encode(given_vals, device=device)

        return ConditionalTorchEncodedSequence(
            data=(len(x), cond_vals, tuple(eobs_vals), tuple(idx_vals), given_enc),
            device=device,
        )


class ConditionalTorchEncodedSequence(TorchEncodedSequence):

    def __init__(
        self,
        data: Tuple[
            int,
            Tuple[Any, ...],
            Tuple[TorchEncodedSequence, ...],
            Tuple[tn.tensor, ...],
            TorchEncodedSequence,
        ],
        device: Optional[tn.device] = None,
    ):
        super().__init__(data=data, device=device)

    def __str__(self) -> str:

        return f"ConditionalTorchEncodedSequence(device={repr(self.device)})"
