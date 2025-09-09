import numpy as np
from typing import Optional, Sequence, Tuple, Any
from dmx.arithmetic import maxint
from numpy.random import RandomState
from dmx.bstats.pdist import (ParameterEstimator,
                               ProbabilityDistribution,
                               StatisticAccumulator,
                               SequenceEncodableAccumulator,
                               DataFrameEncodableDistribution,
                               DataFrameEncodableAccumulator,
                               DataSequenceEncoder,
                               EncodedDataSequence)


class CompositeDistribution(ProbabilityDistribution):

    def __init__(self, dists, name: Optional[str] = None, keys: Optional[str] = None):
        self.dists = dists
        self.count = len(dists)
        self.keys  = keys
        self.set_name(name)

		#self.parents = []
		#for d in dists:
		#	d.add_parent(self)

    def __str__(self):
        return 'CompositeDistribution((%s))' % (','.join(map(str, self.dists)))

    def get_name(self):
        return ','.join([u.get_name() for u in self.dists])

    def get_parameters(self):
        return tuple([d.get_parameters() for d in self.dists])

    def set_parameters(self, params):
        for d,p in zip(self.dists, params):
            d.set_parameters(p)

    def get_prior(self):
        return CompositeDistribution([d.get_prior() for d in self.dists])

    def set_prior(self, prior):
        for d,p in zip(self.dists, prior.dists):
            d.set_prior(p)

    def cross_entropy(self, dist):
        if isinstance(dist, CompositeDistribution):
            # no name checking right now...
            rv = 0
            for u,v in zip(self.dists, dist.dists):
                rv += u.cross_entropy(v)
            return rv
        else:
            rv = 0
            for u in self.dists:
                rv += u.cross_entropy(dist)
            return rv

    def entropy(self):
        rv = 0
        for u in self.dists:
            rv += u.entropy()
        return rv

    def log_density(self, x):
        rv = self.dists[0].log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].log_density(x[i])

        return rv

    def expected_log_density(self, x):
        rv = self.dists[0].expected_log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].expected_log_density(x[i])

        return rv

    def seq_encode(self, x):
        return tuple([self.dists[i].seq_encode([u[i] for u in x]) for i in range(self.count)])

    def seq_log_density(self, x):
        rv = self.dists[0].seq_log_density(x[0])
        for i in range(1, self.count):
            rv += self.dists[i].seq_log_density(x[i])

        return rv

    def seq_expected_log_density(self, x):
        rv = self.dists[0].seq_expected_log_density(x[0])
        for i in range(1, self.count):
            rv += self.dists[i].seq_expected_log_density(x[i])

        return rv

    def df_log_density(self, df):
        rv = self.dists[0].df_log_density(df)
        for i in range(1, self.count):
            rv += self.dists[i].df_log_density(df)

        return rv

    def sampler(self, seed=None):
        return CompositeSampler(self, seed)

    def estimator(self):
        return CompositeEstimator([d.estimator() for d in self.dists])

    def dist_to_encoder(self) -> 'CompositeDataEncoder':

        encoders = tuple([d.dist_to_encoder() for d in self.dists])

        return CompositeDataEncoder(encoders=encoders)

class CompositeSampler(object):

	def __init__(self, dist, seed=None):
		self.dist         = dist
		self.rng          = RandomState(seed)
		self.distSamplers = [d.sampler(seed=self.rng.randint(maxint)) for d in dist.dists]

	def sample(self, size=None):

		if size is None:
			return tuple([d.sample(size=size) for d in self.distSamplers])
		else:
			return list(zip(*[d.sample(size=size) for d in self.distSamplers]))


class CompositeEstimatorAccumulator(SequenceEncodableAccumulator, DataFrameEncodableAccumulator):

    def __init__(self, accumulators, keys=None):
        self.accumulators = accumulators
        self.count        = len(accumulators)
        self.key          = keys

    def update(self, x, weight, estimate):
        if estimate is not None:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, estimate.dists[i])
        else:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, None)

    def initialize(self, x, weight, rng):
        for i in range(0, self.count):
            self.accumulators[i].initialize(x[i], weight, rng)


    def seq_initialize(self, x, weights, rng):
        for i in range(self.count):
            self.accumulators[i].seq_initialize(x[i], weights, rng)

    def seq_update(self, x, weights, estimate):
        for i in range(self.count):
            self.accumulators[i].seq_update(x[i], weights, estimate.dists[i])

    def df_initialize(self, df, weights, rng):
        for i in range(self.count):
            self.accumulators[i].df_initialize(df, weights, rng)

    def df_update(self, df, weights, estimate):
        if estimate is None:
            for i in range(self.count):
                self.accumulators[i].df_update(df, weights, None)
        else:
            for i in range(self.count):
                self.accumulators[i].df_update(df, weights, estimate.dists[i])

    def combine(self, suff_stat):
        for i in range(0, self.count):
            self.accumulators[i].combine(suff_stat[i])
        return self

    def value(self):
        return tuple([x.value() for x in self.accumulators])

    def from_value(self, x):
        self.accumulators = [self.accumulators[i].from_value(x[i]) for i in range(len(x))]
        self.count = len(x)
        return self

    def key_merge(self, stats_dict):

        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict):

        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'CompositeDataEncoder':

        encoders = tuple([acc.acc_to_encoder() for acc in self.accumulators])

        return CompositeDataEncoder(encoders=encoders)

class CompositeAccumulatorFactory():

    def __init__(self, factories, keys):
        self.factories = factories
        self.keys = keys

    def make(self) -> CompositeEstimatorAccumulator:
        return CompositeEstimatorAccumulator([f.make() for f in self.factories], keys=self.keys)


class CompositeEstimator(ParameterEstimator):

    def __init__(self, estimators, name: Optional[str] = None, keys: Optional[str] = None):

        self.estimators  = estimators
        self.count       = len(estimators)
        self.keys        = keys
        self.name        = name

    def get_prior(self):
        return CompositeDistribution([d.get_prior() for d in self.estimators], name=self.keys)

    def set_prior(self, params):
        for d,p in zip(self.estimators, params.dists):
            d.set_prior(p)

    def accumulator_factory(self):
        obj = type('', (object,), {'make': lambda o: CompositeEstimatorAccumulator([x.accumulator_factory().make() for x in self.estimators], self.keys)})()
        #def makeL():
        #	return(CompositeEstimatorAccumulator([x.accumulatorFactory().make() for x in self.estimators]))
        #obj = AccumulatorFactory(makeL)
        return(obj)

    def model_log_density(self, model: CompositeDistribution) -> float:
        return self.get_prior().log_density(model.get_parameters())

    def estimate(self, suff_stat):
        return CompositeDistribution(tuple([est.estimate(ss) for est, ss in zip(self.estimators, suff_stat)]), name=self.name, keys=self.keys)


class CompositeDataEncoder(DataSequenceEncoder):
    def __init__(self, encoders: Sequence[DataSequenceEncoder]):
        self.encoders = encoders

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeDataEncoder):
            return False

        else:

            for i, encoder in enumerate(self.encoders):
                if not encoder == other.encoders[i]:
                    return False

        return True

    def __str__(self) -> str:
        s = 'CompositeDataEncoder(['

        for d in self.encoders[:-1]:
            s += str(d) + ','

        s += str(self.encoders[-1]) + '])'

        return s

    def seq_encode(self, x: Sequence[Tuple[Any, ...]]) -> 'CompositeEncodedData':
        enc_data = []

        for i, encoder in enumerate(self.encoders):
            enc_data.append(encoder.seq_encode([u[i] for u in x]))

        return CompositeEncodedData(data=tuple(enc_data))


class CompositeEncodedData(EncodedDataSequence):

    def __init__(self, data: Tuple[EncodedDataSequence, ...]):
        super().__init__(data)

    def __repr__(self) -> str:
        return f'CompositeEncodedDataSequence(data={self.data})'

