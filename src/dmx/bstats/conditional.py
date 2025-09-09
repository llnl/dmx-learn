from typing import Dict, Optional, Tuple, Sequence, TypeVar
from dmx.arithmetic import maxint
from dmx.bstats.pdist import ProbabilityDistribution, SequenceEncodableAccumulator, \
    ParameterEstimator, EncodedDataSequence, DataSequenceEncoder
from dmx.bstats.nulldist import NullDataEncoder, NullDistribution
from numpy.random import RandomState
import numpy as np
from dmx.bstats.nulldist import null_dist
T1, T0 = TypeVar('T1'), TypeVar('T0')


class ConditionalDistribution(ProbabilityDistribution):

    def __init__(self, dmap, cond_dist, default_dist=null_dist, pass_value=False):

        self.dmap = dmap
        self.cond_dist = cond_dist
        self.default_dist = default_dist
        self.pass_value = pass_value

    def __str__(self):
        return 'ConditionalDistribution(%s, default_dist=%s)' % (
        str({k: str(v) for k, v in self.dmap.items()}), str(self.default_dist))

    def log_density(self, x):
        if self.pass_value:
            return self.dmap.get(x[0], self.default_dist).log_density(x)
        else:
            return self.dmap.get(x[0], self.default_dist).log_density(x[1])

    def seq_log_density(self, x):

        sz, cond_vals, idx_vals, eobs_vals = x

        rv = np.zeros(sz)

        if self.has_default:
            for i in range(len(cond_vals)):
                rv[idx_vals[i]] = self.dmap.get(cond_vals[i], self.default_dist).seq_log_density(eobs_vals[i])
        else:
            for i in range(len(cond_vals)):
                if cond_vals[i] in self.dmap:
                    rv[idx_vals[i]] = self.dmap[cond_vals[i]].seq_log_density(eobs_vals[i])
                else:
                    rv[idx_vals[i]] = -np.inf

        return rv

    def seq_encode(self, x):

        cond_enc = dict()

        for i in range(len(x)):
            xx = x[i]
            vv = xx if self.pass_value else xx[1]
            if xx[0] not in cond_enc:
                cond_enc[xx[0]] = [[vv], [i]]
            else:
                cond_enc_loc = cond_enc[xx[0]]
                cond_enc_loc[0].append(vv)
                cond_enc_loc[1].append(i)

        cond_enc = list(cond_enc.items())

        cond_vals = tuple([u[0] for u in cond_enc])
        eobs_vals = tuple([self.dmap.get(u[0], self.default_dist).seq_encode(u[1][0]) for u in cond_enc])
        idx_vals = tuple([np.asarray(u[1][1]) for u in cond_enc])

        return len(x), cond_vals, idx_vals, eobs_vals

    def sampler(self, seed=None):
        pass

    def estimator(self, pseudo_count=None):
        pass

    def dist_to_encoder(self) -> 'ConditionalDataEncoder':
        e0 = {k: v.dist_to_encoder() for k, v in self.dmap.items()}
        e1 = self.cond_dist.dist_to_encoder()
        e2 = self.default_dist.dist_to_encoder()

        return ConditionalDataEncoder(
                encoder_map=e0,
                given_encoder=e1,
                default_encoder=e2)


class ConditionalDistributionSampler(object):
    def __init__(self, dist, seed=None):
        pass

    def sample(self, size=None):
        pass


class ConditionalDistributionEstimatorAccumulator(SequenceEncodableAccumulator):
    # TODO: add given accumulator
    def __init__(self, accumulator_map, default_accumulator, keys=None):
        self.accumulator_map = accumulator_map
        self.default_accumulator = default_accumulator
        self.key = keys

    def update(self, x, weight, estimate):
        if x[0] in self.accumulator_map:
            self.accumulator_map[x[0]].update(x[1], weight, estimate.dmap[x[0]])
        else:
            if self.default_accumulator is not None:
                self.default_accumulator.update(x[1], weight, estimate.default_dist)

    def initialize(self, x, weight, rng):
        if x[0] in self.accumulator_map:
            self.accumulator_map[x[0]].initialize(x[1], weight, rng)
        else:
            if self.default_accumulator is not None:
                self.default_accumulator.initialize(x[1], weight, rng)

    def seq_update(self, x, weights, estimate):
        sz, cond_vals, idx_vals, eobs_vals = x

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_update(eobs_vals[i], weights[idx_vals[i]],
                                                              estimate.dmap[cond_vals[i]])
            else:
                if self.default_accumulator is not None:
                    self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], estimate.default_dist)

    def combine(self, suff_stat):

        for k, v in suff_stat[0].items():
            if k in self.accumulator_map:
                self.accumulator_map[k].combine(v)
            else:
                self.accumulator_map[k] = v

        if self.default_accumulator is not None and suff_stat[1] is not None:
            self.default_accumulator.combine(suff_stat[1])

        return self

    def value(self):
        rv2 = None if self.default_accumulator is None else self.default_accumulator.value()
        rv1 = {k: v.value() for k, v in self.accumulator_map.items()}
        return rv1, rv2

    def from_value(self, x):
        for k, v in x[0].items():
            self.accumulator_map[k].from_value(v)

        if self.default_accumulator is not None and x[1] is not None:
            self.default_accumulator.from_value(x[1])

        return self

    def key_merge(self, stats_dict):
        for k, v in self.accumulator_map.items():
            v.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        for k, v in self.accumulator_map.items():
            v.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'ConditionalDataEncoder':
        e0 = {k: v.acc_to_encoder() for k, v in self.accumulator_map.items()}
        e1 = NullDataEncoder()
        e2 = self.default_accumulator.acc_to_encoder()

        return ConditionalDataEncoder(
                encoder_map=e0,
                given_encoder=e1,
                default_encoder=e2)


class ConditionalDistributionEstimator(ParameterEstimator):
    def __init__(self, estimator_map, default_estimator=None, keys=None):
        self.estimator_map = estimator_map
        self.default_estimator = default_estimator
        self.keys = keys

    def accumulator_factory(self):
        emap_items = self.estimator_map.items()

        obj = type('', (object,), {'make': lambda o: ConditionalDistributionEstimatorAccumulator(
            {k: v.accumulator_factory().make() for k, v in emap_items},
            None if self.default_estimator is None else self.default_estimator.accumulator_factory().make(),
            self.keys)})()
        # def makeL():
        #	return(CompositeEstimatorAccumulator([x.accumulatorFactory().make() for x in self.estimators]))
        # obj = AccumulatorFactory(makeL)
        return (obj)

    def estimate(self, suff_stat):

        if self.default_estimator is not None:
            default_dist = self.default_estimator.estimate(suff_stat[1])
        else:
            default_dist = None

        dist_map = {k: self.estimator_map[k].estimate(v) for k, v in suff_stat[0].items()}

        return ConditionalDistribution(dist_map, default_dist)


class ConditionalDataEncoder(DataSequenceEncoder):
    def __init__(self,
                 encoder_map: Dict[T0, DataSequenceEncoder],
                 given_encoder: DataSequenceEncoder,
                 default_encoder: DataSequenceEncoder):
        self.encoder_map = encoder_map
        self.default_encoder = default_encoder
        self.given_encoder = given_encoder

        self.null_default_encoder = isinstance(self.default_encoder, NullDataEncoder)
        self.null_given_encoder = isinstance(self.given_encoder, NullDataEncoder)

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, ConditionalDataEncoder):
            return False
        else:
            if not self.encoder_map == other.encoder_map:
                return False

            if not self.default_encoder == other.default_encoder:
                return False

            if not self.given_encoder == other.given_encoder:
                return False

        return True

    def __str__(self) -> str:

        encoder_items = list(self.encoder_map.items())
        encoder_str = 'ConditionalDataEncoder('
        for k, v in encoder_items[:-1]:
            encoder_str += str(k) + ':' + str(v) + ','
        encoder_str += str(encoder_items[-1][0]) + ':' + str(encoder_items[-1][1])

        if not self.null_default_encoder:
            encoder_str += ',default=' + str(self.default_encoder)
        else:
            encoder_str += ',default=None'

        if not self.null_given_encoder:
            encoder_str += ',given=' + str(self.given_encoder)
        else:
            encoder_str += ',given=None)'

        return encoder_str

    def seq_encode(self, x: Sequence[Tuple[T0, T1]]) -> 'ConditionalEncodedData':
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
                    eobs_vals.append(self.encoder_map[u[0]].seq_encode(u[1][0]))
            else:
                eobs_vals.append(self.encoder_map.get(u[0], self.default_encoder).seq_encode(u[1][0]))

            idx_vals.append(np.asarray(u[1][1]))

        given_enc = self.given_encoder.seq_encode(given_vals)

        return ConditionalEncodedData(data=(len(x), cond_vals, tuple(eobs_vals), tuple(idx_vals), given_enc))


class ConditionalEncodedData(EncodedDataSequence):
    def __init__(self, data: Tuple[
        int, Tuple[T0, ...], Tuple[EncodedDataSequence], Tuple[np.ndarray], Optional[EncodedDataSequence]]):
        """ConditionalEncodedDataSequence object.

        Args:
            data (Tuple[int, Tuple[T0, ...], Tuple[EncodedDataSequence], Tuple[np.ndarray], Optional[EncodedDataSequence]]): see above.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'ConditionalEncodedDataSequence(data={self.data})'
