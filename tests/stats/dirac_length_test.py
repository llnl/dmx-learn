"""Tests for DiracMixtureDistribution and related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.categorical import *
from dmx.stats.binomial import * 
from dmx.stats.dirac_length import * 
import numpy as np

def component_log_density_test(dist: DiracMixtureDistribution, encoder: DiracMixtureDataEncoder):
    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:
        s = dist.sampler(seed)
        data = s.sample(size=sz)
        enc_data = encoder.seq_encode(data)
        seq_comp_ll = dist.seq_component_log_density(enc_data)
        lower_bound = -1.0e32
        seq_comp_ll = np.maximum(seq_comp_ll, lower_bound)
        for i in range(sz):
            # handle infs 
            tmp = np.maximum(dist.component_log_density(data[i]), lower_bound)
            seq_comp_ll[i] = np.abs(seq_comp_ll[i] - tmp)
        rv.append(np.max(seq_comp_ll))

    return max(rv) < 1.0e-14, "max(rv) test"

def posterior_test(dist: DiracMixtureDistribution, encoder: DiracMixtureEncodedDataSequence):
    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:
        s = dist.sampler(seed)
        data = s.sample(size=sz)
        enc_data = encoder.seq_encode(data)
        seq_post = dist.seq_posterior(enc_data)

        for i in range(sz):
            if np.all(seq_post[i] == 0):
                seq_post[i] = np.abs(dist.posterior(data[i]))
            else:
                seq_post[i] = np.abs(seq_post[i] - dist.component_log_density(data[i])) / np.abs(seq_post[i])
        rv.append(np.max(seq_post))

    return max(rv) < 1.0e-14, "max(rv) test"


class DiracMixtureDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        self._base_dists = [BinomialDistribution(p=0.9, n=5), CategoricalDistribution({3: 0.1, 4: 0.8, 5: 0.1})]
        self._base_ests = [BinomialEstimator(), CategoricalEstimator()]
        self._base_encoders = [BinomialDataEncoder(), CategoricalDataEncoder()]
        self._base_accs = [BinomialAccumulator(), CategoricalAccumulator()]
        self._base_factories = [BinomialAccumulatorFactory(), CategoricalAccumulatorFactory()]

        self.eval_dists = [
            DiracMixtureDistribution(dist=self._base_dists[0], p=0.20, v=0, name='a', keys=('w', 'comps')),
            DiracMixtureDistribution(dist=self._base_dists[0], p=0.20, v=100, name='a', keys=('w', None)),
            DiracMixtureDistribution(dist=self._base_dists[0], p=0.20, v=0, name='a', keys=(None, 'comps')),
            DiracMixtureDistribution(dist=self._base_dists[0], p=0.20),
            DiracMixtureDistribution(dist=self._base_dists[1], p=0.20, v=0, name='a', keys=('w', 'comps')),
            DiracMixtureDistribution(dist=self._base_dists[1], p=0.20, v=100, name='a', keys=('w', None)),
            DiracMixtureDistribution(dist=self._base_dists[1], p=0.20, v=0, name='a', keys=(None, 'comps')),
            DiracMixtureDistribution(dist=self._base_dists[1], p=0.20)
        ]

        self._ests = [
            DiracMixtureEstimator(estimator=self._base_ests[0], v=0, name='a', keys=('w', 'comps')),
            DiracMixtureEstimator(estimator=self._base_ests[0], v=100, name='a', keys=('w', None)),
            DiracMixtureEstimator(estimator=self._base_ests[0], v=0, name='a', keys=(None, 'comps')),
            DiracMixtureEstimator(estimator=self._base_ests[0]),
            DiracMixtureEstimator(estimator=self._base_ests[1], v=0, name='a', keys=('w', 'comps')),
            DiracMixtureEstimator(estimator=self._base_ests[1], v=100, name='a', keys=('w', None)),
            DiracMixtureEstimator(estimator=self._base_ests[1], v=0, name='a', keys=(None, 'comps')),
            DiracMixtureEstimator(estimator=self._base_ests[1])
        ]
        self._factories = [
            DiracMixtureAccumulatorFactory(factory=self._base_factories[0], v=0, name='a', keys=('w', 'comps')),
            DiracMixtureAccumulatorFactory(factory=self._base_factories[0], v=100, name='a', keys=('w', None)),
            DiracMixtureAccumulatorFactory(factory=self._base_factories[0], v=0, name='a', keys=(None, 'comps')),
            DiracMixtureAccumulatorFactory(factory=self._base_factories[0]),
            DiracMixtureAccumulatorFactory(factory=self._base_factories[1], v=0, name='a', keys=('w', 'comps')),
            DiracMixtureAccumulatorFactory(factory=self._base_factories[1], v=100, name='a', keys=('w', None)),
            DiracMixtureAccumulatorFactory(factory=self._base_factories[1], v=0, name='a', keys=(None, 'comps')),
            DiracMixtureAccumulatorFactory(factory=self._base_factories[1])
        ]
        self._accumulators = [
            DiracMixtureAccumulator(accumulator=self._base_accs[0], v=0, name='a', keys=('w', 'comps')),
            DiracMixtureAccumulator(accumulator=self._base_accs[0], v=100, name='a', keys=('w', None)),
            DiracMixtureAccumulator(accumulator=self._base_accs[0], v=0, name='a', keys=(None, 'comps')),
            DiracMixtureAccumulator(accumulator=self._base_accs[0]),
            DiracMixtureAccumulator(accumulator=self._base_accs[1], v=0, name='a', keys=('w', 'comps')),
            DiracMixtureAccumulator(accumulator=self._base_accs[1], v=100, name='a', keys=('w', None)),
            DiracMixtureAccumulator(accumulator=self._base_accs[1], v=0, name='a', keys=(None, 'comps')),
            DiracMixtureAccumulator(accumulator=self._base_accs[1])
        ]
        self._encoders = [
            DiracMixtureDataEncoder(encoder=self._base_encoders[0], v=0),
            DiracMixtureDataEncoder(encoder=self._base_encoders[0], v=100),
            DiracMixtureDataEncoder(encoder=self._base_encoders[0], v=0),
            DiracMixtureDataEncoder(encoder=self._base_encoders[0], v=0),
            DiracMixtureDataEncoder(encoder=self._base_encoders[1], v=0),
            DiracMixtureDataEncoder(encoder=self._base_encoders[1], v=100),
            DiracMixtureDataEncoder(encoder=self._base_encoders[1], v=0),
            DiracMixtureDataEncoder(encoder=self._base_encoders[1], v=0)
        ]

        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.sampler_dist = self.eval_dists[0]
        self.density_dist_encoder = self.dist_encoder
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder  = [(a, e) for a, e in zip(self._accumulators, self._encoders)]
        self.type_check_keys = [None, 'keys', (None, None, None), (1, 'keys')]
        
    def test_component_log_density(self):
        for x in self.density_dist_encoder:
            self.assertTrue(component_log_density_test(*x))

    def test_posterior(self):
        for x in self.density_dist_encoder:
            self.assertTrue(posterior_test(*x))

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                DiracMixtureEstimator([CategoricalEstimator()]*5, keys=x)
                
            assert str(e.value) == "DiracMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]])."


