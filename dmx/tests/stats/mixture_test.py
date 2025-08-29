"""Test cases for Mixture Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.categorical import *
from dmx.stats.binomial import * 
from dmx.stats.gaussian import * 
from dmx.stats.mixture import * 
import numpy as np

def component_log_density_test(dist, encoder):
    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:
        s = dist.sampler(seed)
        data = s.sample(size=sz)
        enc_data = encoder.seq_encode(data)
        seq_comp_ll = dist.seq_component_log_density(enc_data)

        for i in range(sz):
            
            if np.all(seq_comp_ll[i] == 0):
                seq_comp_ll[i] = np.abs(dist.component_log_density(data[i]))
            else:
                seq_comp_ll[i] = np.abs(seq_comp_ll[i] - dist.component_log_density(data[i])) / np.abs(seq_comp_ll[i])
        rv.append(np.max(seq_comp_ll))

    return max(rv) < 1.0e-14, "max(rv) test"

def posterior_test(dist, encoder):
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


class MixtureDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        
        self._comps = [
            [CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}), 
            CategoricalDistribution(pmap={'a': 0.6, 'b': 0.1, 'c': 0.3}),
            CategoricalDistribution(pmap={'a': 0.3, 'b': 0.6, 'c': 0.1})],
            [CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, name='c'), 
            CategoricalDistribution(pmap={'a': 0.6, 'b': 0.1, 'c': 0.3}, name='c'),
            CategoricalDistribution(pmap={'a': 0.3, 'b': 0.6, 'c': 0.1}, name='c')]

        ]
        self._encoder = MixtureDataEncoder(encoder=CategoricalDataEncoder())
        self._estimators = [CategoricalEstimator(), CategoricalEstimator(name='c'), CategoricalEstimator(keys='c_keys')]
        self._factories = [CategoricalAccumulatorFactory(), CategoricalAccumulatorFactory(name='c'), CategoricalAccumulatorFactory(keys='c_keys')]
        self._accumulators = [CategoricalAccumulator(), CategoricalAccumulator(name='c'), CategoricalAccumulator(keys='c_keys')]

        self.eval_dists = [
            MixtureDistribution(components=self._comps[0], w=[1/3, 4/9, 2/9], name='mix'),
            MixtureDistribution(components=self._comps[0][:2], w=[0.20, 0.80], name='mix'),
            MixtureDistribution(components=self._comps[0][:1], w=[1]*1, name='mix'),
            MixtureDistribution(components=self._comps[1], w=[1/3.]*3, name='mix'),
            MixtureDistribution(components=self._comps[1][:2], w=[1/2.]*2, name='mix'),
            MixtureDistribution(components=self._comps[1][:1], w=[1]*1, name='mix')

        ]
        self.dist_est = [
            (self.eval_dists[0], MixtureEstimator(estimators=[self._estimators[0]]*3, name='mix')),
            (self.eval_dists[1], MixtureEstimator(estimators=[self._estimators[0]]*2, name='mix')),
            (self.eval_dists[2], MixtureEstimator(estimators=[self._estimators[0]]*1, name='mix')),
            (self.eval_dists[3], MixtureEstimator(estimators=[self._estimators[1]]*3, name='mix')),
            (self.eval_dists[4], MixtureEstimator(estimators=[self._estimators[1]]*2, name='mix')),
            (self.eval_dists[5], MixtureEstimator(estimators=[self._estimators[1]]*1, name='mix'))
        ]
        
        self.dist_encoder = [(self.eval_dists[0], self._encoder)]
        self.sampler_dist = self.eval_dists[0]
        self.density_dist_encoder = [(d, self._encoder) for d in self.eval_dists[:3]]
        self.est_factory = [
            (MixtureEstimator(estimators=[self._estimators[0]]*3, name='m', keys=('w', None)), MixtureAccumulatorFactory(factories=[self._factories[0]]*3, name='m', keys=('w', None))),
            (MixtureEstimator(estimators=[self._estimators[0]]*1, name='m', keys=(None, 'comps')), MixtureAccumulatorFactory(factories=[self._factories[0]]*1, name='m', keys=(None, 'comps'))),
            (MixtureEstimator(estimators=[self._estimators[1]]*3, name='m', keys=('w', 'comps')), MixtureAccumulatorFactory(factories=[self._factories[1]]*3, name='m', keys=('w', 'comps'))),
            (MixtureEstimator(estimators=[self._estimators[1]]*1, name='m'), MixtureAccumulatorFactory(factories=[self._factories[1]]*1, name='m', keys=(None, None))),
            (MixtureEstimator(estimators=[self._estimators[2]]*3), MixtureAccumulatorFactory(factories=[self._factories[2]]*3)),
            (MixtureEstimator(estimators=[self._estimators[2]]*1, name='m'), MixtureAccumulatorFactory(factories=[self._factories[2]]*1, name='m')),
        ]
        self.factory_acc = [
            (MixtureAccumulatorFactory(factories=[self._factories[0]]*3, name='m', keys=('w', None)), MixtureAccumulator(accumulators=[self._accumulators[0]]*3, name='m', keys=('w', None))),
            (MixtureAccumulatorFactory(factories=[self._factories[0]]*1, name='m', keys=(None, 'comps')), MixtureAccumulator(accumulators=[self._accumulators[0]]*1, name='m', keys=(None, 'comps'))),
            (MixtureAccumulatorFactory(factories=[self._factories[1]]*3, name='m', keys=('w', 'comps')), MixtureAccumulator(accumulators=[self._accumulators[1]]*3, name='m', keys=('w', 'comps'))),
            (MixtureAccumulatorFactory(factories=[self._factories[1]]*1, name='m'), MixtureAccumulator(accumulators=[self._accumulators[1]]*1, name='m', keys=(None, None))),
            (MixtureAccumulatorFactory(factories=[self._factories[1]]*1, keys=(None, None)), MixtureAccumulator(accumulators=[self._accumulators[1]]*1)),

        ]
        self.acc_encoder = [(MixtureAccumulator(accumulators=[self._accumulators[1]]*3), self._encoder)]
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
                MixtureEstimator([CategoricalEstimator()]*5, keys=x)
                
            assert str(e.value) == "MixtureEstimator requires keys (Tuple[Optional[str], Optional[str]])."
            
        





