"""Tests for Hierarchical Mixture Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.geometric import *
from dmx.stats.gaussian import * 
from dmx.stats.hmixture import * 
from dmx.stats.categorical import * 
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


class HierarchicalMixtureDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:

        self._topics = [
            [CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}), 
            CategoricalDistribution(pmap={'a': 0.6, 'b': 0.1, 'c': 0.3}),
            CategoricalDistribution(pmap={'a': 0.3, 'b': 0.6, 'c': 0.1})],
            [GaussianDistribution(mu=x, sigma2=1.0) for x in [0.0, 5.0, 10.0, 15.0]]
        ]
        self._w = [np.ones(5) / 5., np.ones(2) / 2.]
        self._tw = [
            np.ones((5, 3)) / 3.,
            np.ones((2, 4))  / 4.
        ]
        self._len_dists = [
            CategoricalDistribution({6: 1.0}),
            GeometricDistribution(p=0.40)
        ]
        self._len_ests = [CategoricalEstimator(), GeometricEstimator()]
        self._len_accs = [CategoricalAccumulator(), GeometricAccumulator()]
        self._len_facs = [CategoricalAccumulatorFactory(), GeometricAccumulatorFactory()]
        self._len_encs = [CategoricalDataEncoder(), GeometricDataEncoder()]

        self._ests = [
            [CategoricalEstimator()]*len(self._topics[0]),
            [GaussianEstimator()]*len(self._topics[1])
        ]
        self._facts = [
            [CategoricalAccumulatorFactory()]*len(self._topics[0]), 
            [GaussianAccumulatorFactory()]*len(self._topics[1])
        ]
        self._accs = [
            [CategoricalAccumulator()]*len(self._topics[0]), 
            [GaussianAccumulator()]*len(self._topics[1])
        ]
        self._encs = [CategoricalDataEncoder(), GaussianDataEncoder()]

        self.eval_dists = [
            HierarchicalMixtureDistribution(
                topics=self._topics[0], 
                mixture_weights=self._w[0], 
                topic_weights=self._tw[0], 
                len_dist=self._len_dists[0], 
                name='name', keys=('w', 'comps')),

            HierarchicalMixtureDistribution(
                topics=self._topics[0], 
                mixture_weights=self._w[0], 
                topic_weights=self._tw[0], 
                len_dist=self._len_dists[0], 
                name='name', keys=('w', None)),

            HierarchicalMixtureDistribution(
                topics=self._topics[0], 
                mixture_weights=self._w[0], 
                topic_weights=self._tw[0], 
                len_dist=self._len_dists[0], 
                name='name', keys=(None, 'comps')),

            HierarchicalMixtureDistribution(
                topics=self._topics[0], 
                mixture_weights=self._w[0], 
                topic_weights=self._tw[0], 
                len_dist=self._len_dists[0]),

            HierarchicalMixtureDistribution(
                topics=self._topics[1], 
                mixture_weights=self._w[1], 
                topic_weights=self._tw[1], 
                len_dist=self._len_dists[1], 
                name='name', keys=('w', 'comps'))
        ]
        self._estimators = [
            HierarchicalMixtureEstimator(
                estimators=self._ests[0],
                num_mixtures=self._tw[0].shape[0],
                len_estimator=self._len_ests[0],
                name='name',
                keys=('w', 'comps')
            ),

            HierarchicalMixtureEstimator(
                estimators=self._ests[0],
                num_mixtures=self._tw[0].shape[0],
                len_estimator=self._len_ests[0],
                name='name',
                keys=('w', None)
            ),

            HierarchicalMixtureEstimator(
                estimators=self._ests[0],
                num_mixtures=self._tw[0].shape[0],
                len_estimator=self._len_ests[0],
                name='name',
                keys=(None, 'comps')
            ),

            HierarchicalMixtureEstimator(
                estimators=self._ests[0],
                num_mixtures=self._tw[0].shape[0],
                len_estimator=self._len_ests[0]
            ),

            HierarchicalMixtureEstimator(
                estimators=self._ests[1],
                num_mixtures=self._tw[1].shape[0],
                len_estimator=self._len_ests[1],
                name='name',
                keys=('w', 'comps')
            )
        ]

        self._factories = [
            HierarchicalMixtureEstimatorAccumulatorFactory(
                factories=self._facts[0],
                num_mixtures=self._tw[0].shape[0],
                len_factory=self._len_facs[0],
                name='name',
                keys=('w', 'comps')
            ),

            HierarchicalMixtureEstimatorAccumulatorFactory(
                factories=self._facts[0],
                num_mixtures=self._tw[0].shape[0],
                len_factory=self._len_facs[0],
                name='name',
                keys=('w', None)
            ),

            HierarchicalMixtureEstimatorAccumulatorFactory(
                factories=self._facts[0],
                num_mixtures=self._tw[0].shape[0],
                len_factory=self._len_facs[0],
                name='name',
                keys=(None, 'comps')
            ),

            HierarchicalMixtureEstimatorAccumulatorFactory(
                factories=self._facts[0],
                num_mixtures=self._tw[0].shape[0],
                len_factory=self._len_facs[0]
            ),

            HierarchicalMixtureEstimatorAccumulatorFactory(
                factories=self._facts[1],
                num_mixtures=self._tw[1].shape[0],
                len_factory=self._len_facs[1],
                name='name',
                keys=('w', 'comps')
            )
        ]

        self._accumulators = [
            HierarchicalMixtureEstimatorAccumulator(
                accumulators=self._accs[0],
                num_mixtures=self._tw[0].shape[0],
                len_accumulator=self._len_accs[0],
                name='name',
                keys=('w', 'comps')
            ),

            HierarchicalMixtureEstimatorAccumulator(
                accumulators=self._accs[0],
                num_mixtures=self._tw[0].shape[0],
                len_accumulator=self._len_accs[0],
                name='name',
                keys=('w', None)
            ),

            HierarchicalMixtureEstimatorAccumulator(
                accumulators=self._accs[0],
                num_mixtures=self._tw[0].shape[0],
                len_accumulator=self._len_accs[0],
                name='name',
                keys=(None, 'comps')
            ),

            HierarchicalMixtureEstimatorAccumulator(
                accumulators=self._accs[0],
                num_mixtures=self._tw[0].shape[0],
                len_accumulator=self._len_accs[0]
            ),

            HierarchicalMixtureEstimatorAccumulator(
                accumulators=self._accs[1],
                num_mixtures=self._tw[1].shape[0],
                len_accumulator=self._len_accs[1],
                name='name',
                keys=('w', 'comps')
            ),

        ]
        self._encoders = [
            HierarchicalMixtureDataEncoder(topic_encoder=self._encs[0], len_encoder=self._len_encs[0]),
            HierarchicalMixtureDataEncoder(topic_encoder=self._encs[0], len_encoder=self._len_encs[0]),
            HierarchicalMixtureDataEncoder(topic_encoder=self._encs[0], len_encoder=self._len_encs[0]),
            HierarchicalMixtureDataEncoder(topic_encoder=self._encs[0], len_encoder=self._len_encs[0]),
            HierarchicalMixtureDataEncoder(topic_encoder=self._encs[1], len_encoder=self._len_encs[1])
        ]

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._estimators)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._estimators, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]
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
                HierarchicalMixtureEstimator([CategoricalEstimator()]*5, num_mixtures=1, keys=x)
                
            assert str(e.value) == "HierarchialMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]])."
            



