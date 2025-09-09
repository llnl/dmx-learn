"""Tests for the Heterogeneous Mixture Distribution and its related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.geometric import *
from dmx.stats.binomial import * 
from dmx.stats.gaussian import * 
from dmx.stats.exponential import *
from dmx.stats.heterogeneous_mixture import * 
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


class HeterogeneousMixtureDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        self._dists = [
            [GaussianDistribution(mu=10.0, sigma2=1.0), GaussianDistribution(mu=15.0, sigma2=1.0), ExponentialDistribution(beta=2.0)],
            [GeometricDistribution(p=0.70), GeometricDistribution(p=0.30), BinomialDistribution(p=0.5, min_val=1, n=5), BinomialDistribution(p=0.90, min_val=1, n=5)]
        ]
        self._w = [np.ones(len(x)) / len(x) for x in self._dists]
        self._ests = [
            [GaussianEstimator(), GaussianEstimator(), ExponentialEstimator()],
            [GeometricEstimator(), GeometricEstimator(), BinomialEstimator(), BinomialEstimator()]
        ]
        self._facts = [
            [GaussianAccumulatorFactory()]*2 + [ExponentialAccumulatorFactory()],
            [GeometricAccumulatorFactory()]*2 + [BinomialAccumulatorFactory()]*2
        ]
        self._accs = [
            [GaussianAccumulator()]*2 + [ExponentialAccumulator()],
            [GeometricAccumulator()]*2 + [BinomialAccumulator()]*2
        ]


        self.eval_dists = [
            HeterogeneousMixtureDistribution(components=self._dists[0], name='name', keys=('w', 'comps'), w=self._w[0]),
            HeterogeneousMixtureDistribution(components=self._dists[0], name='name', keys=(None, 'comps'), w=self._w[0]),
            HeterogeneousMixtureDistribution(components=self._dists[0], name='name', keys=('w', None), w=self._w[0]),
            HeterogeneousMixtureDistribution(components=self._dists[0], w=self._w[0]),
            HeterogeneousMixtureDistribution(components=self._dists[1], name='name', keys=('w', 'comps'), w=self._w[1])
        ]
        self._estimators = [
            HeterogeneousMixtureEstimator(estimators=self._ests[0], name='name', keys=('w', 'comps')),
            HeterogeneousMixtureEstimator(estimators=self._ests[0], name='name', keys=(None, 'comps')),
            HeterogeneousMixtureEstimator(estimators=self._ests[0], name='name', keys=('w', None)),
            HeterogeneousMixtureEstimator(estimators=self._ests[0]),
            HeterogeneousMixtureEstimator(estimators=self._ests[1], name='name', keys=('w', 'comps'))
        ]
        self._factories = [
            HeterogeneousMixtureAccumulatorFactory(factories=self._facts[0], name='name', keys=('w', 'comps')),
            HeterogeneousMixtureAccumulatorFactory(factories=self._facts[0], name='name', keys=(None, 'comps')),
            HeterogeneousMixtureAccumulatorFactory(factories=self._facts[0], name='name', keys=('w', None)),
            HeterogeneousMixtureAccumulatorFactory(factories=self._facts[0]),
            HeterogeneousMixtureAccumulatorFactory(factories=self._facts[1], name='name', keys=('w', 'comps'))
        ]
        self._accumulators = [
            HeterogeneousMixtureAccumulator(accumulators=self._accs[0], name='name', keys=('w', 'comps')),
            HeterogeneousMixtureAccumulator(accumulators=self._accs[0], name='name', keys=(None, 'comps')),
            HeterogeneousMixtureAccumulator(accumulators=self._accs[0], name='name', keys=('w', None)),
            HeterogeneousMixtureAccumulator(accumulators=self._accs[0]),
            HeterogeneousMixtureAccumulator(accumulators=self._accs[1], name='name', keys=('w', 'comps'))
        ]
        self._encoders = [
            HeterogeneousMixtureDataEncoder(encoders=[GaussianDataEncoder(), GaussianDataEncoder(), ExponentialDataEncoder()]),
            HeterogeneousMixtureDataEncoder(encoders=[GaussianDataEncoder(), GaussianDataEncoder(), ExponentialDataEncoder()]),
            HeterogeneousMixtureDataEncoder(encoders=[GaussianDataEncoder(), GaussianDataEncoder(), ExponentialDataEncoder()]),
            HeterogeneousMixtureDataEncoder(encoders=[GaussianDataEncoder(), GaussianDataEncoder(), ExponentialDataEncoder()]),
            HeterogeneousMixtureDataEncoder(encoders=[GeometricDataEncoder(), GeometricDataEncoder(), BinomialDataEncoder(), BinomialDataEncoder()])
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
                HeterogeneousMixtureEstimator([CategoricalEstimator()]*5, keys=x)
                
            assert str(e.value) == "HeterogeneousMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]])."



