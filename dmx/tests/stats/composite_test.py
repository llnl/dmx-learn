""""Tests for the CompositeDistribution class and related components."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.categorical import *
from dmx.stats.binomial import * 
from dmx.stats.gaussian import * 
from dmx.stats.composite import * 
import numpy as np

class CompositeDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        
        self._dists = [
            CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}, name='cat'), 
            BinomialDistribution(p=0.50, n=10, name='bino'), 
            GaussianDistribution(mu=0.0, sigma2=3.0, name='g')
        ]
        self._encoders = [
            CategoricalDataEncoder(),
            BinomialDataEncoder(),
            GaussianDataEncoder()
        ]
        self._estimators = [CategoricalEstimator(name='cat'), BinomialEstimator(name='bino'), GaussianEstimator(name='g')]
        self._factories = [
            [CategoricalAccumulatorFactory(name='cat'), 
            BinomialAccumulatorFactory(name='bino'), 
            GaussianAccumulatorFactory(name='g')],

            [CategoricalAccumulatorFactory(name='cat', keys='cat_key'), 
            BinomialAccumulatorFactory(name='bino', keys='bino_keys'), 
            GaussianAccumulatorFactory(name='g', keys='g_keys')]
        ]
        self._accumulators = [
            CategoricalAccumulator(name='cat', keys='cat_key'), 
            BinomialAccumulator(name='bino', keys='bino_keys'), 
            GaussianAccumulator(name='g', keys='g_keys')
        ]

        self.eval_dists = [
            CompositeDistribution(dists=self._dists, name='comp', keys='comp_keys'),
            CompositeDistribution(dists=self._dists[:2], name='comp'),
            CompositeDistribution(dists=self._dists[:1], name='comp')
        ]
        self.dist_est = [
            (self.eval_dists[0], CompositeEstimator(estimators=self._estimators, name='comp', keys="comp_keys")),
            (self.eval_dists[1], CompositeEstimator(estimators=self._estimators[:2], name='comp')),
            (self.eval_dists[2], CompositeEstimator(estimators=self._estimators[:1], name='comp'))
        ]
        
        self.dist_encoder = [(self.eval_dists[0], CompositeDataEncoder(encoders=self._encoders))]
        self.sampler_dist = self.eval_dists[0]
        self.density_dist_encoder = [
            (self.eval_dists[0], CompositeDataEncoder(encoders=self._encoders)),
            (self.eval_dists[1], CompositeDataEncoder(encoders=self._encoders[:2])),
            (self.eval_dists[2], CompositeDataEncoder(encoders=self._encoders[:1]))
        ]
        self.est_factory = [
            (CompositeEstimator(estimators=self._estimators, name='comp'), CompositeAccumulatorFactory(factories=self._factories[0], name='comp')),
            (CompositeEstimator(estimators=self._estimators[:2], name='comp'), CompositeAccumulatorFactory(factories=self._factories[0][:2], name='comp')),
            (CompositeEstimator(estimators=self._estimators[:1], name='comp'), CompositeAccumulatorFactory(factories=self._factories[0][:1], name='comp'))
        ]
        self.factory_acc = [
            (CompositeAccumulatorFactory(factories=self._factories[1], name='comp', keys='comp_key'), CompositeAccumulator(accumulators=self._accumulators, name='comp', keys='comp_key')),
            (CompositeAccumulatorFactory(factories=self._factories[1][:2], name='comp', keys='comp_key'), CompositeAccumulator(accumulators=self._accumulators[:2], name='comp', keys='comp_key')),
            (CompositeAccumulatorFactory(factories=self._factories[1][:1], name='comp', keys='comp_key'), CompositeAccumulator(accumulators=self._accumulators[:1], name='comp', keys='comp_key'))

        ]
        self.acc_encoder = [
            (CompositeAccumulator(accumulators=self._accumulators, name='comp', keys='comp_key'), CompositeDataEncoder(encoders=self._encoders)),
            (CompositeAccumulator(accumulators=self._accumulators[:2], name='comp', keys='comp_key'), CompositeDataEncoder(encoders=self._encoders[:2])),
            (CompositeAccumulator(accumulators=self._accumulators[:1], name='comp', keys='comp_key'), CompositeDataEncoder(encoders=self._encoders[:1]))
        ]
        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]


    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                CompositeEstimator(estimators=[ParameterEstimator(), ParameterEstimator()], keys=x)
                
            assert str(e.value) == "CompositeEstimator requires keys to be of type 'str'."


