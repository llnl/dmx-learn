"Tests for the Hidden Association Distribution and its related classes."
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.categorical import *
from dmx.stats.binomial import * 
from dmx.stats.gaussian import * 
from dmx.stats.conditional import * 
from dmx.stats.catmultinomial import * 
from dmx.stats.hidden_association import *
import numpy as np

class HiddenAssociationDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        self._dists = [
            CategoricalDistribution({'a': 0.90, 'b': 0.05, 'c': 0.05}),
            CategoricalDistribution({'a': 0.05, 'b': 0.90, 'c': 0.05}),
            CategoricalDistribution({'a': 0.05, 'b': 0.05, 'c': 0.90})
        ]
        self._cond_dists = [
            ConditionalDistribution({'a': self._dists[0], 'b': self._dists[1], 'c': self._dists[2]}),
            ConditionalDistribution({'a': self._dists[0], 'b': self._dists[1]})
        ]
        self._cond_ests = [
            ConditionalDistributionEstimator({x: CategoricalEstimator() for x in ['a', 'b', 'c']}),
            ConditionalDistributionEstimator({x: CategoricalEstimator() for x in ['a', 'b']})
        ]
        self._cond_facts = [
            ConditionalDistributionAccumulatorFactory({x: CategoricalAccumulatorFactory() for x in ['a', 'b', 'c']}),
            ConditionalDistributionAccumulatorFactory({x: CategoricalAccumulatorFactory() for x in ['a', 'b']})
        ]
        self._cond_acc = [
            ConditionalDistributionAccumulator({x: CategoricalAccumulator() for x in ['a', 'b', 'c']}),
            ConditionalDistributionAccumulator({x: CategoricalAccumulator() for x in ['a', 'b']})
        ]
        self._given_dists = [
            MultinomialDistribution(CategoricalDistribution({'a': 0.30, 'b': 0.30, 'c': 0.40}), len_dist=CategoricalDistribution({5: 1.0})),
            MultinomialDistribution(CategoricalDistribution({'a': 0.50, 'b': 0.50}), len_dist=CategoricalDistribution({5: 1.0}))
        ]
        self._given_ests = [
            MultinomialEstimator(CategoricalEstimator(), len_estimator=CategoricalEstimator()),
            MultinomialEstimator(CategoricalEstimator(), len_estimator=CategoricalEstimator())
        ]
        self._given_facts = [
            MultinomialAccumulatorFactory(est_factory=CategoricalAccumulatorFactory(), len_factory=CategoricalAccumulatorFactory(), len_normalized=False),
            MultinomialAccumulatorFactory(est_factory=CategoricalAccumulatorFactory(), len_factory=CategoricalAccumulatorFactory(), len_normalized=False)
        ]
        self._given_acc = [
            MultinomialAccumulator(accumulator=CategoricalAccumulator(), len_accumulator=CategoricalAccumulator(), len_normalized=False),
            MultinomialAccumulator(accumulator=CategoricalAccumulator(), len_accumulator=CategoricalAccumulator(), len_normalized=False)
        ]
        self._len_dist = [
            CategoricalDistribution({7: 1.0}), BinomialDistribution(p=0.30, n=10, min_val=5)
        ]
        self._len_est = [
            CategoricalEstimator(),
            BinomialEstimator()
        ]
        self._len_fact = [
            CategoricalAccumulatorFactory(),
            BinomialAccumulatorFactory()
        ]
        self._len_acc = [
            CategoricalAccumulator(),
            BinomialAccumulator()
        ]

        self.eval_dists = [
            HiddenAssociationDistribution(cond_dist=self._cond_dists[0], given_dist=self._given_dists[0], len_dist=self._len_dist[0], name='name', keys=('w', 't')),
            HiddenAssociationDistribution(cond_dist=self._cond_dists[0], given_dist=self._given_dists[0], len_dist=self._len_dist[0], name='name', keys=(None, 't')),
            HiddenAssociationDistribution(cond_dist=self._cond_dists[0], given_dist=self._given_dists[0], len_dist=self._len_dist[0], name='name', keys=('w', None)),
            HiddenAssociationDistribution(cond_dist=self._cond_dists[0], given_dist=self._given_dists[0], len_dist=self._len_dist[0]),
            HiddenAssociationDistribution(cond_dist=self._cond_dists[1], given_dist=self._given_dists[1], len_dist=self._len_dist[1], name='name', keys=('w', 't'))
        ]
        self._estimators = [
            HiddenAssociationEstimator(
                cond_estimator=self._cond_ests[0], 
                given_estimator=self._given_ests[0],
                len_estimator=self._len_est[0],
                name='name',
                keys=('w', 't')
            ),
            HiddenAssociationEstimator(
                cond_estimator=self._cond_ests[0], 
                given_estimator=self._given_ests[0],
                len_estimator=self._len_est[0],
                name='name',
                keys=(None, 't')
            ),
            HiddenAssociationEstimator(
                cond_estimator=self._cond_ests[0], 
                given_estimator=self._given_ests[0],
                len_estimator=self._len_est[0],
                name='name',
                keys=('w', None)
            ),
            HiddenAssociationEstimator(
                cond_estimator=self._cond_ests[0], 
                given_estimator=self._given_ests[0],
                len_estimator=self._len_est[0]
            ),
            HiddenAssociationEstimator(
                cond_estimator=self._cond_ests[1], 
                given_estimator=self._given_ests[1],
                len_estimator=self._len_est[1],
                name='name',
                keys=('w', 't')
            )
        ]
        self._factories = [
            HiddenAssociationAccumulatorFactory(cond_factory=self._cond_facts[0], given_factory=self._given_facts[0], len_factory=self._len_fact[0], name='name', keys=('w', 't')),
            HiddenAssociationAccumulatorFactory(cond_factory=self._cond_facts[0], given_factory=self._given_facts[0], len_factory=self._len_fact[0], name='name', keys=(None, 't')),
            HiddenAssociationAccumulatorFactory(cond_factory=self._cond_facts[0], given_factory=self._given_facts[0], len_factory=self._len_fact[0], name='name', keys=('w', None)),
            HiddenAssociationAccumulatorFactory(cond_factory=self._cond_facts[0], given_factory=self._given_facts[0], len_factory=self._len_fact[0]),
            HiddenAssociationAccumulatorFactory(cond_factory=self._cond_facts[1], given_factory=self._given_facts[1], len_factory=self._len_fact[1], name='name', keys=('w', 't'))
        ]
        self._accumulators = [
            HiddenAssociationAccumulator(cond_acc=self._cond_acc[0], given_acc=self._given_acc[0], size_acc=self._len_acc[0], name='name', keys=('w', 't')),
            HiddenAssociationAccumulator(cond_acc=self._cond_acc[0], given_acc=self._given_acc[0], size_acc=self._len_acc[0], name='name', keys=(None, 't')),
            HiddenAssociationAccumulator(cond_acc=self._cond_acc[0], given_acc=self._given_acc[0], size_acc=self._len_acc[0], name='name', keys=('w', None)),
            HiddenAssociationAccumulator(cond_acc=self._cond_acc[0], given_acc=self._given_acc[0], size_acc=self._len_acc[0]),
            HiddenAssociationAccumulator(cond_acc=self._cond_acc[1], given_acc=self._given_acc[1], size_acc=self._len_acc[1], name='name', keys=('w', 't'))
        ]
        self._encoders = [HiddenAssociationDataEncoder()] * len(self.eval_dists)


        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._estimators)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._estimators, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]
        self.type_check_keys = [None, 'keys', (None, None, None), (1, 'keys')]

    def test_key_exceptions(self):
        est_map = {0: CategoricalDistribution({'a': 0.2, 'b': 0.3, 'c': 0.5})}
        cond_estimator = ConditionalDistributionEstimator(est_map)
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                HiddenAssociationEstimator(cond_estimator=cond_estimator, keys=x)
                
            assert str(e.value) == "HiddenAssociationEstimator requires keys (Tuple[Optional[str], Optional[str]])."
           

  


