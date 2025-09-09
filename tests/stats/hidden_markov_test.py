"""Test cases for Hidden Markov Model distributions and related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.categorical import *
from dmx.stats.binomial import * 
from dmx.stats.gaussian import * 
from dmx.stats.hidden_markov import * 
import numpy as np


class HiddenMarkovDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        
        self._comps = [
            [CategoricalDistribution(pmap={'a': 0.1, 'b': 0.3, 'c': 0.6}), 
            CategoricalDistribution(pmap={'a': 0.6, 'b': 0.1, 'c': 0.3}),
            CategoricalDistribution(pmap={'a': 0.3, 'b': 0.6, 'c': 0.1})],
            [GaussianDistribution(mu=x, sigma2=1.0) for x in [0.0, 5.0, 10.0, 15.0]]
        ]
        self._ests = [
            [CategoricalEstimator()]*len(self._comps[0]),
            [GaussianEstimator()]*len(self._comps[1])
        ]
        self._accs = [
            [CategoricalAccumulator()] * len(self._comps[0]),
            [GaussianAccumulator()] * len(self._comps[1])
        ]
        self._facts = [
            [CategoricalAccumulatorFactory()] * len(self._comps[0]),
            [GaussianAccumulatorFactory()] * len(self._comps[1])
        ]
        self._len_dists = [
            CategoricalDistribution({8: 0.4, 9: 0.5, 10: 0.1}),
            CategoricalDistribution({10: 1.0})
        ]
        
        self._w = [
            np.ones(len(self._comps[0])) / len(self._comps[0]),
            np.ones(len(self._comps[1])) / len(self._comps[1])
        ]
        self._trans = [
            np.ones((len(self._comps[0]), len(self._comps[0]))) / len(self._comps[0]),
            np.ones((len(self._comps[1]), len(self._comps[1]))) / len(self._comps[1])
        ]

        self.eval_dists = [
            HiddenMarkovModelDistribution(
                topics=self._comps[0], 
                w=self._w[0], 
                transitions=self._trans[0], 
                len_dist=self._len_dists[0], 
                name='name', 
                keys=('w', 'trans', 'comps')),
            HiddenMarkovModelDistribution(
                topics=self._comps[0],
                w=self._w[0], 
                transitions=self._trans[0], 
                len_dist=self._len_dists[0], 
                name='name', 
                keys=(None, None, 'comps')),
            HiddenMarkovModelDistribution(
                topics=self._comps[0],
                w=self._w[0], 
                transitions=self._trans[0], 
                len_dist=self._len_dists[0], 
                name='name', 
                keys=(None, 'trans', 'comps')),
            HiddenMarkovModelDistribution(
                topics=self._comps[0], 
                w=self._w[0], 
                transitions=self._trans[0], 
                len_dist=self._len_dists[0], 
                name='name', 
                keys=(None, 'trans', None),
                use_numba=True),
            HiddenMarkovModelDistribution(
                topics=self._comps[0], 
                w=self._w[0], 
                transitions=self._trans[0], 
                len_dist=self._len_dists[0]),
            HiddenMarkovModelDistribution(
                topics=self._comps[1], 
                w=self._w[1], 
                transitions=self._trans[1], 
                len_dist=self._len_dists[1], 
                name='name', 
                keys=('w', None, None))
        ]
        self._encoders = [
            HiddenMarkovDataEncoder(emission_encoder=CategoricalDataEncoder(), len_encoder=CategoricalDataEncoder()),
            HiddenMarkovDataEncoder(emission_encoder=CategoricalDataEncoder(), len_encoder=CategoricalDataEncoder()),
            HiddenMarkovDataEncoder(emission_encoder=CategoricalDataEncoder(), len_encoder=CategoricalDataEncoder()),
            HiddenMarkovDataEncoder(emission_encoder=CategoricalDataEncoder(), len_encoder=CategoricalDataEncoder(), use_numba=True),
            HiddenMarkovDataEncoder(emission_encoder=CategoricalDataEncoder(), len_encoder=CategoricalDataEncoder()),
            HiddenMarkovDataEncoder(emission_encoder=GaussianDataEncoder(), len_encoder=CategoricalDataEncoder())
        ]
        self._estimators = [
            HiddenMarkovEstimator(estimators=self._ests[0], len_estimator=CategoricalEstimator(), name='name', keys=('w', 'trans', 'comps')),
            HiddenMarkovEstimator(estimators=self._ests[0], len_estimator=CategoricalEstimator(), name='name', keys=(None, None, 'comps')), 
            HiddenMarkovEstimator(estimators=self._ests[0], len_estimator=CategoricalEstimator(), name='name', keys=(None, 'trans', 'comps')),
            HiddenMarkovEstimator(estimators=self._ests[0], len_estimator=CategoricalEstimator(), name='name', keys=(None, 'trans', None), use_numba=True),
            HiddenMarkovEstimator(estimators=self._ests[0], len_estimator=CategoricalEstimator()),
            HiddenMarkovEstimator(estimators=self._ests[1], len_estimator=CategoricalEstimator(), name='name', keys=('w', None, None))
        ]
        self._accumulators = [
            HiddenMarkovAccumulator(accumulators=self._accs[0], len_accumulator=CategoricalAccumulator(), name='name', keys=('w', 'trans', 'comps')),
            HiddenMarkovAccumulator(accumulators=self._accs[0], len_accumulator=CategoricalAccumulator(), name='name', keys=(None, None, 'comps')),
            HiddenMarkovAccumulator(accumulators=self._accs[0], len_accumulator=CategoricalAccumulator(), name='name', keys=(None, 'trans', 'comps')),
            HiddenMarkovAccumulator(accumulators=self._accs[0], len_accumulator=CategoricalAccumulator(), name='name', keys=(None, 'trans', None), use_numba=True),
            HiddenMarkovAccumulator(accumulators=self._accs[0], len_accumulator=CategoricalAccumulator()),
            HiddenMarkovAccumulator(accumulators=self._accs[1], len_accumulator=CategoricalAccumulator(), name='name', keys=('w', None, None))
        ]
        self._factories = [
            HiddenMarkovAccumulatorFactory(factories=self._facts[0], len_factory=CategoricalAccumulatorFactory(), name='name', keys=('w', 'trans', 'comps')),
            HiddenMarkovAccumulatorFactory(factories=self._facts[0], len_factory=CategoricalAccumulatorFactory(), name='name', keys=(None, None, 'comps')),
            HiddenMarkovAccumulatorFactory(factories=self._facts[0], len_factory=CategoricalAccumulatorFactory(), name='name', keys=(None, 'trans', 'comps')),
            HiddenMarkovAccumulatorFactory(factories=self._facts[0], len_factory=CategoricalAccumulatorFactory(), name='name', keys=(None, 'trans', None), use_numba=True),
            HiddenMarkovAccumulatorFactory(factories=self._facts[0], len_factory=CategoricalAccumulatorFactory()),
            HiddenMarkovAccumulatorFactory(factories=self._facts[1], len_factory=CategoricalAccumulatorFactory(), name='name', keys=('w', None, None))
        ]

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._estimators)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._estimators, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]
        self.type_check_keys = [None, 'keys', (None, None), ('keys', None, None, None), (1, 'keys', None)]

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                HiddenMarkovEstimator([CategoricalEstimator()]*5, keys=x)
                
            assert str(e.value) == "HiddenMarkovEstimator requires keys (Tuple[Optional[str], Optional[str], Optional[str]])."
            
        

### Test viterbi and numba equivilence of seq_update, seq_initialize, seq_log_density ect. 
