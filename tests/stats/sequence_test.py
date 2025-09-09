"""Test cases for Sequence Distribution and related classes."""
from tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.sequence import *
from dmx.stats.categorical import * 
from dmx.stats.null_dist import * 

import numpy as np
import pytest 


class SequenceDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:

        self._len_dists = [
            CategoricalDistribution({10: 1.0}),
            CategoricalDistribution({5: 0.5, 10: 0.5}),
        ]
        self._len_accs = CategoricalAccumulator()
        self._len_facs = CategoricalAccumulatorFactory()
        self._len_ests = CategoricalEstimator()

        self._dists = [
            CategoricalDistribution({'a': 0.5, 'b': 0.25, 'c': 0.25}),
            CategoricalDistribution({'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25}),
            CategoricalDistribution({'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25})
        ]
        self._accs = CategoricalAccumulator()
        self._facs = CategoricalAccumulatorFactory()
        self._ests = CategoricalEstimator()


        self.eval_dists = [
            SequenceDistribution(dist=self._dists[0], len_dist=self._len_dists[0], len_normalized=False, name='name', keys='keys'),
            SequenceDistribution(dist=self._dists[0], len_dist=self._len_dists[0], len_normalized=True, name='name', keys='keys'),
            SequenceDistribution(dist=self._dists[0], len_dist=self._len_dists[0], len_normalized=False, name='name'),
            SequenceDistribution(dist=self._dists[1], len_dist=self._len_dists[1], keys='keys'),
            SequenceDistribution(dist=self._dists[1], len_dist=self._len_dists[1])
        ]
        self._estimators = [
            SequenceEstimator(estimator=self._ests, len_estimator=self._len_ests, name='name', keys='keys'),
            SequenceEstimator(estimator=self._ests, len_estimator=self._len_ests, len_normalized=True, name='name', keys='keys'),
            SequenceEstimator(estimator=self._ests, len_estimator=self._len_ests, name='name'),
            SequenceEstimator(estimator=self._ests, len_estimator=self._len_ests, keys='keys'),
            SequenceEstimator(estimator=self._ests, len_estimator=self._len_ests)
        ]
        self._factories = [
            SequenceAccumulatorFactory(dist_factory=self._facs, len_factory=self._len_facs, name='name', keys='keys'),
            SequenceAccumulatorFactory(dist_factory=self._facs, len_factory=self._len_facs, len_normalized=True, name='name', keys='keys'),
            SequenceAccumulatorFactory(dist_factory=self._facs, len_factory=self._len_facs, name='name'),
            SequenceAccumulatorFactory(dist_factory=self._facs, len_factory=self._len_facs, keys='keys'),
            SequenceAccumulatorFactory(dist_factory=self._facs, len_factory=self._len_facs)
        ]
        self._accumulators = [
            SequenceAccumulator(accumulator=self._accs, len_accumulator=self._len_accs, name='name', keys='keys'),
            SequenceAccumulator(accumulator=self._accs, len_accumulator=self._len_accs, len_normalized=True, name='name', keys='keys'),
            SequenceAccumulator(accumulator=self._accs, len_accumulator=self._len_accs, name='name'),
            SequenceAccumulator(accumulator=self._accs, len_accumulator=self._len_accs, keys='keys'),
            SequenceAccumulator(accumulator=self._accs, len_accumulator=self._len_accs)
        ]
        self._encoders = [SequenceDataEncoder(encoders=[CategoricalDataEncoder()]*2)] * len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._estimators)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._estimators, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]
    

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "SequenceEncodedDataSequence required for seq_log_density()."
    
    def test_none_estimator(self):
        dists = [
            SequenceDistribution(dist=self._dists[0], name='name', keys='keys'),
            SequenceDistribution(dist=self._dists[0], len_dist=None, name='name', keys='keys'),
            SequenceDistribution(dist=self._dists[0], len_dist=NullDistribution(), name='name', keys='keys')
        ]
        ests = [
            SequenceEstimator(estimator=self._ests, len_estimator=NullEstimator(), name='name', keys='keys'),
            SequenceEstimator(estimator=self._ests, len_estimator=NullEstimator(), name='name', keys='keys'),
            SequenceEstimator(estimator=self._ests, len_estimator=NullEstimator(), name='name', keys='keys')
        ]
        rv = []
        for d, e in zip(dists, ests):
            rv.append(d.estimator() == e)
        
        assert all(rv), rv 

    def test_none_len_dist(self):
        dists = [
            SequenceDistribution(dist=self._dists[0], name='name', keys='keys'),
            SequenceDistribution(dist=self._dists[0], len_dist=None, name='name', keys='keys'),
            SequenceDistribution(dist=self._dists[0], len_dist=NullDistribution(), name='name', keys='keys')
        ]
        rv = []
        for d in dists:
            rv.append(d == eval(str(d)))
        
        assert all(rv), rv 
            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                SequenceEstimator(estimator=ParameterEstimator(), keys=x)
                
            assert str(e.value) == "SequenceEstimator requires keys to be of type 'str'."

