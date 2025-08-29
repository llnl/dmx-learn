""""Tests for the ConditionalDistribution class and related components."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.categorical import *
from dmx.stats.binomial import * 
from dmx.stats.gaussian import * 
from dmx.stats.conditional import * 
import numpy as np

class ConditionalDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        
        self._dmaps = [
            {
                'a': GaussianDistribution(mu=0.0, sigma2=1.0),
                'b': GaussianDistribution(mu=3.0, sigma2=1.0),
                'c': GaussianDistribution(mu=6.0, sigma2=1.0)
            },
            {
                0: CategoricalDistribution({'a': 0.2, 'b': 0.3, 'c': 0.5}),
                1: CategoricalDistribution({'a': 0.7, 'b': 0.1, 'c': 0.2})
            }
        ]
        self._default_dists = [
            GaussianDistribution(mu=-10.0, sigma2=1.0),
            CategoricalDistribution({'a': 0.9, 'b': 0.05, 'c': 0.05})
        ]
        self._given_dists = [
            CategoricalDistribution({'a': 0.2, 'b': 0.3, 'c': 0.5}),
            BinomialDistribution(p=0.5, n=1),
            CategoricalDistribution({'a': 0.2, 'b': 0.3, 'c': 0.3, 'd': 0.1, 'e': 0.1}),
            BinomialDistribution(p=0.5, n=10)
        ]
        self._emaps = [
            {chr(97+x): GaussianEstimator() for x in range(3)},
            {x: CategoricalEstimator() for x in range(2)}
        ]
        self._default_ests = [
            GaussianEstimator(),
            CategoricalEstimator()
        ]
        self._given_ests = [
            CategoricalEstimator(),
            BinomialEstimator()
        ]
        self._encoder_map = [
            {chr(97+x): GaussianDataEncoder() for x in range(3)},
            {x: CategoricalDataEncoder() for x in range(2)}
        ]
        self._default_encoders = [
            GaussianDataEncoder(),
            CategoricalDataEncoder()
        ]
        self._given_encoders = [
            CategoricalDataEncoder(),
            BinomialDataEncoder()
        ]

        self._factory_map = [
            {chr(97+x): GaussianAccumulatorFactory() for x in range(3)},
            {x: CategoricalAccumulatorFactory() for x in range(2)}
        ]
        self._default_factory = [
            GaussianAccumulatorFactory(),
            CategoricalAccumulatorFactory()
        ]
        self._given_factory = [
            CategoricalAccumulatorFactory(),
            BinomialAccumulatorFactory()
        ]

        self._acc_map = [
            {chr(97+x): GaussianAccumulator() for x in range(3)},
            {x: CategoricalAccumulator() for x in range(2)}
        ]
        self._default_acc = [
            GaussianAccumulator(),
            CategoricalAccumulator()
        ]
        self._given_acc = [
            CategoricalAccumulator(),
            BinomialAccumulator()
        ]

        self.eval_dists = [
            ConditionalDistribution(dmap=self._dmaps[0], given_dist=self._given_dists[0], name='c', keys='key'),
            ConditionalDistribution(dmap=self._dmaps[1], given_dist=self._given_dists[1], name='c', keys='key'),
            ConditionalDistribution(dmap=self._dmaps[0], given_dist=self._given_dists[2], default_dist=self._default_dists[0]),
            ConditionalDistribution(dmap=self._dmaps[1], given_dist=self._given_dists[3], default_dist=self._default_dists[1])
        ]
        self._ests = [
            ConditionalDistributionEstimator(estimator_map=self._emaps[0], given_estimator=self._given_ests[0], name='c', keys='key'),
            ConditionalDistributionEstimator(estimator_map=self._emaps[1], given_estimator=self._given_ests[1], name='c', keys='key'),
            ConditionalDistributionEstimator(estimator_map=self._emaps[0], given_estimator=self._given_ests[0], default_estimator=self._default_ests[0])
        ]
        self._factories = [
            ConditionalDistributionAccumulatorFactory(factory_map=self._factory_map[0], given_factory=self._given_factory[0], name='c', keys='key'),
            ConditionalDistributionAccumulatorFactory(factory_map=self._factory_map[1], given_factory=self._given_factory[1], name='c', keys='key'),
            ConditionalDistributionAccumulatorFactory(factory_map=self._factory_map[0], given_factory=self._given_factory[0], default_factory=self._default_factory[0])
        ]
        self._accumulators = [
            ConditionalDistributionAccumulator(accumulator_map=self._acc_map[0], given_accumulator=self._given_acc[0], name='c', keys='key'),
            ConditionalDistributionAccumulator(accumulator_map=self._acc_map[1], given_accumulator=self._given_acc[1], name='c', keys='key'),
            ConditionalDistributionAccumulator(accumulator_map=self._acc_map[0], given_accumulator=self._given_acc[0], default_accumulator=self._default_acc[0])
        ]
        self._encoders = [
            ConditionalDistributionDataEncoder(encoder_map=self._encoder_map[0], given_encoder=self._given_encoders[0]),
            ConditionalDistributionDataEncoder(encoder_map=self._encoder_map[1], given_encoder=self._given_encoders[1]),
            ConditionalDistributionDataEncoder(encoder_map=self._encoder_map[0], default_encoder=self._default_encoders[0], given_encoder=self._given_encoders[0]),
            ConditionalDistributionDataEncoder(encoder_map=self._encoder_map[1], default_encoder=self._default_encoders[1], given_encoder=self._given_encoders[1])
        ]

        self.dist_est = [(self.eval_dists[i], self._ests[i]) for i in range(3)]
        self.dist_encoder = [(self.eval_dists[i], self._encoders[i]) for i in range(4)]
        self.sampler_dist = self.eval_dists[2]
        self.density_dist_encoder = self.dist_encoder
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder  = [(a, e) for a, e in zip(self._accumulators, self._encoders[:3])]

        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [(None, None), 1.0, ('keys', None)]
        
    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.density_dist_encoder[0][0].seq_log_density(x)
            assert str(e.value) == "Requires ConditionalEncodedDataSequence for ConditionalDistribution.seq_log_density()"

            
    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                ConditionalDistributionEstimator(estimator_map=self._dmaps[0], keys=x)
                
            assert str(e.value) == "ConditionalDistributionEstimator requires keys to be of type 'str'."


