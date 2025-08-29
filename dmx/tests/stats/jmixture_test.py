"""Test cases for Joint Mixture Distribution and related classes."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.categorical import *
from dmx.stats.binomial import * 
from dmx.stats.gaussian import * 
from dmx.stats.jmixture import * 
import numpy as np


class JointMixtureDistributionTestCase(StatsTestClass):

    def setUp(self) -> None:
        self._comps0 = [
            [GaussianDistribution(mu=x, sigma2=1.0) for x in [0.0, 5.0, 10.0, 15.0]],
            [GaussianDistribution(mu=x, sigma2=1.0) for x in [0.0, 5.0]]
        ]
        self._comps1 = [
            [GaussianDistribution(mu=x, sigma2=1.0) for x in [0.0, 5.0]],
            [GaussianDistribution(mu=x, sigma2=1.0) for x in [0.0, 5.0, 10.0, 15.0]]
        ]
        self._w0 = [
            np.ones(4) / 4.,
            np.ones(2) / 2.
        ]
        self._w1 = [
            np.ones(2) / 2.,
            np.ones(4) / 4.
        ]

        rng = np.random.RandomState(1)
        self._taus12 = [
            rng.dirichlet(np.ones(2), size=4),
            rng.dirichlet(np.ones(4), size=2)
        ]
        self._taus21 = [
            rng.dirichlet(np.ones(4), size=2),
            rng.dirichlet(np.ones(2), size=4)
        ]

        self._ests0 = [
            [GaussianEstimator()]*4,
            [GaussianEstimator()]*2
        ]
        self._ests1 = [
            [GaussianEstimator()]*2,
            [GaussianEstimator()]*4
        ]
        self._accs0 = [
            [GaussianAccumulator()]*4,
            [GaussianAccumulator()]*2
        ]
        self._accs1 = [
            [GaussianAccumulator()]*2,
            [GaussianAccumulator()]*4
        ]
        self._facs0 = [
            [GaussianAccumulatorFactory()]*4,
            [GaussianAccumulatorFactory()]*2
        ]
        self._facs1 = [
            [GaussianAccumulatorFactory()]*2,
            [GaussianAccumulatorFactory()]*4
        ]

        keys = [('a' if i // 4 == 0 else None, 'b' if (i // 2) % 2 == 0 else None, 'c' if i % 2 == 0 else None) for i in range(8)]
        name = ['name' if i % 2 == 0 else None for i in range(8)]
        self.eval_dists = []
        self._ests = []
        self._factories = []
        self._accumulators = []

        for i in range(8):
            dist = JointMixtureDistribution(
                components1=self._comps0[i%2], 
                components2=self._comps1[i%2],
                w1=self._w0[i%2],
                w2=self._w1[i%2],
                taus12=self._taus12[i%2],
                taus21=self._taus21[i%2],
                name=name[i],
                keys=keys[i]
            )
            estimator = JointMixtureEstimator(
                estimators1=self._ests0[i%2],
                estimators2=self._ests1[i%2],
                name=name[i],
                keys=keys[i]
            )
            factory = JointMixtureEstimatorAccumulatorFactory(
                factories1=self._facs0[i%2],
                factories2=self._facs1[i%2],
                name=name[i],
                keys=keys[i]
            )
            accumulator = JointMixtureEstimatorAccumulator(
                accumulators1=self._accs0[i%2],
                accumulators2=self._accs1[i%2],
                name=name[i],
                keys=keys[i]
            )

            self.eval_dists.append(dist)
            self._ests.append(estimator)
            self._factories.append(factory)
            self._accumulators.append(accumulator)

        self._encoders = [JointMixtureDataEncoder(*tuple([GaussianDataEncoder()]*2))]*len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "JointMixtureEncodedDataSequence required for seq_log_density()."
            
        





