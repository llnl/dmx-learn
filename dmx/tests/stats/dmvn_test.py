"""Tests for the Diagonal Gaussian Distribtion."""
from dmx.tests.stats.stats_tests import * 
from dmx.stats import *
from dmx.stats.dmvn import *
import numpy as np
import pytest 

class DiagonalGaussianDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        self.eval_dists = [
            DiagonalGaussianDistribution(mu=[0.0, 5.0, 10.0], covar=[1.0, 1.0, 1.0], name='name', keys=('mu', 'covar')),
            DiagonalGaussianDistribution(mu=[0.0, 5.0, 10.0], covar=[1.0, 1.0, 1.0], name='name', keys=(None, 'covar')),
            DiagonalGaussianDistribution(mu=[0.0, 5.0, 10.0], covar=[1.0, 1.0, 1.0], name='name', keys=('mu', None)),
            DiagonalGaussianDistribution(mu=[0.0, 5.0, 10.0], covar=[1.0, 1.0, 1.0]),
            DiagonalGaussianDistribution(mu=[0.0, 5.0], covar=[1.0, 1.0])
        ]
        self._ests = [
            DiagonalGaussianEstimator(dim=3, name='name', keys=('mu', 'covar')),
            DiagonalGaussianEstimator(dim=3, name='name', keys=(None, 'covar')),
            DiagonalGaussianEstimator(dim=3, name='name', keys=('mu', None)),
            DiagonalGaussianEstimator(dim=3),
            DiagonalGaussianEstimator(dim=2)
        ]
        self._factories = [
            DiagonalGaussianAccumulatorFactory(dim=3, name='name', keys=('mu', 'covar')),
            DiagonalGaussianAccumulatorFactory(dim=3, name='name', keys=(None, 'covar')),
            DiagonalGaussianAccumulatorFactory(dim=3, name='name', keys=('mu', None)),
            DiagonalGaussianAccumulatorFactory(dim=3),
            DiagonalGaussianAccumulatorFactory(dim=2)
        ]
        self._accumulators = [
            DiagonalGaussianAccumulator(dim=3, name='name', keys=('mu', 'covar')),
            DiagonalGaussianAccumulator(dim=3, name='name', keys=(None, 'covar')),
            DiagonalGaussianAccumulator(dim=3, name='name', keys=('mu', None)),
            DiagonalGaussianAccumulator(dim=3),
            DiagonalGaussianAccumulator(dim=2)
        ]
        self._encoders = [
            DiagonalGaussianDataEncoder(dim=3),
            DiagonalGaussianDataEncoder(dim=3),
            DiagonalGaussianDataEncoder(dim=3),
            DiagonalGaussianDataEncoder(dim=3),
            DiagonalGaussianDataEncoder(dim=2)
        ]

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]
        self.type_check_keys = [None, 'keys', (None, None, None), (1, 'keys')]

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == 'DiagonalGaussianDistribution.seq_log_density() requires DiagonalGaussianEncodedDataSequence.'

    def test_key_exceptions(self):
        for x in self.type_check_keys:
            with pytest.raises(TypeError) as e:
                DiagonalGaussianEstimator(keys=x)
                
            assert str(e.value) == "DiagonalGaussianEstimator requires keys (Tuple[Optional[str], Optional[str]])."