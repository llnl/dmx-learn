from typing import Optional, Union, Sequence, Tuple, NamedTuple, TypeVar, Any, List
from dmx.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from dmx.bstats.mvngamma import MultivariateNormalGammaDistribution
from dmx.bstats.nulldist import null_dist
from dmx.utils.special import digamma
import dmx.utils.vector as vec
import numpy as np
import scipy.linalg

DatumType  = Union[Sequence[float], np.ndarray]
ParamType  = Tuple[np.ndarray, np.ndarray]


class DiagonalGaussianDistribution(ProbabilityDistribution):

    def __init__(self, mu: Union[Sequence[float], np.ndarray], covariance: Union[Sequence[float], np.ndarray], name: Optional[str] = None, prior: Optional[ProbabilityDistribution] = None):

        if prior is None:
            n = len(mu)
            prior = MultivariateNormalGammaDistribution(np.zeros(n), np.ones(n)*1.0e-8, np.ones(n)*0.500001, np.ones(n)*1.0)

        self.conj_prior_params = None
        self.expected_nparams  = None
        self.has_conj_prior    = False
        self.prior             = None

        self.dim      = len(mu)
        self.mu       = np.asarray(mu, dtype=float)
        self.covar    = np.asarray(covariance, dtype=float)
        self.name     = name
        self.log_c    = -0.5*(np.log(2.0*np.pi)*self.dim + np.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c

        self.set_prior(prior)

    def __str__(self):
        mu_str = ','.join(map(str,self.mu.flatten()))
        co_str = ','.join(map(str,self.covar.flatten()))
        return 'DiagonalGaussianDistribution([%s], [%s], name=%s, prior=%s)'%(mu_str, co_str, str(self.name), str(self.prior))

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:

        self.prior = prior

        if isinstance(prior, MultivariateNormalGammaDistribution):

            mu, lam, a, b = prior.get_parameters()

            ea = np.sum(((mu * mu) * (a / b) * 0.5 + (0.5 / lam) + 0.5 * (np.log(b) - digamma(a))))
            e1 = mu * a / b
            e2 = -0.5 * a / b
            eb = -0.5 * np.log(2 * np.pi) * self.dim

            self.conj_prior_params = [mu, lam, a, b]
            self.expected_nparams  = [ea, eb, e1, e2]
            self.has_conj_prior    = True

        else:

            self.conj_prior_params = None
            self.expected_nparams  = None
            self.has_conj_prior    = False

    def get_parameters(self) -> ParamType:
        return self.mu, self.covar

    def set_parameters(self, value: ParamType) -> None:

        mu, covariance = value

        self.dim      = len(mu)
        self.mu       = np.asarray(mu, dtype=float)
        self.covar    = np.asarray(covariance, dtype=float)
        self.log_c    = -0.5*(np.log(2.0*np.pi)*self.dim + np.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c

    def log_density(self, x):
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

    def expected_log_density(self, x) -> float:

        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return np.dot(x, e1) + np.dot(np.power(x,2), e2) - ea + eb
        else:
            raise Exception('dmvn expected_log_density not implemented.')

    def seq_log_density(self, x):
        rv = np.dot(x[1], self.ca)
        rv += np.dot(x[0], self.cb)
        rv += self.cc
        return rv

    def seq_expected_log_density(self, x):
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return np.dot(x[0], e1) + np.dot(x[1], e2) - ea + eb
        else:
            raise Exception('General seq_expected_log_density not implemented.')

    def seq_encode(self, x) -> Tuple[np.ndarray, np.ndarray]:
        xv = np.reshape(x, (-1, self.dim))
        return xv, xv*xv

    def sampler(self, seed=None):
        return DiagonalGaussianSampler(self, seed)

    def estimator(self):
        return DiagonalGaussianEstimator()


class DiagonalGaussianSampler(object):

    def __init__(self, dist, seed=None):
        self.rng  = np.random.RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        if size is None:
            rv = self.rng.standard_normal(size=self.dist.dim)*np.sqrt(self.dist.covar) + self.dist.mu
            return rv.tolist()
        else:
            rv = self.rng.standard_normal(size=(size,self.dist.dim)) * np.sqrt(self.dist.covar) + self.dist.mu
            return [u.tolist() for u in rv]


class DiagonalGaussianAccumulator(StatisticAccumulator):

    def __init__(self, dim=None):

        self.dim     = dim
        self.count   = 0.0

        if dim is not None:
            self.sum  = vec.zeros(dim)
            self.sum2 = vec.zeros(dim)
        else:
            self.sum = None
            self.sum2 = None

    def update(self, x, weight, estimate):

        if self.dim is None:
            self.dim  = len(x)
            self.sum  = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        xWeight    = x*weight
        self.count += weight
        self.sum  += xWeight
        xWeight *= x
        self.sum2 += xWeight

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_update(self, x, weights, estimate):

        if self.dim is None:
            self.dim  = x[0].shape[1]
            self.sum  = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        self.count += weights.sum()
        self.sum   += np.dot(x[0].T, weights)
        self.sum2  += np.dot(x[1].T, weights)

    def combine(self, suff_stat):

        if suff_stat[0] is not None and self.sum is not None:
            self.sum  += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            self.sum  = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]

        return self

    def value(self):
        return self.sum, self.sum2, self.count

    def from_value(self, x):
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]


class DiagonalGaussianEstimator(ParameterEstimator):

    def __init__(self, dim: Optional[int] = None, name: Optional[str] = None, prior: Optional[ProbabilityDistribution] = None):

        if (prior is None) and (dim is not None):
            prior = MultivariateNormalGammaDistribution(np.zeros(dim), np.ones(dim)*1.0e-8, np.ones(dim)*0.500001, np.ones(dim)*1.0)

        self.dim            = dim
        self.name           = name
        self.prior          = None
        self.has_conj_prior = None

        self.set_prior(prior)

    def accumulator_factory(self):
        dim = self.dim
        obj = type('', (object,), {'make': lambda o: DiagonalGaussianAccumulator(dim=dim)})()
        return(obj)

    def set_prior(self, prior):
        self.prior = prior
        self.has_conj_prior = isinstance(prior, MultivariateNormalGammaDistribution)

    def get_prior(self):
        return self.prior

    def estimate(self, suff_stat):

        sum_x, sum_xx, nobs_loc1 = suff_stat
        sum_xxx   = sum_x
        nobs_loc2 = nobs_loc1

        if self.has_conj_prior:

            old_mu, old_lam, old_a, old_b = self.prior.get_parameters()

            new_n  = old_lam + nobs_loc1
            new_a  = old_a + (nobs_loc2 / 2.0)
            new_nn = old_lam + nobs_loc2

            if nobs_loc1 > 0:
                sample_mean1 = (sum_x/nobs_loc1)
            else:
                sample_mean1 = 0

            if nobs_loc2 > 0:
                sample_mean2 = (sum_xxx/nobs_loc2)
            else:
                sample_mean2 = 0

            new_mu = (sum_x + old_mu*old_lam)/(old_lam + nobs_loc1)

            new_b0 = (sum_xx - sample_mean2*sum_xxx)
            new_b1 = (old_lam*nobs_loc1/new_n)*np.power(sample_mean1-old_mu,2)
            new_b  = old_b + 0.5*(new_b0 + new_b1)

            new_sigma2 = (new_b/(new_a - 0.5))

            new_prior  = MultivariateNormalGammaDistribution(new_mu, new_n, new_a, new_b)

            return DiagonalGaussianDistribution(new_mu, new_sigma2, name=self.name, prior=new_prior)

        else:

            if nobs_loc1 == 0:
                mu = 0.0
            else:
                mu = sum_x / nobs_loc1

            if nobs_loc2 == 0:
                sigma2 = 0
            else:
                mu2 = sum_xxx/nobs_loc2
                sigma2 = (sum_xx / nobs_loc2) - mu2*mu2

            return DiagonalGaussianDistribution(mu, sigma2, name=self.name)