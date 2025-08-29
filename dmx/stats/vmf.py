""""Create, estimate, and sample from a von Mises-Fisher distribution.

Defines the VonMisesFisherDistribution, VonMisesFisherSampler, VonMisesFisherAccumulatorFactory,
VonMisesFisherAccumulator, VonMisesFisherEstimator, and the VonMisesFisherDataEncoder classes for use with dmx-learn.

Data type: Union[Sequence[float], np.ndarray].

The von Mises-Fisher (vmf) distribution on the (p-1) sphere in R^{p}. Assume x_mat = (X_1,..,X_p) follows a vmf
distribution with mean direction vector mu = (mu_1, mu_2, ..., mu_p) s.t. ||mu||=1 and concentration parameter
kappa > 0. The vmf log-density if given by

    log(f(x; mu, kappa)) = log(c_p(kappa)) + kappa * dot(mu, x),

where dot is a dot product and
    log(c_p(kappa)) = (p/2-1)log(kappa) - (p/2)*log(2*pi) + log(B_{p/2-1}(kappa)), where

log(B_{p/2-1}(kappa)) = denotes the modified Bessel function of the first kind at order p/2-1.

"""
from dmx.arithmetic import *
from dmx.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, EncodedDataSequence
from numpy.random import RandomState
import dmx.utils.vector as vec
import numpy as np
import scipy.linalg
import scipy.special
from scipy.special import gammaln
import sys

from typing import Union, Sequence, Any, Optional, Dict, Tuple


def lniv_z(v: float, ln_z: float) -> float:
    """Computes the modified Bessel function approximation for large values of z.

    Args:
        v (float): Order of the Bessel function.
        ln_z (float): Logarithm of z.

    Returns:
        float: Approximation of the modified Bessel function.
    """
    return v * (ln_z - np.log(2.0)) - gammaln(np.exp(ln_z) + 1.0)


def lniv_h(v: float, ln_z: float) -> float:
    """Computes the modified Bessel function approximation for small values of z.

    Args:
        v (float): Order of the Bessel function.
        ln_z (float): Logarithm of z.

    Returns:
        float: Approximation of the modified Bessel function.
    """
    pt = ln_z * 0.0
    cc = 1.0
    rv = 1.0
    for i in range(10):
        num = np.log(4.0 * v * v - (2 * i + 1) ** 2)
        den = np.log((i + 1) * 8) + ln_z
        cc *= 1.0
        pt += (num - den)
        rv += cc * np.exp(pt)
    rv = np.log(rv)
    rv += (np.exp(ln_z) - 0.5 * (np.log(2.0 * np.pi) + ln_z))
    return rv


def lniv(v: float, ln_z: float) -> float:
    """Computes the logarithm of the modified Bessel function of the first kind.

    Args:
        v (float): Order of the Bessel function.
        ln_z (float): Logarithm of z.

    Returns:
        float: Logarithm of the modified Bessel function.
    """
    z = np.exp(ln_z)
    if np.isfinite(ln_z):
        rv0 = scipy.special.ive(v, z)
        if rv0 == 0:
            rv = lniv_z(v, ln_z)
        elif np.isposinf(rv0):
            rv = lniv_h(v, ln_z)
        else:
            rv = np.log(rv0) + z
    else:
        rv = 0
    return rv

class VonMisesFisherDistribution(SequenceEncodableProbabilityDistribution):
    """Represents a von Mises-Fisher distribution.

    Attributes:
        name (Optional[str]): Name for the distribution instance.
        dim (int): Dimension of the mean direction vector.
        mu (np.ndarray): Mean direction vector (norm should be 1.0).
        kappa (float): Positive concentration parameter.
        log_const (float): Normalizing constant for the vmf distribution.
        keys (Optional[str]): Optional keys for the distribution instance.
    """

    def __init__(self, mu: Union[Sequence[float], np.ndarray], kappa: float, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """
        Args:
            mu (Union[Sequence[float], np.ndarray]): Mean direction vector. Norm should be 1.0.
            kappa (float): Positive concentration parameter.
            name (Optional[str]): Optional name for the distribution instance.
            keys (Optional[str]): Optional keys for the distribution instance.
        """
        dim = len(mu)
        mu = np.asarray(mu).copy()

        if kappa > 0:
            log_kappa = np.log(kappa)
            cc = log_kappa * ((dim / 2.0) - 1) - lniv((dim / 2.0) - 1.0, log_kappa)

            log_kappa0 = -10000
            # This is a hack to identify the limiting constant as k -> 0
            cc0 = log_kappa0 * ((dim / 2.0) - 1) - lniv((dim / 2.0) - 1.0, log_kappa0)
            cc = (cc - cc0) + gammaln(dim / 2.0)
        else:
            cc = gammaln(dim / 2.0)

        self.name = name
        self.dim = dim
        self.mu = mu
        self.kappa = kappa
        self.log_const = cc - np.log(2.0 * pi) * (dim / 2.0)
        self.keys = keys

    def __str__(self) -> str:
        s1 = repr(list(self.mu))
        s2 = repr(self.kappa)
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return 'VonMisesFisherDistribution(%s, %s, name=%s, keys=%s)' % (s1, s2, s3, s4)

    def density(self, x: Union[Sequence[float], np.ndarray]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[float], np.ndarray]) -> float:
        z = np.asarray(x).copy()
        return np.dot(z, self.mu) * self.kappa + self.log_const

    def seq_log_density(self, x: 'VonMisesFisherEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, VonMisesFisherEncodedDataSequence):
            raise Exception('VonMisesFisherEncodedDataSequence required for seq_log_density().')
        return np.dot(x.data, self.mu) * self.kappa + self.log_const

    def sampler(self, seed: Optional[int] = None) -> 'VonMisesFisherSampler':
        return VonMisesFisherSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'VonMisesFisherEstimator':

        if pseudo_count is None:
            return VonMisesFisherEstimator(dim=self.dim, name=self.name, keys=self.keys)
        else:
            return VonMisesFisherEstimator(dim=self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'VonMisesFisherDataEncoder':
        return VonMisesFisherDataEncoder()


class VonMisesFisherSampler(DistributionSampler):
    """Sampler for the von Mises-Fisher distribution.

    Attributes:
        rng (RandomState): Random number generator.
        dist (VonMisesFisherDistribution): The vmf distribution instance.
    """
    def __init__(self, dist: 'VonMisesFisherDistribution', seed: Optional[int] = None) -> None:
        """
        Args:
            dist (VonMisesFisherDistribution): The vmf distribution instance.
            seed (Optional[int]): Random seed for sampling.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Generates samples from the vmf distribution.

        Args:
            size (Optional[int]): Number of samples to generate. If None, generates a single sample.

        Returns:
            np.ndarray: Generated samples.
        """
        rng1 = np.random.RandomState(self.rng.randint(maxrandint))
        rng2 = np.random.RandomState(self.rng.randint(maxrandint))
        rng3 = np.random.RandomState(self.rng.randint(maxrandint))

        d = self.dist.dim
        mu = self.dist.mu
        k = self.dist.kappa

        t1 = np.sqrt(4.0 * k * k + (d - 1.0) * (d - 1.0))
        # b = (d-1.0)/(t1 + 2*k)
        b = (t1 - 2 * k) / (d - 1.0)
        x0 = (1.0 - b) / (1.0 + b)

        m = (d - 1.0) / 2.0
        c = k * x0 + (d - 1.0) * np.log(1 - x0 * x0)

        sz = 1 if size is None else size
        rv = np.zeros((sz, d))

        QQ = np.zeros((d, d), dtype=float)
        QQ[0, :] = mu
        _, s, vh = scipy.linalg.svd(QQ)
        QQ = vh[np.abs(s) < 0.1, :].T

        for i in range(sz):

            t = c - 1
            u = 1

            while (t - c) < np.log(u):
                z = rng1.beta(m, m)
                u = rng2.rand()
                w = (1.0 - (1.0 + b) * z) / (1.0 - (1 - b) * z)
                t = k * w + (d - 1) * np.log(1.0 - x0 * w)

            v = rng3.randn(d - 1)
            v = np.dot(QQ, v)
            v /= np.sqrt(np.dot(v, v))
            rv[i, :] = np.sqrt(1 - w * w) * v + w * mu

        if size is None:
            return rv[0, :]
        else:
            return rv


class VonMisesFisherAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of the vmf distribution.

    Attributes:
        dim (Optional[int]): Dimension of the data.
        count (float): Total weight of observations.
        ssum (np.ndarray): Sum of weighted observations.
        key (Optional[str]): Key for the accumulator instance.
        name (Optional[str]): Name for the accumulator instance.
    """

    def __init__(self, dim: Optional[int] = None, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """
        Args:
            dim (Optional[int]): Dimension of the data.
            name (Optional[str]): Name for the accumulator instance.
            keys (Optional[str]): Key for the accumulator instance.
        """
        self.dim = dim
        self.count = 0.0

        if dim is not None:
            self.ssum = vec.zeros(dim)
        else:
            self.ssum = None

        self.key = keys
        self.name = name

    def update(self, x: Union[Sequence[float], np.ndarray], weight: float,
               estimate: Optional[VonMisesFisherDistribution]) -> None:
        if self.dim is None:
            self.dim = len(x)
            self.ssum = vec.zeros(self.dim)

        self.ssum += x * weight
        self.count += weight

    def initialize(self, x: Union[Sequence[float], np.ndarray], weight: float, rng: RandomState) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: 'VonMisesFisherEncodedDataSequence', weights: np.ndarray, estimate: Optional[VonMisesFisherDistribution]) -> None:
        if self.dim is None:
            self.dim = x.data.shape[1]
            self.ssum = vec.zeros(self.dim)

        good_w = np.bitwise_and(np.isfinite(weights), weights >= 0)
        if np.all(good_w):
            x_weight = np.multiply(x.data.T, weights)
        else:
            x_weight = np.multiply(x.data[good_w, :].T, weights[good_w])

        self.count += weights.sum()
        self.ssum += x_weight.sum(axis=1)

    def seq_initialize(self, x: 'VonMisesFisherEncodedDataSequence', weights: np.ndarray, rng: RandomState) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, np.ndarray]) -> 'VonMisesFisherAccumulator':

        if suff_stat[1] is not None and self.ssum is not None:
            self.ssum += suff_stat[1]
            self.count += suff_stat[0]

        elif suff_stat[1] is not None and self.ssum is None:
            self.ssum = suff_stat[1]
            self.count = suff_stat[0]

        return self

    def value(self) -> Tuple[float, np.ndarray]:
        return self.count, self.ssum

    def from_value(self, x: Tuple[float, np.ndarray]) -> 'VonMisesFisherAccumulator':
        self.ssum = x[1]
        self.count = x[0]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key].value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'VonMisesFisherDataEncoder':
        return VonMisesFisherDataEncoder()


class VonMisesFisherAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory class for creating VonMisesFisherAccumulator instances.

    Attributes:
        dim (Optional[int]): Dimension of the data.
        name (Optional[str]): Name for the factory instance.
        keys (Optional[str]): Key for the factory instance.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """
        Args:
            dim (Optional[int]): Dimension of the data.
            name (Optional[str]): Name for the factory instance.
            keys (Optional[str]): Key for the factory instance.
        """
        self.dim = dim
        self.keys = keys
        self.name = name

    def make(self) -> 'SequenceEncodableStatisticAccumulator':
        """Creates a new VonMisesFisherAccumulator instance.

        Returns:
            SequenceEncodableStatisticAccumulator: A new accumulator instance.
        """
        return VonMisesFisherAccumulator(dim=self.dim, name=self.name, keys=self.keys)


class VonMisesFisherEstimator(ParameterEstimator):
    """Estimator for the von Mises-Fisher distribution.

    Attributes:
        dim (Optional[int]): Dimension of the data.
        pseudo_count (Optional[float]): Pseudo count for estimation.
        name (Optional[str]): Name for the estimator instance.
        keys (Optional[str]): Key for the estimator instance.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        pseudo_count: Optional[float] = None,
        name: Optional[str] = None,
        keys: Optional[str] = None,
    ) -> None:
        """
        Args:
            dim (Optional[int]): Dimension of the data.
            pseudo_count (Optional[float]): Pseudo count for estimation.
            name (Optional[str]): Name for the estimator instance.
            keys (Optional[str]): Key for the estimator instance.
        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("VonMisesFisherEstimator requires keys to be of type 'str'.")

        self.dim = dim
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> VonMisesFisherAccumulatorFactory:
        """Creates a factory for VonMisesFisherAccumulator instances.

        Returns:
            VonMisesFisherAccumulatorFactory: A factory instance.
        """
        return VonMisesFisherAccumulatorFactory(dim=self.dim, name=self.name, keys=self.keys)

    def estimate(
        self, nobs: Optional[float], suff_stat: Tuple[float, np.ndarray]
    ) -> 'VonMisesFisherDistribution':
        """Estimates the parameters of the vmf distribution.

        Args:
            nobs (Optional[float]): Number of observations.
            suff_stat (Tuple[float, np.ndarray]): Sufficient statistics for estimation.

        Returns:
            VonMisesFisherDistribution: Estimated vmf distribution.
        """
        count, ssum = suff_stat
        dim = len(ssum)

        def _newton(p, r, k):
            """Newton's method for solving the concentration parameter."""
            k = max(sys.float_info.min, k)
            apk = np.exp(lniv(p / 2.0, np.log(k)) - lniv((p / 2.0) - 1.0, np.log(k)))
            rv = k - (apk - r) / (1.0 - apk * apk - ((p - 1.0) / k) * apk)
            rv = max(sys.float_info.min, rv)
            return rv

        ssum_norm = np.sqrt(np.dot(ssum, ssum))
        if ssum_norm > 0 and count > 0:
            rhat = ssum_norm / count
            mu = ssum / ssum_norm
            if rhat != 1.0:
                k = rhat * (dim - (rhat * rhat)) / (1.0 - (rhat * rhat))
            else:
                k = 1.1 * (dim - 1.1 ** 2) / (1.0 - (1.1 ** 2))
            for _ in range(3):  # Perform a fixed number of iterations
                k = _newton(dim, rhat, k)
        else:
            mu = np.ones(dim) / np.sqrt(dim)
            k = 0.0
        return VonMisesFisherDistribution(mu, k, name=self.name)


class VonMisesFisherDataEncoder(DataSequenceEncoder):
    """Encoder for data sequences used in the vmf distribution."""

    def __str__(self) -> str:
        """String representation of the encoder."""
        return 'VonMisesFisherDataEncoder'

    def __eq__(self, other) -> bool:
        """Checks equality between encoders.

        Args:
            other (Any): Another encoder instance.

        Returns:
            bool: True if equal, False otherwise.
        """
        return isinstance(other, VonMisesFisherDataEncoder)

    def seq_encode(self, x: Union[Sequence[float], np.ndarray]) -> 'VonMisesFisherEncodedDataSequence':
        """Encodes a sequence of data.

        Args:
            x (Union[Sequence[float], np.ndarray]): Input data sequence.

        Returns:
            VonMisesFisherEncodedDataSequence: Encoded data sequence.
        """
        rv = np.asarray(x).copy()
        return VonMisesFisherEncodedDataSequence(data=rv)


class VonMisesFisherEncodedDataSequence(EncodedDataSequence):
    """Represents an encoded data sequence for the vmf distribution.

    Attributes:
        data (np.ndarray): Encoded data array.
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        Args:
            data (np.ndarray): Encoded data array.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        """String representation of the encoded data sequence."""
        return f'VonMisesFisherEncodedDataSequence(data={self.data})'