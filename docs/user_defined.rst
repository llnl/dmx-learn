.. _user_defined:
User Defined Classes
----------------------

.. code-block:: python

    import sys 
    import os
    

    sys.path.insert(0, os.path.abspath('path/to/dmx-learn'))

    import numpy as np 
    from numpy.random import RandomState
    from typing import Optional, Sequence, Dict, Union, Tuple, Any
    from dmx.arithmetic import * 
    from dmx.stats.pdist import (SequenceEncodableProbabilityDistribution, 
                                   ParameterEstimator, 
                                   DistributionSampler, 
                                   StatisticAccumulatorFactory, 
                                   SequenceEncodableStatisticAccumulator, 
                                   DataSequenceEncoder, 
                                   EncodedDataSequence)
    from dmx.utils.estimation import optimize
    import matplotlib.pyplot as plt

Outline: User-Defined dmx-learn Class
=========================================

SequenceEncodableProbabilityDistribution
=========================================
1. **Log-likelihood**: First we work out the log-likelihood for a single observation.
2. **Vectorize the log-likelihood**: Next we think about how we can format our data for fast vectorized updates.
3. **Create DataEncodedSequence**: Stores encoded data.
4. **Write the DataSequenceEncoder**: This processes our data for use with vectorized calls.
5. **Write a Sampler**: This allows us to draw samples from the distribution.

SequenceEncodableStatisticAccumulator
=======================================
1. **Determine Sufficient Statistics**: Follows from exponential family, defines the member variables of the object.
2. **Write Key Functionality**: Allows us to share parameters with other distribution instances.
3. **Update**: Define method for sufficient statistic accumulation.
4. **Initialize**: Define initialization for the sufficient statistics.
5. **Write a factory object**: Standard factory object for the accumulator.

ParameterEstimator
====================
1. **Define the wrapper**: This is what users most commonly interact with to estimate distributions.
2. **Form estimates**: Use sufficient statistics gathered in the accumulator to estimate the distribution.

Once we have filled out our univariate Gaussian distribution, we can compare it with the standard dmx-learn style mixture wrapper.

One-Dimensional Gaussian Mixture Model (GMM)
==============================================

Introduction
============

A One-Dimensional Gaussian Mixture Model (GMM) is a probabilistic model that assumes that the data is generated from a mixture of several one-dimensional Gaussian distributions with unknown parameters. This model is useful for clustering and density estimation in one-dimensional data.

Mathematical Formulation
=========================

A one-dimensional GMM is defined as follows:

.. math::

    p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)

where:
- **K** is the number of Gaussian components.
- **π_k** is the weight of the k-th Gaussian component, satisfying \(\sum_{k=1}^{K} \pi_k = 1\).
- **\mathcal{N}(x | \mu_k, \sigma_k^2)** is the one-dimensional Gaussian distribution with mean **μ_k** and variance **σ_k²**.

The one-dimensional Gaussian distribution is given by:

.. math::

    \mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

where:
- **σ²** is the variance of the Gaussian distribution.


SequenceEncodableProbabilityDistribution
==========================================

We first define a skeleton of the :class:`SequenceEncodableProbabilityDistribution`. We know that the distribution requires parameters **μ**, **σ²**, and mixing weights **π**. These must be passed to the constructor. Note that the argument **name** must also be included. We won't get into the reason for this, but make sure to include it for consistency with other dmx-learn distributions.

.. code-block:: python

    class GmmDistribution(SequenceEncodableProbabilityDistribution):
        
        def __init__(self, mu: Union[Sequence[float], np.ndarray], sigma2: Union[Sequence[float], np.ndarray], w: Union[Sequence[float], np.ndarray], name: Optional[str] = None):
            self.mu = np.asarray(mu)
            self.sigma2 = np.asarray(sigma2)
            self.w = np.asarray(w)
            self.name = name

            self.log_const = -0.5 * np.log(2.0 * np.pi)

        def __str__(self) -> str:
            return 'GmmDistribution(mu=%s, sigma2=%s, w=%s, name=%s)' % (repr(self.mu.tolist()), repr(self.sigma2.tolist()), repr(self.w.tolist()), repr(self.name))
        
        def log_density(self, x: float) -> float:
            pass

        def density(self, x: float) -> float:
            return np.exp(self.log_density(x))

        def seq_log_density(self, x) -> np.ndarray:
            pass

        def sampler(self, seed: Optional[int] = None):
            pass
        
        def dist_to_encoder(self):
            pass

        def estimator(self, pseudo_count: Optional[float] = None):
            pass

Log Density of a Univariate Gaussian Mixture Model
========================================================

The next step is generally to define the likelihood on the log scale in terms of the parameters set as member variables for the :class:`SequenceEncodableProbabilityDistribution`. This is detailed below.

A univariate Gaussian mixture model (GMM) can be expressed as:

.. math::

    p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)

where:

- :math:`K` is the number of Gaussian components.
- :math:`\pi_k` is the mixing weight for component :math:`k` (with :math:`\sum_{k=1}^{K} \pi_k = 1`).
- :math:`\mathcal{N}(x | \mu_k, \sigma_k^2)` is the Gaussian density function given by:

.. math::

    \mathcal{N}(x | \mu_k, \sigma_k^2) = \frac{1}{\sqrt{2\pi \sigma_k^2}} \exp\left(-\frac{(x - \mu_k)^2}{2\sigma_k^2}\right)

To evaluate the log density of the GMM, we can use the `logsumexp` trick to avoid numerical underflow or overflow when dealing with exponentials. The log density can be computed as follows:

1. **Compute the log densities for each component**:

   .. math::

       \log p_k(x) = \log \pi_k + \log \mathcal{N}(x | \mu_k, \sigma_k^2)

   This expands to:

   .. math::

       \log p_k(x) = \log \pi_k - \frac{1}{2} \log(2\pi \sigma_k^2) - \frac{(x - \mu_k)^2}{2\sigma_k^2}

2. **Use `logsumexp` to compute the log density of the mixture**:

   The log density of the GMM can be computed as:

   .. math::

       \log p(x) = \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2) \right)

   Using the `logsumexp` function, we can rewrite this as:

   .. math::

       \log p(x) = \log \left( \sum_{k=1}^{K} \exp\left(\log \pi_k + \log \mathcal{N}(x | \mu_k, \sigma_k^2)\right) \right)

   This can be expressed as:

   .. math::

       \log p(x) = \log \left( \sum_{k=1}^{K} \exp\left(\log \pi_k - \frac{1}{2} \log(2\pi \sigma_k^2) - \frac{(x - \mu_k)^2}{2\sigma_k^2}\right) \right)

3. **Final Expression**:

   Therefore, the log density of the univariate Gaussian mixture model can be computed using:

   .. math::

       \log p(x) = \log \left( \sum_{k=1}^{K} \exp\left(\log \pi_k - \frac{1}{2} \log(2\pi \sigma_k^2) - \frac{(x - \mu_k)^2}{2\sigma_k^2}\right) \right)

This formulation allows for stable computation of the log density of a Gaussian mixture model using the `logsumexp` trick, which is particularly useful in practice to avoid numerical issues.

.. code-block:: python

    class GmmDistribution(SequenceEncodableProbabilityDistribution):
        
        def __init__(self, mu: Union[Sequence[float], np.ndarray], sigma2: Union[Sequence[float], np.ndarray], w: Union[Sequence[float], np.ndarray], name: Optional[str] = None):
            self.mu = np.asarray(mu)
            self.sigma2 = np.asarray(sigma2)
            self.w = np.asarray(w)
            self.name = name

            self.log_const = -0.5*np.log(2.0 * np.pi)

        def __str__(self) -> str:
            return 'GmmDistribution(mu=%s, sigma2=%s, w=%s, name=%s)' % (repr(self.mu.tolist()), repr(self.sigma2.tolist()), repr(self.w.tolist()), repr(self.name))
        
        def log_density(self, x: float) -> float:
            # eval log-density for each component
            ll = self.log_const - 0.50*(x-self.mu) ** 2 / self.sigma2 - 0.5*np.log(self.sigma2) + np.log(self.w)
            max_ = np.max(ll)
            # subtract max and exponentiate
            np.exp(ll-max_, out=ll)
            # finish log-sum-exp
            rv = np.log(np.sum(ll)) + max_ 
            return rv

        def density(self, x: float) -> float:
            return np.exp(self.log_density(x))

        def seq_log_density(self, x) -> np.ndarray:
            pass

        def sampler(self, seed: Optional[int] = None):
            pass
        
        def dist_to_encoder(self):
            pass

        def estimator(self, pseudo_count: Optional[float] = None):
            pass

EncodedDataSequence and the DataSequenceEncoder objects
=======================================================

To make calculations fast, we want to think of vectorizing our ``log_density`` function call. Under the hood, this requires us to encode our data. Put another way, we want to pre-process the data passed to our object so we can perform fast vectorized operations. The :class:`DataSequenceEncoder` object is responsible for encoding our data into a format useful for repeated vectorized operations. The output data is stored in an :class:`EncodedDataSequence` object. This object also allows for type checking if desired.

A good way to think about how this will all look is to first consider a vectorized form of the ``log_density`` function. Assume the data is a one-dimensional numpy array (this is the form the encoded data will take). We can write out the density in vectorized form as seen below.

.. code-block:: python

    def seq_log_density_(x: np.ndarray) -> np.ndarray:
    
        ll = -0.5 * (x[:, None] - self.mu) ** 2 / self.sigma2 - 0.5 * np.log(self.sigma2) + self.log_const + np.log(self.w)
        max_ = np.max(ll, axis=1, keepdims=True)
        np.exp(ll - max_, out=ll)
        ll = np.log(np.sum(ll, axis=1, keepdims=False))
        ll += max_.flatten()
    
        return ll

The :class:`EncodedDataSequence` should store the processed data (which happens to be a numpy array for floats).

.. code-block:: python

    class GmmEncodedDataSequence(EncodedDataSequence):
    
        def __init__(self, data: np.ndarray):
            super().__init__(data=data)
        
        def __repr__(self) -> str:
            return f'GmmEncodedDataSequence(data={self.data})'

The :class:`DataSequenceEncoder` object must implement :meth:`__str__`, :meth:`__eq__`, and :meth:`seq_encode`. The method :meth:`seq_encode` should take the data and encode it. The result is returned as an :class:`EncodedDataSequence` object. Note that this is also the place to check for data compatibility (i.e. GMM can't handle NaN or inf values).

The :meth:`__eq__` method is implemented to check if two :class:`DataSequenceEncoder` objects are the same. This helps with avoiding multiple encodings under the hood when nesting dmx-learn functions.

.. code-block:: python

    class GmmDataEncoder(DataSequenceEncoder):

        def __str__(self) -> str:
            return 'GmmDataEncoder'
        
        def __eq__(self, other) -> bool:
            return isinstance(other, GmmDataEncoder)
        
        def seq_encode(self, x: Union[Sequence[float], np.ndarray]) -> 'GmmEncodedDataSequence':
            rv = np.asarray(x, dtype=float)
            
            if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
                raise Exception('GmmDistribution requires support x in (-inf,inf).')
            
            return GmmEncodedDataSequence(data=rv)

We can now fill in the :meth:`seq_log_density` function with proper type hints. We also fill out :meth:`dist_to_encoder`, which returns the appropriate :class:`DataSequenceEncoder` object for encoding data.

.. code-block:: python
 
 class GmmDistribution(SequenceEncodableProbabilityDistribution):
     
     def __init__(self, mu: Union[Sequence[float], np.ndarray], sigma2: Union[Sequence[float], np.ndarray], w: Union[Sequence[float], np.ndarray], name: Optional[str] = None):
         self.mu = np.asarray(mu)
         self.sigma2 = np.asarray(sigma2)
         self.w = np.asarray(w)
         self.name = name
 
         self.log_const = -0.5 * np.log(2.0 * np.pi)
 
     def __str__(self) -> str:
         return 'GmmDistribution(mu=%s, sigma2=%s, w=%s, name=%s)' % (repr(self.mu.tolist()), repr(self.sigma2.tolist()), repr(self.w.tolist()), repr(self.name))
     
     def log_density(self, x: float) -> float:
         # eval log-density for each component
         ll = self.log_const - 0.5 * (x - self.mu) ** 2 / self.sigma2 - 0.5 * np.log(self.sigma2) + np.log(self.w)
         max_ = np.max(ll)
         # subtract max and exponentiate
         np.exp(ll - max_, out=ll)
         # finish log-sum-exp
         rv = np.log(np.sum(ll)) + max_ 
         return rv
 
     def density(self, x: float) -> float:
         return np.exp(self.log_density(x))
 
     def seq_log_density(self, x: GmmEncodedDataSequence) -> np.ndarray:
         # Type check
         if not isinstance(x, GmmEncodedDataSequence):
             raise Exception('GmmEncodedDataSequence requires for seq_log_density.')
         
         # Evaluate the vectorized log-density as before
         ll = -0.5 * (x.data[:, None] - self.mu) ** 2 / self.sigma2 - 0.5 * np.log(self.sigma2) + self.log_const + np.log(self.w)
         max_ = np.max(ll, axis=1, keepdims=True)
         np.exp(ll - max_, out=ll)
         ll = np.log(np.sum(ll, axis=1, keepdims=False))
         ll += max_.flatten()
 
         return ll
     
     def dist_to_encoder(self) -> GmmDataEncoder:
         return GmmDataEncoder()
     
     def sampler(self, seed: Optional[int] = None):
         pass
 
     def estimator(self, pseudo_count: Optional[float] = None):
         pass

DistributionSampler
====================

Next we create the ``DistributionSampler``. The sampler allows us to draw samples from a fitted distribution. ``DistributionSampler`` objects are generally realized through the method ``sampler`` in the ``SequenceEncodableProbabilityDistribution``.

Sampling the GMM
=================

1. Draw a label from the mixture weights: :math:`z_i \sim \boldsymbol{\pi}`
2. Sample an observation conditioned on the label drawn: :math:`x_i \vert z_i = k \sim N\left(\mu_k, \sigma^2_k \right)`

Below is a vectorized implementation of GMM sampling in the ``sample`` method.
 
.. code-block:: python

        class GmmSampler(DistributionSampler):

            def __init__(self, dist: GmmDistribution, seed: Optional[int] = None):
                self.rng = RandomState(seed)
                self.dist = dist
            
            def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
                ncomps = len(self.dist.w)
                if size:
                    rv = np.zeros(size)
                    idx = np.arange(size)
                    self.rng.shuffle(idx)
                    z = self.rng.choice(ncomps, p=self.dist.w, replace=True, size=size)
                    z = np.bincount(z, minlength=ncomps)

                    i0 = 0
                    for xi, xc in enumerate(z):
                        if xc > 0:
                            i1 = i0 + xc
                            rv[idx[i0:i1]] = self.rng.normal(loc=self.dist.mu[xi], scale=np.sqrt(self.dist.sigma2[xi]), size=xc)
                            i0 += xc
                        
                    return rv 
                else:
                    z = self.rng.choice(ncomps, p=self.dist.w)
                    rv = self.rng.randn() * np.sqrt(self.dist.sigma2[z]) + self.dist.mu[z]

                    return float(rv)

We can now update the ``sampler`` function in the ``SequenceEncodableProbabilityDistribution``.

.. code-block:: python

    class GmmDistribution(SequenceEncodableProbabilityDistribution):
        
        def __init__(self, mu: Union[Sequence[float], np.ndarray], sigma2: Union[Sequence[float], np.ndarray], w: Union[Sequence[float], np.ndarray], name: Optional[str] = None):
            self.mu = np.asarray(mu)
            self.sigma2 = np.asarray(sigma2)
            self.w = np.asarray(w)
            self.name = name

            self.log_const = -0.5*np.log(2.0 * np.pi)

        def __str__(self) -> str:
            return 'GmmDistribution(mu=%s, sigma2=%s, w=%s, name=%s)' % (repr(self.mu.tolist()), repr(self.sigma2.tolist()), repr(self.w.tolist()), repr(self.name))
        
        def log_density(self, x: float) -> float:
            # eval log-density for each component
            ll = self.log_const - 0.5*(x-self.mu) ** 2 / self.sigma2 - 0.5*np.log(self.sigma2) + np.log(self.w)
            max_ = np.max(ll)
            # subtract max and exponentiate
            np.exp(ll-max_, out=ll)
            # finish log-sum-exp
            rv = np.log(np.sum(ll)) + max_ 
            return rv

        def density(self, x: float) -> float:
            return np.exp(self.log_density(x))

        def seq_log_density(self, x: GmmEncodedDataSequence) -> np.ndarray:
            # Type check
            if not isinstance(x, GmmEncodedDataSequence):
                raise Exception('GmmEncodedDataSequence requires for seq_log_density.')
            
            # Evaluate the vetorized log-density as before
            ll = -0.5*(x.data[:, None] - self.mu)**2 / self.sigma2 - 0.5*np.log(self.sigma2) + self.log_const + np.log(self.w)
            max_ = np.max(ll, axis=1, keepdims=True)
            np.exp(ll-max_, out=ll)
            ll = np.log(np.sum(ll, axis=1, keepdims=False))
            ll += max_.flatten()

            return ll
        
        def dist_to_encoder(self) -> GmmDataEncoder:
            return GmmDataEncoder()
        
        def sampler(self, seed: Optional[int] = None) -> GmmSampler:
            return GmmSampler(dist=self, seed=seed)

        def estimator(self, pseudo_count: Optional[float] = None):
            pass


We need to write the estimator to complete the distribution. We will return to this later.

SequenceEncodableStatisticAccumulator
=======================================
Next we will write the ``SequenceEncodableStatisticAccumulator`` which is used to aggregate sufficient statistics. To identify the sufficient statistics and the calculation involved in tracking them, we can refer to the exponential family form of the distribution. In the case of the univariate GMM, it is easier to refer back to the E-step of the EM algorithm.

Expectation-Maximization Algorithm
===================================
To estimate the parameters of a one-dimensional GMM, we typically use the Expectation-Maximization (EM) algorithm, which consists of two steps:

1. **Expectation Step (E-step)**: Calculate the expected value of the log-likelihood function, given the current estimates of the parameters.

   .. math::

       \gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n | \mu_j, \sigma_j^2)}

where :math:`\gamma_{nk}` is the responsibility that component ``k`` takes for data point ``n``.

2. **Maximization Step (M-step)**: Update the parameters using the expected values computed in the E-step.

   .. math::

       \pi_k = \frac{N_k}{N}

   .. math::

       \mu_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk} x_n

   .. math::

       \sigma_k^2 = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk} (x_n - \mu_k)^2 = \gamma_{nk} x_n^2 - 2 x_n \mu_k + \mu_k^2

where :math:`N_k = \sum_{n=1}^{N} \gamma_{nk}` is the effective number of points assigned to component :math:`k`.

Sufficient Statistics
=====================
In dmx-learn, the accumulator tracks the sufficient statistics, which are used to perform the estimation step. Following the above, we see that 

.. math::

   \sum_{n=1}^{N} \gamma_{nk}

.. math::

   \sum_{n=1}^{N} \gamma_{nk} x_n

.. math::

   \sum_{n=1}^{N} \gamma_{nk} x_n^2

are required to estimate :math:`\pi_k`, :math:`\mu_k`, :math:`\sigma^2_k`. Our accumulator class will aggregate these sufficient statistics. We will denote the sufficient statistics with member variables: ``comp_counts`` = :math:`\sum_{n=1}^{N} \gamma_{nk}`, ``x`` = :math:`\sum_{n=1}^{N} \gamma_{nk} x_n`, and ``x2`` = :math:`\sum_{n=1}^{N} \gamma_{nk} x_n^2`. Note that each of these member variables are ``k`` dimensional vectors where ``k`` is the number of mixture components.

Below is a skeleton of the ``SequenceEncodableStatisticAccumulator``. The methods ``value``, ``combine``, and ``from_value`` all must be implemented. They return the sufficient stats from the accumulator, combine the current suff stats of the object instance with another set of suff stats, and assign the Accumulator instance suff stats respectively.

Another interesting thing to note is the passing of the variable ``keys``. For the Gaussian Mixture model implementation, we allow the user to pass keys specifying whether the mixture weights or means and variances should be shared across any other distributions (with matching keys). You must implement the two methods ``key_merge`` and ``key_replace``.

The member function ``acc_to_encoder`` is similar to ``dist_to_encoder`` from the distribution class. It must be included here, as we currently require data encodings to be available in the initialization step.

.. code-block:: python

    class GmmAccumulator(SequenceEncodableStatisticAccumulator):

        def __init__(self, num_comps: int, keys: Optional[Tuple[Optional[str], Optional[str]]] = None, name: Optional[str] = None):
            self.x = np.zeros(num_comps)
            self.x2 = np.zeros(num_comps)
            self.comp_counts = np.zeros(num_comps)
            self.ncomps = num_comps
            self.weight_keys = keys[0] if keys else None
            self.param_keys = keys[1] if keys else None

        def initialize(self, x: float, weight: float, rng: Optional[RandomState]):
            pass

        def update(self, x: float, weight: float, estimate: GmmDistribution):
            pass

        def seq_initialize(self, x: GmmEncodedDataSequence, weights: np.ndarray, rng: Optional[RandomState]):
            pass

        def seq_update(self, x: GmmEncodedDataSequence, weights: np.ndarray, estimate: GmmDistribution):
            pass

        # Return the sufficient statistics
        def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            return self.comp_counts, self.x, self.x2
        
        # Combine suff stats with the accumulators
        def combine(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]):
            self.comp_counts += x[0]
            self.x += x[1]
            self.x2 += x[2]

            return self
        
        # assign sufficient statistics from a value
        def from_value(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]):
            self.comp_counts = x[0]
            self.x = x[1]
            self.x2 = x[2]

        # This allows for merging of suff stats with parameters that have the same keys
        def key_merge(self, stats_dict: Dict[str, Any]):
            if self.weight_keys is not None:
                if self.weight_keys in stats_dict:
                    self.comp_counts += stats_dict[self.weight_keys]
                else:
                    stats_dict[self.weight_keys] = self.comp_counts
            
            if self.param_keys is not None:
                if self.param_keys in stats_dict:
                    x, x2 = stats_dict[self.param_keys]
                    self.x += x
                    self.x2 += x2
                else:
                    stats_dict[self.param_keys] = (self.x, self.x2)

        # Set the sufficient statistics of the accumulator to suff stats with matching keys.
        def key_replace(self, stats_dict: Dict[str, Any]):
            if self.weight_keys is not None:
                if self.weight_keys in stats_dict:
                    self.comp_counts = stats_dict[self.weight_keys]

            if self.param_keys is not None:
                if self.param_keys in stats_dict:
                    self.param_keys = stats_dict[self.param_keys]
        
        # Create a DataSequenceEncoder object for seq initialize encodings.
        def acc_to_encoder(self):
            return GmmDataEncoder()

            
Implementing Update
===================
**Recall: Expectation Step (E-step)**: Calculate the expected value of the log-likelihood function, given the current estimates of the parameters.

.. math::

   \gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n | \mu_j, \sigma_j^2)}

where :math:`\gamma_{nk}` is the responsibility that component ``k`` takes for data point ``n``.

For the ``update`` function, we must calculate the posterior :math:`\gamma_{nk}` for each observation :math:`x_n`. This is done using a log-sum-exp trick. Once we have :math:`\gamma_{nk}`, we simply update the accumulators sufficient stats ``x``, ``x2``, and ``comp_counts`` accordingly. Note that ``weight`` is also multiplied to the :math:`\gamma_{nk}` values, as this allows for nesting with other dmx-learn classes.

We must also implement the vectorized ``seq_update``, which takes the ``GmmEncodedDataSequence`` previously defined.

.. code-block:: python

    class GmmAccumulator(SequenceEncodableStatisticAccumulator):

        def __init__(self, num_comps: int, keys: Optional[Tuple[Optional[str], Optional[str]]] = None, name: Optional[str] = None):
            self.x = np.zeros(num_comps)
            self.x2 = np.zeros(num_comps)
            self.comp_counts = np.zeros(num_comps)
            self.ncomps = num_comps
            self.weight_keys = keys[0] if keys else None
            self.param_keys = keys[1] if keys else None

        def update(self, x: float, weight: float, estimate: GmmDistribution):
            mu, s2, w = estimate.mu, estimate.sigma2, estimate.w

            gamma = -0.5*(x-mu)**2 / s2 - 0.5*np.log(s2) + np.log(w)
            max_ = np.max(gamma)

            if not np.isinf(max_):
                # log-sum-exp back to exp
                gamma = np.exp(gamma-max_, out=gamma)
                gamma /= np.sum(gamma)
                # multiply by weight to allow for down stream nesting with other dmx classes
                gamma *= weight
                self.comp_counts += gamma
                self.x += x*gamma
                self.x2 += x**2*gamma 

        def seq_update(self, x: GmmEncodedDataSequence, weights: np.ndarray, estimate: GmmDistribution):
            mu, s2, log_w = estimate.mu, estimate.sigma2, np.log(estimate.w)
            gammas = -0.5*(x.data[:, None] - mu)**2 / s2 - 0.5*np.log(s2)
            gammas += log_w[None, :]

            # check for 0 weights
            zw = np.isinf(log_w)
            if np.any(zw):
                gammas[:, zw] = -np.inf
            
            max_ = np.max(gammas, axis=1, keepdims=True)

            # correct for any posterior containing all -np.inf values.
            bad_rows = np.all(np.isinf(gammas), axis=1).flatten()
            gammas[bad_rows, :] = log_w.copy()
            max_[bad_rows] = np.max(log_w)

            # logsumexp and multiply by weights passed 
            gammas -= max_
            np.exp(gammas, out=gammas)
            np.sum(gammas, axis=1, keepdims=True, out=max_)
            np.divide(weights[:, None], max_, out=max_)
            gammas *= max_

            # update the sufficient stats
            wsum = gammas.sum(axis=0)
            self.comp_counts += wsum
            self.x += np.dot(x.data, gammas)
            self.x2 += np.dot(x.data**2, gammas)

        def initialize(self, x: float, weight: float, rng: Optional[RandomState]):
            pass

        def seq_initialize(self, x: GmmEncodedDataSequence, weights: np.ndarray, rng: Optional[RandomState]):
            pass

        # Return the sufficient statistics
        def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            return self.comp_counts, self.x, self.x2
        
        # Combine suff stats with the accumulators
        def combine(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]):
            self.comp_counts += x[0]
            self.x += x[1]
            self.x2 += x[2]

            return self
        
        # assign sufficient statistics from a value
        def from_value(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]):
            self.comp_counts = x[0]
            self.x = x[1]
            self.x2 = x[2]

        # This allows for merging of suff stats with parameters that have the same keys
        def key_merge(self, stats_dict: Dict[str, Any]):
            if self.weight_keys is not None:
                if self.weight_keys in stats_dict:
                    self.comp_counts += stats_dict[self.weight_keys]
                else:
                    stats_dict[self.weight_keys] = self.comp_counts
            
            if self.param_keys is not None:
                if self.param_keys in stats_dict:
                    x, x2 = stats_dict[self.param_keys]
                    self.x += x
                    self.x2 += x2
                else:
                    stats_dict[self.param_keys] = (self.x, self.x2)

        # Set the sufficient statistics of the accumulator to suff stats with matching keys.
        def key_replace(self, stats_dict: Dict[str, Any]):
            if self.weight_keys is not None:
                if self.weight_keys in stats_dict:
                    self.comp_counts = stats_dict[self.weight_keys]

            if self.param_keys is not None:
                if self.param_keys in stats_dict:
                    self.param_keys = stats_dict[self.param_keys]
        
        # Create a DataSequenceEncoder object for seq initialize encodings.
        def acc_to_encoder(self):
            return GmmDataEncoder()

Initialize
===========
The sufficient statistics must be initialized. The `SequenceEncodableStatisticAccumulator` object allows for randomized initialization of the sufficient statistics for the observed data. The methods required here are `initialize` and the vectorized version `seq_initialize`. It is up to you to define how these methods are implemented. Below we outline the method for the case of the univariate Gaussian mixture model.

Initialization of GMM sufficient statistics
=============================================
1. Draw :math:`\boldsymbol{\gamma}_i \sim \text{Dirichlet}\left(\left( \frac{1}{k}, \frac{1}{k}, \ldots, \frac{1}{k} \right)\right)`.

2. Multiply by passed `weight` (this is for nesting with other dmx-learn distributions):

   .. math::

      \boldsymbol{\gamma}_i = \boldsymbol{\gamma}_i * \text{weight}_i

3. Update the sufficient statistic member variables of the accumulator:

   .. math::

      \text{comp_counts}[k] \text{ += } \gamma_{i, k}

   .. math::

      x[k] \text{ += } \gamma_{i,k} * x_i

   .. math::

      x2[k] \text{ += } \gamma_{i,k} * x^2_i

This is quite trivial to vectorize and implement in `seq_initialize` using our encoded data previously defined as `GmmEncodedDataSequence`. The code is filled out below. One small comment, the value `c` is used in the initialization to avoid numeric issues encountered when sampling from a Dirichlet distribution.

.. code-block:: python

 class GmmAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_comps: int, keys: Optional[Tuple[Optional[str], Optional[str]]] = None, name: Optional[str] = None):
        
        self.x = np.zeros(num_comps)
        self.x2 = np.zeros(num_comps)
        self.comp_counts = np.zeros(num_comps)
        self.ncomps = num_comps
        self.weight_keys = keys[0] if keys else None
        self.param_keys = keys[1] if keys else None

    def update(self, x: float, weight: float, estimate: GmmDistribution):
        mu, s2, w = estimate.mu, estimate.sigma2, estimate.w

        gamma = -0.5*(x-mu)**2 / s2 - 0.5*np.log(s2) + np.log(w)
        max_ = np.max(gamma)

        if not np.isinf(max_):
            # log-sum-exp back to exp
            gamma = np.exp(gamma-max_, out=gamma)
            gamma /= np.sum(gamma)
            # multiply by weight to allow for down stream nesting with other dmx classes
            gamma *= weight
            self.comp_counts += gamma
            self.x += x*gamma
            self.x2 += x**2*gamma 

    def seq_update(self, x: GmmEncodedDataSequence, weights: np.ndarray, estimate: GmmDistribution):

        mu, s2, log_w = estimate.mu, estimate.sigma2, np.log(estimate.w)
        gammas = -0.5*(x.data[:, None] - mu)**2 / s2 - 0.5*np.log(s2)
        gammas += log_w[None, :]

        # check for 0 weights
        zw = np.isinf(log_w)
        if np.any(zw):
            gammas[:, zw] = -np.inf
        
        max_ = np.max(gammas, axis=1, keepdims=True)

        # correct for any posterior containing all -np.inf values.
        bad_rows = np.isinf(max_.flatten())
        gammas[bad_rows, :] = log_w.copy()
        max_[bad_rows] = np.max(log_w)

        # logsumexp and multiply by weights passed 
        gammas -= max_
        np.exp(gammas, out=gammas)
        np.sum(gammas, axis=1, keepdims=True, out=max_)
        np.divide(weights[:, None], max_, out=max_)
        gammas *= max_

        # update the sufficient stats
        wsum = gammas.sum(axis=0)
        self.comp_counts += wsum
        self.x += np.dot(x.data, gammas)
        self.x2 += np.dot(x.data**2, gammas)

    def initialize(self, x: float, weight: float, rng: RandomState):

        # generate random posterior values
        c = 20 ** 2 if self.ncomps > 20 else self.ncomps**2
        ww = rng.dirichelt(np.ones(self.ncomps) / c)
        ww *= weight

        # update suff stats
        self.x += x * ww
        self.x2 += x**2 * ww
        self.comp_counts += ww


    def seq_initialize(self, x: GmmEncodedDataSequence, weights: np.ndarray, rng: Optional[RandomState]):

        # only generate random posteriors for weights that are non-zero
        sz = len(weights)
        c = 20 ** 2 if self.ncomps > 20 else self.ncomps ** 2

        ww = rng.dirichlet(np.ones(self.ncomps) / c, size=sz)
        ww *= weights[:, None]
        w_sum = ww.sum(axis=0)

        # initialize suff stats
        self.comp_counts += w_sum
        self.x += np.dot(x.data, ww)
        self.x2 += np.dot(x.data ** 2, ww)


    # Return the sufficient statistics
    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.comp_counts, self.x, self.x2
    
    # Combine suff stats with the accumulators
    def combine(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.comp_counts += x[0]
        self.x += x[1]
        self.x2 += x[2]

        return self
    
    # assign sufficient statistics from a value
    def from_value(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.comp_counts = x[0]
        self.x = x[1]
        self.x2 = x[2]

    # This allows for merging of suff stats with parameters that have the same keys
    def key_merge(self, stats_dict: Dict[str, Any]):
        
        if self.weight_keys is not None:
            if self.weight_keys in stats_dict:
                self.comp_counts += stats_dict[self.weight_keys]
            else:
                stats_dict[self.weight_keys] = self.comp_counts
        
        if self.param_keys is not None:
            if self.param_keys in stats_dict:
                x, x2 = stats_dict[self.param_keys]
                self.x += x
                self.x2 += x2
            else:
                stats_dict[self.param_keys] = (self.x, self.x2)

    # Set the sufficient statistics of the accumulator to suff stats with matching keys.
    def key_replace(self, stats_dict: Dict[str, Any]):
        if self.weight_keys is not None:
            if self.weight_keys in stats_dict:
                self.comp_counts = stats_dict[self.weight_keys]

        if self.param_keys is not None:
            if self.param_keys in stats_dict:
                self.param_keys = stats_dict[self.param_keys]
    
    # Create a DataSequenceEncoder object for seq initialize encodings.
    def acc_to_encoder(self):
        return GmmDataEncoder()
    
StatisticAccumulatorFactory
============================

In programming, a factory object is a design pattern used to create instances of objects. Instead of calling a constructor directly to create an object, a factory provides a method that returns an instance of a class. We define the `StatisticAccumulatorFactory` as a method for creating `SequenceEncodableStatisticAccumulator` objects.

.. code-block:: python

 class GmmAccumulatorFactory(StatisticAccumulatorFactory):

    # same constructor as the GmmAccumulator object
    def __init__(self, num_comps: int, keys: Optional[Tuple[Optional[str], Optional[str]]] = None, name: Optional[str] = None):
        self.num_comps = num_comps
        self.keys = keys
        self.name = name

    # creates a GmmAccumulator object 
    def make (self) -> 'GmmAccumulator':
        return GmmAccumulator(num_comps=self.num_comps, keys=self.keys, name=self.name)
    

The only thing left to do is to implement the `ParameterEstimator` class and fill out the `estimator` function call in the `SequenceEncodableProbabilityDistribution`. So let's implement the `ParameterEstimator` and tie everything together.

.. code-block:: python

 class GmmEstimator(ParameterEstimator):

    def __init__(self, num_comps: int, suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]] = (None, None, None), pseudo_count: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None), keys: Tuple[Optional[str], Optional[str]] = (None, None)):
        self.ncomps = num_comps
        self.suff_stat = suff_stat
        self.pseudo_count = pseudo_count
        self.keys = keys

    def accumulator_factory(self) -> GmmAccumulatorFactory:
        return GmmAccumulatorFactory(num_comps=self.ncomps, keys=self.keys)
    
    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> GmmDistribution:
        counts, xw, x2w = suff_stat

        # regularize weights without suff stat passed to it
        if self.pseudo_count[0] and not self.suff_stat[0]:
            p = self.pseudo_count[0] / self.ncomps
            w = counts + p
            w /= w.sum()
        # regularize weights with suff stat passed
        elif self.pseudo_count[0] and self.suff_stat[0]:
            w = (counts + self.suff_stat[0]*self.pseudo_count[0]) / (counts.sum() + self.pseudo_count[0]*self.suff_stat[0].sum())
        # dont regularize weights
        else:
            wsum = counts.sum()

            if wsum == 0.0:
                w = np.ones(self.ncomps) / float(self.ncomps)
            else:
                w = counts.copy() / wsum

        # flatten the mean estimates
        if self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            mu = (xw + self.pseudo_count[1] * self.suff_stat[1]) / (counts + self.pseudo_count * np.sum(self.suff_stats[1]))
        else:
            wsum = counts.copy()
            wsum[wsum==0.0] = 1.0
            mu = xw / wsum
        
        # flatten/regularize the variance estimates
        if self.pseudo_count[2] and self.suff_stat[2]:
            s2 = (x2w - mu**2 * counts * self.pseudo_count[2] * self.suff_stat[2]) / (counts + self.pseudo_count[2] * np.sum(self.suff_stat[2]))
        else:
            wsum = counts.copy()
            wsum[wsum==0.0] = 1.0
            s2 = x2w / wsum - mu * mu 

        return GmmDistribution(mu=mu, sigma2=s2, w=w)
        
Completed SequenceEncodableProbabilityDistribution
==================================================

We can now fill out `estimate` in the `SequenceEncodableProbabilityDistribution`. This completes the GMM class for use in dmx-learn!

.. code-block:: python

 class GmmDistribution(SequenceEncodableProbabilityDistribution):
    
    def __init__(self, mu: Union[Sequence[float], np.ndarray], sigma2: Union[Sequence[float], np.ndarray], w: Union[Sequence[float], np.ndarray], name: Optional[str] = None):
        self.mu = np.asarray(mu)
        self.sigma2 = np.asarray(sigma2)
        self.w = np.asarray(w)
        self.name = name

        self.log_const = -0.5*np.log(2.0 * np.pi)

    def __str__(self) -> str:
        return 'GmmDistribution(mu=%s, sigma2=%s, w=%s, name=%s)' % (repr(self.mu.tolist()), repr(self.sigma2.tolist()), repr(self.w.tolist()), repr(self.name))
    
    def log_density(self, x: float) -> float:
        # eval log-density for each component
        ll = self.log_const - 0.5*(x-self.mu) ** 2 / self.sigma2 - 0.5*np.log(self.sigma2) + np.log(self.w)
        max_ = np.max(ll)
        # subtract max and exponentiate
        np.exp(ll-max_, out=ll)
        # finish log-sum-exp
        rv = np.log(np.sum(ll)) + max_ 
        return rv

    def density(self, x: float) -> float:
        return np.exp(self.log_density(x))

    def seq_log_density(self, x: GmmEncodedDataSequence) -> np.ndarray:
        # Type check
        if not isinstance(x, GmmEncodedDataSequence):
            raise Exception('GmmEncodedDataSequence requires for seq_log_density.')
        
        # Evaluate the vetorized log-density as before
        ll = -0.5*(x.data[:, None] - self.mu)**2 / self.sigma2 + self.log_const + np.log(self.w) - 0.5*np.log(self.sigma2)
        max_ = np.max(ll, axis=1, keepdims=True)
        np.exp(ll-max_, out=ll)
        ll = np.log(np.sum(ll, axis=1, keepdims=False))
        ll += max_.flatten()

        return ll
    
    def dist_to_encoder(self) -> GmmDataEncoder:
        return GmmDataEncoder()
    
    def sampler(self, seed: Optional[int] = None) -> GmmSampler:
        return GmmSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None):
        pc = (pseudo_count, pseudo_count, pseudo_count)
        return GmmEstimator(num_comps=len(self.w), pseudo_count=pc)


Proof of Concept
================

Let's walk through the standard dmx-learn pipeline. First we declare the model and simulate some data. We then declare the estimator and fit the model using `optimize`.

.. code-block:: python

        N = 1000
        k = 3
        w = np.ones(k) / float(k)
        mu = np.linspace(-5, 5, k)
        sigma2 = np.ones(k) / 1.

        dist = GmmDistribution(mu=mu, sigma2=sigma2, w=w)

        sampler = dist.sampler(seed=1)
        data = sampler.sample(N)

        est = GmmEstimator(num_comps=k)
        fit = optimize(data=data, estimator=est, max_its=10000, print_iter=100, rng=RandomState(1))

This wraps things up. Keep in mind you are free to add other member functions to the ``SequenceEncodableProbabilityDistribution`` class that improve your quality of life. One thing you can try out is implementing a ``posterior`` and vectorized version ``seq_posterior`` that computes the posterior probability of component membership.

