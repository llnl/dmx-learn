.. _base_distributions:

Base Distributions
===========================
Below is a list of distributions aranged by data type. This should help pick a suitable probability distribution for you data. Note that there may be restrictions on the support of each distribution. This information will be contained in the associated references for each individual distribution.

Data Type: int
--------------------------------------------------

.. toctree::
   :maxdepth: 1

   /stats/binomial.rst
   /stats/geometric.rst
   /stats/poisson.rst
   /stats/intrange.rst
   /stats/int_spike.rst


Data Type: float
------------------------

.. toctree::
   :maxdepth: 1

   /stats/gaussian.rst
   /stats/exponential.rst
   /stats/gamma.rst
   /stats/log_gaussian.rst


Data Type: Sequence[float]
------------------------------

.. toctree::
   :maxdepth: 1

   /stats/dmvn.rst
   /stats/mvn.rst
   /stats/dirichlet.rst


Data Type: str
--------------------------

.. toctree::
   :maxdepth: 1

   /stats/categorical.rst


Data Type: Sequence[int]
------------------------------

.. toctree::
   :maxdepth: 1

   /stats/intsetdist.rst


Data Type: Sequence[str]
------------------------------

.. toctree::
   :maxdepth: 1

   /stats/setdist.rst


Data Type: Sequence[Tuple[int, float]]
----------------------------------------

.. toctree::
   :maxdepth: 1

   /stats/intmultinomial.rst


Data Type: Sequence[Tuple[int, str]]
----------------------------------------

.. toctree::
   :maxdepth: 1

   /stats/catmultinomial.rst

Data Type: Sequence[Any]
-----------------------------------

.. toctree::
   :maxdepth: 1

   /stats/markovchain.rst
