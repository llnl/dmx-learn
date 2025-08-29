Spike and Slab Distribution
============================

Data Type: int

The spike and slab distribution places a spike of probability *p* on integer value *k* in the range of values [*min_val*, *max_val*]. The remaining *n-1* follow a uniform distribution. This distribution is great for cases where you encounter integer valued data with a spike on certain values in the range.

.. math::

   f(x|k, p) = \left\{ \begin{array}{ll}
        p, & x=k \\
        \frac{1-p}{n-1}, & x \neq k \;\& \; x \in [min\_val, max\_val] \\
        0, & else
   \end{array} \right.


In the above we have assumed the length of [*min_val*, *max_val*] is *n*. 

SpikeAndSlabDistribution
---------------------------------

.. autoclass:: dmx.stats.int_spike.SpikeAndSlabDistribution
   :members:
   :special-members: __init__

SpikeAndSlabEstimator
-----------------------------

.. autoclass:: dmx.stats.int_spike.SpikeAndSlabEstimator
   :members:
   :special-members: __init__

SpikeAndSlabSampler
--------------------------

.. autoclass:: dmx.stats.int_spike.SpikeAndSlabSampler
   :members:

