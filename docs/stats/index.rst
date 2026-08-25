``dmx.stats`` API
=================

.. automodule:: dmx.stats

The guides below combine model introductions with the API reference pages for
the most commonly used distributions. The remaining groups expose specialized
models and integration helpers directly from their cleaned source docstrings.

Package-level workflows
-----------------------

.. autofunction:: dmx.stats.initialize

.. autofunction:: dmx.stats.estimate

.. autofunction:: dmx.stats.seq_encode

.. autofunction:: dmx.stats.seq_log_density

.. autofunction:: dmx.stats.seq_log_density_sum

.. autofunction:: dmx.stats.seq_estimate

.. autofunction:: dmx.stats.seq_initialize

Core interfaces and model guides
--------------------------------

.. toctree::
   :maxdepth: 2

   /pdist
   /base_distributions
   /combinators
   /mixture_models

Additional distributions and wrappers
-------------------------------------

.. toctree::
   :maxdepth: 1

   dirac_length
   int_edit_setdist
   int_edit_stepsetdist
   null_dist
   select
   spearman_rho
   vmf
   weighted

Mixture and topic models
------------------------

.. toctree::
   :maxdepth: 1

   dmvn_mixture
   gmm
   int_plsi
   lda
   ss_mixture

Association and sequential models
---------------------------------

.. toctree::
   :maxdepth: 1

   hidden_association
   icltree
   int_hidden_association
   int_hidden_markov
   int_markovchain
   look_back_hmm
   sparse_markov_transform
   tree_hmm

Distributed helpers
-------------------

.. toctree::
   :maxdepth: 1

   rdd_sampler
