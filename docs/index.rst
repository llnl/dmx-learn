.. dmx-learn_demo documentation master file, created by
   sphinx-quickstart on Wed Nov 13 10:20:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dmx-learn Documentation
=======================

   *The expert for mixture models on messy tables. Every model it fits is one
   you can read, rerun, and scale to a cluster.*

*dmx-learn* is a dedicated Python package for latent-variable graphical models
on heterogeneous tabular data. It fits mixtures, hierarchical mixtures, HMMs,
and tree-structured models over records that mix continuous values, counts,
categories, sets, and variable-length sequences. Each field keeps its own type,
so there is no one-hot encoding. Every fit uses exact EM or variational
updates, and the same model runs locally, with MPI, or on Spark.

Workflow: from table to agent
-----------------------------

The primary user surface is a conversation with an expert agent backed by a
small, deterministic tool API:

#. Load a table and agree on a typed schema for its columns.
#. Fit a baseline and a latent-variable model.
#. Compare, diagnose, and refine the model structure.
#. Explain the result: clusters, dependencies, anomalies, search depth, and
   embeddings.
#. Export plain dmx Python code that reproduces the fit exactly.
#. Scale the same model to MPI or Spark when the data grows.

.. note::

   The agent (``dmx-mcp``) is in development for the 1.2 release. The modeling
   engine documented below (``dmx.stats``, ``dmx.bstats``, ``dmx.torch_stats``,
   ``dmx.utils``, and ``dmx.mpi4py``) is available today and can be used
   directly from Python.

Contents
--------


.. toctree::
   :maxdepth: 1

   scope.rst
   installation.rst
   seeds.rst
   docstring_style.rst
   stats/index.rst
   bstats/index.rst
   torch_stats/index.rst
   utils/index.rst
   torch_utils/index.rst
   mpi4py/index.rst
   user_defined.rst
   mpi4py_example.rst
