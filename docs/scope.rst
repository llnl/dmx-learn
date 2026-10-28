Scope and Non-Goals
===================

dmx-learn is deliberately narrow. It is a dedicated tool for latent-variable
graphical models on heterogeneous tabular data, and it aims to be exact,
transparent, and easy to reproduce. It does not try to be a general
probabilistic or neural modeling platform. This page describes what belongs in
dmx-learn and what does not.

In Scope
--------

- **Exponential-family leaves** for typed fields: continuous values, counts,
  categories, sets, and variable-length sequences. Each field keeps its own
  type, so data never needs one-hot encoding.
- **Structured compositions of those leaves**: mixtures, hierarchical and
  joint mixtures, hidden Markov models, tree-structured models, and composites
  over record fields.
- **Exact EM or variational updates**: every fit uses closed-form
  expectation-maximization or variational (for example, conjugate-prior and
  Dirichlet process mixture) updates that you can inspect.
- **MPI and Spark scale-out**: the same sufficient-statistics path runs
  locally, with ``mpi4py``, or on Spark, so a model does not change when the
  data grows.
- **The expert agent**: a conversational surface, backed by a small,
  deterministic tool API, that profiles a table, fits and compares models,
  explains the results, and exports plain dmx code that reproduces each fit.

Non-Goals
---------

dmx-learn will not build the following. These are deliberate choices, not
missing features.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Non-goal
     - Why it is out of scope
   * - Neural densities (normalizing flows, VAEs, transformers)
     - These have no closed-form sufficient statistics, and their fits are
       hard to read or reproduce exactly.
   * - Probabilistic programming language (PPL) syntax
     - Models are built by composing typed estimators in plain Python, not
       written in a modeling language.
   * - LLM distillation
     - Large language models are never used as teachers or as model
       components. The agent orchestrates deterministic tools and does not
       produce the fit itself.
   * - MCMC and NUTS sampling
     - Inference is exact EM or variational, not sampling-based.
   * - MLOps serving
     - Model deployment, gateways, and serving infrastructure belong in other
       tools.
   * - Copulas and vines
     - Dependence between fields is modeled with latent variables and
       structure (mixtures, trees, and blocks), not with copula constructions.

The Rule for New Models
-----------------------

A new model belongs in dmx-learn only if it can be estimated with
**closed-form or variational sufficient statistics accumulated through**
``seq_update``.

In practice, the model's accumulator must:

#. collect weighted sufficient statistics from encoded data in ``seq_update``;
#. merge partial statistics with ``combine``, ``value``, and ``from_value``;
#. compute the new model in the estimator's ``estimate`` from the merged
   statistics alone, without another pass over the data.

A model that meets this rule can be used inside mixtures and composites, and
it distributes over MPI and Spark with no extra work, because those back ends
only move and merge sufficient statistics. See :doc:`user_defined` for a
worked example of writing such a distribution.
