# Repo Entry Points

Use this file when you already know the model family and need the fastest path
to repo-native fitting, diagnostics, or example code.

## Contents

- Core fitting helpers
- Bayesian and automatic fitting helpers
- Vectorized model use
- Diagnostics helpers
- Embedding helpers
- Example files to adapt
- Cautions

## Core Fitting Helpers

### `src/dmx/utils/estimation.py`

- `partition_data`: split local observations into train and validation partitions.
- `optimize`: standard local EM loop for one estimator.
- `best_of`: rerun EM from multiple initializations and keep the best validation fit.
- `empirical_kl_divergence`: compare two fitted models on encoded data when a
  reference model exists.

Typical local pattern:

1. Build an explicit estimator.
2. Split data with `partition_data` if needed.
3. Fit with `optimize`.
4. Upgrade to `best_of` if initialization sensitivity appears.
5. Evaluate on held-out data with encoded likelihood calls.

This remains the default route for ordinary non-Bayesian local modeling.

## Bayesian And Automatic Fitting Helpers

### `src/dmx/bstats/bestimation.py`

- `optimize`: local variational optimization for an explicit `dmx.bstats`
  estimator, including truncated DPM estimators.
- `best_of`: repeated local variational fits with validation selection.
- `partition_data`: local train/validation partitioning for Bayesian models.
- `empirical_kl_divergence`: comparison on encoded support shared by two
  Bayesian models.

Use these helpers when priors, expected log-density behavior, or direct control
of the variational estimator tree matters.

### `src/dmx/utils/automatic.py`

- `get_estimator(data, use_bstats=True)`: infer a supported `dmx.bstats`
  estimator tree. Keep `use_bstats=True` explicit so the intended surface is
  unambiguous.
- `get_dpm_mixture(data, estimator=..., ...)`: fit a local truncated DPM with
  `dmx.bstats.bestimation.optimize`, discard components below the requested
  count threshold, and return a finite `dmx.bstats.MixtureDistribution`.

Use automatic construction when it is part of the requirement, especially as
the front end to a Bayesian mixture. If the model family is already known and
the task is ordinarily non-Bayesian, construct the `dmx.stats` estimator
explicitly instead.

### DPM Over Composite Distributions

Treat this as the primary high-value `bstats` pattern for heterogeneous local
data with an unknown or flexible latent subtype count:

1. Build a `dmx.bstats.CompositeEstimator` from field-level `bstats`
   estimators, or obtain the supported structure with
   `get_estimator(..., use_bstats=True)`.
2. Supply that estimator to `get_dpm_mixture`, or repeat it inside an explicit
   `dmx.bstats.DirichletProcessMixtureEstimator`.
3. Use `dmx.bstats.bestimation.optimize` for explicit variational control; use
   `get_dpm_mixture` when automatic fitting and finite-mixture conversion are
   desired.

Tests that exercise these cleaned routes:

- `tests/bstats/dpm_test.py`
- `tests/bstats/composite_test.py`
- `tests/bstats/structural_test.py`
- `tests/bstats/discrete_primitives_test.py`

All helpers in this section are local. `get_dpm_mixture_mpi` and modules under
`src/dmx/mpi4py` are MPI-specific counterparts and are not interchangeable
with this local recipe.

## Composite Mixture Pattern

For heterogeneous local problems, prefer this construction order:

1. Choose a field-level estimator for each attribute.
2. Combine them with `CompositeEstimator`.
3. Add latent subtypes with `MixtureEstimator`.
4. Only after that decide whether labels or conditioning variables need an
   outer `ConditionalDistributionEstimator`.

This keeps the composite mixture as the main modeling object instead of hiding
it inside many isolated conditional branches.

## Keyed Shared Components

### `examples/stats_examples/mixture_example.py`

Use this file as the reference implementation for keyed mixtures where
components are shared but weights can differ.

Key rule:

- `MixtureEstimator(..., keys=(None, "shared-components"))`

Interpretation:

- first key position controls mixture weights
- second key position controls component distributions
- setting the component key ties the components across contexts while leaving
  weights free to vary

For labeled heterogeneous data, use this to build a low-rank approximation to
the conditional distribution:

1. Build one mixture-of-composites template.
2. Tie the mixture components with a shared component key.
3. Let each label or condition retain its own mixture weights.
4. Train on all available labels when shared structure estimation is valuable.
5. Restrict evaluation or ranking to the target label subset if needed.

## Vectorized Model Use

### `src/dmx/stats/__init__.py`

- `seq_encode`: encode local data using a model or encoder.
- `seq_initialize`: initialize a model from encoded data and an estimator.
- `seq_estimate`: run one estimation step from encoded data.
- `seq_log_density`: compute vectorized log densities.
- `seq_log_density_sum`: compute aggregate likelihood summaries.

After fitting, prefer vectorized paths over repeated scalar `log_density` calls.

## Diagnostics Helpers

### `src/dmx/utils/metrics.py`

- `classify`: evaluate conditional or label-aware models on local labeled data.
- `roc_curve`: build ROC coordinates from positive and negative scores.
- `roc_percentiles`: inspect ROC operating points at requested percentiles.
- `ranking_depth`: measure where the correct item lands in a scored ranking.

### `src/dmx/utils/pvalues.py`

- `binomial_rank`: approximate composite binomial rank tails.

Use `pvalues.py` only when the user explicitly wants this narrow style of
approximate significance or rank calculation. Do not present it as a generic
LLR testing framework.

## Embedding Helpers

### `src/dmx/utils/automatic.py`

- `prepare_mixture_model`: fit or validate a mixture model and return
  posteriors. With no supplied model, it uses the local `get_dpm_mixture`
  `bstats` route.

### `src/dmx/utils/htsne.py`

- `htsne`: produce heterogeneous t-SNE-style embeddings from mixture posteriors.

### `src/dmx/utils/humap.py`

- `humap`: produce UMAP embeddings from mixture posteriors.

Recommended embedding pattern:

1. Confirm the user wants exploratory visualization, not only a predictive fit.
2. Fit or reuse a mixture model.
3. Inspect posteriors and component counts first.
4. Run `htsne` or `humap`.
5. Return both the embedding array and the fitted mixture model.

## Example Files To Adapt

- `examples/stats_examples/gaussian_example.py`: smallest local fit-and-score example.
- `examples/stats_examples/mixture_example.py`: mixed latent structure and keyed estimators.
- `examples/detailed_estimation_example.py`: validation likelihood tracking and repeated estimation.
- `examples/stats_examples/spearman_rho_example.py`: ranking model fit.
- `examples/htsne_example.py`: local embedding workflow.
- `tests/bstats/dpm_test.py`: local variational DPM and automatic conversion.
- `tests/bstats/composite_test.py`: Bayesian composites inside mixture and DPM
  containers.
- `tests/bstats/structural_test.py`: automatic structural `bstats` routing.

## Cautions

- Prefer explicit `dmx.stats` estimators for explainable, ordinary
  non-Bayesian model choices. Prefer `dmx.bstats` and its automatic helpers
  when Bayesian, variational, or DPM behavior is actually requested.
- Keep this skill local. Do not route to Spark or MPI examples or substitute
  `get_dpm_mixture_mpi` for `get_dpm_mixture`.
- When model choice is uncertain, present 2 or 3 candidate estimators and explain
  the tradeoffs before writing full code.
- Do not assume keyed shared components always help. They are a structural bias
  toward shared latent factors, so compare them against untied mixtures when
  label-specific structure may dominate.
