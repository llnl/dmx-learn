---
name: dmx-local-modeling
description: Help users choose, fit, diagnose, and use local `dmx-learn` models with user-provided data, especially APIs in `src/dmx/stats` and `src/dmx/utils`. Use for question-first sessions about estimator selection, coding model fits, validation splits, likelihood diagnostics, clustering, ranking, sequence models, classification metrics, embedding plots, composite mixtures, or keyed shared-component mixtures on local and in-memory data. Do not use for Spark, MPI, or other distributed estimation workflows.
---

# Dmx Local Modeling

## Overview

Use this skill to run a short QA loop before writing code, then turn the answers
into explicit `dmx.stats` estimators, fitting code, and diagnostics. Keep the
scope local: assume the user already has data loaded or can provide concrete
Python objects, dtypes, schemas, and representative rows.

## Start With Questions

- Confirm that the task is local-only and not Spark, MPI, or distributed fitting.
- Ask at most 2 to 4 high-value questions at a time.
- Prefer questions about the actual Python object over abstract modeling jargon.
- Infer what you can from the user's sample data before asking follow-ups.

### Minimum First-Pass Questions

1. What does one observation look like?
   Ask for a short schema, dtype summary, and 3 to 5 representative rows or items.
2. What is the modeling goal?
   Ask whether they want density estimation, clustering, classification,
   ranking, sequence modeling, topic modeling, embedding, model comparison,
   or diagnostics.
3. Which parts of the observation are continuous, counts, categorical, vector,
   set-like, sequential, ranking-like, or optional and missing?
4. What supervision exists?
   Ask whether they have no labels, full labels, partial labels, known groups,
   or conditioned pairs.
5. What constraints matter?
   Ask about sample size, missingness, expected number of clusters or states,
   interpretability, and whether a train and validation split already exists.

## Build A Modeling Sketch Before Coding

- Restate the observation type and task in `dmx-learn` terms.
- Name the tentative estimator or estimators and explain why they match.
- State any unresolved modeling choices.
- If model choice is ambiguous, offer 2 or 3 concrete candidates with tradeoffs.
- For heterogeneous supervised tasks, explicitly consider whether a
  mixture-of-composites is a better primary abstraction than an independent
  per-label conditional model.
- Read `references/model-routing.md` when choosing a model family.
- Read `references/repo-entry-points.md` when turning the plan into code.

## Prefer Explicit Local Fitting Paths

1. Build explicit `dmx.stats` estimators instead of relying on broad auto-detection.
2. Use `dmx.utils.estimation.partition_data` when the user needs a validation split.
3. Use `dmx.utils.estimation.optimize` for the standard local EM loop.
4. Use `dmx.utils.estimation.best_of` when initialization sensitivity is likely,
   especially for mixtures and other latent-variable models.
5. Use `seq_encode`, `dist_to_encoder`, `seq_log_density`,
   `seq_log_density_sum`, and `seq_posterior` for fast post-fit evaluation.
6. Use `src/dmx/utils/automatic.py` selectively. It is useful for
   `prepare_mixture_model` and embedding helpers, but `get_estimator` and
   `get_dpm_mixture` route through `dmx.bstats` and are not the default path
   for this skill.
7. For heterogeneous labeled data, prefer explicit `CompositeEstimator` plus
   `MixtureEstimator` constructions over ad hoc condition-specific models.
8. When labels are numerous and likely share latent structure, prefer keyed
   shared-component mixtures as a low-rank approximation to `p(x | y)` instead
   of fitting fully independent mixtures per label.

## Decide Diagnostics Up Front

- Choose the success criterion before writing code: held-out log likelihood,
  class probabilities, rank depth, cluster structure, posterior inspection,
  embedding separation, or parameter sanity.
- For held-out likelihood and repeated fits, prefer
  `partition_data`, `optimize`, `best_of`, and `empirical_kl_divergence`
  when a reference model exists.
- For classification-style evaluation, inspect `src/dmx/utils/metrics.py`.
- For ranking data, prefer `SpearmanRankingEstimator` plus rank-based diagnostics.
- For embedding plots, fit or reuse a mixture model, then use
  `dmx.utils.htsne.htsne` or `dmx.utils.humap.humap`.
- Treat formal likelihood-ratio and p-value requests narrowly. There is no
  general LLR orchestration helper here. If the user wants significance or rank
  approximations, inspect `src/dmx/utils/pvalues.py` and explain the limits.

## Route By Task And Data Shape

- Scalar continuous data: start with `GaussianEstimator`,
  `ExponentialEstimator`, `GammaEstimator`, or `LogGaussianEstimator`.
- Counts and nonnegative integers: start with `PoissonEstimator`,
  `GeometricEstimator`, or `BinomialEstimator`.
- Unordered categorical data: start with `CategoricalEstimator`,
  `IntegerCategoricalEstimator`, `MultinomialEstimator`, or
  `IntegerMultinomialEstimator`.
- Continuous vectors: start with `MultivariateGaussianEstimator`,
  `DiagonalGaussianEstimator`, `GaussianMixtureEstimator`, or
  `DiagonalGaussianMixtureEstimator`.
- Heterogeneous rows or records: combine field-level estimators with
  `CompositeEstimator`, and use wrappers such as `OptionalEstimator`,
  `IgnoredEstimator`, `WeightedEstimator`, or `ConditionalDistributionEstimator`
  when needed.
- Heterogeneous rows with latent subtypes: treat mixture-of-composites as a
  first-class default. Start with `MixtureEstimator([CompositeEstimator(...)])`
  before reaching for more fragmented per-field or per-label constructions.
- Labeled heterogeneous ranking or classification: usually prefer a shared
  mixture-of-composites workflow over a `ConditionalDistributionEstimator`
  whose branches are fully separate composite mixtures. If labels likely share
  latent substructure, use keyed mixture components to learn a low-rank
  approximation where labels keep their own mixture weights but share component
  distributions.
- Latent clustering: start with `MixtureEstimator`,
  `GaussianMixtureEstimator`, `HierarchicalMixtureEstimator`,
  `HeterogeneousMixtureEstimator`, or `SemiSupervisedMixtureEstimator`.
- Sequences: start with `MarkovChainEstimator`, `HiddenMarkovEstimator`,
  `IntegerHiddenMarkovEstimator`, `LookbackHiddenMarkovEstimator`, or
  `SequenceEstimator`.
- Sets or edit-distance-like discrete objects: start with
  `BernoulliSetEstimator`, `IntegerBernoulliSetEstimator`,
  `IntegerBernoulliEditEstimator`, or `IntegerStepBernoulliEditEstimator`.
- Rankings and permutations: start with `SpearmanRankingEstimator`.
- Topic-style discrete mixtures: inspect `IntegerPLSIEstimator` and `LDAEstimator`.

## Reuse Repo Examples Aggressively

- Prefer adapting a nearby runnable example over inventing fresh scaffolding.
- Check `examples/stats_examples/gaussian_example.py` for simple local fitting.
- Check `examples/stats_examples/mixture_example.py` for structured latent mixtures.
- Check `examples/detailed_estimation_example.py` for train and validation loops.
- Check `examples/stats_examples/mixture_example.py` specifically for keyed
  shared-component mixtures. That pattern is the main reference for low-rank
  conditional approximations built from shared mixture components.
- Check `examples/stats_examples/spearman_rho_example.py` for ranking models.
- Check `src/dmx/utils/metrics.py`, `src/dmx/utils/htsne.py`, and
  `src/dmx/utils/humap.py` for downstream diagnostics and visualization.

## Output Expectations

- Produce a short restatement of assumptions before coding.
- Write runnable code, not only model advice.
- Include both fitting and model-use snippets when relevant.
- Show how to inspect the fitted object, evaluate likelihoods, or compute the
  requested diagnostic.
- Point to the exact example, utility, or stats module that informed the answer.
