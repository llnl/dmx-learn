---
name: dmx-local-modeling
description: Implementation-oriented local `dmx-learn` fitting skill for coding, fitting, diagnosing, and using models on local or in-memory data once the problem structure is known. Prefer explicit `dmx.stats` estimators for ordinary non-Bayesian work and use `dmx.bstats` for Bayesian, variational, DPM, and automatic mixture workflows. Do not use for Spark, MPI, or other distributed estimation workflows.
---

# Dmx Local Modeling

Use this skill for implementation-heavy local modeling work after the problem
has already been scoped to a concrete local `dmx-learn` path.

This skill is narrow on purpose:

- it writes explicit estimators, fitting loops, and diagnostics
- it assumes the task is local and in scope for `dmx-learn`
- it does not own broad intake, lightweight EDA, or high-level routing policy

When the problem framing is still ambiguous, hand that work to
[`dmx-expert-orchestrator`](../dmx-expert-orchestrator/SKILL.md) first.

## When To Use This Skill

Use this skill when at least one of these is already known:

- the observation structure is clear enough to code directly
- the model family has already been chosen upstream
- the user wants runnable fitting code, diagnostics, or post-fit usage
- the main question is about implementation details in `dmx.stats`,
  `dmx.bstats`, or `dmx.utils`

Do not use this skill as the main router for vague modeling requests. For:

- intake and lightweight local-data inspection
- structure-first routing across base, composite, mixture, sequence, joint, or
  grouped models
- deciding when to fit one primary model plus one baseline
- deciding whether the task should stay broad and reusable instead of going
  straight to a narrow conditional path

read [`../dmx-expert-orchestrator/SKILL.md`](../dmx-expert-orchestrator/SKILL.md)
and its
[`references/hierarchy-and-data-structure.md`](../dmx-expert-orchestrator/references/hierarchy-and-data-structure.md)
first.

## Default Workflow

### 1. Confirm The Implementation Target

Before coding, restate only the facts needed for implementation:

- what one observation is
- the intended estimator family or estimator tree
- the fitting objective or evaluation target
- any known validation split, restart budget, or scale constraint

If those facts are still uncertain, stop routing locally and defer back to the
orchestrator instead of improvising your own broad routing pass here.

### 2. Choose The Local Modeling Surface

- Prefer explicit `dmx.stats` estimator construction for ordinary
  non-Bayesian estimation when the model family is known.
- Use `dmx.bstats` when the implementation needs explicit priors, expected
  log-density calculations, local variational fitting, a truncated DPM, or an
  automatic estimator that will participate in a Bayesian mixture workflow.
- `dmx.utils.automatic.get_estimator(data, use_bstats=True)` is a valid
  first-class route when automatic Bayesian structure inference is intentional.
  Keep `use_bstats=True` explicit in generated code even though the current
  helper default is `True`.
- `dmx.utils.automatic.get_dpm_mixture(data, ...)` is the first-class local
  automatic DPM route. It fits a truncated `dmx.bstats` DPM variationally,
  removes components below its count threshold, and returns a finite
  `dmx.bstats.MixtureDistribution`.
- Keep the hierarchy chosen upstream intact. Do not flatten a composite,
  sequence, or grouped problem into a simpler estimator only because it is
  quicker to code.
- Read [`references/model-routing.md`](references/model-routing.md) when you
  need the concrete estimator family map after the structure is already known.
- Reuse the orchestrator's structure-first philosophy: hierarchy choice comes
  first, field-level estimators second.

For advanced structure selection, grouped-sharing policy, or joint-model-first
reasoning, go back to the orchestrator reference instead of re-deriving that
logic in this skill.

### 3. Use Repo-Native Local Fitting Helpers

Read [`references/repo-entry-points.md`](references/repo-entry-points.md) for
the concrete repo paths.

Default non-Bayesian implementation path:

1. Build an explicit estimator in `dmx.stats`.
2. Split data with `dmx.utils.estimation.partition_data` when held-out
   validation is needed.
3. Fit with `dmx.utils.estimation.optimize`.
4. Upgrade to `dmx.utils.estimation.best_of` when initialization sensitivity is
   plausible, especially for mixtures or other latent-variable models.
5. Use vectorized post-fit calls such as `seq_encode`, `seq_log_density`,
   `seq_log_density_sum`, or related posterior helpers instead of repeated
   scalar scoring.

Bayesian and automatic implementation paths:

- Build explicit `dmx.bstats` estimators and fit them with local helpers in
  `dmx.bstats.bestimation` when priors or variational behavior must be
  controlled directly.
- Use `get_estimator(..., use_bstats=True)` for automatic Bayesian estimator
  construction, including composite, optional, and sequence structures.
- Use `get_dpm_mixture` for a concise local automatic mixture workflow.
- `prepare_mixture_model` is a first-class automatic mixture route for
  embedding flows; without a supplied model it delegates to the local
  `get_dpm_mixture` path.

Keep the execution boundary explicit: these are local helpers. Do not route to
`dmx.mpi4py.bstats`, `dmx.mpi4py.utils.bestimation`, or
`get_dpm_mixture_mpi`; MPI workflows require separate distributed guidance.

### 4. Keep Shared-Structure Implementations Explicit

This skill should still be useful for coding and fitting advanced local models,
but not for deciding whether they are the right first abstraction.

Implementation defaults:

- for heterogeneous records, build field estimators explicitly and combine them
  with `CompositeEstimator`
- for latent subtypes over heterogeneous records, implement a
  `CompositeEstimator` inside a `MixtureEstimator`
- for Bayesian latent subtypes over heterogeneous records, prefer a
  `dmx.bstats.CompositeEstimator` as the base estimator for a truncated DPM;
  this DPM-over-composite path is the primary high-value `bstats` workflow
- for many-label problems with shared latent structure, use keyed shared
  components explicitly instead of hiding the structure inside many unrelated
  conditional branches

Use [`references/model-routing.md`](references/model-routing.md) and
[`references/repo-entry-points.md`](references/repo-entry-points.md) for the
keyed mixture and composite-mixture patterns. Use the orchestrator references
for the higher-level question of when those patterns should be preferred.

### 5. Code Diagnostics And Post-Fit Usage

Choose the success criterion before writing the final code path:

- held-out log likelihood or likelihood comparisons
- classification metrics or ranking depth
- posterior inspection or cluster structure
- embedding-oriented exploratory views
- parameter sanity and fit stability

Repo-native helpers:

- `src/dmx/utils/metrics.py` for classification and ranking-style evaluation
- `src/dmx/utils/pvalues.py` only for narrow approximate significance or rank
  calculations, not as a generic testing framework
- `src/dmx/utils/htsne.py` and `src/dmx/utils/humap.py` for embedding workflows

For cleaned `bstats` workflow evidence, use:

- `tests/bstats/dpm_test.py` for deterministic DPM initialization, variational
  updates, local optimization, and `get_dpm_mixture` conversion
- `tests/bstats/composite_test.py` for a composite distribution inside finite
  mixture and DPM containers
- `tests/bstats/structural_test.py` and
  `tests/bstats/discrete_primitives_test.py` for
  `get_estimator(..., use_bstats=True)` routing

The MPI import smoke assertion in `tests/bstats/dpm_test.py` is compatibility
coverage only; it does not turn the local fitting recipe into an MPI workflow.

## Example Reuse Policy

- Prefer adapting nearby runnable examples over inventing new scaffolding.
- Start with `examples/stats_examples/gaussian_example.py` for the smallest
  fit-and-score flow.
- Use `examples/detailed_estimation_example.py` for validation and repeated-fit
  loops.
- Use `examples/stats_examples/mixture_example.py` for structured mixtures and
  keyed shared-component implementations.
- Use `examples/stats_examples/spearman_rho_example.py` for ranking workflows.
- For Bayesian and DPM behavior, prefer the focused `tests/bstats` files above;
  the current runnable examples primarily demonstrate `dmx.stats` models.

## Output Expectations

- Restate the implementation assumptions briefly before coding.
- Produce runnable code, not only estimator advice.
- Include fitting, scoring, and model-use snippets when they are relevant.
- Show how to inspect the fitted model or compute the requested diagnostic.
- Cite the exact repo example, utility, or stats module that informed the code.
