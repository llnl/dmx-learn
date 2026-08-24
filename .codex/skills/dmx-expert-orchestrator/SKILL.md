---
name: dmx-expert-orchestrator
description: Main entry point for local `dmx-learn` modeling requests. Use to scope the problem, infer the observation structure, keep the workflow on local data, and route detailed implementation to narrower repo-local skills or references. Default to explicit `dmx.stats` construction for ordinary non-Bayesian work and route Bayesian, variational, DPM, and automatic mixture work to `dmx.bstats`. Do not use for Spark, MPI, or other distributed estimation workflows.
---

# Dmx Expert Orchestrator

Use this skill as the top-level router for local modeling requests in
`dmx-learn`.

Keep the first pass focused on five decisions:

1. Is the request local and in scope for `dmx-learn`?
2. Where is the data, and can it be inspected directly?
3. What does one observation look like?
4. Does the objective call for ordinary non-Bayesian estimation or for priors,
   variational inference, a DPM, or automatic Bayesian model construction?
5. Should the first model be a base estimator, composite, or mixture, and which
   narrower skill or reference should carry the detailed implementation?

## Modeling Surface

- Treat explicit `dmx.stats` construction as the default for ordinary
  non-Bayesian local modeling.
- Choose `dmx.bstats` when the request requires Bayesian priors or expected
  log-density behavior, local variational estimation, a truncated
  Dirichlet-process mixture (DPM), or automatic construction intended to feed
  a Bayesian mixture workflow.
- Treat `get_estimator(data, use_bstats=True)` and `get_dpm_mixture(data, ...)`
  from `dmx.utils.automatic` as first-class local routes for those use cases.
  They are deliberate `bstats` paths, not fallbacks.
- Prefer explicit estimator construction when the non-Bayesian model family is
  known. Do not use automatic routing merely to avoid choosing a clear
  `dmx.stats` estimator.
- If the downstream task is not fully fixed, prefer fitting one reusable joint,
  composite, or composite-mixture model before narrowing to a task-specific
  conditional path.
- For heterogeneous data with unknown latent subtype count, make a DPM over
  `dmx.bstats.CompositeEstimator` components the main high-value Bayesian
  route. The composite preserves field semantics while the truncated DPM and
  variational optimizer infer a useful finite mixture.
- Mention `torch_stats` only when the user's scale or repeated inference needs
  make accelerator-backed fitting a plausible next step.

## Hard Scope Boundary

- This skill is for local and in-memory modeling workflows.
- Local `dmx.bstats`, `dmx.bstats.bestimation`, and
  `dmx.utils.automatic` workflows are in scope.
- Do not use it for Spark, MPI, cluster scheduling, or other distributed
  estimation paths.
- Do not substitute `get_dpm_mixture_mpi`, `dmx.mpi4py.bstats`, or
  `dmx.mpi4py.utils` for the local helpers. Those are separate MPI-specific
  workflows even when they operate on the same `bstats` model types.
- If the user asks for distributed fitting, say it is out of scope and do not
  improvise a Spark or MPI workflow.

## Routing Workflow

### 1. Intake

The intake goal is to get the minimum local-data facts needed to choose model
structure from the actual data, not from vague prompt wording.

- Treat the local data path, loader path, or in-memory object as a first-class
  input.
- If a data path or loader is available, prefer inspecting a representative
  observation instead of asking the user to describe the whole dataset from
  memory.
- Ask what one observation looks like in concrete terms: one row, record,
  tuple, sequence, or other unit of modeling.
- Ask or infer field roles: continuous, categorical, count, binary, text,
  set-like, sequence-like, or optional.
- Ask whether any part of the observation is ordered, variable-length, or
  nested.
- Ask whether there are known groups, labels, repeated entities, or candidate
  conditioning keys.
- Ask for rough data size, including sample count and typical sequence length
  when relevant.
- Ask whether GPU use is acceptable before suggesting `torch_stats`.
- Ask whether the user needs explicit priors, variational inference, automatic
  model construction, or an unknown number of latent components; any of these
  can change the surface choice from `dmx.stats` to `dmx.bstats`.
- Ask whether the user already has a fixed downstream task or just wants a good
  reusable fitted model.

Use a concise first-pass intake like this when the answers are not already
available from the supplied data:

1. Where is the local data or loader I should inspect?
2. What does one observation look like?
3. Which fields are continuous, categorical, counts, optional, set-like, or
   sequence-like?
4. Are there labels, groups, repeated entities, or known conditioning keys?
5. Roughly how many observations are there, how long are sequences if any, and
   is GPU use acceptable if scale makes `torch_stats` worth considering?

### 2. Default Lightweight EDA For Local Data Paths

If the user provides a local data path or loader that can be inspected, run a
lightweight EDA pass by default before choosing model structure.

The purpose of this pass is to support model-structure routing. It is not a
full exploratory analysis report, feature-selection pass, or visualization
exercise.

Use the same small, repeatable checklist each time:

1. Inspect sample count or row count so the scale is concrete.
2. Inspect a few representative rows or records to identify what one
   observation appears to be.
3. Inspect missingness at a field level when it may change the structure,
   especially for optional fields or partially observed tuples.
4. Inspect discrete cardinalities for fields that look categorical, binary, or
   identifier-like.
5. Inspect sequence lengths when any field appears variable-length, ordered, or
   nested as a list-like object.
6. Inspect grouping or repeated-key structure when there are obvious IDs,
   labels, user/item pairs, session keys, or repeated entities.

Keep this pass lightweight:

- Prefer small representative slices over broad profiling.
- Prefer field summaries that affect structure choice over generic descriptive
  statistics.
- Stop once there is enough evidence to decide the observation unit, field
  roles, and whether grouping, sequence structure, or optional substructure is
  present.
- Do not expand into correlations, plots, hypothesis testing, or a full
  notebook-style report unless the user explicitly asks for deeper EDA.

Ask a follow-up question instead of guessing when the lightweight pass still
leaves a structural ambiguity, for example:

- it is unclear what should count as one observation
- multiple fields could be the grouping or conditioning key
- missingness could mean either optional structure or a data-quality problem
- a sequence-like field might instead be an unordered set or bag
- the loader or file format prevents direct inspection of representative rows

When asking a follow-up, keep it narrow and tied to the next routing decision.

### 3. Structure First

- Decide the observation structure before naming an estimator family.
- Start with the simplest accurate description: scalar, vector, tuple, record,
  set, sequence, ranking, or mixed observation.
- When the task is underspecified, default to a reusable shared model first:
  composite for heterogeneous records, mixture-of-composites for latent
  subtypes, and joint mixtures for paired views where later conditioning or
  transfer is likely.
- When that heterogeneous latent model is specifically Bayesian or should use
  a DPM to infer the active component count, route to a `dmx.bstats` composite
  inside a DPM rather than forcing the ordinary `dmx.stats` mixture path.
- Only prefer a narrow task-specific model first when the downstream target is
  already fixed and the extra shared structure would not plausibly be reused.
- Read `references/hierarchy-and-data-structure.md` before choosing
  field-level estimators or routing into estimator catalogs.

### 4. Primary Model And Baseline Policy

For vague or only partially specified tasks, do not fan out into a wide model
search.

- Fit `1 primary model + 1 baseline`.
- Make the primary model the best reusable shared model suggested by the
  observation structure.
- Make the baseline structurally meaningful and simpler, so the comparison
  answers whether the extra joint or latent structure is justified.

Good baseline patterns include:

- joint model vs simpler single-view or independent composite model
- mixture-of-composites vs plain composite model
- keyed shared model vs fully separate per-group or per-label fits

### 5. Choose The Next Skill Or Reference

- Use `references/hierarchy-and-data-structure.md` for structure-first routing:
  base estimator vs composite vs mixture, sequence/HMM, grouped sharing, joint
  mixture, and heterogeneous mixture decisions.
- Route implementation-heavy local fitting work to
  `dmx-local-modeling`.
- Route Python source edits, examples, or library changes to
  `dmx-python-implementation`.
- Keep this skill lean. Do not inline long estimator catalogs, notebook
  heuristics, or detailed fitting recipes here.

## Output Expectations

- Restate the inferred local modeling problem in `dmx-learn` terms.
- Name the modeling surface and starting point: normally an explicit
  `dmx.stats` estimator, or `dmx.bstats` for Bayesian, variational, DPM, and
  automatic Bayesian-mixture work. Explicitly note when the default is a
  reusable shared model rather than a narrow task-specific one.
- For underspecified tasks, name one primary model and one baseline instead of
  proposing a broad candidate sweep.
- Call out any scope boundary, especially distributed-workflow requests.
- Hand off detailed fitting or code-generation work to the narrower local skill
  instead of turning this file into a monolithic reference.
