---
name: dmx-expert-orchestrator
description: Main entry point for local `dmx-learn` modeling requests. Use to scope the problem, infer the observation structure, keep the workflow on local data, and route detailed implementation to narrower repo-local skills or references. Default to `dmx.stats` for explicit model construction. Do not use for Spark, MPI, or other distributed estimation workflows.
---

# Dmx Expert Orchestrator

Use this skill as the top-level router for local modeling requests in
`dmx-learn`.

Keep the first pass focused on five decisions:

1. Is the request local and in scope for `dmx-learn`?
2. Where is the data, and can it be inspected directly?
3. What does one observation look like?
4. Should the first model be a direct `dmx.stats` estimator or a composite or
   mixture built from `dmx.stats` parts?
5. Which narrower skill or reference should carry the detailed implementation?

## Default Surface

- Treat `dmx.stats` as the default modeling surface.
- Prefer explicit estimator construction over broad automatic routing.
- Mention `torch_stats` only when the user's scale or repeated inference needs
  make accelerator-backed fitting a plausible next step.
- Keep `bstats` out of the default path for ordinary local modeling.

## Hard Scope Boundary

- This skill is for local and in-memory modeling workflows.
- Do not use it for Spark, MPI, cluster scheduling, or other distributed
  estimation paths.
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

### 2. Structure First

- Decide the observation structure before naming an estimator family.
- Start with the simplest accurate description: scalar, vector, tuple, record,
  set, sequence, ranking, or mixed observation.
- When the task is underspecified, prefer a reusable joint or composite view of
  the observation over a narrow one-off objective.

### 3. Choose The Next Skill Or Reference

- Route implementation-heavy local fitting work to
  `dmx-local-modeling`.
- Route Python source edits, examples, or library changes to
  `dmx-python-implementation`.
- Keep this skill lean. Do not inline long estimator catalogs, notebook
  heuristics, or detailed fitting recipes here.

## Output Expectations

- Restate the inferred local modeling problem in `dmx-learn` terms.
- Name the most likely `dmx.stats` starting point.
- Call out any scope boundary, especially distributed-workflow requests.
- Hand off detailed fitting or code-generation work to the narrower local skill
  instead of turning this file into a monolithic reference.
