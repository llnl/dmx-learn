---
name: dmx-expert-orchestrator
description: Main entry point for local `dmx-learn` modeling requests. Use to scope the problem, infer the observation structure, keep the workflow on local data, and route detailed implementation to narrower repo-local skills or references. Default to `dmx.stats` for explicit model construction. Do not use for Spark, MPI, or other distributed estimation workflows.
---

# Dmx Expert Orchestrator

Use this skill as the top-level router for local modeling requests in
`dmx-learn`.

Keep the first pass focused on four decisions:

1. Is the request local and in scope for `dmx-learn`?
2. What does one observation look like?
3. Should the first model be a direct `dmx.stats` estimator or a composite or
   mixture built from `dmx.stats` parts?
4. Which narrower skill or reference should carry the detailed implementation?

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

- Ask for or infer the data path, loader path, or in-memory object.
- Ask what one observation looks like in concrete terms.
- Ask whether the user already has a fixed downstream task or just wants a good
  reusable fitted model.
- Ask about groups, labels, repeated entities, and rough data size only when
  those details affect the model family choice.

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
