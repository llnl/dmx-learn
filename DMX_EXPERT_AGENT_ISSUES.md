# dmx-learn Expert Agent: GitHub Epics and Issues

Source plan: `DMX_EXPERT_AGENT_ROADMAP.md`. Decision references (D1–D11) point
to section 12 of that file.

This file is formatted for `scripts/create_github_issue_plan.py`. Each epic is
one run of the script: it creates the epic as a parent issue, creates the
child issues in that epic's range, and then fills the parent with an ordered
checklist. Issue numbers below are **source numbers**. GitHub assigns its own
numbers when the issues are created. Dependencies are written as source
numbers plus titles so they stay resolvable.

Branching: open one branch per child issue using GitHub's **Create a branch**
button (it produces `<issue-number>-<title-slug>`, matching existing branches
like `197-add-pdf-of-docs`). Each issue lists a suggested short slug.

Common labels: `agentic-dmx,enhancement`. Add area labels per epic as shown.

## Epic index

| Epic | Release | Source issues | Extra labels |
|---|---|---|---|
| 1. Positioning and packaging | 1.2 | 1–4 | `docs` |
| 2. Engine hardening for scoring and quiet tools | 1.2 | 5–9 | `tests` |
| 3. Tabular ingestion and Schema layer | 1.2 | 10–15 | `tests` |
| 4. Agent MVP: dmx-mcp vertical slice | 1.2 | 16–25 | |
| 5. Diagnostics and structure search | 1.3 | 26–34 | `tests` |
| 6. Model queries and exploration | 1.4 | 35–43 | `tests` |
| 7. Distributed scale-out for the agent | 1.5 | 44–48 | |
| 8. dmx-learn 2.0 release | 2.0 | 49–52 | `docs` |
| 9. Future research: deep latent structure | v2 | 53 | `bayesian` |

Dry-run any epic first by adding `--dry-run`.

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement,docs" --start-issue 1 --end-issue 4 --parent-title "Epic 1: Positioning and packaging (v1.2)" --parent-summary "Reposition dmx-learn as the dedicated expert tool for latent-variable graphical models on heterogeneous tabular data, document explicit non-goals that separate it from broad platforms, and slim core dependencies (pyspark to an extra; new viz and agent extras). See DMX_EXPERT_AGENT_ROADMAP.md sections 1 and 12 (D5, D7)."
```

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement,tests" --start-issue 5 --end-issue 9 --parent-title "Epic 2: Engine hardening for scoring and quiet tools (v1.2)" --parent-summary "Add the engine capabilities the agent needs on day one: parameter counts and BIC/AIC/ICL scoring, structured logging with a verbose switch instead of stdout prints, and seed discipline on every stochastic entry point. All changes are additive and keep current defaults (D3). See DMX_EXPERT_AGENT_ROADMAP.md section 4."
```

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement,tests" --start-issue 10 --end-issue 15 --parent-title "Epic 3: Tabular ingestion and Schema layer (v1.2)" --parent-summary "Create dmx.tabular so users bring a CSV or DataFrame, negotiate an editable typed Schema, and get dmx estimators without one-hot encoding. Includes role inference, candidate families per field, record conversion, and a schema test corpus. See DMX_EXPERT_AGENT_ROADMAP.md section 5."
```

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement" --start-issue 16 --end-issue 25 --parent-title "Epic 4: Agent MVP - dmx-mcp vertical slice (v1.2)" --parent-summary "Ship a working end-to-end expert agent as an MCP server usable from any MCP client, including on-prem models (D1). It wraps existing fit, automatic, embedding, and search-depth APIs, keeps a session registry, enforces tiered privacy (D8), returns PNG plots (D6), exports reproducible code, and ships tool-oriented skills. See DMX_EXPERT_AGENT_ROADMAP.md sections 8 and 11."
```

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement,tests" --start-issue 26 --end-issue 34 --parent-title "Epic 5: Diagnostics and structure search (v1.3)" --parent-summary "Move model selection beyond the automatic estimator: diagnostics (per-field fit, residual dependence, component health), a structure grammar over existing classes (D4), a deterministic beam search the LLM can steer (D2), a SearchReport with lineage, and a benchmark suite against one-hot sklearn GMMs and StepMix (D10). See DMX_EXPERT_AGENT_ROADMAP.md sections 6 and 10."
```

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement,tests" --start-issue 35 --end-issue 43 --parent-title "Epic 6: Model queries and exploration (v1.4)" --parent-summary "Add posterior parity, marginal and conditional queries, imputation, a JSON model spec and graph rendering, plus exploration functions (cluster profiles, anomalies with per-field blame, group comparisons, search depth in both directions) and the matching agent tools. See DMX_EXPERT_AGENT_ROADMAP.md sections 4 and 7."
```

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement" --start-issue 44 --end-issue 48 --parent-title "Epic 7: Distributed scale-out for the agent (v1.5)" --parent-summary "Let the same model spec run locally, on MPI, or on Spark through an Engine abstraction, add a subsample-search-then-refit workflow and a scale_out_plan tool, and replace the skills' distributed out-of-scope boundary with a distributed playbook. See DMX_EXPERT_AGENT_ROADMAP.md section 9."
```

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement,docs" --start-issue 49 --end-issue 52 --parent-title "Epic 8: dmx-learn 2.0 release" --parent-summary "Consolidate the 1.x additive work into a 2.0 release: API cleanup and deprecation removal, agent-first user guide and tutorials, a tech report with published benchmark results, and release packaging. See DMX_EXPERT_AGENT_ROADMAP.md section 11."
```

```bash
python scripts/create_github_issue_plan.py --repo llnl/dmx-learn --issue-file DMX_EXPERT_AGENT_ISSUES.md --link-sub-issues --labels "agentic-dmx,enhancement,bayesian" --start-issue 53 --end-issue 53 --parent-title "Epic 9: Future research - deep latent structure (v2)" --parent-summary "Research track for model classes beyond the v1 composition vocabulary, starting with a latent tree of discrete latent variables over three or more field blocks. Driven by benchmark gaps found in Epic 5 (D4)."
```

---

## Issue 1

**Title:** Rewrite README and package metadata around the expert-agent positioning

**Body:**

Epic: 1. Positioning and packaging. Suggested branch slug: `readme-expert-positioning`.

Reposition dmx-learn as the dedicated, exact, transparent tool for
latent-variable graphical models on heterogeneous tabular data, with an
expert agent as the primary user surface (roadmap section 1).

Scope:

- rewrite the top of `README.md` to lead with the table-to-agent workflow and
  the tagline
- update `description` and `keywords` in `pyproject.toml`
- update `docs/index.rst` introduction to match
- keep installation and example sections accurate

Acceptance criteria:

- README opens with the positioning statement and a short workflow sketch
- package description no longer reads as a generic density library
- docs build passes

---

## Issue 2

**Title:** Add a scope and non-goals page to the docs

**Body:**

Epic: 1. Positioning and packaging. Suggested branch slug: `docs-scope-non-goals`.

Document what dmx-learn is and is deliberately not (roadmap section 1).

Scope:

- add `docs/scope.rst` and link it from `docs/index.rst` and `README.md`
- in scope: exponential-family leaves composed into mixtures, HMMs, trees,
  composites; exact EM or variational updates; MPI and Spark scale-out; the
  expert agent
- non-goals: neural densities, PPL syntax, LLM distillation, MCMC/NUTS,
  MLOps serving, copulas and vines
- state the rule for new models: closed-form or variational sufficient
  statistics through `seq_update`

Acceptance criteria:

- page renders in the docs build and is linked from README
- every non-goal from the roadmap is listed

---

## Issue 3

**Title:** Move pyspark from core dependencies to a `[spark]` extra

**Body:**

Epic: 1. Positioning and packaging. Suggested branch slug: `pyspark-extra`.

pyspark is currently a core dependency in `pyproject.toml`, which forces a
large install and a Java runtime on every user (decision D5).

Scope:

- move `pyspark` to an optional `[spark]` extra; add `[distributed]` as
  spark plus mpi4py; include both in `[all]`
- update `poetry.lock`
- guard Spark imports (`src/dmx/stats/rdd_sampler.py` and any other Spark
  paths) with a clear install hint
- update the README extras table, `.github/workflows/*` installs, and Spark
  examples instructions
- add a release-note entry calling out the packaging change

Acceptance criteria:

- `pip install .` succeeds without pyspark and `import dmx` works
- importing a Spark path without pyspark raises an error naming
  `pip install dmx-learn[spark]`
- CI passes with and without the extra

---

## Issue 4

**Title:** Add `[viz]` and `[agent]` optional extras with import guards

**Body:**

Epic: 1. Positioning and packaging. Suggested branch slug: `viz-agent-extras`.

Prepare the optional dependency groups used by later epics (D5, D6).

Scope:

- `[viz]`: matplotlib
- `[agent]`: the official `mcp` Python SDK plus everything in `[viz]`
- shared helper for optional-import errors with install hints
- document both extras in the README table

Acceptance criteria:

- extras install cleanly in a fresh environment
- core install has no matplotlib or mcp dependency
- missing-extra imports fail with a helpful message

---

## Issue 5

**Title:** Add `num_params()` to all `dmx.stats` distributions

**Body:**

Epic: 2. Engine hardening. Suggested branch slug: `stats-num-params`.

BIC, AIC, and ICL need free-parameter counts (roadmap section 4).

Scope:

- add `num_params()` to the `ProbabilityDistribution` interface in
  `src/dmx/stats/pdist.py`
- implement for every leaf distribution in `src/dmx/stats/`
- implement recursively for combinators: composite, mixture, hierarchical
  mixture, joint mixture, heterogeneous mixture, semi-supervised mixture,
  optional, sequence, conditional, HMM variants, ICL tree
- count keyed (shared) parameters once
- ignored and null distributions return 0

Acceptance criteria:

- a test per distribution checks the count against a hand-computed value
- keyed sharing is covered by a test
- no change to existing behavior

---

## Issue 6

**Title:** Add `num_params()` to `dmx.bstats` distributions and DPM effective components

**Body:**

Epic: 2. Engine hardening. Suggested branch slug: `bstats-num-params`.

Depends on: Issue 5 (`num_params()` for `dmx.stats`).

Scope:

- implement `num_params()` across `src/dmx/bstats/`
- add an effective component count for the DPM (weights above a threshold)
  and document how it relates to `get_dpm_mixture` pruning

Acceptance criteria:

- tests mirror the `dmx.stats` coverage for every bstats distribution
- DPM effective count is tested on a planted-K synthetic dataset

---

## Issue 7

**Title:** Add model scoring helpers: held-out LL, k-fold LL, BIC, AIC, ICL

**Body:**

Epic: 2. Engine hardening. Suggested branch slug: `model-scores`.

Depends on: Issue 5, Issue 6.

Scope:

- new module `src/dmx/utils/scores.py`
- `heldout_log_likelihood`, `kfold_log_likelihood` (reusing
  `partition_data` / `k_fold_split_index`), `bic`, `aic`, `icl`
- ICL uses posterior entropy for mixture-type models via `seq_posterior`
- return plain dataclasses, no printing

Acceptance criteria:

- scores match hand calculations on small Gaussian and categorical mixtures
- BIC selects the planted K on a synthetic composite mixture test
- docstrings follow the repo's Google style and pydocstyle passes

---

## Issue 8

**Title:** Replace stdout prints with structured logging and a `verbose` switch

**Body:**

Epic: 2. Engine hardening. Suggested branch slug: `quiet-logging`.

Fitting and embedding helpers currently `print` progress, which breaks
tool outputs (roadmap section 4).

Scope:

- `src/dmx/utils/estimation.py`, `src/dmx/bstats/bestimation.py`,
  `src/dmx/utils/automatic.py`, `src/dmx/utils/htsne.py`,
  `src/dmx/utils/humap.py`, and `src/dmx/mpi4py/utils/*`
- route messages through `logging` (logger per module)
- add `verbose: bool = True` (current behavior) and an optional
  `progress_callback` receiving structured events (iteration, LL, delta)
- keep existing `out=` parameters working

Acceptance criteria:

- with `verbose=False`, these functions write nothing to stdout or stderr
  (tested with `capsys`)
- default output is unchanged so existing answer-key tests pass
- callback receives one event per iteration in a test

---

## Issue 9

**Title:** Enforce seed discipline on every stochastic entry point

**Body:**

Epic: 2. Engine hardening. Suggested branch slug: `seed-discipline`.

Reproducibility is required for `export_code` (Issue 22).

Scope:

- audit `optimize`, `best_of`, `iterate`, `get_dpm_mixture`,
  `prepare_mixture_model`, `htsne`, `humap`, samplers, and bstats helpers
- accept `seed` or `rng` consistently; never fall back to global numpy state
  silently
- return or expose the seed actually used

Acceptance criteria:

- running each entry point twice with the same seed gives identical results
  (tested)
- an audit table of entry points and their seed parameters is added to the
  docs

---

## Issue 10

**Title:** Define `Schema` and `FieldSpec` with roles and JSON round-trip

**Body:**

Epic: 3. Tabular layer. Suggested branch slug: `tabular-schema`.

Create the editable schema the user and agent negotiate (roadmap section 5).

Scope:

- new package `src/dmx/tabular/` with `schema.py`
- `FieldSpec(name, role, family, candidates, optional, options)`
- roles: continuous, positive-continuous, count, binary, categorical,
  high-cardinality ID, text-bag, text-sequence, set, ordered sequence,
  datetime, ignored, group key, label, sensitive flag
- `Schema.to_json()` / `Schema.from_json()` with a version field
- edit helpers: set role, set family, drop or ignore field

Acceptance criteria:

- JSON round-trip test for every role
- invalid role and family combinations raise clear errors

---

## Issue 11

**Title:** Infer field roles from pandas DataFrames

**Body:**

Epic: 3. Tabular layer. Suggested branch slug: `tabular-role-inference`.

Depends on: Issue 10.

Generalize the heuristics in `DatumNode` (`src/dmx/utils/automatic.py`) to
DataFrame columns.

Scope:

- `infer_schema(df) -> Schema`
- integer cardinality vs. range, skew and positivity tests, ID detection
  (cardinality close to n), list-like string detection with delimiter
  inference, datetime parsing, missingness rates
- all thresholds defined in one documented config object
- each inferred field records a short reason string for the agent to show

Acceptance criteria:

- unit tests per heuristic
- reasons are present for every field

---

## Issue 12

**Title:** Provide candidate distribution families per field role

**Body:**

Epic: 3. Tabular layer. Suggested branch slug: `tabular-candidate-families`.

Depends on: Issue 10.

Structure search (Epic 5) needs alternatives per field, not one choice.

Scope:

- role-to-candidates table, for example continuous: Gaussian, LogGaussian,
  Gamma; count: Poisson, Geometric, IntegerCategorical; text-bag:
  Multinomial or Sequence of Categorical
- factory that builds a `dmx.stats` or `dmx.bstats` estimator for a
  FieldSpec and chosen family, with sensible pseudo-counts
- wrap in `OptionalEstimator` when the field has missing values

Acceptance criteria:

- every role has at least one candidate and a default
- each candidate estimator fits a small synthetic column in a test

---

## Issue 13

**Title:** Add `load_table` and `Table.to_records(schema)`

**Body:**

Epic: 3. Tabular layer. Suggested branch slug: `tabular-load-table`.

Depends on: Issue 10, Issue 11.

Scope:

- `load_table(path_or_df) -> Table` for CSV, parquet, and DataFrames via
  pandas
- `Table.to_records(schema)` yields the tuples dmx estimators consume:
  missing values to `None`, list-like strings to lists, text to token bags or
  sequences, datetimes decomposed per options, ignored fields dropped
- stable field ordering recorded in the schema

Acceptance criteria:

- records from each repo CSV in `data/` fit with their schema's estimator
- no one-hot encoding anywhere in the path

---

## Issue 14

**Title:** Build estimators from a Schema: composite, mixture, and DPM

**Body:**

Epic: 3. Tabular layer. Suggested branch slug: `tabular-to-estimator`.

Depends on: Issue 12, Issue 13.

Scope:

- `schema.to_estimator(structure="composite" | "mixture" | "dpm", k=...)`
- composite of field estimators; mixture of composites; DPM over a
  `dmx.bstats` composite
- parity check against `get_estimator(data, use_bstats=True)` on repo data

Acceptance criteria:

- tests fit all three structures on a mixed-type synthetic table
- parity test documents any intentional differences from `get_estimator`

---

## Issue 15

**Title:** Create a schema-inference test corpus of 10 diverse tables

**Body:**

Epic: 3. Tabular layer. Suggested branch slug: `tabular-schema-corpus`.

Depends on: Issue 11.

Scope:

- small synthetic tables covering IDs, skewed positives, low and high
  cardinality integers, list-like strings, free text, datetimes, heavy
  missingness, and a group key
- expected schemas stored as JSON next to each table under `tests/tabular/`
- include tables derived from `data/search_depth`, `data/subset_keys`, and
  `data/process_sequences`

Acceptance criteria:

- inference matches the expected schema on all 10 tables
- corpus runs in under 30 seconds in CI

---

## Issue 16

**Title:** Scaffold `dmx.agent` and the `dmx-mcp` stdio server

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-mcp-scaffold`.

Depends on: Issue 4 (`[agent]` extra).

Scope:

- `src/dmx/agent/` with a tool registry: each tool has a name, JSON input
  schema, and a deterministic handler returning JSON (and optional images)
- `dmx-mcp` console entry point using the official `mcp` SDK over stdio
- a `ping`/`version` tool and a smoke test using an in-process client
- setup docs for Claude Code, Claude Desktop, and one local-model MCP client

Acceptance criteria:

- `dmx-mcp` starts and lists tools in an MCP client
- tool handlers never print to stdout (protocol channel)

---

## Issue 17

**Title:** Add session and model registry under `.dmx/`

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-session-registry`.

Depends on: Issue 16.

Scope:

- on-disk session directory: datasets, schemas, models, plots, experiment log
- model entries store pickle, `str` repr, scores, seed, schema id, privacy
  level, and parent model id (lineage)
- append-only JSONL experiment log of every tool call and result summary

Acceptance criteria:

- a session survives a server restart and models reload identically
- registry is covered by tests using a temp directory

---

## Issue 18

**Title:** Enforce tiered privacy levels in tool outputs

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-privacy-levels`.

Depends on: Issue 17. Implements decision D8.

Scope:

- `summaries` (default): names and aggregates; category values only below a
  cardinality threshold and never for sensitive or ID fields
- `aliased`: stable tokens for column names and category values, alias map
  kept in `.dmx/`, resolved only in local exports
- `full`: explicit opt-in, sample rows allowed
- a single output filter applied to every tool result; privacy level stamped
  on every result

Acceptance criteria:

- property tests assert that no raw value of a sensitive field appears in any
  tool output under `summaries` or `aliased`
- alias map round-trips in exported reports

---

## Issue 19

**Title:** Agent data tools: load, profile, propose and update schema

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-data-tools`.

Depends on: Issue 13, Issue 18.

Scope:

- `load_table`, `profile_table`, `propose_schema`, `update_schema`
- profile returns counts, missingness, cardinalities, and sequence lengths,
  filtered by privacy level
- `propose_schema` returns roles with reasons and candidates

Acceptance criteria:

- tools work end to end on two repo datasets in tests
- schema edits are persisted in the session

---

## Issue 20

**Title:** Agent fit tools over existing APIs: fit, auto model, compare

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-fit-tools`.

Depends on: Issue 7, Issue 8, Issue 14, Issue 17.

Scope:

- `fit_model` (structure, K, restarts via `best_of`, seed, validation split)
- `auto_model` wrapping `get_estimator` and `get_dpm_mixture`
- `compare_models` returning held-out LL, BIC, ICL, and K for registered
  models as a leaderboard
- quiet execution with progress events

Acceptance criteria:

- fitted models are registered with scores and seeds
- compare output is deterministic for fixed seeds

---

## Issue 21

**Title:** Agent exploration MVP tools with PNG plots

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-explore-mvp`.

Depends on: Issue 20. Implements decision D6.

Scope:

- `embed` using `htsne` or `humap` on a registered model (reuse, never refit)
- `cluster_summary` with component weights and posterior assignment counts
- `search_depth` via `dmx.utils.metrics.ranking_depth`
- return tidy data, a PNG image (matplotlib from `[viz]`), and the saved path
  under `.dmx/plots/`

Acceptance criteria:

- each tool returns valid MCP image content and JSON in tests
- embeddings reproduce for a fixed seed

---

## Issue 22

**Title:** Implement `export_code` for reproducible fit scripts

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-export-code`.

Depends on: Issue 9, Issue 20.

Scope:

- generate a standalone Python script using only public dmx APIs that loads
  the data, applies the schema, builds the estimator, and fits with the same
  seed
- resolve aliases locally so the script uses real column names

Acceptance criteria:

- a test runs the exported script and matches the registered model's
  log-likelihood exactly
- exported code passes the repo's lint configuration

---

## Issue 23

**Title:** Port skills to `AGENTS.md` and `.claude/skills`, rewritten around tools

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-skills-port`.

Depends on: Issue 19, Issue 20, Issue 21.

Scope:

- single source of truth for skills; generate `.codex/skills` and
  `.claude/skills` copies plus an agent-agnostic `AGENTS.md`
- rewrite the workflow around tools: intake, profile, schema negotiation,
  baseline plus primary model, diagnostics, explain, export
- MVP playbooks: choosing K and trusting clusters, field family selection,
  when to use keys, when to go Bayesian (DPM)
- keep `_references/` as the cited evidence base

Acceptance criteria:

- generator script keeps copies in sync (checked in CI)
- skills reference only tools and APIs that exist

---

## Issue 24

**Title:** Add `dmx chat` launcher and a Jupyter `Session` interface

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-chat-session`.

Depends on: Issue 16, Issue 17.

Scope:

- `dmx chat <table>`: thin, vendor-neutral launcher that starts an
  MCP-capable client with `dmx-mcp` attached (D1)
- `dmx.agent.Session` for calling the same tools directly from Python or
  Jupyter

Acceptance criteria:

- launcher documented with at least two clients
- `Session` runs the full MVP tool chain in a notebook test

---

## Issue 25

**Title:** MVP end-to-end evaluation with Claude and one local model

**Body:**

Epic: 4. Agent MVP. Suggested branch slug: `agent-mvp-eval`.

Depends on: Issues 19–24.

Scope:

- scripted conversations on three repo datasets (for example search_depth,
  process_sequences, malware)
- run with Claude and one open-weight local model through MCP
- extend `_benchmarks/evaluation-rubric.md` with schema correctness,
  explanation faithfulness, and export reproducibility
- store transcripts and scores

Acceptance criteria:

- both models complete all three conversations
- exported code reproduces each final fit
- findings feed follow-up issues

---

## Issue 26

**Title:** Per-field fit diagnostics and family-swap suggestions

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `diag-per-field`.

Depends on: Issue 7, Issue 12.

Scope:

- per-field held-out log-likelihood contributions from composite models
- compare against single-field fits of each candidate family
- emit a ranked list of family-swap suggestions with evidence

Acceptance criteria:

- detects a planted wrong family (Gaussian on log-normal data) in a test

---

## Issue 27

**Title:** Residual dependence matrix: empirical vs. model-implied MI

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `diag-residual-mi`.

Depends on: Issue 13.

Scope:

- pairwise empirical mutual information between fields (discretizing
  continuous fields)
- model-implied mutual information by sampling from the fitted model
- gap matrix with bootstrap noise floor; heatmap via `[viz]`

Acceptance criteria:

- planted dependence between two fields appears above the noise floor and
  disappears after fitting a model that captures it

---

## Issue 28

**Title:** Component health diagnostics

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `diag-component-health`.

Depends on: Issue 7.

Scope:

- near-empty and near-duplicate components (weight and KL thresholds)
- posterior entropy summary
- group-weight divergence for a group-key field
- irrelevant fields (low between-component KL)

Acceptance criteria:

- each diagnostic triggers on a planted synthetic case and stays quiet on a
  clean case

---

## Issue 29

**Title:** Structure grammar and candidate builder over existing classes

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `search-structure-grammar`.

Depends on: Issue 14. Implements decision D4.

Scope:

- serializable structure nodes: Composite, Mixture_K, Hierarchical,
  independent blocks (Composite of Mixtures), coupled two-block
  (`JointMixtureDistribution`), Keyed sharing by group, ICL tree over fields,
  Sequence/HMM for ordered fields
- builder from Schema plus structure node to an estimator, including record
  regrouping for blocks

Acceptance criteria:

- every node type builds and fits on synthetic data in tests
- structure nodes round-trip through JSON

---

## Issue 30

**Title:** Deterministic beam search driver with warm starts and steering

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `search-beam-driver`.

Depends on: Issues 26–29. Implements decision D2.

Scope:

- moves proposed from diagnostics (family swap, change K, add tree edge,
  split blocks, keyed sharing, drop irrelevant field)
- budgeted beam search with warm starts via `init_estimator`
- steering arguments: pinned or forbidden moves, fixed families, K range,
  objective, time and fit budget
- automatic subsample-then-refit above the interactive scale target (D11)

Acceptance criteria:

- same seed gives the same search path
- respects budgets in tests
- recovers planted structure on the synthetic suite (Issue 33)

---

## Issue 31

**Title:** SearchReport with leaderboard, lineage, and rationale

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `search-report`.

Depends on: Issue 30.

Scope:

- leaderboard with fold-level confidence intervals
- lineage tree of moves with the diagnostic that triggered each move
- JSON and Markdown renderers; registered in the session

Acceptance criteria:

- report for a synthetic run is readable and complete in a snapshot test

---

## Issue 32

**Title:** Agent tools and playbooks for diagnostics and search

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `agent-search-tools`.

Depends on: Issue 23, Issue 31.

Scope:

- tools: `run_structure_search`, `diagnose_model`, `propose_moves`,
  `residual_dependence`
- playbooks: detecting missing dependence, when to use keys, reading a
  SearchReport, steering the search

Acceptance criteria:

- a local model can run and explain a search end to end on one dataset

---

## Issue 33

**Title:** Synthetic ground-truth benchmark suite

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `bench-synthetic-suite`.

Scope:

- generators under `tests/agent_bench/` with planted K, field families,
  inter-field dependence, group-specific weights, and anomalies
- metrics: structure recovery, held-out LL, adjusted Rand index
- fast CI subset plus a full offline run

Acceptance criteria:

- suite runs `get_dpm_mixture` and the search driver and writes a results
  table

---

## Issue 34

**Title:** External baselines: one-hot sklearn GMM and StepMix

**Body:**

Epic: 5. Diagnostics and structure search. Suggested branch slug: `bench-external-baselines`.

Depends on: Issue 33. Implements decision D10.

Scope:

- baseline harness for sklearn `GaussianMixture`/`BayesianGaussianMixture`
  on standardized one-hot data with a change-of-variables correction for
  held-out LL
- StepMix baseline on the same tables
- benchmark-only dependencies, not added to dmx extras
- results report comparing dmx search, `get_dpm_mixture`, and baselines

Acceptance criteria:

- report regenerates from one command
- headline claim is stated with the observed win rate, whatever it is

---

## Issue 35

**Title:** Uniform `seq_posterior` across mixture-type models

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `posterior-parity`.

Scope:

- consistent `seq_posterior` for mixture, hierarchical, joint, heterogeneous,
  semi-supervised, DPM, and HMM (state marginals)
- document return shapes

Acceptance criteria:

- posteriors sum to one and match brute-force enumeration on small models

---

## Issue 36

**Title:** Marginalize fields out of composite and mixture models

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `marginalize-fields`.

Depends on: Issue 35.

Scope:

- `marginalize(dist, keep_fields)` for composite, mixture, optional, and
  nested combinations
- returns a valid distribution over the kept fields

Acceptance criteria:

- marginal densities match numerical integration or summation in tests

---

## Issue 37

**Title:** Conditional queries and imputation

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `conditional-impute`.

Depends on: Issue 36.

Scope:

- `conditional(dist, observed)` returning the posterior-reweighted mixture
  over unobserved fields (see `_references/posterior-reweighting.md`)
- `impute(dist, records)` with mode, mean, or sample strategies

Acceptance criteria:

- conditional densities verified against Bayes' rule on small models
- imputation beats column-mean baseline on a synthetic test

---

## Issue 38

**Title:** JSON model spec: `to_spec()` and `from_spec()`

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `model-json-spec`.

Scope:

- versioned, human-readable JSON for every `dmx.stats` and `dmx.bstats`
  distribution (family, parameters, keys, children, seed)
- keep pickle for speed; spec is for review, diffing, and exchange

Acceptance criteria:

- round-trip test for every distribution with identical log densities

---

## Issue 39

**Title:** Render models as graphical models with `to_graph()`

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `model-to-graph`.

Depends on: Issue 38.

Scope:

- emit latent nodes, observed fields, and shared keys as Mermaid and Graphviz
- PNG rendering via `[viz]` for the agent

Acceptance criteria:

- snapshot tests for composite, mixture, hierarchical, joint, and HMM models

---

## Issue 40

**Title:** Cluster profiles and distinguishing fields

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `explore-cluster-profiles`.

Depends on: Issue 35.

Scope:

- per-component field summaries from parameters and posterior-weighted data
- top distinguishing fields per component by KL against the global marginal
- tidy DataFrame output plus plot helpers

Acceptance criteria:

- planted distinguishing fields rank first in a synthetic test

---

## Issue 41

**Title:** Anomaly scoring with per-field blame

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `explore-anomalies`.

Scope:

- rank records by log-density
- per-field contribution to each anomaly score from composite structure

Acceptance criteria:

- planted anomalies rank in the top percentile and blame the planted field

---

## Issue 42

**Title:** Group comparisons and two-direction search depth

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `explore-groups-search-depth`.

Depends on: Issue 37.

Scope:

- compare component weights across a group field with bootstrap confidence
  intervals (keyed mixtures)
- search depth in both conditioning directions with a recommendation (see
  `_references/variable-sequence-length-search-depth.md`)

Acceptance criteria:

- detects planted group differences
- reproduces the notebook's direction finding on `data/search_depth`

---

## Issue 43

**Title:** Agent tools for queries and exploration

**Body:**

Epic: 6. Queries and exploration. Suggested branch slug: `agent-explore-tools`.

Depends on: Issues 37–42.

Scope:

- tools: `conditional_query`, `impute`, `cluster_profiles`, `anomalies`,
  `group_comparison`, `render_model_graph`, `export_report`
- privacy filtering applied to every output
- playbook updates for search depth direction choice

Acceptance criteria:

- each tool covered by an in-process MCP test
- `export_report` produces a local Markdown report with plots

---

## Issue 44

**Title:** Engine abstraction for local, MPI, Spark, and torch fitting

**Body:**

Epic: 7. Distributed scale-out. Suggested branch slug: `engine-abstraction`.

Scope:

- `Engine` interface accepted by fit and search functions
- adapters over `optimize_mpi`, `best_of_mpi`, `get_dpm_mixture_mpi`,
  `humap_mpi`, the Spark path, and `torch_stats` where supported
- model spec unchanged across engines

Acceptance criteria:

- the same spec fits locally and on MPI (4 ranks) with matching results
  within tolerance

---

## Issue 45

**Title:** Subsample-search-then-refit workflow

**Body:**

Epic: 7. Distributed scale-out. Suggested branch slug: `subsample-refit`.

Depends on: Issue 30, Issue 44.

Scope:

- stratified subsample for search, refit top candidates on full data with a
  chosen engine
- report how rankings change between subsample and full data

Acceptance criteria:

- tested on a synthetic dataset larger than the interactive target (D11)

---

## Issue 46

**Title:** `scale_out_plan` tool with job script generation

**Body:**

Epic: 7. Distributed scale-out. Suggested branch slug: `scale-out-plan`.

Depends on: Issue 44.

Scope:

- estimate memory and time per EM iteration from n, schema, and structure
- emit `srun`/`mpirun` or `spark-submit` scripts and a sharding recipe
- expose as an agent tool

Acceptance criteria:

- generated MPI script runs the exported model on a test cluster or local
  `mpirun`

---

## Issue 47

**Title:** Distributed skill and playbook

**Body:**

Epic: 7. Distributed scale-out. Suggested branch slug: `distributed-skill`.

Depends on: Issue 46.

Scope:

- replace the distributed out-of-scope boundary in the orchestrator skill
- playbook: when to scale out, engine choice, subsample tradeoffs

Acceptance criteria:

- skills generator keeps all copies in sync
- an agent transcript scales a fit using the playbook

---

## Issue 48

**Title:** End-to-end distributed agent test

**Body:**

Epic: 7. Distributed scale-out. Suggested branch slug: `distributed-e2e`.

Depends on: Issues 44–47.

Scope:

- one conversation that searches locally and refits on MPI with 4 or more
  ranks
- verify identical model spec and comparable scores

Acceptance criteria:

- test documented and runnable in the MPI CI job or a documented local run

---

## Issue 49

**Title:** API cleanup and deprecation removal for 2.0

**Body:**

Epic: 8. 2.0 release. Suggested branch slug: `v2-api-cleanup`.

Scope:

- review signatures frozen during 1.x (D3)
- remove deprecated paths, rename inconsistent parameters, and finalize
  `verbose` and seed defaults
- migration guide

Acceptance criteria:

- migration guide covers every breaking change
- full test suite passes

---

## Issue 50

**Title:** Agent-first user guide and tutorials

**Body:**

Epic: 8. 2.0 release. Suggested branch slug: `v2-user-guide`.

Scope:

- user guide for the table-to-agent workflow
- tutorial notebooks using `dmx.tabular`, search, and explore
- update existing notebooks to the new APIs where helpful

Acceptance criteria:

- docs build passes with the guide and tutorials linked from the index

---

## Issue 51

**Title:** Tech report and published benchmark results

**Body:**

Epic: 8. 2.0 release. Suggested branch slug: `v2-tech-report`.

Depends on: Issue 34.

Scope:

- write up agent-guided structure search, diagnostics, and benchmark
  results against `get_dpm_mixture`, one-hot GMMs, and StepMix
- publish results tables with the regeneration command

Acceptance criteria:

- report draft reviewed and linked from the docs

---

## Issue 52

**Title:** Release dmx-learn 2.0

**Body:**

Epic: 8. 2.0 release. Suggested branch slug: `release-2-0`.

Depends on: Issues 49–51.

Scope:

- version bump, changelog, docs PDF, PyPI release with `[agent]`, `[viz]`,
  `[spark]`, and `[distributed]` extras

Acceptance criteria:

- tagged release and published package install cleanly with each extra

---

## Issue 53

**Title:** Research: `LatentTreeMixture` over three or more field blocks

**Body:**

Epic: 9. Future research. Suggested branch slug: `latent-tree-mixture`.

Depends on: Issue 34 (benchmark gaps).

Scope:

- tree of discrete latent variables over field blocks with exact sum-product
  EM through `seq_update`
- structure learning for block assignment and latent tree edges
- add to the search grammar if benchmarks justify it

Acceptance criteria:

- prototype recovers a planted three-block latent tree
- benchmark shows where it beats the v1 composition vocabulary, or the issue
  closes with documented findings
