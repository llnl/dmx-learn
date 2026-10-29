# dmx-learn Expert Agent Roadmap

**Goal:** turn dmx-learn into a dedicated expert system for deep graphical
models on heterogeneous tabular data. A user brings a table and talks with an
AI agent. The agent profiles the data, proposes a typed schema with no one-hot
encoding, fits and compares structured latent-variable models, explains what
it finds (clusters, dependencies, anomalies, search depth, embeddings), and
scales the same model to MPI or Spark when the data grows.

---

## 1. Positioning: dmx-learn vs. mixle

mixle forked dmx-learn and grew it into a broad platform: about 90 families,
neural densities, flows, VAEs, transformers, a PPL, LLM distillation
(`task.solve`), copulas and vines, NUTS, an MLOps gateway, and five scale-out
backends. dmx-learn should not compete on breadth. It should be the
**narrow, exact, transparent** tool, with an expert agent built in.

| Axis | mixle | dmx-learn (target) |
|---|---|---|
| Scope | General probabilistic and neural platform | Latent-variable graphical models for heterogeneous records, nothing else |
| Model class | Classical, neural, PPL, and LLM teachers mixed together | Exponential-family leaves composed into mixtures, HMMs, trees, and composites; exact EM or variational updates only |
| Primary user surface | Python API and PPL | A conversation with an expert agent, backed by a small deterministic tool API |
| Model selection | `propose(data)` and information criteria | Explicit, inspectable structure search with a leaderboard, lineage, and diagnostics-driven moves |
| Output | A fitted model object | A fitted model, a written analysis, and plain dmx Python code that reproduces the fit exactly |
| Distributed | Many backends behind one argument | MPI and Spark built for HPC (LLNL), with the same sufficient-statistics path as local fits |
| Dependencies | About 20 extras | Core: numpy, scipy, numba, mpmath, pandas. Extras: `[torch]`, `[spark]`, `[optional]` (mpi4py, umap-learn), `[distributed]` (pyspark, mpi4py), `[viz]`, `[agent]`, and `[all]`. pyspark moved out of core in Epic 1 (D5) |

**Things we deliberately won't build** (documented as out of scope in
`docs/scope.rst`):
neural densities, PPL syntax, LLM distillation, MCMC/NUTS, MLOps serving,
copulas or vines. Anything we add has to be a closed-form or variational
sufficient-statistics model that runs through `seq_update` and therefore
distributes for free.

The tagline is something like *"The expert for mixture models on messy
tables. Every model it fits is one you can read, rerun, and scale to a
cluster."*

---

## 2. Current assets (what we build on)

- **Engine:** `dmx.stats` (about 55 distributions: composite, mixture,
  hierarchical, joint, semi-supervised, HMM variants, tree HMM, ICL tree,
  LDA, PLSI, sets, rankings, optional, conditional), `dmx.bstats` (conjugate
  priors, DPM, variational), `torch_stats`, `mpi4py`, and the Spark sampler.
- **Fitting:** `utils/estimation.py` (`optimize`, `best_of`, `hill_climb`,
  `partition_data`, `empirical_kl_divergence`) and `bstats/bestimation.py`.
- **Automatic structure:** `utils/automatic.py`: `DatumNode`,
  `get_estimator`, `get_dpm_mixture`, `prepare_mixture_model`.
- **Exploration:** `utils/htsne.py` (htsne, dpmsne), `utils/humap.py`, and
  `utils/metrics.py` (`ranking_depth`, ROC, classify).
- **Agent knowledge already written:** `.codex/skills/*` (orchestrator, local
  modeling, implementation, test), `_references/*` (notebook distillations),
  and `_benchmarks/*` (canonical prompts and rubric).

**Gaps that block an agent from being an expert:**

1. No parameter counting, so no BIC, AIC, or ICL. *(Addressed in Epic 2:
   #206, #207, #208.)*
2. No general marginal or conditional query API on fitted composites and
   mixtures (for example, P(field_j | other fields) or imputation).
3. Fitting helpers `print` to stdout, so there are no structured progress
   events or quiet mode for tools. *(Addressed in Epic 2: #209.)*
4. `DatumNode` infers a schema implicitly. There is no editable, serializable
   schema the user and agent can negotiate.
5. No DataFrame or CSV ingestion. Users have to hand-build tuples.
6. Models round-trip only through `__str__`/`eval` or pickle. There is no
   stable JSON model spec and no graph or plate rendering.
7. Skills exist only for Codex and explicitly exclude distributed work.

---

## 3. Target architecture

```
                ┌─────────────────────────────────────────────┐
  User  ◄─────► │ Agent client (Claude Code / Desktop / any   │
                │ MCP client, optional `dmx chat` CLI)        │
                └──────────────┬──────────────────────────────┘
                               │ MCP (JSON tools) + skills/playbooks
                ┌──────────────▼──────────────────────────────┐
                │ dmx.agent  - session, registry, tools, MCP  │  Layer 4
                ├─────────────────────────────────────────────┤
                │ dmx.explore - profiles, queries, embeddings │  Layer 3
                ├─────────────────────────────────────────────┤
                │ dmx.search  - scoring + structure search    │  Layer 2
                ├─────────────────────────────────────────────┤
                │ dmx.tabular - ingestion, Schema, encoding   │  Layer 1
                ├─────────────────────────────────────────────┤
                │ dmx.stats / bstats / torch_stats / mpi4py / │  Layer 0
                │ spark  (existing engine, hardened)          │
                └─────────────────────────────────────────────┘
```

Each layer is plain Python and useful without the agent. The agent only
orchestrates deterministic tools. That keeps results reproducible and makes
`export_code` possible.

---

## 4. Layer 0: engine hardening (prerequisites)

| Item | Detail |
|---|---|
| `num_params()` on every distribution | Free-parameter count, recursive through composite and mixture. Enables BIC, AIC, and ICL. **Done in Epic 2** for every `dmx.stats` and `dmx.bstats` distribution, with keyed parameters counted once (#206, #207). |
| Structured logging | Replace `print` in `optimize`, `best_of`, `get_dpm_mixture`, `htsne`, and `humap` with a callback or `logging` plus `verbose=` (default behavior unchanged). Tools get JSON progress events. **Done in Epic 2** (#209): `verbose=` and `progress_callback` on the fitting and embedding helpers (including `bstats` and MPI variants), DEBUG logging per module, and `dmx.utils.progress.ProgressEvent` with `to_dict()` for JSON. |
| Marginal and conditional ops | `marginalize(dist, keep_fields)` for composite, mixture, and optional; `conditional(dist, observed)` returning the posterior-reweighted mixture over unobserved fields; `impute`. The math is already in `_references/posterior-reweighting.md`. |
| Posterior API parity | Uniform `seq_posterior` for mixture, hierarchical, joint, DPM, and HMM (state marginals) so explore tools are type-agnostic. |
| JSON model spec | `to_spec()` / `from_spec()`: a stable, versioned, human-readable tree (family, params, keys, children). Pickle stays for speed. |
| `to_graph()` | Emit the estimator tree as a graphical model (latent nodes, observed fields, shared keys) in Mermaid or Graphviz, so the agent can show the model. |
| Seed discipline | Every stochastic entry point takes `seed`/`rng` and records it in the spec. **Done in Epic 2** (#210) except the spec: no entry point falls back to NumPy's global state, an unseeded call draws and reports its seed (a `"start"` progress event, `sampler.seed`), and `docs/seeds.rst` audits every entry point. Recording the seed in the JSON spec waits for `to_spec`. |

Exit criteria: tests for each op on composite, mixture, and DPM, plus a
round-trip of every distribution through `to_spec`/`from_spec`.

---

## 5. Layer 1: `dmx.tabular` (no one-hot, ever)

**`Schema`**: an editable, JSON-serializable list of `FieldSpec(name, role,
family, options)`.

- **Roles:** continuous, positive-continuous, count, binary, categorical,
  high-cardinality ID, text (to token bag or sequence), set or multi-label,
  ordered sequence, datetime (decomposed into cyclic or ordinal parts),
  ignored, group key, label.
- **Missingness:** any field can be wrapped in `Optional`, with a missingness
  rate recorded.
- **Inference:** generalize `DatumNode` to columns, with heuristics
  documented in one place: integer cardinality vs. range, skew and positivity
  tests, delimiter detection for list-like strings, and ID detection
  (cardinality ≈ n).
- **Candidate families per field**, not just one, for example
  `continuous → [Gaussian, LogGaussian, Gamma]` and
  `count → [Poisson, Geometric, IntegerCategorical]`. Search (Layer 2) uses
  these.

**Ingestion:** `load_table(path_or_df) -> Table`. pandas is already a core
dependency, so it is used directly (CSV and parquet go through pandas). `Table.to_records(schema)` yields the tuples dmx
already consumes. `schema.to_estimator(structure)` builds the composite.

**Why this is the key UX:** the user edits a schema in plain language ("treat
zip code as an ID, not a number") and never sees an encoding step.

---

## 6. Layer 2: `dmx.search`, model selection beyond "automatic"

### 6.1 Scoring

- Held-out log-likelihood (k-fold through `k_fold_split_index`), the default.
  `heldout_log_likelihood` and `kfold_log_likelihood` are in `dmx.utils.scores`
  (Epic 2, #208).
- BIC, AIC, and ICL (needs `num_params`). ICL penalizes posterior entropy
  and is good for "are these clusters real?" Implemented in `dmx.utils.scores`
  (Epic 2, #208) on the deviance scale (lower is better). ICL covers
  mixture-type models with an `(n, K)` `seq_posterior`; HMMs need the
  state-path entropy from pairwise posteriors, after posterior parity (#240).
- ELBO for `bstats`, and effective component count for DPM. The DPM
  effective component count shipped in Epic 2 (#207); ELBO is still open.
- Task scores: `ranking_depth`, classification accuracy, and imputation
  error.
- Stability: adjusted Rand index across restarts and seeds, plus
  bootstrap K-stability.

### 6.2 A structure grammar

Each candidate is a node in a small grammar the agent can reason about:

```
M := Composite(fields)                   # independence baseline
   | Mixture_K(M)                        # one latent class
   | Hierarchical(K1, K2, M)             # nested latent classes
   | TreeOverFields(M)                   # Chow-Liu / ICL tree dependencies
   | Blocks([fields_a]->Mixture_Ka, ...) # multiple latent variables over field blocks
   | Keyed(M, group_field, shared=...)   # group-specific weights, shared components
   | Sequence/HMM(M)                     # for ordered fields
```

### 6.3 Search moves (diagnostics-driven, not brute force)

| Diagnostic | Suggests move |
|---|---|
| Per-field held-out LL much worse than a single-field model | Swap that field's family |
| High residual mutual information between fields given Z | Add a tree edge or move the fields into a shared block |
| Near-empty or duplicate components | Decrease K, or switch to a DPM |
| Posterior entropy high everywhere | Clusters are weak: try blocks or report "no structure" |
| Component weights differ strongly by a group column | Keyed sharing by group |
| Field irrelevant to all components (low KL across comps) | Mark it as conditionally independent (move it outside the mixture) |

**Code drives, the LLM steers (D2).** The search driver is deterministic
code: a budgeted beam or greedy search over these moves, reusing warm starts
(`init_estimator`). It has to produce good models even when the client LLM is
a weak on-prem model. The LLM steers through arguments (pin or forbid moves,
fix field families, set the budget or K range, choose the objective) and can
explore manually with step-level tools (`diagnose_model`, `propose_moves`,
`fit_model`). A strong LLM adds judgment on top; it is never needed for a
correct result.

**v1 move vocabulary uses existing classes only (D4):** independent blocks as
`Composite([Mixture(Composite(block_a)), Mixture(Composite(block_b))])`, two
coupled blocks as `JointMixtureDistribution` (latent coupling `taus12`), and
nesting via `HierarchicalMixtureDistribution`. A general `LatentTreeMixture`
(a tree of discrete latent variables over three or more blocks, exact
sum-product EM) is a v2 research item, built once benchmarks show where the
two-block limit hurts.

The search runs interactively at about 100k rows × 50 fields, with a full
search in about 5 minutes on a laptop. Above that it automatically
subsamples, searches, and refits the top candidates on the full data. It returns a **`SearchReport`**: a
leaderboard, the lineage tree of moves, why each move was proposed, scores
with confidence intervals across folds, and the selected model.
`get_estimator`/`get_dpm_mixture` become *one starting point* in the search,
not the answer.

### 6.4 Residual dependence diagnostic (new, core differentiator)

Compute empirical pairwise MI between fields (discretized where needed) vs.
model-implied MI (by sampling, or analytically where available). The gap
matrix is the agent's main evidence for *where structure is missing*, and it
renders well as a heatmap.

---

## 7. Layer 3: `dmx.explore`, what users actually ask for

| Capability | Built on |
|---|---|
| Fit a joint density, score rows, flag anomalies (lowest LL, per-field blame) | `seq_log_density`, per-field contributions from composite |
| Cluster profiles: per-component field summaries, top distinguishing fields (KL vs. global) | posteriors and component params |
| Conditional queries ("given A and B, what's the distribution of C?"), imputation, what-if sampling | Layer 0 `conditional` |
| Group and latent comparisons: how cluster mix differs by group | keyed mixtures, weight comparisons with bootstrap CIs |
| Search depth and ranking (paired or retrieval tasks, both conditioning directions) | `metrics.ranking_depth`, `_references/variable-sequence-length-search-depth.md` |
| Low-dimensional embeddings | `htsne`, `dpmsne`, `humap` (reuse the selected model, never refit) |
| Model graph view | `to_graph()` |

**Plots (D6):** every explore function returns tidy pandas DataFrames (core,
no plotting dependency). A `[viz]` extra (matplotlib, included in `[agent]`)
adds `.plot()` helpers. MCP tools return PNG image content, the saved file
path under `.dmx/plots/`, and summary JSON, which works in every MCP client.
There is no Vega-Lite in v1.

---

## 8. Layer 4: `dmx.agent`, the expert

### 8.1 Tool server

Ship `dmx.agent` as an **MCP server** (`dmx-mcp`, optional extra
`dmx-learn[agent]`). Any MCP client works, including clients backed by
on-prem or open-weight models (D1). We test against Claude and at least one
local open-weight model in CI-adjacent evals. There is no bundled chat app.
That makes it a dedicated tool rather than a chat app we have to maintain.

Session state lives on disk in `.dmx/` (datasets, schemas, model registry
with spec, pickle, scores, and seed, plus an experiment log).

Tools (JSON in, JSON out, all quiet and deterministic):

```
load_table, profile_table, propose_schema, update_schema,
fit_model, compare_models, run_structure_search, diagnose_model,
residual_dependence, cluster_profiles, anomalies, conditional_query,
impute, group_comparison, search_depth, embed, render_model_graph,
export_code, export_report, scale_out_plan
```

`export_code` emits a standalone script using only public dmx APIs that
reproduces the selected fit bit-for-bit (same seed). This is the
transparency guarantee and the clearest contrast with black-box platforms.

### 8.2 Expert knowledge (skills and playbooks)

- Port `.codex/skills/*` to an agent-agnostic `AGENTS.md` plus
  `.claude/skills/` (keeping Codex copies generated from one source).
- Rewrite the skills around tools instead of hand-written code: *intake →
  profile → schema negotiation → baseline + primary → diagnostics → search
  moves → explain → export*.
- Add playbooks: **choosing K and when to trust clusters**, **when to use
  keys**, **field family selection**, **detecting missing dependence**,
  **search depth direction choice**, **when to go Bayesian (DPM)**, and
  **when to scale out**.
- Keep `_references/` as the evidence base the skills cite.

### 8.3 Data handling (local first, D8)

The LLM never sees raw rows by default. A `privacy` level on the session
controls what every tool returns:

- **`summaries` (default):** column names and aggregate statistics are
  shared. Category values are included only below a cardinality threshold,
  and never for fields with the `sensitive` or ID role.
- **`aliased`:** column names and category values are replaced with stable
  tokens (`col_3`, `cat_17`). The alias map stays in `.dmx/`, and locally
  exported reports and code resolve the real names.
- **`full`:** explicit opt-in. Sample rows may be shared.

The privacy level is recorded in every report and registry entry. Privacy
filtering happens in the tool layer, never in prompts. This matters for LLNL
and sensitive data.

### 8.4 Optional thin front-end

`dmx chat <table.csv>` is a thin wrapper that launches an MCP-capable client
with `dmx-mcp` attached. It does not depend on a particular LLM vendor.
Notebook integration: a `dmx.agent.Session` you can drive from Jupyter.

---

## 9. Distributed computing

- Introduce an `Engine` interface (`local`, `mpi`, `spark`, `torch`) that
  `fit_model` and `run_structure_search` accept. The model spec never
  changes, only the engine. This builds on the existing `optimize_mpi`,
  `best_of_mpi`, `get_dpm_mixture_mpi`, `humap_mpi`, and the Spark sampler.
- `scale_out_plan` tool: given n, field types, and the model, estimate memory
  and time per EM iteration and emit an `srun`/`mpirun` or `spark-submit`
  script plus a data-sharding recipe.
- Search on big data: run structure search on a stratified subsample locally,
  then refit only the top candidates distributed. The agent explains this
  tradeoff.
- New distributed skill, replacing the current "out of scope" boundary in
  the orchestrator skill.

---

## 10. Evaluation

1. **Synthetic ground-truth suite** (`tests/agent_bench/`): generators with a
   planted K, planted field families, planted inter-field dependence, planted
   group-specific weights, and planted anomalies. Metrics: structure recovery,
   held-out LL vs. the `get_dpm_mixture` baseline, and ARI to true clusters.
2. **Real datasets** already in `data/` (search_depth, process_sequences,
   malware, subset_keys, geotweet, embedding).
3. **Agent transcripts** scored with an extended
   `_benchmarks/evaluation-rubric.md` (add: schema correctness,
   search-reasoning quality, explanation faithfulness, export_code runs and
   reproduces).
4. **External baselines (D10):**
   - **One-hot + sklearn** `GaussianMixture`/`BayesianGaussianMixture` on
     standardized, one-hot data. This directly tests the "no one-hot" claim on
     held-out log-likelihood (with a change-of-variables correction back to
     the original scale) and on cluster recovery.
   - **StepMix**, a mixed-data latent class tool in Python, for parity with
     domain-standard tooling.
   - mixle `propose` is run internally to track differentiation, but is not
     part of published claims.
5. **Headline claim to prove:** on the synthetic suite, agent-driven search
   beats `get_dpm_mixture` alone, one-hot GMMs, and StepMix on held-out
   log-likelihood and structure recovery in most cases, with a readable
   rationale.

---

## 11. Phased delivery (vertical slice first, D9)

Get a working end-to-end agent in front of users first, then deepen it. Real
transcripts drive the search and explore design.

| Release | Scope | Exit criteria |
|---|---|---|
| **1.2: MVP slice** (6–8 wk) | `num_params` (BIC/ICL); quiet `verbose=` logging; pyspark moved to `[spark]`; `Schema` + `load_table`; `dmx-mcp` wrapping existing `optimize`/`best_of`, `get_estimator`/`get_dpm_mixture`, `htsne`/`humap`, and `ranking_depth`; session registry; `export_code`; `summaries`/`aliased` privacy; PNG plots; skills ported to `AGENTS.md`/`.claude/skills`; README repositioning | End-to-end conversation on 3 repo datasets with Claude and one local model; exported code reproduces each fit with the same seed |
| **1.3: Diagnostics + search** (5–7 wk) | Per-field LL, residual MI matrix, component health; deterministic beam search over the D4 vocabulary; `SearchReport`; steering arguments; synthetic benchmark suite with sklearn GMM + StepMix baselines | Search beats `get_dpm_mixture` and both external baselines on most synthetic cases; lineage is readable |
| **1.4: Queries + explore** (4–5 wk) | `marginalize`/`conditional`/`impute`; posterior parity; cluster profiles, anomalies with per-field blame, group comparison, search depth both ways; `to_spec`/`to_graph` | Each capability demoed through the agent on a repo dataset |
| **1.5: Distributed** (3–4 wk) | `Engine` (local/mpi/spark); subsample-then-refit; `scale_out_plan`; distributed skill | The same conversation scales a fit to MPI (4+ ranks) with an identical spec |
| **2.0: Release** | API cleanup, docs, tech report on agent-guided structure search | Tagged release; benchmark results published |
| **v2 research** | `LatentTreeMixture` over 3+ field blocks | Driven by benchmark gaps found in 1.3 |

All 1.x releases are additive: existing signatures stay frozen and defaults
keep current behavior (D3). Breaking changes wait for 2.0.

### Progress

| Epic | Issue | Status | Delivered |
|---|---|---|---|
| 1. Positioning and packaging (1.2) | #200 | Complete | README and package metadata repositioned around the expert agent (#201); `docs/scope.rst` scope and non-goals page (#202); pyspark moved from core to the `[spark]` and `[distributed]` extras with import guards (#203); `[viz]` and `[agent]` extras plus the shared `dmx.utils.optional` import-guard helper (#204) |
| 2. Engine hardening for scoring and quiet tools (1.2) | #205 | Complete | `num_params()` on every `dmx.stats` distribution, with keyed parameters counted once (#206); `num_params()` on every `dmx.bstats` distribution plus DPM `effective_num_components` / `effective_num_params` (#207); `dmx.utils.scores` with held-out and k-fold log-likelihood, BIC, AIC, and ICL (#208); `verbose=` and `progress_callback` with `dmx.utils.progress` events and per-module DEBUG logging, default output unchanged (#209); seed discipline with `dmx.utils.seeding`, reported seeds, `sampler.seed`, and the `docs/seeds.rst` audit (#210). Also fixed the pylint gate so `examples/` is linted (#270) |

---

## 12. Decision log (resolved during review, 2026-10-27)

| # | Decision | Resolution |
|---|---|---|
| D1 | Agent host / LLM access | MCP server first; must work with any MCP client, including on-prem/open-weight models. No bundled chat app. |
| D2 | Where model-selection expertise lives | Deterministic code runs the search; the LLM steers through arguments and step-level tools, and explains. |
| D3 | Backward compatibility | 1.x releases are additive with frozen signatures; breaks wait for 2.0. |
| D4 | Multi-latent / block models | v1 composes existing classes (Composite-of-Mixtures, JointMixture, HierarchicalMixture); `LatentTreeMixture` is v2 research. |
| D5 | Packaging | One package; tabular, search and explore in core; `[agent]` extra for MCP. pyspark moves from core to `[spark]`. pandas stays core (already a dependency). Packaging part implemented in Epic 1 (#203, #204); MPI and UMAP stay together in the existing `[optional]` extra. |
| D6 | Plot output | Tidy DataFrames in core; matplotlib in `[viz]`; MCP returns PNG + path + JSON. No Vega-Lite in v1. |
| D7 | Naming | Keep `dmx-learn` / `dmx` / `dmx-mcp` everywhere. |
| D8 | Privacy default | Tiered: `summaries` default, `aliased` opt-in, `full` explicit; enforced in the tool layer. |
| D9 | Sequencing | Vertical slice first (1.2 MVP), then search, queries/explore, distributed. |
| D10 | External baselines | One-hot + sklearn GMM and StepMix are published; mixle is tracked internally only. |
| D11 | Interactive scale target | About 100k rows × 50 fields with a search in about 5 min on a laptop; auto subsample-then-refit above that. |
