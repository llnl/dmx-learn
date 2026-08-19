# Compute And Scaling

`dmx.stats` is the default modeling surface in this repo. Do not drift to
`torch_stats` just because PyTorch exists, because the user mentioned speed in
the abstract, or because GPU acceleration sounds attractive in principle.

The routing rule from [../AGENTIC_DMX_PLAN.md](../AGENTIC_DMX_PLAN.md) is
stricter:

- start with `dmx.stats`
- suggest `torch_stats` only when there is a clear compute reason
- ask about GPU preference and device routing before recommending that switch

This reference is about when that exception is actually justified.

## Default Stance

Stay in `dmx.stats` by default for:

- first-pass model selection
- small and medium local datasets
- one-off fits where the main problem is model structure, not raw throughput
- CPU-only work unless the user is already committed to a torch-backed path
- tasks where the repo examples and references already map cleanly to
  `dmx.stats`

Treat `torch_stats` as an acceleration path, not as the default modeling API.
The agent should still describe the model structure in `dmx.stats` terms first,
then note that a torch-backed implementation may be worth using if the compute
profile is large enough.

## When `torch_stats` Is Worth Suggesting

Do not suggest `torch_stats` unless at least one compute trigger is clearly
present and the user is open to accelerator use.

### Trigger 1: Large Observation-By-Component Work

For mixture-like, composite-mixture, or joint-mixture fitting, `torch_stats`
becomes worth mentioning when the repeated E-step workload is plainly large.

Use this rule of thumb:

- stay with `dmx.stats` when `num_observations * num_components < 1e7`
- consider `torch_stats` when `num_observations * num_components >= 1e7`
- recommend asking about GPU/device routing immediately when
  `num_observations * num_components >= 5e7`

Examples:

- `20_000` observations x `20` components: stay in `dmx.stats`
- `250_000` observations x `64` components: `torch_stats` is worth discussing
- `1_000_000` observations x `100` components: strongly consider
  `torch_stats` if the user allows GPU use

### Trigger 2: Large Sequence Or HMM Work

Sequence models get expensive faster, especially when latent-state count grows.

Use these rough thresholds:

- stay with `dmx.stats` when `total_time_steps * num_states < 1e7` and
  `total_time_steps * num_states^2 < 1e8`
- consider `torch_stats` when either threshold is exceeded
- escalate the suggestion when both thresholds are exceeded and the user wants
  repeated fitting, decoding, or posterior passes

This is the main case where "sequence length x state count" matters more than
raw sample count.

### Trigger 3: Repeated Inference Or Many Restarts

Even if one fit is manageable in `dmx.stats`, repeated passes can create a real
compute reason.

Mention `torch_stats` when any of these are true:

- the user wants `best_of` or restart-heavy fitting with `10+` trials on a
  large model
- the workflow will score or compute posteriors over `1e6+` observations more
  than once
- the user expects repeated interactive re-fitting on the same large dataset
- the task is clearly throughput-bound rather than modeling-bound

If the workload is a single exploratory fit plus a few summaries, this is not a
good enough reason by itself.

## When Not To Suggest `torch_stats`

Do not suggest `torch_stats` when:

- the user has not said GPU use is acceptable
- the device target is unknown and the speedup claim depends on GPU
- the model is still being figured out and the dataset is not obviously large
- the workload is small enough that routing complexity will dominate runtime
- the user wants the most repo-native and broadly documented path first

If the user says they want CPU only, keep the default with `dmx.stats` unless
they explicitly ask for a torch-specific implementation.

## How To Ask About GPU Preference

Before recommending `torch_stats`, ask directly about accelerator use and
device routing. Do not bury this in a long intake list.

The minimum questions are:

1. Is GPU or accelerator use acceptable for this fit, or should I stay on CPU?
2. If accelerator use is acceptable, should I target `cuda`, `cuda:0`, `mps`,
   or leave device choice automatic?
3. Is the goal faster fitting only, or do you also need repeated scoring or
   posterior inference on that device?

Those device names match the repo’s torch test conventions in
[../tests/torch_stats/README.md](../tests/torch_stats/README.md).

If the user cannot answer the device question, default back to `dmx.stats`
instead of assuming a GPU path.

## Recommended Suggestion Pattern

When the thresholds say `torch_stats` is worth mentioning, do not replace the
main recommendation outright. Use this shape:

1. name the `dmx.stats` model structure first
2. say why the workload may justify a torch-backed implementation
3. ask about GPU preference and device routing
4. only then pivot to `torch_stats` if the user confirms that preference

That keeps the modeling recommendation stable while treating acceleration as a
second decision.

## Numba JIT Stance During Agent Testing

For routine agent testing, prefer disabling Numba JIT.

That repo stance is already visible in:

- [../AGENTIC_DMX_PLAN.md](../AGENTIC_DMX_PLAN.md)
- [../tests/examples/__init__.py](../tests/examples/__init__.py)
- [../tests/torch_stats/conftest.py](../tests/torch_stats/conftest.py)
- [../setup_and_test.py](../setup_and_test.py)

The default testing behavior is:

- set `NUMBA_DISABLE_JIT=1` for ordinary example and torch-backed test runs
- keep tests deterministic and easier to debug while validating agent changes
- avoid turning "did the model choice make sense?" into "did a JIT kernel warm
  up correctly?"

Only leave Numba enabled when the requested model materially depends on a
Numba-backed path or when the point of the test is specifically to validate
that path. The current example tests already make that exception for
`hidden_markov_example.py` and `int_plsi_example.py` in
[../tests/examples/example_stats_test.py](../tests/examples/example_stats_test.py).

So the practical rule is:

- default: `NUMBA_DISABLE_JIT=1`
- exception: enable Numba only for models or validations that genuinely depend
  on those kernels

## Bottom Line

The agent should be conservative here:

- default to `dmx.stats`
- bring up `torch_stats` only for clearly large or repeated workloads
- ask about GPU preference and exact device routing before recommending it
- keep Numba JIT off during ordinary agent testing unless the code path being
  validated really depends on Numba
