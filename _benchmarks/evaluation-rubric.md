# Agentic DMX Evaluation Rubric

Use this rubric to review answers against the five canonical benchmark prompts.
Keep the review lightweight: score each dimension as `2 = expert`, `1 = partial`,
or `0 = miss/regression`.

## Dimensions

1. **Model structure choice**
   The answer correctly identifies what one observation is and chooses a
   primary `dmx.stats` structure that matches it: composite, mixture,
   joint-mixture, sequence, HMM, or another package-native route. For vague
   tasks, it prefers one reusable shared model plus one meaningful baseline
   instead of a broad model sweep.

2. **API correctness**
   The proposed construction is actually valid `dmx-learn`: the estimator
   types, nesting, and post-fit operations are package-native and coherent.
   A strong answer names the right estimator families and avoids impossible or
   internally inconsistent API claims.

3. **`keys` judgment**
   The answer uses `keys` only when the statistical claim is shared structure
   with non-shared group-specific parts. It clearly states what is shared, what
   is not, and why. It should avoid over-sharing, especially on mixture weights,
   transitions, or sequence-length structure when those are meant to differ.

4. **Initialization and fit hygiene**
   The fit plan uses sane initialization for the chosen structure: reasonable
   pseudo-counts or smoothing, stabilized initial estimators, and repeated
   restarts such as `best_of` when mixtures or hierarchical fits are fragile.
   A weak answer treats initialization as irrelevant.

5. **Downstream-task usefulness**
   The fitted model is usable for the task the benchmark cares about. Strong
   answers explain the post-fit path clearly, usually through posterior reuse,
   conditioning, reweighting, or component inspection rather than proposing a
   separate refit for each downstream query.

6. **Compute-path judgment**
   The answer makes a sensible complexity choice: generic vs specialized
   estimator, simpler baseline vs richer primary model, and ordinary
   `dmx.stats` vs `torch_stats` only when the scale justifies it. A strong
   answer earns extra structure or compute cost with a concrete benefit.

## Benchmark-Aligned Pass Signal

A benchmark answer should usually be considered expert only if it has no `0`
on `Model structure choice`, `API correctness`, or `Downstream-task usefulness`.
Across the full suite, the rubric should reward the same behaviors emphasized in
the prompts: structure-first routing, correct package-native APIs, justified
sharing through `keys`, stable fitting, reusable post-fit workflows, and
practical compute judgment.
