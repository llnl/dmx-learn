# Initialization, Sufficient Statistics, And Pseudo-Counts

Mixture, hierarchical, sequence, and keyed models in `dmx-learn` are often
won or lost before the main EM loop settles. The important choices are:

- what estimator you use to initialize the fit
- which sufficient statistics you bake into that estimator
- how strong the pseudo-count regularization should be
- how many randomized restarts you run with `best_of`
- how parameter sharing through `keys` changes all of the above

The project plan in [../AGENTIC_DMX_PLAN.md](../AGENTIC_DMX_PLAN.md) calls
these out as first-class modeling decisions. The notebooks and source agree.

## The Default Pattern

The repo repeatedly uses the same high-level fitting pattern:

1. build a more regularized `init_estimator`
2. run `best_of(...)` to try several randomized starts
3. keep the best validation fit
4. continue with `optimize(...)` or `iterate(...)` using the target estimator

You can see that directly in:

- [../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb)
- [../notebooks/dmx_advanced_plsi.ipynb](../notebooks/dmx_advanced_plsi.ipynb)
- [../notebooks/dmx_basics_hmm.ipynb](../notebooks/dmx_basics_hmm.ipynb)
- [../src/dmx/utils/estimation.py](../src/dmx/utils/estimation.py)

The important API detail from
[../src/dmx/utils/estimation.py](../src/dmx/utils/estimation.py) is:

- `best_of` initializes from `init_estimator` when provided
- each trial uses randomized initialization via `seq_initialize(..., init_p)`
- the returned model is the trial with the best validation log-likelihood
- if `vdata` is omitted, selection falls back to the training data

So `best_of` is not just "run EM five times." It is "search for a good basin
with randomized starts, scored on held-out data when possible."

## Init-Estimator Design

The initialization estimator should usually be easier to fit and more
regularized than the final estimator, while still matching the same broad model
shape.

Good repo-native initialization patterns are:

- flattened or pooled discrete estimators with explicit support smoothing
- mixture estimators with uniform or near-uniform component weights
- shared component or topic subtrees across groups using `keys`
- simpler conditional structure for the first pass, followed by the fuller fit

Examples from the notebooks:

- In [../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb),
  the initial mixture uses `suff_stat=np.ones(K) / K` and `pseudo_count=1.0`
  for stable near-uniform starting weights.
- In [../notebooks/dmx_basics_hmm.ipynb](../notebooks/dmx_basics_hmm.ipynb),
  the initial emission estimator is a flattened `CategoricalEstimator` built
  from pooled corpus counts, then reused across HMM states.
- In [../notebooks/dmx_advanced_plsi.ipynb](../notebooks/dmx_advanced_plsi.ipynb),
  `best_of` is run with `init_estimator=est`, which is still valuable because
  the randomized start is happening inside `best_of` even when the init and
  target estimators are the same object.

The practical rule is:

1. keep the initializer structurally compatible with the final estimator
2. make the initializer more stable, more pooled, or more smoothed
3. do not make the initializer so different that EM starts in the wrong model

## Sufficient Statistics As Prior Targets

In this codebase, `suff_stat` is often the target that pseudo-count mass is
pulled toward, not just a cached empirical statistic.

That matters because the same pseudo-count can mean very different things:

- with `pseudo_count` only, the estimator smooths toward a built-in default such
  as uniform mass in mixture weights
- with both `pseudo_count` and `suff_stat`, the estimator shrinks toward the
  supplied sufficient-stat target

For example, [../src/dmx/stats/mixture.py](../src/dmx/stats/mixture.py)
estimates weights as:

- uniform smoothing across components when `pseudo_count` is set and
  `suff_stat` is `None`
- shrinkage toward `suff_stat` when both are provided

That is why the notebooks often use:

```python
MixtureEstimator(
    [iest_tuple] * 10,
    suff_stat=np.ones(10) / 10.0,
    pseudo_count=1.0,
)
```

The modeling claim is not just "smooth the weights." It is "start near a
balanced mixture unless the data quickly say otherwise."

For discrete leaves, pooled empirical frequencies are often the right
`suff_stat` for initialization. The HMM notebook builds a flattened letter
distribution first, then uses that as the emission prior target for every
state:

```python
iest0 = CategoricalEstimator(pseudo_count=1.0, suff_stat=suff_stat)
```

That is a strong pattern for sparse text, counts, or low-support categorical
data: use pooled global sufficient statistics for the initializer so every
component begins with sensible support.

## Pseudo-Count Heuristics

Pseudo-counts are most useful when you need one of these outcomes:

- avoid zero probabilities in sparse discrete models
- prevent early component collapse in mixtures
- keep HMM initial-state and transition estimates away from degenerate corners
- stabilize pooled/shared parameters before responsibilities become sensible

Reasonable heuristics in this repo’s style are:

1. Use larger pseudo-counts for the initializer than for the final estimator.
2. Put pseudo-count mass on the parameters most likely to collapse early:
   mixture weights, categorical probabilities, HMM initial states, and
   transitions.
3. When you know a good reference distribution, pair `pseudo_count` with
   `suff_stat` instead of smoothing blindly.
4. If the model is discrete and sparse, keep at least a tiny final
   pseudo-count instead of forcing exact zeros.

The notebooks show this "strong start, weak finish" pattern clearly. In
[../notebooks/dmx_basics_hmm.ipynb](../notebooks/dmx_basics_hmm.ipynb), the
initial HMM uses:

```python
pseudo_count=(1.0, 1.0)
```

while the later estimator weakens that to:

```python
pseudo_count=(1.0e-6, 1.0e-6)
```

That is usually the right interpretation:

- `1.0`-scale pseudo-counts are for support stabilization and sane starts
- tiny pseudo-counts are for a nearly unregularized final fit that still avoids
  pathological zeros

## Regularized Starts Versus Final Fits

For mixture and hierarchical models, "regularized initialization" and "final
estimation" should usually be treated as separate stages.

Use the regularized start to:

- establish support
- align shared components or topics
- keep weights and transitions from collapsing too early
- give every keyed subtree enough pooled mass to begin learning

Then use the final estimator to reduce or remove that regularization once EM is
already in a good basin.

The usual workflow is:

```python
_, best_model = best_of(
    data=train_data,
    vdata=valid_data,
    est=final_est,
    init_estimator=init_est,
    trials=5,
    init_p=0.10,
    max_its=100,
    delta=1.0e-6,
    rng=np.random.RandomState(10),
)

fit = optimize(
    data=train_data,
    estimator=final_est,
    prev_estimate=best_model,
    max_its=100,
)
```

This is the pattern used in both
[../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb](../notebooks/dmx_basics_conditional_vs_composite_mixture.ipynb)
and [../notebooks/dmx_advanced_plsi.ipynb](../notebooks/dmx_advanced_plsi.ipynb).

When should the final fit be fully unregularized?

- when the model has enough data per free parameter
- when zero-mass states are not a numerical or statistical hazard
- when the regularized start was only needed to find the basin

When should you keep weak regularization in the final fit?

- sparse categorical or count models
- HMM transitions or initial states with weak support
- high-component mixtures where some components see little mass
- keyed/shared parameters that pool uneven evidence across groups

In practice, "final unregularized refit" often means "much weaker
regularization," not necessarily "exactly zero pseudo-count everywhere."

## What `best_of` Is For

`best_of` matters most when the objective is non-convex and the model can get
stuck in poor local optima. That is exactly the situation for:

- mixtures
- composite mixtures
- HMMs and sequence models
- keyed shared-component models
- hierarchical models with several interacting latent blocks

Use more than one restart whenever:

- component labels can permute
- some components can collapse or go empty
- shared parameters depend on early responsibility assignments
- the fit quality varies noticeably across random seeds

The repo notebooks commonly use `trials=5` as a practical default. That is not
magic, but it is a good first pass when the model is moderately expensive and a
single bad start is costly.

The `init_p` argument also matters. Smaller values make the randomized
initialization noisier and more diverse; larger values make it closer to a
full-data initialization. For complex mixture models, `0.1` is a common repo
default because it encourages trial diversity without making starts purely
random.

## Interactions With `keys`

`keys` change the sufficient-statistics story directly. As explained in
[keys-and-parameter-sharing.md](./keys-and-parameter-sharing.md), repeated keys
mean pooled sufficient statistics and shared parameter updates.

That has four implications for initialization.

### 1. Shared Keys Mean Shared Initialization Pressure

If several subtrees share a key, then the initializer is not choosing separate
starting points for them. It is choosing one pooled starting point for the
shared parameter block.

This is often exactly what you want for:

- shared mixture components with group-specific weights
- shared HMM emissions with separate initial-state or transition parameters
- shared sequence-length models across groups

### 2. Pseudo-Counts On Shared Parameters Act On The Pooled Block

With keyed sharing, pseudo-counts regularize the shared update, not each group
independently. So stronger pseudo-counts are often justified on shared
component distributions because each early EM update is affecting many parent
models at once.

### 3. Pooled `suff_stat` Targets Should Match The Sharing Claim

If components are shared across groups, the initialization `suff_stat` should
usually come from pooled data that represent the shared component family.

If you instead seed shared components from one narrow group, the shared fit can
start biased toward that group and take longer to recover.

### 4. Shared Components, Separate Weights Is The Usual Win Pattern

The notebooks repeatedly use keys like:

```python
keys=(None, "comps")
```

or for HMMs:

```python
keys=(None, None, "topics")
```

The point is:

- keep the reusable latent parts shared
- let weights, initial states, or transitions remain group-specific

This is the most important keyed interaction with initialization because early
collapse of shared components is expensive: one bad shared start can degrade
every group at once. That is a strong reason to use a regularized
`init_estimator` and repeated restarts.

## Recommended Defaults

If you need one default policy for a new mixture or hierarchical fit in
`dmx-learn`, use this:

1. Start with an initializer that is slightly simpler, more pooled, and more
   regularized than the target estimator.
2. Give mixture weights or shared categorical parts an explicit `suff_stat`
   target instead of relying on accidental defaults.
3. Use meaningful pseudo-counts for initialization and weaker pseudo-counts for
   the final estimator.
4. Run `best_of` with validation data if you have it.
5. Continue from the best returned model with `optimize(...)`.
6. When using `keys`, smooth the shared subtrees conservatively because their
   early mistakes propagate widely.

That policy matches the existing notebooks and is the right first choice unless
you have strong evidence for a different initialization strategy.
