# Mixture Models

This notebook distillation captures the main lessons from
[../notebooks/dmx_basics_mixture_models.ipynb](../notebooks/dmx_basics_mixture_models.ipynb):
start from the observation structure, build mixtures from package-native
`dmx.stats` parts, use composite mixtures for heterogeneous conditionally
independent records, use keyed inner mixtures only when several outer models
should share the same latent components, and use a joint mixture when the real
goal is fast posterior transfer between two fitted views.

## Structure-First Routing

The notebook walks through several mixture families, but the routing rule is
simple:

- use a plain `MixtureEstimator` for one homogeneous scalar or vector view
- wrap field-level estimators in `CompositeEstimator` before mixing when one
  observation is a heterogeneous tuple
- use a hierarchical or keyed inner-mixture route when each observation is
  itself a mixture over shared inner topics
- use `JointMixtureEstimator` when one observation is a paired view `(x, y)`
  and the downstream task needs to move posterior mass from one view to the
  other

The package-native preference is compositional: reuse the standard estimator
building blocks instead of inventing custom conditional logic too early.

## Generic Composite-Mixture Pattern

The notebook's main reusable pattern is a mixture of composite observations:

```python
e0 = MixtureEstimator([GaussianEstimator()] * 2)
e1 = OptionalEstimator(GeometricEstimator(), est_prob=True)
e2 = MarkovChainEstimator(len_estimator=PoissonEstimator())
e3 = BernoulliSetEstimator()

comp_est = CompositeEstimator((e0, e1, e2, e3))
est = MixtureEstimator([comp_est] * 3)
```

Interpretation:

- one observation is a heterogeneous tuple
- each field gets the estimator that matches its local structure
- the outer mixture captures latent subtypes of the whole record

This is preferred over ad hoc task-specific conditionals because it keeps one
shared latent model that can later be reused for:

- missing-value imputation by posterior reweighting
- posterior embeddings on the simplex
- downstream conditional views formed by "pop off one field, keep the fitted
  components"

That is the package-native reason to start with a composite mixture when the
record is heterogeneous and the downstream task is not fully fixed.

## Why `optimize` First And `best_of` When Needed

The notebook uses both:

- `optimize(...)` for one vectorized EM fit
- `best_of(...)` when initialization sensitivity matters enough to justify
  multiple randomized starts on a validation split

The preferred default is:

1. use `optimize` for a first stable fit
2. switch to `best_of` when the mixture is non-convex enough that local optima
   matter

This is consistent with the repo-wide guidance in
[init-suffstats-pseudocounts.md](./init-suffstats-pseudocounts.md): stabilize
the start, then keep the best basin instead of overcomplicating the estimator
surface.

## Keyed Inner-Mixture Pattern

The notebook shows two ways to model "a mixture of sequences of mixtures that
share the same inner topics."

### Preferred Route: Specialized Hierarchical Mixture

```python
est = HierarchicalMixtureEstimator([GaussianEstimator()] * 3, 4)
model = optimize(data, est, max_its=200, rng=np.random.RandomState(1))
```

Use this when one observation is:

```python
list[float]
```

but each sequence is generated from:

- one outer mixture over sequence types
- one inner mixture over reusable topic components

Why it is preferred:

- it expresses the intended structure directly with outer weights `w` and
  inner weights `taus`
- it is much faster than spelling out the same sharing with nested generic
  estimators
- it makes the shared-topic claim explicit without manual parameter plumbing

### Generic Fallback: Key The Inner Components

```python
est0 = MixtureEstimator([GaussianEstimator()] * 3, keys=(None, "comps"))
est = MixtureEstimator([SequenceEstimator(est0)] * 4)
model = optimize(data, est, 500, rng=np.random.RandomState(1))
```

Meaning:

- do not share the inner mixture weights
- do share the inner component subtree

This route is slower, but it is still important because it explains the
package's native sharing semantics: if several outer mixtures should reuse the
same latent components, put the key on the inner component slot and leave the
weights unshared.

Use the keyed inner-mixture route when:

- you need the same sharing idea inside a more custom composite or sequence
  construction
- there is no specialized estimator that captures the exact structure

## Joint-Mixture Pattern

The joint-mixture section models one paired observation:

```python
(
    x_view,
    y_view,
)
```

with separate component families for each view and learned transition weights
between them:

```python
est1 = CompositeEstimator([CategoricalEstimator(pseudo_count=1.0), GaussianEstimator()])
est2 = SequenceEstimator(GaussianEstimator(), PoissonEstimator())
est = JointMixtureEstimator([est1] * 3, [est2] * 3,
                            pseudo_count=(0.001, 0.001, 0.001))
```

Prefer `JointMixtureEstimator` over two separate fits when:

- both views are observed together during training
- later inference needs to transfer evidence from one view to the other
- the main downstream win is reweighting, not independent per-view density
  estimation

Why the joint route is worth the added complexity:

- it learns aligned latent structure for both views in one fit
- it exposes `taus12` and `taus21`, which provide a fast bridge between view
  posteriors
- it avoids training separate conditional models for every downstream query

The notebook's key post-fit pattern is:

```python
comp1 = MixtureDistribution(model.components1, model.w1)
weights2 = np.dot(comp1.posterior(x_obs), model.taus12)
weights2 /= weights2.sum()
pred_view2 = MixtureDistribution(model.components2, weights2)
```

That is the package-native reason to choose a joint mixture: fit once, then
move posterior mass across views cheaply.

## Preferred Modeling Rules

Use this notebook as the routing guide for mixture construction:

1. Start from the shape of one observation, not from the downstream query.
2. Use a composite mixture when one record is heterogeneous and its fields are
   conditionally independent given a latent subtype.
3. Use keyed inner mixtures only for "same latent components, different outer
   weights" claims, and keep the key off the weight slot.
4. Prefer the specialized hierarchical estimator when the model really is a
   mixture over shared inner topic mixtures, because it is the same claim with
   much less cost.
5. Use a joint mixture when two views are trained together and later
   conditioning or cross-view retrieval should be done by posterior transfer
   rather than by refitting.
