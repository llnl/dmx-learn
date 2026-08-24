# Geotweet Joint Mixtures

This notebook distillation captures the main lessons from
[../notebooks/dmx_advanced_geotweet.ipynb](../notebooks/dmx_advanced_geotweet.ipynb):
when each training case is a paired text-plus-location observation and later
inspection needs to move between those views, the extra complexity of a joint
mixture is worth it because one fit learns reusable topic structure, reusable
spatial structure, and the bridge between them.

## Observation Structure

After preprocessing, one modeled observation is:

```python
(
    hashtag_ids: list[int],
    location: tuple[latitude: float, longitude: float],
)
```

The notebook starts from raw tweet records with text, timestamp, and
coordinates, then:

- extracts hashtags and filtered words from tweet text
- builds a finite hashtag vocabulary
- maps each tweet to integer hashtag ids
- keeps only tweets with at least one in-vocabulary hashtag and finite
  coordinates

The important multi-view claim is:

- `hashtag_ids` is sequence-like discrete content
- `location` is a continuous 2D view
- both views are observed together during training
- the likely downstream questions are cross-view questions such as "where do
  tweets with this topic or hashtag tend to occur?" and "which topics dominate
  this region?"

That is exactly the setting where a joint model is more useful than two
separate single-view fits.

## Joint-Mixture Route

The notebook models the hashtag view and location view with different local
estimators, then couples them with `JointMixtureEstimator`:

```python
word_est = IntegerCategoricalEstimator(
    min_val=0,
    max_val=len(word_map),
    pseudo_count=1.0,
)
hashtag_est = SequenceEstimator(estimator=word_est)

lat_lon_est = CompositeEstimator(
    [GaussianEstimator(pseudo_count=(0.0, 1.0), suff_stat=(0.0, 0.1))] * 2
)

init_est = JointMixtureEstimator(
    estimators1=[hashtag_est] * nmix1,
    estimators2=[lat_lon_est] * nmix2,
    pseudo_count=(1.0, 1.0, 1.0),
)

est = JointMixtureEstimator(
    estimators1=[hashtag_est] * nmix1,
    estimators2=[lat_lon_est] * nmix2,
)
```

Interpretation:

- `Z1` is a latent hashtag-topic mixture component
- `Z2` is a latent geographic mixture component
- `taus12[i, j]` learns how hashtag topic `i` connects to geographic region
  `j`

The smoothing choices matter:

- the hashtag categorical leaf uses `pseudo_count=1.0` so rare hashtags do not
  collapse too aggressively during EM
- the Gaussian leaves use variance regularization through `pseudo_count` and
  `suff_stat` so many spatial components can be fit without degenerate tiny
  variances
- the initialized joint estimator uses smoothed priors for both view mixtures
  and the coupling matrix

## Why The Joint Route Is Worth The Complexity

A simpler baseline is to fit the hashtag topics and location mixture
independently:

```python
hashtag_baseline = MixtureEstimator([hashtag_est] * nmix1)
location_baseline = MixtureEstimator([lat_lon_est] * nmix2)
```

That baseline can summarize each view separately, but it cannot answer the
notebook's main questions without extra ad hoc conditioning logic because it
never learns which text topics align with which spatial regions.

The joint route is worth it when:

- training observations contain both views together
- the downstream task is inspection or reuse across views, not just marginal
  density estimation inside one view
- you want one reusable fit for topic-to-map and map-to-topic queries

Why it wins in this notebook:

- it learns topic structure for hashtags and region structure for coordinates
  in one model
- it exposes the learned bridge through `taus12`, so `P(location | topic)` and
  related conditional views are cheap after fitting
- it supports inspection at both topic level and individual hashtag level
  without refitting a separate conditional model
- it preserves a reusable latent representation even when the immediate query
  changes from topic inspection to hashtag lookup or region-based comparison

## Post-Fit Topic And Component Inspection

The notebook uses the fitted joint model in three inspection patterns.

### Topic-Level Spatial Inspection

For each hashtag topic `i`, form the conditional location mixture:

```python
MixtureDistribution(model.components2, model.taus12[i, :])
```

and visualize its density on a latitude-longitude grid.

This answers:

- which geographic regions are associated with each latent hashtag topic
- which hashtags dominate each topic, by ranking under
  `model.components1[i].dist`

That is the main topic inspection workflow: inspect the text side by top
hashtags and the spatial side by the reweighted location mixture.

### Hashtag-To-Location Queries

For a specific hashtag or small hashtag set, evaluate the joint density over a
spatial grid:

```python
model.seq_log_density(model.dist_to_encoder().seq_encode(query_pairs))
```

and normalize over grid points.

This gives a practical map for queries like:

- `P(location | #food)`
- `P(location | #london)`
- `P(location | [#love, #food])`

### Location-To-Hashtag Inspection Over A Candidate Set

The notebook also normalizes several hashtag densities across the candidate
hashtags at each grid point to inspect:

```python
P(hashtag | location)
```

over a chosen lookup set.

This is useful when the downstream need is comparative interpretation such as
"which of these candidate hashtags is most characteristic of this region?"

## Preferred Routing Rule

Use this notebook as the routing guide when one observation is "discrete
content plus continuous location" and both views are available at training
time:

1. model the discrete content and spatial view with separate local estimators
2. use `JointMixtureEstimator` as the primary route when cross-view inspection
   or posterior transfer is part of the goal
3. keep independent per-view mixtures only as a baseline when you need to show
   whether the learned coupling is actually earning its complexity
4. inspect fitted topics by reweighting the location components with
   `taus12` and inspect specific hashtags by evaluating the joint density over
   a spatial grid
