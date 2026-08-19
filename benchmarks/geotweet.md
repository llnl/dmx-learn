# Benchmark: Geotweet Joint Mixtures

Source notebook:
[../notebooks/dmx_advanced_geotweet.ipynb](../notebooks/dmx_advanced_geotweet.ipynb)

## Prompt

The user points you at a local preprocessed geotweet file:
`data/benchmarks/geotweet/preprocessed_geotweets.pkl`

Start by inspecting that file and confirm that one observation is:

```python
(
    hashtag_ids: list[int],
    location: tuple[latitude: float, longitude: float],
)
```

`hashtag_ids` comes from a finite hashtag vocabulary and may contain multiple
hashtags per tweet. `location` is a latitude-longitude pair. Both views are
observed together during training. The main downstream needs from that local
dataset are:

1. inspect which latent hashtag topics map to which regions
2. estimate where a given hashtag or small hashtag set is most likely to
   appear
3. compare a small set of candidate hashtags by `P(hashtag | location)` over a
   map

Recommend one primary `dmx.stats` modeling route and one meaningful baseline.
Explain:

1. why this is a multi-view observation rather than a single flat record
2. how to build the joint route from package-native estimators
3. why the joint route is worth its added complexity
4. what smoothing or regularization choices help keep the fit stable
5. how to inspect the fitted model after training for both topic-level and
   hashtag-level geographic queries

## Expected Success Characteristics

A strong answer should:

- identify the data as paired discrete hashtag content plus continuous spatial
  coordinates
- choose `JointMixtureEstimator` as the primary route, with a
  `SequenceEstimator(IntegerCategoricalEstimator(...))` for hashtags and a
  `CompositeEstimator` of Gaussian leaves for latitude and longitude
- use independent per-view mixtures as the simpler baseline:
  `MixtureEstimator([hashtag_est] * nmix1)` and
  `MixtureEstimator([lat_lon_est] * nmix2)`
- explain that the joint route is preferred because it learns reusable
  structure for both views plus the coupling between them through `taus12`
- mention hashtag pseudo-count smoothing and Gaussian variance regularization
  as practical stabilizers
- describe topic-level inspection by forming
  `MixtureDistribution(model.components2, model.taus12[i, :])` for each
  hashtag topic and pairing that with top-ranked hashtags from
  `model.components1[i].dist`
- describe hashtag-to-location queries by evaluating the joint density on a
  spatial grid and normalizing over locations
- describe candidate-set `P(hashtag | location)` inspection by normalizing the
  hashtag-specific densities across the queried hashtags at each grid point

## Regression Checks

Treat these as failures:

- recommending separate independent hashtag and location fits as the primary
  route without explaining the loss of cross-view transfer
- describing the observation as if latitude and longitude were just extra flat
  features inside a single text model
- omitting the learned coupling or failing to mention `taus12`
- missing the post-fit inspection story and suggesting separate conditional
  refits for hashtag-to-location queries
- failing to explain why the joint route earns its complexity for this
  notebook's downstream tasks
