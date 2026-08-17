# Repo Canonical Examples

Use this as the first stop when an agent needs a package-native pattern.
Prefer these repo examples and utilities before writing fresh scaffolding.

## Task Map

| Task family | Start from | Reuse these entry points | Notes |
| --- | --- | --- | --- |
| Keyed mixtures and shared components | [keys-and-parameter-sharing.md](./keys-and-parameter-sharing.md), [conditional-vs-composite-mixture.md](./conditional-vs-composite-mixture.md), [process-sequences.md](./process-sequences.md), [../examples/stats_examples/mixture_example.py](../examples/stats_examples/mixture_example.py) | `MixtureEstimator(..., keys=(None, "comps"))`, `HiddenMarkovEstimator(..., keys=(None, None, "topics"))`, keyed length estimators | Default to shared components or emissions with unshared weights unless the modeling claim really says to share weights too. |
| Joint mixtures and cross-view transfer | [mixture-models.md](./mixture-models.md), [geotweet.md](./geotweet.md), [posterior-reweighting.md](./posterior-reweighting.md), [../examples/stats_examples/jmixture_example.py](../examples/stats_examples/jmixture_example.py) | `JointMixtureEstimator`, `JointMixtureDistribution`, `MixtureDistribution(model.components1, model.w1)`, `model.taus12`, `model.taus21` | Use when one observation has paired views and post-fit inference must move posterior mass from one view to the other. |
| Train/validation EM loops | [init-suffstats-pseudocounts.md](./init-suffstats-pseudocounts.md), [mixture-models.md](./mixture-models.md), [../examples/detailed_estimation_example.py](../examples/detailed_estimation_example.py) | `partition_data`, `optimize`, `best_of`, `seq_encode`, `seq_log_density_sum`, `seq_estimate` from `dmx.utils.estimation` and `dmx.stats` | Use `optimize` for one fit, `best_of` for repeated randomized starts, and explicit encoded loops only when the example's lower-level diagnostics are needed. |
| Ranking and search reduction | [reduced-search-space.md](./reduced-search-space.md), [variable-sequence-length-search-depth.md](./variable-sequence-length-search-depth.md), [process-sequences.md](./process-sequences.md) | `seq_posterior`, `posterior`, `seq_component_log_density`, `MixtureDistribution(..., w=posterior_weights)` | Fit one shared model, cache posterior summaries or component scores, then rank in that latent space instead of fitting per-key models. |
| Embeddings | [posterior-reweighting.md](./posterior-reweighting.md), [process-sequences.md](./process-sequences.md), [plsi.md](./plsi.md), [../examples/htsne_example.py](../examples/htsne_example.py) | `model.seq_posterior(...)`, averaged posterior vectors, `IntegerPLSIEstimator.state_mat`, `dmx.utils.htsne.htsne`, `dmx.utils.humap.humap` | Prefer posterior-simplex embeddings from an already fitted mixture; use `htsne`/`humap` for visualization after the probabilistic model is defined. |
| Sequence models | [process-sequences.md](./process-sequences.md), [variable-sequence-length-search-depth.md](./variable-sequence-length-search-depth.md), [../examples/stats_examples/sequence_example.py](../examples/stats_examples/sequence_example.py), [../examples/stats_examples/hidden_markov_example.py](../examples/stats_examples/hidden_markov_example.py), [../examples/stats_examples/int_hidden_markov_example.py](../examples/stats_examples/int_hidden_markov_example.py) | `SequenceEstimator`, `HiddenMarkovEstimator`, `IntegerHiddenMarkovEstimator`, `MarkovChainEstimator`, `len_estimator`, `seq_encode` | Choose `SequenceEstimator` for conditionally independent elements, HMM estimators when transition structure matters, and explicit `len_estimator` when length carries or regularizes signal. |

## Utility Defaults

- Use `dmx.stats` as the default modeling surface.
- Use `CompositeEstimator` before mixing heterogeneous records.
- Use `partition_data(data, [0.9, 0.1], rng)` for ordinary train/validation
  splits.
- Use `optimize(...)` for a single EM fit and `best_of(...)` when local optima
  or initialization sensitivity matter.
- Encode once with `seq_encode(...)` before repeated scoring, posterior, or
  search loops.
- For downstream conditionals, reuse fitted components and replace mixture
  weights with posterior-derived weights instead of refitting.

## If Unsure

Identify the shape of one observation first: scalar, vector, tuple, sequence,
ranking pair, or paired views. Then return here and pick the closest
repo-native example.
