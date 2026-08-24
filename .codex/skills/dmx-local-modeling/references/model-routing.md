# Model Routing

Use this file when the user has described the data well enough that you need to
map it to a concrete local `dmx.stats` or `dmx.bstats` family. Default to an
explicit `dmx.stats` estimator for ordinary non-Bayesian work; select
`dmx.bstats` when priors, variational inference, a DPM, or intentional automatic
Bayesian construction is part of the objective.

## Contents

- Scalar observations
- Vector observations
- Mixed and structured observations
- Sequence and ranking observations
- Bayesian, variational, and DPM routing
- Latent structure and embeddings
- Example file map
- Out of scope

## Scalar Observations

### Continuous scalar values

- `GaussianEstimator`: default first pass for roughly symmetric real-valued data.
- `ExponentialEstimator`: positive values with strong right tail and memoryless assumptions.
- `GammaEstimator`: positive skewed data when `ExponentialEstimator` is too rigid.
- `LogGaussianEstimator`: positive data that look log-normal.

Examples:
- `examples/stats_examples/gaussian_example.py`
- `examples/stats_examples/exponential_example.py`
- `examples/stats_examples/gamma_example.py`
- `examples/stats_examples/log_gaussian_example.py`

### Counts and binary outcomes

- `PoissonEstimator`: event counts with rate-like behavior.
- `GeometricEstimator`: counts until first success.
- `BinomialEstimator`: bounded integer successes out of fixed trials.

Examples:
- `examples/stats_examples/poisson_example.py`
- `examples/stats_examples/geometric_example.py`
- `examples/stats_examples/binomial_example.py`

### Unordered categories and bags

- `CategoricalEstimator`: string-like or hashable labels.
- `IntegerCategoricalEstimator`: integer-coded categories.
- `MultinomialEstimator`: bag or count-vector observations.
- `IntegerMultinomialEstimator`: integer-indexed bag or count-vector observations.

Examples:
- `examples/stats_examples/categorical_example.py`
- `examples/stats_examples/intrange_example.py`
- `examples/stats_examples/catmultinomial_example.py`
- `examples/stats_examples/intmultinomial_example.py`

## Vector Observations

### Continuous vectors

- `MultivariateGaussianEstimator`: dense continuous vectors with full covariance.
- `DiagonalGaussianEstimator`: continuous vectors with approximately independent dimensions.
- `GaussianMixtureEstimator`: continuous clustering with full-covariance components.
- `DiagonalGaussianMixtureEstimator`: continuous clustering with diagonal covariance.

Examples:
- `examples/stats_examples/mvn_example.py`
- `examples/stats_examples/dmv_example.py`
- `examples/stats_examples/gmm_example.py`
- `examples/stats_examples/dmvn_mixture_example.py`

## Mixed And Structured Observations

### Heterogeneous tuples or records

Build one field-level estimator per attribute, then combine them with
`CompositeEstimator`.

Common wrappers:
- `OptionalEstimator`: handle missing or optional values.
- `IgnoredEstimator`: leave a field in place but exclude it from modeling.
- `WeightedEstimator`: reweight a component when the likelihood contribution needs scaling.
- `ConditionalDistributionEstimator`: model label-conditioned observations.

Examples:
- `examples/stats_examples/composite_example.py`
- `examples/stats_examples/conditional_example.py`
- `examples/stats_examples/optional_example.py`
- `examples/stats_examples/heterogeneous_mixture_example.py`

### Composite mixtures as the default latent model

When an observation is heterogeneous and you expect latent subtypes, treat
mixture-of-composites as the primary model family rather than a special case.

Preferred pattern:

1. Build one estimator per field.
2. Combine them with `CompositeEstimator`.
3. Wrap the composite in `MixtureEstimator`.

This is usually the right first abstraction for mixed data because it keeps the
field semantics explicit while still giving the model a latent subtype
mechanism.

Examples:
- `examples/stats_examples/mixture_example.py`
- `examples/detailed_estimation_example.py`

### Mixtures over structured observations

Use a mixture when you expect latent subpopulations.

- `MixtureEstimator`: generic latent clustering.
- `HierarchicalMixtureEstimator`: nested mixture structure.
- `HeterogeneousMixtureEstimator`: mixed data with heterogeneous components.
- `SemiSupervisedMixtureEstimator`: some supervision or partial labeling.

Examples:
- `examples/stats_examples/mixture_example.py`
- `examples/stats_examples/hierarchical_mixture_example.py`
- `examples/stats_examples/heterogeneous_mixture_example.py`
- `examples/stats_examples/semi_supervised_mixture_example.py`
- `examples/detailed_estimation_example.py`

### Low-rank conditional approximations with keyed components

For heterogeneous supervised problems with many labels, do not default to
fitting a fully separate mixture inside every conditional branch.

Prefer this question:

- Are the labels likely to share a smaller library of latent components, with
  only the mixture weights changing by label?

If yes, treat the model as a low-rank approximation to `p(x | y)`:

1. Build a shared mixture-of-composites.
2. Use `MixtureEstimator(..., keys=(None, "shared-components"))` so component
   distributions are tied.
3. Let each label keep its own mixture weights.
4. Use `ConditionalDistributionEstimator` only as the thin outer wrapper over
   labels, not as the main source of modeling complexity.

This pattern is especially useful for:

- ranking over many candidate labels
- conditional density estimation with heterogeneous records
- keeping non-target-label data in training so it can help estimate shared
  components
- forming a low-rank conditional model that is smaller and often more stable
  than fully independent per-label mixtures

Examples:
- `examples/stats_examples/mixture_example.py`

## Sequence And Ranking Observations

### Ordered sequences

- `MarkovChainEstimator`: discrete first-order transitions with explicit length modeling.
- `HiddenMarkovEstimator`: latent state sequences over emissions.
- `IntegerHiddenMarkovEstimator`: integer-coded HMM data.
- `LookbackHiddenMarkovEstimator`: sequence dependence that needs more history.
- `SequenceEstimator`: wrap a base estimator when the observation is a sequence of iid-like items.

Examples:
- `examples/stats_examples/markov_chain_example.py`
- `examples/stats_examples/hidden_markov_example.py`
- `examples/stats_examples/int_hidden_markov_example.py`
- `examples/stats_examples/sequence_example.py`

### Sets and edit-style discrete objects

- `BernoulliSetEstimator`: unordered set membership.
- `IntegerBernoulliSetEstimator`: integer-coded set membership.
- `IntegerBernoulliEditEstimator`: edit-style set differences.
- `IntegerStepBernoulliEditEstimator`: step-aware edit behavior.

Examples:
- `examples/stats_examples/set_example.py`
- `examples/stats_examples/intsetdist_example.py`
- `examples/stats_examples/set_edit_example.py`
- `examples/stats_examples/stepset_edit_example.py`

### Rankings and topic-style objects

- `SpearmanRankingEstimator`: ranked permutations and order data.
- `IntegerPLSIEstimator`: topic-like latent structure for integer count data.
- `LDAEstimator`: topic modeling when the user explicitly wants LDA-style latent topics.

Examples:
- `examples/stats_examples/spearman_rho_example.py`
- `examples/stats_examples/int_plsi_example.py`
- `examples/stats_examples/lda_example.py`

### Heterogeneous ranking over labels

For tasks that rank candidate labels using heterogeneous observations:

- start from a mixture-of-composites view of the observation,
- then decide whether labels should have fully separate models or keyed shared
  components,
- prefer keyed shared components when the label set is large and latent
  structure is plausibly shared.

Do not force `SpearmanRankingEstimator` onto these problems unless the
observation itself is a ranking or permutation.

## Bayesian, Variational, And DPM Routing

Choose `dmx.bstats` instead of `dmx.stats` when at least one of these is
material to the requested model:

- informative or inspectable parameter priors
- expected log-density or other Bayesian distribution behavior
- local variational estimation through `dmx.bstats.bestimation`
- a truncated Dirichlet-process mixture
- automatic estimator construction specifically intended for a Bayesian or
  automatic mixture workflow

The main high-value route is a DPM over composite distributions for
heterogeneous observations:

1. Represent each record with a `dmx.bstats.CompositeEstimator` whose children
   match the field types.
2. Use that composite as the repeated base estimator in a
   `DirichletProcessMixtureEstimator`, or pass it as `estimator=` to
   `dmx.utils.automatic.get_dpm_mixture`.
3. Fit locally with `dmx.bstats.bestimation.optimize`, or let
   `get_dpm_mixture` perform the variational fit and convert active components
   to a finite `dmx.bstats.MixtureDistribution`.

For intentional automatic structure inference, use
`dmx.utils.automatic.get_estimator(data, use_bstats=True)`. This is a
first-class `bstats` route for supported primitive and structural observations,
not a substitute for explicit `dmx.stats` construction in a routine
non-Bayesian task.

Evidence anchors:

- `tests/bstats/dpm_test.py`: DPM initialization, variational updates, local
  optimization, and `get_dpm_mixture`
- `tests/bstats/composite_test.py`: composite distributions inside finite
  mixtures and DPM containers
- `tests/bstats/structural_test.py` and
  `tests/bstats/discrete_primitives_test.py`: automatic `bstats` routing

These are local paths. Do not replace them with `get_dpm_mixture_mpi` or APIs
under `dmx.mpi4py`; those belong to MPI-specific workflows.

## Latent Structure And Embeddings

For 2D exploratory views of heterogeneous or mixture-like data:

1. Fit an explicit mixture model yourself, or use
   `dmx.utils.automatic.prepare_mixture_model`.
2. Inspect posteriors and component structure first.
3. Use `dmx.utils.htsne.htsne` or `dmx.utils.humap.humap` for embeddings.

Notes:
- `prepare_mixture_model` is useful for local exploratory embedding workflows.
- Use `get_estimator(..., use_bstats=True)` when Bayesian automatic routing is
  intended. Continue to prefer an explicit `dmx.stats` estimator for ordinary
  non-Bayesian modeling.

Examples:
- `examples/htsne_example.py`
- `notebooks/dmx_advanced_embedding_plots.ipynb`

## Example File Map

- Simple local EM fit: `examples/stats_examples/gaussian_example.py`
- Nested or keyed mixtures: `examples/stats_examples/mixture_example.py`
- Validation loop and held-out likelihood: `examples/detailed_estimation_example.py`
- Ranking: `examples/stats_examples/spearman_rho_example.py`
- Local embeddings: `examples/htsne_example.py`
- Bayesian automatic routing: `tests/bstats/structural_test.py` and
  `tests/bstats/discrete_primitives_test.py`
- Local variational DPM and automatic conversion: `tests/bstats/dpm_test.py`
- Composite distributions in mixtures and DPMs:
  `tests/bstats/composite_test.py`

## Out Of Scope

- `examples_spark/`
- `examples_mpi4py/`
- `src/dmx/stats/rdd_sampler.py`
- `dmx.mpi4py.bstats`, `dmx.mpi4py.utils`, and `get_dpm_mixture_mpi`
- Distributed or remote data loading

Hand off those workflows to a separate distributed estimation skill.
