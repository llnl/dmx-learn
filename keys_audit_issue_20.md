# Issue 20: `keys` Docstring Audit for `src/dmx/stats`

This audit is limited to estimator-facing `keys` documentation in `src/dmx/stats`.
It is an audit only. No docstrings are changed here.

## Inspected areas

- High-impact keyed estimators and wrappers: `mixture.py`, `jmixture.py`,
  `hidden_markov.py`, `hmixture.py`, `lda.py`, `composite.py`,
  `conditional.py`, `sequence.py`, `optional.py`.
- Existing examples that already demonstrate keyed behavior:
  `examples/stats_examples/mixture_example.py`,
  `examples/stats_examples/hidden_markov_example.py`,
  `examples/stats_examples/lda_example.py`.
- Low-level single-key estimators such as `gaussian.py`, `categorical.py`,
  `gamma.py`, `binomial.py`, and similar modules were spot-checked. Their
  `keys` docstrings are terse, but the highest-value ambiguity sits in the
  composite/mixture wrappers listed below.

## Unclear semantics

- `src/dmx/stats/mixture.py`
  Targets: `MixtureDistribution`, `MixtureAccumulator`, `MixtureEstimator`.
  Gap: the two tuple positions are only described as "weights" and
  "components". The docstrings do not explain that `keys[0]` shares outer
  mixture counts, while `keys[1]` shares component accumulators positionally
  across mixture components with the same shape.

- `src/dmx/stats/jmixture.py`
  Targets: `JointMixtureDistribution`, `JointMixtureEstimatorAccumulator`,
  `JointMixtureEstimator`.
  Gap: the three tuple positions are not explained concretely. The code uses
  one key for joint weight statistics and separate keys for the `X1` and `X2`
  component banks, but the docstrings never say what is actually shared.

- `src/dmx/stats/hidden_markov.py`
  Targets: `HiddenMarkovModelDistribution`, `HiddenMarkovAccumulator`,
  `HiddenMarkovEstimator`.
  Gap: "initial states, transitions, and emissions" is still too vague. The
  docstrings do not say that `keys[2]` shares emission accumulators by hidden
  state index, or that length statistics are outside this tuple.

- `src/dmx/stats/hmixture.py`
  Targets: `HierarchicalMixtureDistribution`,
  `HierarchicalMixtureEstimatorAccumulator`, `HierarchicalMixtureEstimator`.
  Gap: the tuple meaning is underspecified. The code shares the outer
  mixture-by-topic count matrix separately from the shared topic estimators,
  but the docstrings do not make that structure explicit.

- `src/dmx/stats/lda.py`
  Targets: `LDADistribution`, `LDAEstimatorAccumulator`, `LDAEstimator`.
  Gap: this is the thinnest keyed documentation in the inspected set. The
  `keys` argument is barely described, and there is no estimator docstring that
  explains `alpha_key` versus `topics_key`.

- `src/dmx/stats/composite.py`
  Targets: `CompositeDistribution`, `CompositeAccumulator`, `CompositeEstimator`.
  Gap: "shared parameters" and "merging sufficient statistics" do not explain
  that the whole tuple-shaped sufficient statistic is shared positionally, then
  child accumulators still recurse through their own keys.

- `src/dmx/stats/conditional.py`
  Targets: `ConditionalDistribution`, `ConditionalDistributionAccumulator`,
  `ConditionalDistributionEstimator`.
  Gap: current wording overstates the role of the top-level `keys` string. The
  accumulator's `key_merge` delegates to the keyed child accumulators and does
  not use `self.key` directly, so the docstrings are ambiguous and may be
  misleading.

- `src/dmx/stats/sequence.py`
  Targets: `SequenceDistribution`, `SequenceAccumulator`, `SequenceEstimator`.
  Gap: the docstrings do not say whether the wrapper key shares only the base
  distribution, only the length model, or both. The implementation shares the
  wrapper-level sufficient statistic and then also recurses into the base and
  length accumulators.

- `src/dmx/stats/optional.py`
  Targets: `OptionalDistribution`, `OptionalEstimatorAccumulator`,
  `OptionalEstimator`.
  Gap: "keys for parameters" is unclear about what is shared. The
  implementation merges both missing/non-missing counts and the wrapped
  estimator's sufficient statistics under the same outer key.

## Missing examples

- `src/dmx/stats/mixture.py`
  Missing docstring example for nested component sharing such as
  `MixtureEstimator([GaussianEstimator()] * 2, keys=(None, "comps0"))`.
  Existing example source: `examples/stats_examples/mixture_example.py`.

- `src/dmx/stats/hidden_markov.py`
  Missing docstring example for sharing initial-state counts and transition
  counts while leaving emissions unshared, e.g.
  `keys=("init_key", "trans_key", None)`.
  Existing example source: `examples/stats_examples/hidden_markov_example.py`.

- `src/dmx/stats/lda.py`
  Missing docstring example that distinguishes sharing topic-word
  distributions from sharing document-topic priors.
  Existing example source: `examples/stats_examples/lda_example.py` is too weak
  for this because it uses `keys=(None, None)` and does not teach the keyed
  cases.

- `src/dmx/stats/hmixture.py`
  Missing docstring example for sharing a topic bank across multiple outer
  mixtures while keeping outer weights independent.

- `src/dmx/stats/composite.py`
  Missing docstring example showing positional sharing across tuple fields, and
  how composite-level keys interact with child estimator keys.

- `src/dmx/stats/sequence.py`
  Missing docstring example showing that a wrapper-level key shares both token
  statistics and sequence-length statistics unless more specific child keys are
  used.

- `src/dmx/stats/optional.py`
  Missing docstring example showing that the key shares missingness counts plus
  the wrapped estimator's statistics.

## Missing notes on sharing consequences

- `src/dmx/stats/mixture.py`, `src/dmx/stats/jmixture.py`,
  `src/dmx/stats/hidden_markov.py`, `src/dmx/stats/hmixture.py`, `src/dmx/stats/lda.py`
  Missing note: keyed sharing ties parameter updates across model instances, so
  only the explicitly keyed statistics are pooled while unkeyed statistics stay
  separate.

- `src/dmx/stats/mixture.py`, `src/dmx/stats/jmixture.py`,
  `src/dmx/stats/hidden_markov.py`, `src/dmx/stats/hmixture.py`
  Missing note: tuple positions do not mean "share the entire model"; they
  correspond to specific statistic blocks, so partial sharing is possible and
  often the intended use.

- `src/dmx/stats/composite.py`, `src/dmx/stats/sequence.py`,
  `src/dmx/stats/optional.py`
  Missing note: wrapper-level keys can silently over-share because they merge
  outer wrapper statistics and then recurse into keyed children.

- `src/dmx/stats/conditional.py`
  Missing note: the current top-level `keys` story should be clarified before
  improving docstrings, because the implementation and wording are not obviously
  aligned.

## Suggested follow-up order for Issue 21

1. `src/dmx/stats/mixture.py`
2. `src/dmx/stats/hidden_markov.py`
3. `src/dmx/stats/lda.py`
4. `src/dmx/stats/jmixture.py`
5. `src/dmx/stats/hmixture.py`
6. `src/dmx/stats/composite.py`
7. `src/dmx/stats/sequence.py`
8. `src/dmx/stats/optional.py`
9. `src/dmx/stats/conditional.py` after clarifying intended behavior
