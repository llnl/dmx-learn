# Hierarchy And Data Structure

Use this file before choosing individual estimator families.

The main decision is the structure of one observation:

- scalar vs vector vs tuple vs set vs sequence
- ordered vs unordered parts
- optional vs always-present fields
- repeated vs single-instance substructure
- grouped vs ungrouped observations

Choose the observation hierarchy first. Only then choose field-level
estimators such as Gaussian, categorical, Poisson, or multinomial models.

## Contents

- Structure-first rule
- Base estimators
- Composite observations
- Mixtures over structured observations
- Sequences and HMMs
- Grouped observations and sharing
- Joint mixtures
- Heterogeneous mixtures
- Hand-off to estimator selection

## Structure-First Rule

Use this routing order:

1. Decide what one observation is.
2. Decide whether its parts are ordered or unordered.
3. Decide whether any part is optional, repeated, or nested.
4. Decide whether observations come with groups, labels, or paired views.
5. Choose the hierarchical model family.
6. Only after that choose estimator families for each scalar or vector field.

Keep these two questions separate:

- Structure choice: should the model be a base estimator, composite,
  mixture-of-composites, sequence model, HMM, joint mixture, or
  heterogeneous mixture?
- Field-level choice: should one field use a Gaussian, categorical, Poisson,
  multinomial, or other estimator?

If the user has not fixed a narrow downstream task, prefer a reusable joint or
composite latent model over a one-off conditional branch.

## Base Estimators

Use a base estimator when one observation is already a single atomic object and
there is no higher-level substructure to preserve.

Typical routes:

- scalar continuous, count, binary, or categorical values
- fixed-length dense vectors
- one unordered bag or count vector
- one ranking or permutation object
- one set-like object

Use this route when:

- there is no natural decomposition into separate named fields
- there is no repeated nested part that needs its own length or transition model
- latent structure, if needed, can be added later with a mixture over the same
  atomic observation type

Do not jump straight to a base estimator just because each field looks simple.
If one observation is really a record or tuple with different semantic parts,
that is composite structure, not a flat scalar choice.

## Composite Observations

Use a composite model when one observation is a tuple, record, or heterogeneous
row with multiple parts that should stay distinct.

Common signals:

- mixed continuous, categorical, count, or set-like fields in one record
- a tuple like `(category, real value)` or `(metadata, feature vector)`
- fields that are optional or ignorable
- repeated subfields that are still part of one observation, such as a profile
  plus a bag of events

Preferred route:

1. Describe the observation as a tuple or record.
2. Assign one estimator to each field or subfield.
3. Combine them with `CompositeEstimator`.
4. Add wrappers such as `OptionalEstimator`, `IgnoredEstimator`, or
   `WeightedEstimator` only where the structure requires them.

This is the main route for separating hierarchy choice from estimator choice:

- hierarchy choice says "this is a heterogeneous record"
- field-level choice says which estimator each record field should use

## Mixtures Over Structured Observations

Once the observation hierarchy is composite, ask whether latent subtypes are
part of the problem.

Use a mixture-of-composites when:

- heterogeneous observations likely come from latent clusters or regimes
- the user wants one reusable model for clustering, scoring, and downstream
  reweighting
- the task is classification or ranking but labels may share latent structure

Preferred route:

1. Build a `CompositeEstimator` that matches one observation.
2. Wrap that composite in `MixtureEstimator`.
3. Treat this as the default first latent model for heterogeneous local data.

This is usually the right first abstraction for mixed observations because the
latent structure sits above the fields instead of replacing them.

Use `HierarchicalMixtureEstimator` when the latent structure itself is nested,
not just flat clustering.

Use `SemiSupervisedMixtureEstimator` when some observations have partial or weak
label information that should influence the mixture rather than define fully
separate conditional branches.

## Sequences And HMMs

If one observation is a repeated ordered list, route to sequence structure
before deciding emission estimators.

Ask these questions:

- Is order part of the data-generating process?
- Does sequence length vary?
- Are items approximately conditionally independent given length, or do nearby
  items influence each other?
- Is there an unobserved latent state process?

Routes:

- `SequenceEstimator`: ordered variable-length data where the items are
  iid-like given the item model and a separate length model.
- `MarkovChainEstimator`: discrete observed-state transitions matter directly.
- `HiddenMarkovEstimator` or `IntegerHiddenMarkovEstimator`: there is a latent
  state sequence generating emissions.
- `LookbackHiddenMarkovEstimator`: more than first-order latent dependence is
  materially important.

Keep this separation explicit:

- hierarchy choice says "this observation is an ordered sequence" or "this is
  an HMM"
- field-level choice says what each emitted item or sequence element looks like

If the repeated items are unordered, do not route to sequence models. Use a set
or bag route instead.

## Grouped Observations And Sharing

Grouped data do not automatically require a conditional model.

First ask whether groups, labels, or repeated entities change the observation
hierarchy or only suggest parameter sharing.

Common cases:

- ungrouped iid observations: ordinary base, composite, or mixture route
- grouped observations with shared latent factors: keep one main model and use
  estimator `keys` for parameter sharing
- many labels with related structure: prefer shared-component mixtures over
  fully separate per-label composite models

If the same latent library should recur across labels, users, or conditions:

1. choose the observation hierarchy first, often composite or sequence
2. choose whether a mixture sits above that hierarchy
3. then decide which subtree parameters should share via `keys`

Grouping is a sharing decision layered on top of structure choice, not a
replacement for structure choice.

## Joint Mixtures

Use `JointMixtureEstimator` when one observation naturally has two linked views
or modalities and you want a single latent model for the pair.

Common signals:

- paired observations such as `(profile view, event-sequence view)`
- two modalities observed together for each case
- downstream tasks need transfer, conditioning, or reweighting across views
- separate models would lose useful cross-view latent alignment

Use this route when:

- each view has its own internal structure and estimator tree
- the views are coupled through shared or linked mixture structure
- the goal is a reusable joint latent model rather than isolated marginal fits

For each view:

1. choose its own hierarchy first, such as composite or sequence
2. choose field-level estimators inside that view
3. combine the two views with `JointMixtureEstimator`

Do not collapse a multi-view problem into one flat composite if the main
structure is really "paired views with linked latent states."

## Heterogeneous Mixtures

Use `HeterogeneousMixtureEstimator` when the observation is a single atomic
object but different mixture components come from different distribution
families.

This is different from mixture-of-composites:

- mixture-of-composites: each component has the same heterogeneous record
  structure, but parameters differ by component
- heterogeneous mixture: different components themselves belong to different
  estimator families

Typical signal:

- one scalar observation might come from a Poisson-like regime or a Binomial-like
  regime, and that family difference is part of the latent structure

Do not use `HeterogeneousMixtureEstimator` just because the data record has
different field types. That situation usually calls for `CompositeEstimator`
first.

## Hand-Off To Estimator Selection

After choosing the hierarchy:

1. route to `.codex/skills/dmx-local-modeling/references/model-routing.md`
2. pick estimators for each atomic field or emission type
3. reuse repo examples that match the chosen hierarchy

Useful example anchors:

- `examples/stats_examples/composite_example.py`
- `examples/stats_examples/mixture_example.py`
- `examples/stats_examples/hierarchical_mixture_example.py`
- `examples/stats_examples/heterogeneous_mixture_example.py`
- `examples/stats_examples/sequence_example.py`
- `examples/stats_examples/markov_chain_example.py`
- `examples/stats_examples/hidden_markov_example.py`
- `examples/stats_examples/int_hidden_markov_example.py`
- `examples/stats_examples/jmixture_example.py`
