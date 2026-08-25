Docstring style guide
=====================

dmx-learn uses Google-style docstrings, parsed by Sphinx Napoleon and checked
with pydocstyle.  Write for callers: describe the contract, statistical
meaning, and non-obvious constraints without restating the implementation.

Structure and section order
---------------------------

Start with a short imperative or descriptive summary that ends with
punctuation.  Add a blank line and explanatory prose only when it helps a
caller.  When applicable, use sections in this order:

#. ``Args``
#. ``Attributes`` (for public class attributes)
#. ``Returns`` or ``Yields``
#. ``Raises``
#. ``Note``
#. ``Examples``

Omit empty sections.  Use the exact section names and a colon, indent entries
by four spaces, and leave a blank line before and after each section.  Keep
constructor arguments in ``Args`` on the class docstring.  A function that
only has a useful summary does not need an ``Args`` or ``Returns`` section.

One-line docstrings
-------------------

A one-line docstring is sufficient when the complete caller-facing contract
is obvious from the name, signature, annotations, and surrounding protocol.
This normally applies to simple accessors, trivial properties, and small
protocol methods with no special shapes, side effects, or failure modes::

    def name(self) -> str:
        """Return the distribution name."""

Do not expand a clear one-line docstring merely to satisfy a template.  Use a
longer docstring when a caller needs parameterization, support, shape,
mutation, randomness, device or dtype behavior, initialization semantics, or
exceptions to use the API correctly.

Repository policies
-------------------

``__init__``
    Put the complete construction contract and ``Args`` section on the class
    so autodoc presents it in one place.  When pydocstyle requires a
    constructor docstring, use a short compliant description such as
    ``"""Initialize the Gaussian distribution."""`` rather than duplicating
    the class documentation.

Magic methods
    Give public magic methods a concise, useful description when pydocstyle
    requires one.  Describe the observable result, for example, ``"""Return a
    reproducible representation of the distribution."""``.  Do not document
    Python mechanics that are already self-evident.

Inherited protocol methods
    Document what the concrete implementation does.  State its accepted input
    form, result, shapes, or implementation-specific behavior; do not copy the
    abstract base class text without adapting it.  For example, a concrete
    ``seq_log_density`` docstring should identify that implementation's encoded
    sequence layout and output shape.

Private helpers
    A private helper needs a docstring only when its contract, mutation, shape
    transformation, or algorithm is not obvious.  Keep it short and useful to
    maintainers.  Public APIs always require documentation even if a related
    private helper is documented.

Types
    Prefer function annotations as the source of types.  Do not repeat a type
    in a docstring unless the prose adds a constraint that the annotation
    cannot express, such as accepted dtypes or scalar-versus-array behavior.

Shapes
    State shapes whenever they are part of the contract.  Use names such as
    ``n`` for observations, ``d`` for features, and ``k`` for components,
    define them in prose, and distinguish scalar input from encoded sequence
    input.  Also document sufficient-statistic layouts exposed through
    ``keys`` and device or dtype rules in ``torch_stats``.

``Raises``
    List exceptions that the function deliberately raises as part of its
    public contract and say precisely when they occur.  Do not invent failure
    cases or list every incidental exception that a dependency might propagate.

Raw docstrings
    Prefix a docstring with ``r`` whenever it contains backslashes, especially
    LaTeX commands such as ``\sum`` or ``\sigma``.  This prevents Python escape
    processing and avoids invalid-escape warnings.  Sphinx roles and directives
    still work normally inside a raw docstring.

Statistical API content
-----------------------

For distributions, estimators, accumulators, samplers, and encoders, document
the caller-visible items that apply:

* parameterization and support;
* scalar and encoded-sequence input forms;
* array or tensor shapes;
* public sufficient-statistic layout;
* pseudo-count and initialization semantics;
* device and dtype behavior for ``torch_stats``; and
* exceptional inputs or failure conditions.

Keep derivations in narrative documentation unless a short equation is
necessary to define the API.

Examples
--------

Distribution
~~~~~~~~~~~~

This class docstring records parameterization, support, and attributes without
repeating annotated types::

    class BernoulliDistribution:
        """Represent a Bernoulli distribution on zero and one.

        ``prob`` is the probability of observing one. Scalar density methods
        accept values in ``{0, 1}``; sequence methods consume the representation
        returned by this distribution's data encoder.

        Args:
            prob: Probability of one. Must lie in ``[0, 1]``.
            name: Optional identifier used when composing models.

        Attributes:
            prob: Probability of one.
            name: Identifier used when composing models.
        """

        def __init__(self, prob: float, name: str | None = None) -> None:
            """Initialize the Bernoulli distribution."""

Estimator
~~~~~~~~~

An estimator should explain the statistic layout, initialization semantics,
and meaningful failure modes::

    class BernoulliEstimator:
        """Estimate a Bernoulli distribution from weighted observations.

        Args:
            pseudo_count: Effective sample size of the optional prior. A value
                of ``None`` disables smoothing.
        """

        def estimate(
            self,
            nobs: float | None,
            suff_stat: tuple[float, float],
        ) -> BernoulliDistribution:
            """Estimate a distribution from sufficient statistics.

            Args:
                nobs: Effective observation count, or ``None`` when the count
                    is encoded in ``suff_stat``.
                suff_stat: Pair ``(sum_of_ones, total_weight)``.

            Returns:
                Distribution with its probability set to the weighted mean.

            Raises:
                ValueError: If the total weight is not positive.
            """

Utility function
~~~~~~~~~~~~~~~~

For array utilities, annotations give the type while prose defines the shape
contract::

    def normalize_rows(values: np.ndarray) -> np.ndarray:
        """Normalize each row to sum to one.

        Args:
            values: Nonnegative array of shape ``(n, d)``, where ``n`` is the
                number of observations and ``d`` is the feature dimension.

        Returns:
            New array of shape ``(n, d)`` with unit row sums.

        Raises:
            ValueError: If ``values`` is not two-dimensional or contains a row
                whose sum is zero.
        """

Sphinx and MathJax math
~~~~~~~~~~~~~~~~~~~~~~~

Use ``:math:`` for short inline expressions and ``.. math::`` for a displayed
definition.  Define every symbol, and use a raw docstring because LaTeX uses
backslashes::

    def log_density(self, x: float) -> float:
        r"""Evaluate the Gaussian log density at :math:`x`.

        The mean is :math:`\mu` and the positive variance is :math:`\sigma^2`:

        .. math::

            \log p(x) = -\frac{1}{2}
                \left[\log(2\pi\sigma^2)
                + \frac{(x-\mu)^2}{\sigma^2}\right].

        Args:
            x: Scalar observation on the real line.

        Returns:
            Log density at ``x``.
        """

Validation
----------

Run pydocstyle on every Python source file changed by an issue, using the
smallest relevant target while iterating::

    poetry run pydocstyle src/dmx/stats/gaussian.py
    poetry run pydocstyle src/dmx/utils

The repository configuration selects the Google convention, ignores module
rule ``D100``, and excludes ``__init__.py`` files.  Consequently, manually
review module and package overview docstrings even when pydocstyle reports no
violations.  Run the repository-wide check used by CI and pre-commit with::

    poetry run pydocstyle src/dmx

Build all documentation with warnings treated as errors to validate Napoleon,
cross-references, and MathJax markup::

    poetry run sphinx-build -W -b html docs/ docs/_build/html

Inspect the rendered HTML for representative signatures, lists, and equations.
Do not commit generated files under ``docs/_build``.
