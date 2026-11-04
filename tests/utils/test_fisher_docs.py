"""Keep the Fisher support table in the docs in sync with the test registries.

The table of distributions with Fisher scores in
``docs/utils/fisher_embedding.rst`` is generated from the contract-test
registries (``STATS_CASES`` and ``BSTATS_CASES``): a family is listed when a
registry case of it passes :func:`dmx.utils.fisher.has_fisher_score`. After
adding a family to a registry, regenerate the table with::

    PYTHONPATH=.:src python tests/utils/test_fisher_docs.py --write
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dmx.utils.fisher import has_fisher_score
from tests.bstats.fisher_score_test import BSTATS_CASES
from tests.stats.fisher_score_test import STATS_CASES

DOC = Path(__file__).resolve().parents[2] / "docs" / "utils" / "fisher_embedding.rst"
START = ".. fisher-support-table-start"
END = ".. fisher-support-table-end"

# Display group of each family; unknown families are listed under "Other".
GROUPS: Dict[str, str] = {
    "GaussianDistribution": "Numeric leaves",
    "LogGaussianDistribution": "Numeric leaves",
    "ExponentialDistribution": "Numeric leaves",
    "GammaDistribution": "Numeric leaves",
    "PoissonDistribution": "Count leaves",
    "GeometricDistribution": "Count leaves",
    "BinomialDistribution": "Count leaves",
    "BernoulliDistribution": "Count leaves",
    "CategoricalDistribution": "Categorical leaves",
    "IntegerCategoricalDistribution": "Categorical leaves",
    "MultivariateGaussianDistribution": "Multivariate leaves",
    "DiagonalGaussianDistribution": "Multivariate leaves",
    "DirichletDistribution": "Multivariate leaves",
    "CompositeDistribution": "Combinators",
    "OptionalDistribution": "Combinators",
    "IgnoredDistribution": "Combinators",
    "NullDistribution": "Combinators",
    "SequenceDistribution": "Variable-length",
    "BernoulliSetDistribution": "Variable-length",
    "IntegerMultinomialDistribution": "Variable-length",
    "MultinomialDistribution": "Variable-length",
    "MarkovChainDistribution": "Variable-length",
    "MixtureDistribution": "Latent (empirically whitened)",
    "HeterogeneousMixtureDistribution": "Latent (empirically whitened)",
    "HiddenMarkovModelDistribution": "Latent (empirically whitened)",
    "IntegerHiddenMarkovModelDistribution": "Latent (empirically whitened)",
}
GROUP_ORDER = [
    "Numeric leaves",
    "Count leaves",
    "Categorical leaves",
    "Multivariate leaves",
    "Combinators",
    "Variable-length",
    "Latent (empirically whitened)",
    "Other",
]


def supported_families() -> Tuple[Set[str], Set[str]]:
    """Return the class names with Fisher scores in dmx.stats and dmx.bstats."""
    found: List[Set[str]] = []
    for cases in (STATS_CASES, BSTATS_CASES):
        names = set()
        for case in cases:
            dist = case.make()
            if has_fisher_score(dist, case.options):
                names.add(type(dist).__name__)
        found.append(names)
    return found[0], found[1]


def support_table() -> str:
    """Return the reStructuredText support table, between its markers."""
    stats_names, bstats_names = supported_families()
    rows: Dict[str, List[str]] = {}
    for name in sorted(stats_names | bstats_names):
        rows.setdefault(GROUPS.get(name, "Other"), []).append(name)
    lines = [
        START,
        "",
        ".. list-table:: Distributions with Fisher scores",
        "   :header-rows: 1",
        "   :widths: 30 40 15 15",
        "",
        "   * - Group",
        "     - Distribution",
        "     - ``dmx.stats``",
        "     - ``dmx.bstats``",
    ]
    for group in GROUP_ORDER:
        for name in rows.get(group, []):
            lines += [
                f"   * - {group}",
                f"     - ``{name}``",
                f"     - {'yes' if name in stats_names else '-'}",
                f"     - {'yes' if name in bstats_names else '-'}",
            ]
    lines += ["", END]
    return "\n".join(lines)


def _replace_table(text: str, table: str) -> str:
    """Return ``text`` with the block between the markers replaced."""
    start = text.index(START)
    end = text.index(END) + len(END)
    return text[:start] + table + text[end:]


def test_docs_support_table_matches_registries() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert START in text and END in text
    assert _replace_table(text, support_table()) == text, (
        "The Fisher support table in docs/utils/fisher_embedding.rst is out of "
        "date; regenerate it with "
        "`PYTHONPATH=.:src python tests/utils/test_fisher_docs.py --write`."
    )


if __name__ == "__main__":
    if "--write" in sys.argv:
        DOC.write_text(
            _replace_table(DOC.read_text(encoding="utf-8"), support_table()),
            encoding="utf-8",
        )
        print(f"Updated {DOC}")
    else:
        print(support_table())
