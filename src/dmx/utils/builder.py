"""Build field parsers and estimators for comma-delimited Spark input."""

from typing import Any, Callable, List, Optional, Tuple, cast

from dmx import stats

_STATS_NAMESPACE = {name: getattr(stats, name) for name in stats.__all__}


def _eval_stats_expr(expr: str) -> Any:
    """Evaluate a trusted expression in the public :mod:`dmx.stats` namespace."""
    # pylint: disable=eval-used
    return eval(expr, _STATS_NAMESPACE.copy())


def _build_value_mapper(mapstr: str) -> Optional[Callable[[Any], Any]]:
    """Build a trusted value-mapping function from a field specification."""
    if mapstr == "":
        return None
    # pylint: disable=eval-used
    return cast(
        Callable[[Any], Any], eval("lambda x: " + mapstr, _STATS_NAMESPACE.copy())
    )


def read_index_csv(filename: str) -> List[List[str]]:
    """Read field specifications from a comma-delimited text file.

    Text after ``#`` on each line is discarded. Only lines that split into
    exactly four fields (index, name, mapping expression, and estimator
    expression) are returned; blank and malformed lines are ignored.

    Args:
        filename: Path to the UTF-8 encoded field specification.

    Returns:
        The four string fields from each valid line, without further
        conversion.

    Raises:
        OSError: If the file cannot be opened or read.
    """
    with open(filename, "r", encoding="utf-8") as fin:
        line_parts = map(
            lambda v: v.split("#", 1)[0].split(",", 3), fin.read().split("\n")
        )
    field_rows = filter(lambda v: len(v) == 4, line_parts)
    return list(field_rows)


def get_indexed_rdd_pne(
    field_info: Optional[List[List[str]]] = None, filename: Optional[str] = None
) -> Tuple[Any, Callable[[str], Optional[Tuple[Any, ...]]]]:
    """Create a composite estimator and parser from field specifications.

    Each specification has the form ``[index, name, mapping, estimator]``.
    Estimator and mapping expressions are evaluated as trusted Python input in
    the public :mod:`dmx.stats` namespace. The returned parser splits input on
    literal commas and returns ``None`` when a line has too few fields.

    Args:
        field_info: Field specifications. Used directly when provided.
        filename: File passed to :func:`read_index_csv` when ``field_info`` is
            absent.

    Returns:
        A composite estimator containing the non-null estimator expressions
        and a callable that converts a line to the corresponding value tuple.

    Raises:
        ValueError: If neither ``field_info`` nor ``filename`` supplies field
            specifications.
        OSError: If ``filename`` cannot be read.
        SyntaxError: If a trusted mapping or estimator expression is invalid.
        NameError: If an expression references an unavailable name.
    """
    if filename is not None and field_info is None:
        field_info = read_index_csv(filename)
    if field_info is None:
        raise ValueError("field_info or filename is required.")

    def entry_lambda(idx: int, mapstr: str) -> Callable[[List[str]], Any]:
        """Create a field extractor with an optional value expression.

        Args:
            idx: Zero-based field index.
            mapstr: Python expression evaluated with the extracted string
                available as ``x``. An empty expression returns the string
                unchanged.

        Returns:
            A callable that extracts and optionally converts one split field.
        """
        temp_lambda_0 = _build_value_mapper(mapstr)
        if temp_lambda_0 is not None:

            def mapped_entry(u: List[str]) -> Any:
                return temp_lambda_0(u[idx])

            return mapped_entry

        def direct_entry(u: List[str]) -> str:
            return u[idx]

        return direct_entry

    parser_list: List[Callable[[List[str]], Any]] = []
    estimator_list: List[Any] = []
    max_idx = -1

    for entry in field_info:
        idx, _name, lam, dist = entry
        estimator = _eval_stats_expr(dist)
        if estimator is not None:
            idx_i = int(idx)
            parser_list.append(entry_lambda(idx_i, lam.strip()))
            estimator_list.append(estimator)
            max_idx = idx_i if idx_i > max_idx else max_idx

    def line_parser(line: str) -> Optional[Tuple[Any, ...]]:
        """Parse and convert the configured fields from one input line.

        Args:
            line: Comma-delimited input text.

        Returns:
            Converted values in specification order, or ``None`` if the line
            does not contain the greatest configured field index.
        """
        parts = line.split(",")
        if len(parts) < (max_idx + 1):
            return None
        return tuple(parser(parts) for parser in parser_list)

    estimator = stats.CompositeEstimator(tuple(estimator_list))
    return estimator, line_parser
