"""Helper functions for building RDD for pyspark estimation."""

from dmx import stats

_STATS_NAMESPACE = {name: getattr(stats, name) for name in stats.__all__}


def _eval_stats_expr(expr: str):
    """Evaluates a trusted stats expression from builder input."""
    # pylint: disable=eval-used
    return eval(expr, _STATS_NAMESPACE.copy())


def _build_value_mapper(mapstr: str):
    """Builds a trusted value-mapping function from builder input."""
    if mapstr == "":
        return None
    # pylint: disable=eval-used
    return eval("lambda x: " + mapstr, _STATS_NAMESPACE.copy())


def read_index_csv(filename: str):
    """
    Reads a CSV file and extracts field information.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        list: A list of tuples, where each tuple contains four elements
        extracted from the CSV file (index, name, lambda expression, distribution).
    """
    with open(filename, "r", encoding="utf-8") as fin:
        lines = map(lambda v: v.split("#", 1)[0].split(",", 3), fin.read().split("\n"))
    lines = filter(lambda v: len(v) == 4, lines)
    return list(lines)


def get_indexed_rdd_pne(field_info=None, filename=None):
    """
    Creates an indexed RDD parser and estimator based on field information.

    Args:
        field_info (list, optional): List of tuples containing field information
        (index, name, lambda expression, distribution). Defaults to None.
        filename (str, optional): Path to the CSV file containing field
            information. Defaults to None.

    Returns:
        tuple: A tuple containing the CompositeEstimator and a line parser function.
    """
    if filename is not None and field_info is None:
        field_info = read_index_csv(filename)

    def entry_lambda(idx, mapstr):
        """
        Creates a lambda function for mapping values.

        Args:
            idx (int): Index of the field to map.
            mapstr (str): Lambda expression as a string.

        Returns:
            function: A lambda function to process the entry.
        """
        temp_lambda_0 = _build_value_mapper(mapstr)
        if temp_lambda_0 is not None:

            def mapped_entry(u):
                return temp_lambda_0(u[idx])

            return mapped_entry

        def direct_entry(u):
            return u[idx]

        return direct_entry

    parser_list = []
    estimator_list = []
    max_idx = -1

    for entry in field_info:
        idx, _name, lam, dist = entry
        estimator = _eval_stats_expr(dist)
        if estimator is not None:
            idx_i = int(idx)
            parser_list.append(entry_lambda(idx_i, lam.strip()))
            estimator_list.append(estimator)
            max_idx = idx_i if idx_i > max_idx else max_idx

    def line_parser(line: str):
        """
        Parses a line and applies the defined parsers.

        Args:
            line (str): A line of text to parse.

        Returns:
            tuple or None: A tuple of parsed values, or None if the line is invalid.
        """
        parts = line.split(",")
        if len(parts) < (max_idx + 1):
            return None
        return tuple(parser(parts) for parser in parser_list)

    estimator = stats.CompositeEstimator(tuple(estimator_list))
    return estimator, line_parser
