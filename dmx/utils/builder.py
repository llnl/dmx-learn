"""Helper functions for building RDD for pyspark estimation."""
from dmx.stats import *

def read_index_csv(filename: str):
    """
    Reads a CSV file and extracts field information.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        list: A list of tuples, where each tuple contains four elements 
        extracted from the CSV file (index, name, lambda expression, distribution).
    """
    with open(filename, 'r') as fin:
        lines = map(lambda v: v.split('#', 1)[0].split(',', 3), fin.read().split('\n'))
    lines = filter(lambda v: len(v) == 4, lines)
    return list(lines)


def get_indexed_rdd_pne(field_info=None, filename=None):
    """
    Creates an indexed RDD parser and estimator based on field information.

    Args:
        field_info (list, optional): List of tuples containing field information 
        (index, name, lambda expression, distribution). Defaults to None.
        filename (str, optional): Path to the CSV file containing field information. Defaults to None.

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
        if mapstr != '':
            temp_lambda_0 = eval('lambda x: ' + mapstr)
            temp_lambda = lambda u: temp_lambda_0(u[idx])
        else:
            temp_lambda = lambda u: u[idx]
        return temp_lambda

    parser_list = []
    estimator_list = []
    max_idx = -1

    for entry in field_info:
        idx, name, lam, dist = entry
        estimator = eval(dist)
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
        parts = line.split(',')
        if len(parts) < (max_idx + 1):
            return None
        return tuple([parser(parts) for parser in parser_list])

    estimator = CompositeEstimator(tuple(estimator_list))
    return estimator, line_parser