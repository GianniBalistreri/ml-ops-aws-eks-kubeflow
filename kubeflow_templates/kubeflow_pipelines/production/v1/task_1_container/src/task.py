"""
Task: ... (Function to run in container)
"""

from typing import NamedTuple


def add(a: float, b: float) -> NamedTuple('outputs', [('sum', float)]):
    """
    Calculates sum of two arguments

    :param a: float
        Numeric value

    :param b: float
        Numeric value

    :return: NamedTuple
        Output values
    """
    return [a + b]
