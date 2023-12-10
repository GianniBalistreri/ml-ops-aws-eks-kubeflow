"""
Task: ... (Function to run in container)
"""

from typing import NamedTuple


def multiply(a: float, b: float) -> NamedTuple('outputs', [('multiplication', float)]):
    """
    Calculates multiplication of two arguments

    :param a: float
        Numeric value

    :param b: float
        Numeric value

    :return: NamedTuple
        Output values
    """
    return [a * b]
