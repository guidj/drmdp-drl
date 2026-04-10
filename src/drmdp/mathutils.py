"""
Utils for combinatorial problems.
"""

import math
from typing import Sequence

import numpy as np
from scipy import stats


def sequence_to_integer(space_size: int, sequence: Sequence[int]) -> int:
    """
    Uses the positional system of integers to generate a unique
    sequence of numbers given represetation integer - `index`.
    Based on https://2ality.com/2013/03/permutations.html.

    Horner's method implementation that processes digits from left to right.
    Args:
        space_size: the number of possible digits
        sequence_size: the length of the sequence of digits.
    """
    integer = 0
    for digit in sequence:
        integer = integer * space_size + digit
    return integer


def interger_to_sequence(
    space_size: int, sequence_length: int, index: int
) -> Sequence[int]:
    """
    Uses the positional system of integers to generate a unique
    sequence of numbers given represetation integer - `index`.

    Based on https://2ality.com/2013/03/permutations.html.

    Args:
        space_size: the number of possible digits
        sequence_length: the length of the sequence of digits.
        index: the index of the unique sequence.
    """
    xs = []
    for pw in reversed(range(sequence_length)):
        if pw == 0:
            xs.append(index)
        else:
            mult = space_size**pw
            digit = math.floor(index / mult)
            xs.append(digit)
            index = index % mult
    return tuple(xs)


def hashtrick(xs, dim: int):
    """Hash a sparse feature vector to a dense vector of the given dimension."""
    if dim <= 0:
        raise ValueError("`dim` must be positive")
    # Get indices of non-zero elements directly
    idx = np.nonzero(xs)[0]

    # Use modulo operation on all indices at once
    hashed_idx = idx % dim

    # Use bincount to count occurrences of each index
    # Specify minlength to ensure output has correct size
    return np.bincount(hashed_idx, minlength=dim)


def poisson_exact_confidence_interval(observed_value: int, confidence: float = 0.95):
    """Compute an exact confidence interval for a Poisson-distributed observed value."""
    # Calculate alpha (the complement of the confidence level)
    alpha = 1 - confidence

    # Lower bound: find the value where CDF is approximately alpha/2
    lower_bound = 0
    while stats.poisson.cdf(lower_bound, observed_value) < alpha / 2:
        lower_bound += 1

    # Upper bound: find the value where 1 - CDF is approximately alpha/2
    upper_bound = observed_value
    while 1 - stats.poisson.cdf(upper_bound, observed_value) > alpha / 2:
        upper_bound += 1

    return lower_bound, upper_bound
