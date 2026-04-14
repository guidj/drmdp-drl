import numpy as np
import pytest

from drmdp import mathutils


def test_hashtrick_empty():
    # Test with empty input
    xs = np.zeros(10)
    result = mathutils.hashtrick(xs, dim=5)
    assert np.array_equal(result, np.zeros(5))


def test_hashtrick_single_value():
    # Test with single 1 value
    xs = np.zeros(10)
    xs[3] = 1
    result = mathutils.hashtrick(xs, dim=5)
    expected = np.zeros(5)
    expected[3] = 1
    assert np.array_equal(result, expected)


def test_hashtrick_collision():
    # Test hash collision (5 % 3 = 2, 2 % 3 = 2)
    xs = np.zeros(10)
    xs[5] = 1
    xs[2] = 1
    result = mathutils.hashtrick(xs, dim=3)
    expected = np.zeros(3)
    expected[2] = 2  # Two values hash to same bucket
    assert np.array_equal(result, expected)


def test_hashtrick_full_range():
    # Test with 1s across full range
    xs = np.zeros(10)
    xs[1] = 1  # 1 % 4 = 1
    xs[4] = 1  # 4 % 4 = 0
    xs[7] = 1  # 7 % 4 = 3
    result = mathutils.hashtrick(xs, dim=4)
    expected = np.array([1, 1, 0, 1])  # 1->1, 4->0, 7->3
    assert np.array_equal(result, expected)


def test_hashtrick_invalid_input():
    # Test with invalid inputs
    with pytest.raises(ValueError):
        mathutils.hashtrick(np.zeros(10), dim=0)  # dim must be positive

    with pytest.raises(ValueError):
        mathutils.hashtrick(np.zeros(10), dim=-1)  # dim must be positive


def test_sequence_to_integer_empty():
    # Test with empty sequence
    result = mathutils.sequence_to_integer(10, [])
    assert result == 0


def test_sequence_to_integer_single():
    # Test with single digit
    result = mathutils.sequence_to_integer(10, [5])
    assert result == 5


def test_sequence_to_integer_multiple():
    # Test with multiple digits
    # For base 10, [1,2,3] -> 123
    result = mathutils.sequence_to_integer(10, [1, 2, 3])
    assert result == 123


def test_sequence_to_integer_binary():
    # Test with base 2 (binary)
    # [1,0,1] in binary -> 5 in decimal
    result = mathutils.sequence_to_integer(2, [1, 0, 1])
    assert result == 5


def test_sequence_to_integer_base3():
    # Test with base 3
    # [1,2,0] in base 3 -> 15 in decimal (1*9 + 2*3 + 0*1)
    result = mathutils.sequence_to_integer(3, [1, 2, 0])
    assert result == 15


def test_sequence_to_integer_zeros():
    # Test with all zeros
    result = mathutils.sequence_to_integer(10, [0, 0, 0])
    assert result == 0


def test_sequence_to_integer_leading_zeros():
    # Test with leading zeros
    result = mathutils.sequence_to_integer(10, [0, 1, 2])
    assert result == 12


class TestIntergerToSequence:
    def test_single_digit_base10(self):
        result = mathutils.interger_to_sequence(10, 1, 5)
        assert result == (5,)

    def test_multiple_digits_base10(self):
        # 123 in base 10 with length 3 → [1, 2, 3]
        result = mathutils.interger_to_sequence(10, 3, 123)
        assert result == (1, 2, 3)

    def test_binary_roundtrip(self):
        # [1, 0, 1] in base 2 = 5; recover [1, 0, 1]
        integer = mathutils.sequence_to_integer(2, [1, 0, 1])
        result = mathutils.interger_to_sequence(2, 3, integer)
        assert result == (1, 0, 1)

    def test_base3_roundtrip(self):
        sequence = [1, 2, 0]
        integer = mathutils.sequence_to_integer(3, sequence)
        result = mathutils.interger_to_sequence(3, 3, integer)
        assert tuple(result) == tuple(sequence)

    def test_zero_index(self):
        result = mathutils.interger_to_sequence(10, 3, 0)
        assert result == (0, 0, 0)


class TestPoissonExactConfidenceInterval:
    def test_returns_lower_and_upper_bounds(self):
        lower, upper = mathutils.poisson_exact_confidence_interval(5)
        assert isinstance(lower, int)
        assert isinstance(upper, int)

    def test_lower_bound_leq_observed_value(self):
        lower, upper = mathutils.poisson_exact_confidence_interval(10)
        assert lower <= 10
        assert upper >= 10

    def test_lower_while_loop_executes_for_large_observed_value(self):
        # For observed_value >= 4, CDF(0, observed_value) < 0.025, so lower_bound increases
        lower, upper = mathutils.poisson_exact_confidence_interval(10)
        assert lower > 0  # lower_bound was incremented in the while loop

    def test_confidence_interval_contains_observed_value(self):
        observed = 7
        lower, upper = mathutils.poisson_exact_confidence_interval(
            observed, confidence=0.95
        )
        assert lower <= observed <= upper
