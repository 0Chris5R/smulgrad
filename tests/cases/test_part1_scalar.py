"""
Part 1: Scalar Values and Basic Operations (15 points)

These tests verify your Value class implementation for scalar autodiff.
"""

import pytest
import numpy as np
from tests.adapters import (
    create_value,
    get_value_data,
    get_value_grad,
    run_add,
    run_mul,
    run_neg,
    run_sub,
    run_pow,
    run_div,
)


class TestValueCreation:
    """Test Value object creation (2 points)."""

    def test_value_creation_positive(self):
        """Test creating a Value with a positive number."""
        v = create_value(3.0)
        assert get_value_data(v) == 3.0
        assert get_value_grad(v) == 0.0

    def test_value_creation_negative(self):
        """Test creating a Value with a negative number."""
        v = create_value(-5.5)
        assert get_value_data(v) == -5.5
        assert get_value_grad(v) == 0.0

    def test_value_creation_zero(self):
        """Test creating a Value with zero."""
        v = create_value(0.0)
        assert get_value_data(v) == 0.0
        assert get_value_grad(v) == 0.0

    def test_value_creation_integer(self):
        """Test creating a Value with an integer (should be stored as float)."""
        v = create_value(7)
        assert get_value_data(v) == 7.0


class TestValueAdd:
    """Test Value addition (2 points)."""

    def test_value_add_two_values(self):
        """Test adding two Value objects."""
        a = create_value(2.0)
        b = create_value(3.0)
        c = run_add(a, b)
        assert get_value_data(c) == 5.0

    def test_value_add_value_and_scalar_right(self):
        """Test adding a Value and a scalar (Value + scalar)."""
        a = create_value(2.0)
        c = run_add(a, 3.0)
        assert get_value_data(c) == 5.0

    def test_value_add_value_and_scalar_left(self):
        """Test adding a scalar and a Value (scalar + Value)."""
        a = create_value(2.0)
        c = run_add(3.0, a)
        assert get_value_data(c) == 5.0

    def test_value_add_negative(self):
        """Test adding negative values."""
        a = create_value(-2.0)
        b = create_value(5.0)
        c = run_add(a, b)
        assert get_value_data(c) == 3.0

    def test_value_add_chain(self):
        """Test chaining multiple additions."""
        a = create_value(1.0)
        b = create_value(2.0)
        c = create_value(3.0)
        d = run_add(run_add(a, b), c)
        assert get_value_data(d) == 6.0


class TestValueMul:
    """Test Value multiplication (2 points)."""

    def test_value_mul_two_values(self):
        """Test multiplying two Value objects."""
        a = create_value(2.0)
        b = create_value(3.0)
        c = run_mul(a, b)
        assert get_value_data(c) == 6.0

    def test_value_mul_value_and_scalar_right(self):
        """Test multiplying a Value and a scalar (Value * scalar)."""
        a = create_value(2.0)
        c = run_mul(a, 3.0)
        assert get_value_data(c) == 6.0

    def test_value_mul_value_and_scalar_left(self):
        """Test multiplying a scalar and a Value (scalar * Value)."""
        a = create_value(2.0)
        c = run_mul(3.0, a)
        assert get_value_data(c) == 6.0

    def test_value_mul_negative(self):
        """Test multiplying with negative values."""
        a = create_value(-2.0)
        b = create_value(3.0)
        c = run_mul(a, b)
        assert get_value_data(c) == -6.0

    def test_value_mul_zero(self):
        """Test multiplying by zero."""
        a = create_value(5.0)
        c = run_mul(a, 0.0)
        assert get_value_data(c) == 0.0


class TestValueNegSub:
    """Test Value negation and subtraction (3 points)."""

    def test_value_neg(self):
        """Test negating a Value."""
        a = create_value(3.0)
        b = run_neg(a)
        assert get_value_data(b) == -3.0

    def test_value_neg_negative(self):
        """Test negating a negative Value."""
        a = create_value(-3.0)
        b = run_neg(a)
        assert get_value_data(b) == 3.0

    def test_value_neg_zero(self):
        """Test negating zero."""
        a = create_value(0.0)
        b = run_neg(a)
        assert get_value_data(b) == 0.0

    def test_value_sub_two_values(self):
        """Test subtracting two Value objects."""
        a = create_value(5.0)
        b = create_value(3.0)
        c = run_sub(a, b)
        assert get_value_data(c) == 2.0

    def test_value_sub_value_and_scalar_right(self):
        """Test subtracting a scalar from a Value."""
        a = create_value(5.0)
        c = run_sub(a, 3.0)
        assert get_value_data(c) == 2.0

    def test_value_sub_value_and_scalar_left(self):
        """Test subtracting a Value from a scalar."""
        a = create_value(3.0)
        c = run_sub(5.0, a)
        assert get_value_data(c) == 2.0

    def test_value_sub_result_negative(self):
        """Test subtraction resulting in negative value."""
        a = create_value(3.0)
        b = create_value(5.0)
        c = run_sub(a, b)
        assert get_value_data(c) == -2.0


class TestValuePow:
    """Test Value power operation (3 points)."""

    def test_value_pow_integer(self):
        """Test raising a Value to an integer power."""
        a = create_value(2.0)
        b = run_pow(a, 3)
        assert get_value_data(b) == 8.0

    def test_value_pow_float(self):
        """Test raising a Value to a float power."""
        a = create_value(4.0)
        b = run_pow(a, 0.5)
        np.testing.assert_almost_equal(get_value_data(b), 2.0)

    def test_value_pow_negative_exponent(self):
        """Test raising a Value to a negative power."""
        a = create_value(2.0)
        b = run_pow(a, -1)
        assert get_value_data(b) == 0.5

    def test_value_pow_zero(self):
        """Test raising a Value to the power of zero."""
        a = create_value(5.0)
        b = run_pow(a, 0)
        assert get_value_data(b) == 1.0

    def test_value_pow_one(self):
        """Test raising a Value to the power of one."""
        a = create_value(5.0)
        b = run_pow(a, 1)
        assert get_value_data(b) == 5.0

    def test_value_pow_square(self):
        """Test squaring a Value."""
        a = create_value(3.0)
        b = run_pow(a, 2)
        assert get_value_data(b) == 9.0


class TestValueDiv:
    """Test Value division (3 points)."""

    def test_value_div_two_values(self):
        """Test dividing two Value objects."""
        a = create_value(6.0)
        b = create_value(2.0)
        c = run_div(a, b)
        assert get_value_data(c) == 3.0

    def test_value_div_value_by_scalar(self):
        """Test dividing a Value by a scalar."""
        a = create_value(6.0)
        c = run_div(a, 2.0)
        assert get_value_data(c) == 3.0

    def test_value_div_scalar_by_value(self):
        """Test dividing a scalar by a Value."""
        a = create_value(2.0)
        c = run_div(6.0, a)
        assert get_value_data(c) == 3.0

    def test_value_div_non_integer_result(self):
        """Test division with non-integer result."""
        a = create_value(5.0)
        b = create_value(2.0)
        c = run_div(a, b)
        assert get_value_data(c) == 2.5

    def test_value_div_by_negative(self):
        """Test division by negative value."""
        a = create_value(6.0)
        b = create_value(-2.0)
        c = run_div(a, b)
        assert get_value_data(c) == -3.0


class TestValueCombined:
    """Test combining multiple operations."""

    def test_combined_add_mul(self):
        """Test expression: a * b + c"""
        a = create_value(2.0)
        b = create_value(3.0)
        c = create_value(4.0)
        result = run_add(run_mul(a, b), c)
        assert get_value_data(result) == 10.0

    def test_combined_sub_div(self):
        """Test expression: (a - b) / c"""
        a = create_value(10.0)
        b = create_value(4.0)
        c = create_value(2.0)
        result = run_div(run_sub(a, b), c)
        assert get_value_data(result) == 3.0

    def test_combined_pow_mul(self):
        """Test expression: a^2 * b"""
        a = create_value(3.0)
        b = create_value(2.0)
        result = run_mul(run_pow(a, 2), b)
        assert get_value_data(result) == 18.0

    def test_complex_expression(self):
        """Test expression: (a + b) * (c - d) / e"""
        a = create_value(1.0)
        b = create_value(2.0)
        c = create_value(5.0)
        d = create_value(2.0)
        e = create_value(3.0)
        result = run_div(run_mul(run_add(a, b), run_sub(c, d)), e)
        assert get_value_data(result) == 3.0
