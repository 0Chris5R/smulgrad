"""
Part 3: More Operations (15 points)

These tests verify additional operations like ReLU, exp, log, and tanh.
"""

import pytest
import math
import numpy as np
from .conftest import assert_close
from tests.adapters import (
    create_value,
    get_value_data,
    get_value_grad,
    run_add,
    run_mul,
    run_sub,
    run_div,
    run_pow,
    run_relu,
    run_exp,
    run_log,
    run_tanh,
    run_backward,
)


class TestReLU:
    """Test ReLU activation (3 points)."""

    def test_relu_positive(self):
        """Test ReLU with positive input."""
        a = create_value(3.0)
        b = run_relu(a)
        assert get_value_data(b) == 3.0

    def test_relu_negative(self):
        """Test ReLU with negative input."""
        a = create_value(-3.0)
        b = run_relu(a)
        assert get_value_data(b) == 0.0

    def test_relu_zero(self):
        """Test ReLU with zero input."""
        a = create_value(0.0)
        b = run_relu(a)
        assert get_value_data(b) == 0.0

    def test_relu_backward_positive(self):
        """Test ReLU gradient with positive input."""
        a = create_value(3.0)
        b = run_relu(a)
        run_backward(b)
        # d(relu)/da = 1 when a > 0
        assert_close(get_value_grad(a), 1.0)

    def test_relu_backward_negative(self):
        """Test ReLU gradient with negative input."""
        a = create_value(-3.0)
        b = run_relu(a)
        run_backward(b)
        # d(relu)/da = 0 when a < 0
        assert_close(get_value_grad(a), 0.0)

    def test_relu_in_expression(self):
        """Test ReLU within a larger expression."""
        a = create_value(-2.0)
        b = create_value(5.0)
        # c = relu(a + b) = relu(3) = 3
        c = run_relu(run_add(a, b))
        assert get_value_data(c) == 3.0

        run_backward(c)
        # dc/da = dc/db = 1 (since a+b > 0)
        assert_close(get_value_grad(a), 1.0)
        assert_close(get_value_grad(b), 1.0)


class TestExp:
    """Test exponential function (3 points)."""

    def test_exp_zero(self):
        """Test exp(0) = 1."""
        a = create_value(0.0)
        b = run_exp(a)
        assert_close(get_value_data(b), 1.0)

    def test_exp_one(self):
        """Test exp(1) = e."""
        a = create_value(1.0)
        b = run_exp(a)
        assert_close(get_value_data(b), math.e)

    def test_exp_negative(self):
        """Test exp with negative input."""
        a = create_value(-1.0)
        b = run_exp(a)
        assert_close(get_value_data(b), 1.0 / math.e)

    def test_exp_two(self):
        """Test exp(2) = e^2."""
        a = create_value(2.0)
        b = run_exp(a)
        assert_close(get_value_data(b), math.e ** 2)

    def test_exp_backward(self):
        """Test exp gradient: d(e^x)/dx = e^x."""
        a = create_value(2.0)
        b = run_exp(a)
        run_backward(b)
        # d(exp(a))/da = exp(a)
        assert_close(get_value_grad(a), math.e ** 2)

    def test_exp_in_expression(self):
        """Test exp within a larger expression."""
        a = create_value(1.0)
        b = create_value(2.0)
        # c = exp(a) * b
        c = run_mul(run_exp(a), b)
        assert_close(get_value_data(c), math.e * 2)

        run_backward(c)
        # dc/da = b * exp(a) = 2e
        # dc/db = exp(a) = e
        assert_close(get_value_grad(a), 2 * math.e)
        assert_close(get_value_grad(b), math.e)


class TestLog:
    """Test natural logarithm (3 points)."""

    def test_log_one(self):
        """Test log(1) = 0."""
        a = create_value(1.0)
        b = run_log(a)
        assert_close(get_value_data(b), 0.0)

    def test_log_e(self):
        """Test log(e) = 1."""
        a = create_value(math.e)
        b = run_log(a)
        assert_close(get_value_data(b), 1.0)

    def test_log_positive(self):
        """Test log with positive input."""
        a = create_value(2.0)
        b = run_log(a)
        assert_close(get_value_data(b), math.log(2.0))

    def test_log_backward(self):
        """Test log gradient: d(ln(x))/dx = 1/x."""
        a = create_value(2.0)
        b = run_log(a)
        run_backward(b)
        # d(log(a))/da = 1/a = 0.5
        assert_close(get_value_grad(a), 0.5)

    def test_log_backward_other(self):
        """Test log gradient with different value."""
        a = create_value(4.0)
        b = run_log(a)
        run_backward(b)
        # d(log(a))/da = 1/a = 0.25
        assert_close(get_value_grad(a), 0.25)

    def test_log_in_expression(self):
        """Test log within a larger expression."""
        a = create_value(math.e)
        b = create_value(3.0)
        # c = log(a) * b = 1 * 3 = 3
        c = run_mul(run_log(a), b)
        assert_close(get_value_data(c), 3.0)

        run_backward(c)
        # dc/da = b * (1/a) = 3/e
        # dc/db = log(a) = 1
        assert_close(get_value_grad(a), 3.0 / math.e)
        assert_close(get_value_grad(b), 1.0)


class TestTanh:
    """Test hyperbolic tangent (3 points)."""

    def test_tanh_zero(self):
        """Test tanh(0) = 0."""
        a = create_value(0.0)
        b = run_tanh(a)
        assert_close(get_value_data(b), 0.0)

    def test_tanh_positive(self):
        """Test tanh with positive input."""
        a = create_value(1.0)
        b = run_tanh(a)
        assert_close(get_value_data(b), math.tanh(1.0))

    def test_tanh_negative(self):
        """Test tanh with negative input."""
        a = create_value(-1.0)
        b = run_tanh(a)
        assert_close(get_value_data(b), math.tanh(-1.0))

    def test_tanh_large(self):
        """Test tanh with large input (approaches 1)."""
        a = create_value(5.0)
        b = run_tanh(a)
        assert_close(get_value_data(b), math.tanh(5.0))

    def test_tanh_backward(self):
        """Test tanh gradient: d(tanh(x))/dx = 1 - tanh(x)^2."""
        a = create_value(1.0)
        b = run_tanh(a)
        run_backward(b)
        # d(tanh(a))/da = 1 - tanh(a)^2
        expected = 1.0 - math.tanh(1.0) ** 2
        assert_close(get_value_grad(a), expected)

    def test_tanh_backward_zero(self):
        """Test tanh gradient at zero."""
        a = create_value(0.0)
        b = run_tanh(a)
        run_backward(b)
        # d(tanh(0))/da = 1 - 0^2 = 1
        assert_close(get_value_grad(a), 1.0)

    def test_tanh_in_expression(self):
        """Test tanh within a larger expression."""
        a = create_value(0.5)
        b = create_value(2.0)
        # c = tanh(a) * b
        c = run_mul(run_tanh(a), b)
        assert_close(get_value_data(c), math.tanh(0.5) * 2.0)

        run_backward(c)
        # dc/da = b * (1 - tanh(a)^2)
        # dc/db = tanh(a)
        expected_grad_a = 2.0 * (1.0 - math.tanh(0.5) ** 2)
        assert_close(get_value_grad(a), expected_grad_a)
        assert_close(get_value_grad(b), math.tanh(0.5))


class TestComplexExpr:
    """Test complex expressions combining multiple operations (3 points)."""

    def test_complex_expr_1(self):
        """Test: y = relu(a * b + c^2)"""
        a = create_value(2.0)
        b = create_value(3.0)
        c = create_value(-1.0)
        # y = relu(a*b + c^2) = relu(6 + 1) = relu(7) = 7
        y = run_relu(run_add(run_mul(a, b), run_pow(c, 2)))
        assert get_value_data(y) == 7.0

        run_backward(y)
        # dy/da = b * 1 = 3 (relu is active)
        # dy/db = a * 1 = 2
        # dy/dc = 2c * 1 = -2
        assert_close(get_value_grad(a), 3.0)
        assert_close(get_value_grad(b), 2.0)
        assert_close(get_value_grad(c), -2.0)

    def test_complex_expr_2(self):
        """Test: y = exp(log(a)) = a"""
        a = create_value(3.0)
        y = run_exp(run_log(a))
        assert_close(get_value_data(y), 3.0)

        run_backward(y)
        # dy/da = exp(log(a)) * (1/a) = a * (1/a) = 1
        assert_close(get_value_grad(a), 1.0)

    def test_complex_expr_3(self):
        """Test: y = tanh(a * b) - exp(-c)"""
        a = create_value(0.5)
        b = create_value(2.0)
        c = create_value(1.0)

        # tanh(a*b) = tanh(1) ≈ 0.7616
        # exp(-c) = exp(-1) ≈ 0.3679
        # y ≈ 0.3937
        ab = run_mul(a, b)
        tanh_ab = run_tanh(ab)
        neg_c = run_mul(c, -1.0)
        exp_neg_c = run_exp(neg_c)
        y = run_sub(tanh_ab, exp_neg_c)

        expected = math.tanh(1.0) - math.exp(-1.0)
        assert_close(get_value_data(y), expected)

        run_backward(y)

        # dy/d(ab) = 1 - tanh(ab)^2
        # dy/da = dy/d(ab) * b
        # dy/db = dy/d(ab) * a
        # dy/dc = -(-exp(-c)) = exp(-c)
        dtanh = 1.0 - math.tanh(1.0) ** 2
        assert_close(get_value_grad(a), dtanh * 2.0)
        assert_close(get_value_grad(b), dtanh * 0.5)
        assert_close(get_value_grad(c), math.exp(-1.0))

    def test_complex_expr_4(self):
        """Test: y = log(exp(a) + exp(b)) - logsumexp-like"""
        a = create_value(1.0)
        b = create_value(2.0)

        # y = log(exp(a) + exp(b))
        exp_a = run_exp(a)
        exp_b = run_exp(b)
        sum_exp = run_add(exp_a, exp_b)
        y = run_log(sum_exp)

        expected = math.log(math.exp(1.0) + math.exp(2.0))
        assert_close(get_value_data(y), expected)

        run_backward(y)

        # dy/da = exp(a) / (exp(a) + exp(b)) = softmax(a)
        # dy/db = exp(b) / (exp(a) + exp(b)) = softmax(b)
        denom = math.exp(1.0) + math.exp(2.0)
        assert_close(get_value_grad(a), math.exp(1.0) / denom)
        assert_close(get_value_grad(b), math.exp(2.0) / denom)

    def test_complex_expr_5(self):
        """Test neural network like expression: relu(w*x + b)"""
        x = create_value(2.0)
        w = create_value(0.5)
        b = create_value(-0.5)

        # y = relu(w*x + b) = relu(1 - 0.5) = relu(0.5) = 0.5
        y = run_relu(run_add(run_mul(w, x), b))
        assert get_value_data(y) == 0.5

        run_backward(y)

        # Since relu output is positive, gradient passes through
        # dy/dw = x * 1 = 2
        # dy/dx = w * 1 = 0.5
        # dy/db = 1 * 1 = 1
        assert_close(get_value_grad(w), 2.0)
        assert_close(get_value_grad(x), 0.5)
        assert_close(get_value_grad(b), 1.0)

    def test_complex_expr_numerical_check(self):
        """Verify complex expression gradients numerically."""
        def f(a_val, b_val):
            a = create_value(a_val)
            b = create_value(b_val)
            # y = tanh(a) * exp(b)
            y = run_mul(run_tanh(a), run_exp(b))
            return get_value_data(y)

        a_val, b_val = 0.5, 0.3

        # Analytical
        a = create_value(a_val)
        b = create_value(b_val)
        y = run_mul(run_tanh(a), run_exp(b))
        run_backward(y)
        grad_a_ana = get_value_grad(a)
        grad_b_ana = get_value_grad(b)

        # Numerical
        eps = 1e-5
        grad_a_num = (f(a_val + eps, b_val) - f(a_val - eps, b_val)) / (2 * eps)
        grad_b_num = (f(a_val, b_val + eps) - f(a_val, b_val - eps)) / (2 * eps)

        assert_close(grad_a_ana, grad_a_num, rtol=1e-4)
        assert_close(grad_b_ana, grad_b_num, rtol=1e-4)
