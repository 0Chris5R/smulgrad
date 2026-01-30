"""
Part 2: The Backward Pass (20 points)

These tests verify your backpropagation implementation.
"""

import pytest
import numpy as np
from .conftest import assert_close, numerical_gradient
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
    run_backward,
)


class TestBackwardAdd:
    """Test _backward for addition (3 points).

    These tests only test the local _backward function, not the full backward() method.
    We manually set out.grad and call out._backward() directly.
    """

    def test_backward_add_simple(self):
        """Test _backward of c = a + b."""
        a = create_value(2.0)
        b = create_value(3.0)
        c = run_add(a, b)

        # Manually set output gradient and call _backward
        c.grad = 1.0
        c._backward()

        # dc/da = 1, dc/db = 1
        assert_close(get_value_grad(a), 1.0)
        assert_close(get_value_grad(b), 1.0)

    def test_backward_add_with_scalar(self):
        """Test _backward of c = a + 5."""
        a = create_value(2.0)
        c = run_add(a, 5.0)

        c.grad = 1.0
        c._backward()

        # dc/da = 1
        assert_close(get_value_grad(a), 1.0)

    def test_backward_add_with_upstream_grad(self):
        """Test _backward with non-unit upstream gradient."""
        a = create_value(2.0)
        b = create_value(3.0)
        c = run_add(a, b)

        # Simulate upstream gradient of 5.0
        c.grad = 5.0
        c._backward()

        # dc/da = 1, so a.grad = 5.0 * 1 = 5.0
        assert_close(get_value_grad(a), 5.0)
        assert_close(get_value_grad(b), 5.0)


class TestBackwardMul:
    """Test _backward for multiplication (3 points).

    These tests only test the local _backward function, not the full backward() method.
    """

    def test_backward_mul_simple(self):
        """Test _backward of c = a * b."""
        a = create_value(2.0)
        b = create_value(3.0)
        c = run_mul(a, b)

        c.grad = 1.0
        c._backward()

        # dc/da = b = 3, dc/db = a = 2
        assert_close(get_value_grad(a), 3.0)
        assert_close(get_value_grad(b), 2.0)

    def test_backward_mul_with_scalar(self):
        """Test _backward of c = a * 5."""
        a = create_value(2.0)
        c = run_mul(a, 5.0)

        c.grad = 1.0
        c._backward()

        # dc/da = 5
        assert_close(get_value_grad(a), 5.0)

    def test_backward_mul_with_upstream_grad(self):
        """Test _backward with non-unit upstream gradient."""
        a = create_value(2.0)
        b = create_value(3.0)
        c = run_mul(a, b)

        c.grad = 2.0
        c._backward()

        # dc/da = b = 3, so a.grad = 2.0 * 3 = 6.0
        # dc/db = a = 2, so b.grad = 2.0 * 2 = 4.0
        assert_close(get_value_grad(a), 6.0)
        assert_close(get_value_grad(b), 4.0)


class TestBackwardOps:
    """Test _backward for other operations (4 points).

    These tests only test the local _backward function, not the full backward() method.
    """

    def test_backward_neg(self):
        """Test _backward of b = -a."""
        a = create_value(3.0)
        b = run_neg(a)

        b.grad = 1.0
        b._backward()

        # db/da = -1
        assert_close(get_value_grad(a), -1.0)

    def test_backward_sub(self):
        """Test _backward of c = a - b.

        Note: If sub is implemented as a + (-b), we need to propagate
        through the intermediate negation node as well.
        """
        a = create_value(5.0)
        b = create_value(3.0)
        c = run_sub(a, b)

        c.grad = 1.0
        c._backward()

        # If sub is composed (a + (-b)), propagate through intermediate nodes
        for node in c._prev:
            if node._prev:  # has children = intermediate node
                node._backward()

        # dc/da = 1, dc/db = -1
        assert_close(get_value_grad(a), 1.0)
        assert_close(get_value_grad(b), -1.0)

    def test_backward_pow_square(self):
        """Test _backward of b = a^2."""
        a = create_value(3.0)
        b = run_pow(a, 2)

        b.grad = 1.0
        b._backward()

        # db/da = 2*a = 6
        assert_close(get_value_grad(a), 6.0)

    def test_backward_pow_cube(self):
        """Test _backward of b = a^3."""
        a = create_value(2.0)
        b = run_pow(a, 3)

        b.grad = 1.0
        b._backward()

        # db/da = 3*a^2 = 12
        assert_close(get_value_grad(a), 12.0)

    def test_backward_pow_negative(self):
        """Test _backward of b = a^(-1)."""
        a = create_value(2.0)
        b = run_pow(a, -1)

        b.grad = 1.0
        b._backward()

        # db/da = -1 * a^(-2) = -0.25
        assert_close(get_value_grad(a), -0.25)

    def test_backward_pow_fractional(self):
        """Test _backward of b = a^0.5 (sqrt)."""
        a = create_value(4.0)
        b = run_pow(a, 0.5)

        b.grad = 1.0
        b._backward()

        # db/da = 0.5 * a^(-0.5) = 0.25
        assert_close(get_value_grad(a), 0.25)

    def test_backward_div(self):
        """Test _backward of c = a / b.

        Note: If div is implemented as a * (b ** -1), we need to propagate
        through the intermediate power node as well.
        """
        a = create_value(6.0)
        b = create_value(2.0)
        c = run_div(a, b)

        c.grad = 1.0
        c._backward()

        # If div is composed (a * b**-1), propagate through intermediate nodes
        for node in c._prev:
            if node._prev:  # has children = intermediate node
                node._backward()

        # dc/da = 1/b = 0.5
        # dc/db = -a/b^2 = -1.5
        assert_close(get_value_grad(a), 0.5)
        assert_close(get_value_grad(b), -1.5)

    def test_backward_rsub(self):
        """Test _backward of c = scalar - a (calls __rsub__)."""
        a = create_value(3.0)
        c = 5.0 - a  # = 2.0, calls __rsub__

        c.grad = 1.0
        c._backward()

        # If rsub is composed, propagate through intermediate nodes
        for node in c._prev:
            if node._prev:
                node._backward()

        # d(5 - a)/da = -1
        assert_close(get_value_grad(a), -1.0)

    def test_backward_rtruediv(self):
        """Test _backward of c = scalar / a (calls __rtruediv__)."""
        a = create_value(2.0)
        c = 6.0 / a  # = 3.0, calls __rtruediv__

        c.grad = 1.0
        c._backward()

        # If rtruediv is composed, propagate through intermediate nodes
        for node in c._prev:
            if node._prev:
                node._backward()
                for n2 in node._prev:
                    if n2._prev:
                        n2._backward()

        # d(6/a)/da = -6/a^2 = -6/4 = -1.5
        assert_close(get_value_grad(a), -1.5)


class TestBackwardFull:
    """Test full backward pass with complex graphs (5 points)."""

    def test_backward_full_add_mul(self):
        """Test gradient of d = a * b + a."""
        a = create_value(2.0)
        b = create_value(3.0)
        c = run_mul(a, b)
        d = run_add(c, a)
        run_backward(d)

        # d = a*b + a
        # dd/da = b + 1 = 4
        # dd/db = a = 2
        assert_close(get_value_grad(a), 4.0)
        assert_close(get_value_grad(b), 2.0)

    def test_backward_full_complex_1(self):
        """Test gradient of y = x1 + x2*x3*x1."""
        x1 = create_value(1.0)
        x2 = create_value(2.0)
        x3 = create_value(3.0)
        y = run_add(x1, run_mul(run_mul(x2, x3), x1))
        run_backward(y)

        # y = x1 + x2*x3*x1
        # dy/dx1 = 1 + x2*x3 = 1 + 6 = 7
        # dy/dx2 = x3*x1 = 3
        # dy/dx3 = x2*x1 = 2
        assert_close(get_value_grad(x1), 7.0)
        assert_close(get_value_grad(x2), 3.0)
        assert_close(get_value_grad(x3), 2.0)

    def test_backward_full_complex_2(self):
        """Test gradient of y = x1 + x2*x3*x4."""
        x1 = create_value(1.0)
        x2 = create_value(2.0)
        x3 = create_value(3.0)
        x4 = create_value(4.0)
        y = run_add(x1, run_mul(run_mul(x2, x3), x4))
        run_backward(y)

        # y = x1 + x2*x3*x4
        # dy/dx1 = 1
        # dy/dx2 = x3*x4 = 12
        # dy/dx3 = x2*x4 = 8
        # dy/dx4 = x2*x3 = 6
        assert_close(get_value_grad(x1), 1.0)
        assert_close(get_value_grad(x2), 12.0)
        assert_close(get_value_grad(x3), 8.0)
        assert_close(get_value_grad(x4), 6.0)

    def test_backward_full_complex_3(self):
        """Test gradient of y = z*z + x3 where z = x2*x2 + x2 + x3 + 3."""
        x2 = create_value(2.0)
        x3 = create_value(3.0)

        # z = x2*x2 + x2 + x3 + 3 = 4 + 2 + 3 + 3 = 12
        z = run_add(run_add(run_add(run_mul(x2, x2), x2), x3), 3.0)

        # y = z*z + x3 = 144 + 3 = 147
        y = run_add(run_mul(z, z), x3)
        run_backward(y)

        z_val = 2.0 * 2.0 + 2.0 + 3.0 + 3.0  # 12
        # dy/dx2 = dy/dz * dz/dx2 = 2*z * (2*x2 + 1) = 24 * 5 = 120
        # dy/dx3 = dy/dz * dz/dx3 + 1 = 24 * 1 + 1 = 25
        expected_grad_x2 = 2 * z_val * (2 * 2.0 + 1)
        expected_grad_x3 = 2 * z_val + 1

        assert_close(get_value_grad(x2), expected_grad_x2)
        assert_close(get_value_grad(x3), expected_grad_x3)

    def test_backward_with_pow_and_div(self):
        """Test gradient of complex expression with pow and div."""
        a = create_value(2.0)
        b = create_value(4.0)
        # y = a^2 / b + b
        y = run_add(run_div(run_pow(a, 2), b), b)
        run_backward(y)

        # y = a^2/b + b
        # dy/da = 2a/b = 1
        # dy/db = -a^2/b^2 + 1 = -4/16 + 1 = 0.75
        assert_close(get_value_grad(a), 1.0)
        assert_close(get_value_grad(b), 0.75)


class TestGradAccumulation:
    """Test gradient accumulation (3 points)."""

    def test_grad_accumulation_add_same(self):
        """Test gradient when same value is added to itself."""
        a = create_value(3.0)
        b = run_add(a, a)
        run_backward(b)

        # b = a + a = 2a
        # db/da = 2
        assert_close(get_value_grad(a), 2.0)

    def test_grad_accumulation_mul_same(self):
        """Test gradient when value is multiplied by itself."""
        a = create_value(3.0)
        b = run_mul(a, a)
        run_backward(b)

        # b = a * a = a^2
        # db/da = 2a = 6
        assert_close(get_value_grad(a), 6.0)

    def test_grad_accumulation_complex(self):
        """Test gradient with multiple uses of same variable."""
        a = create_value(2.0)
        b = create_value(3.0)
        # y = a*b + a*a + a
        y = run_add(run_add(run_mul(a, b), run_mul(a, a)), a)
        run_backward(y)

        # y = a*b + a^2 + a
        # dy/da = b + 2a + 1 = 3 + 4 + 1 = 8
        # dy/db = a = 2
        assert_close(get_value_grad(a), 8.0)
        assert_close(get_value_grad(b), 2.0)

    def test_grad_accumulation_three_paths(self):
        """Test gradient with three computational paths."""
        x = create_value(2.0)
        # y = x + x + x
        y = run_add(run_add(x, x), x)
        run_backward(y)

        # dy/dx = 3
        assert_close(get_value_grad(x), 3.0)

    def test_grad_accumulation_mixed(self):
        """Test gradient with mixed operations on same variable."""
        x = create_value(2.0)
        # y = x * x * x (x^3)
        y = run_mul(run_mul(x, x), x)
        run_backward(y)

        # y = x^3
        # dy/dx = 3x^2 = 12
        assert_close(get_value_grad(x), 12.0)


class TestGradOfGrad:
    """Test second derivatives / gradient of gradient (2 points - bonus)."""

    def test_grad_of_grad_simple(self):
        """Test second derivative of x^2."""
        x2 = create_value(2.0)
        x3 = create_value(3.0)

        # y = x2^2 + x2*x3
        y = run_add(run_mul(x2, x2), run_mul(x2, x3))
        run_backward(y)

        # dy/dx2 = 2*x2 + x3 = 4 + 3 = 7
        # dy/dx3 = x2 = 2
        assert_close(get_value_grad(x2), 7.0)
        assert_close(get_value_grad(x3), 2.0)

        # Note: For full second derivative support, we'd need to be able to
        # differentiate the gradient expressions themselves. This is more
        # advanced and typically requires careful implementation.
        # This test just verifies first-order gradients are correct.


class TestNumericalGradientCheck:
    """Verify gradients against numerical approximations."""

    def test_numerical_check_mul(self):
        """Verify multiplication gradient numerically."""
        def f(x_val):
            x = create_value(x_val)
            y = run_mul(x, x)  # x^2
            return get_value_data(y)

        x_val = 3.0
        x = create_value(x_val)
        y = run_mul(x, x)
        run_backward(y)

        analytical = get_value_grad(x)
        numerical = (f(x_val + 1e-5) - f(x_val - 1e-5)) / (2e-5)

        assert_close(analytical, numerical, rtol=1e-4)

    def test_numerical_check_complex(self):
        """Verify complex expression gradient numerically."""
        def f(a_val, b_val):
            a = create_value(a_val)
            b = create_value(b_val)
            # y = (a + b) * (a - b) = a^2 - b^2
            y = run_mul(run_add(a, b), run_sub(a, b))
            return get_value_data(y)

        a_val, b_val = 3.0, 2.0

        # Analytical gradients
        a = create_value(a_val)
        b = create_value(b_val)
        y = run_mul(run_add(a, b), run_sub(a, b))
        run_backward(y)
        grad_a_analytical = get_value_grad(a)
        grad_b_analytical = get_value_grad(b)

        # Numerical gradients
        eps = 1e-5
        grad_a_numerical = (f(a_val + eps, b_val) - f(a_val - eps, b_val)) / (2 * eps)
        grad_b_numerical = (f(a_val, b_val + eps) - f(a_val, b_val - eps)) / (2 * eps)

        assert_close(grad_a_analytical, grad_a_numerical, rtol=1e-4)
        assert_close(grad_b_analytical, grad_b_numerical, rtol=1e-4)
