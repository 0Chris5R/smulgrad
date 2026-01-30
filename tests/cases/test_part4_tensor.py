"""
Part 4: Tensor Support (20 points)

These tests verify your Tensor class implementation for multi-dimensional autodiff.
"""

import pytest
import numpy as np
from .conftest import assert_close, numerical_gradient
from tests.adapters import (
    create_tensor,
    get_tensor_data,
    get_tensor_grad,
    run_tensor_add,
    run_tensor_mul,
    run_tensor_sum,
    run_tensor_mean,
    run_tensor_max,
    run_tensor_reshape,
    run_tensor_transpose,
    run_tensor_backward,
)


class TestTensorCreation:
    """Test Tensor object creation (2 points)."""

    def test_tensor_creation_1d(self):
        """Test creating a 1D Tensor."""
        data = np.array([1.0, 2.0, 3.0])
        t = create_tensor(data)
        np.testing.assert_array_equal(get_tensor_data(t), data)
        np.testing.assert_array_equal(get_tensor_grad(t), np.zeros_like(data))

    def test_tensor_creation_2d(self):
        """Test creating a 2D Tensor."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = create_tensor(data)
        np.testing.assert_array_equal(get_tensor_data(t), data)
        assert get_tensor_grad(t).shape == data.shape

    def test_tensor_creation_3d(self):
        """Test creating a 3D Tensor."""
        data = np.random.randn(2, 3, 4)
        t = create_tensor(data)
        np.testing.assert_array_equal(get_tensor_data(t), data)
        assert get_tensor_grad(t).shape == data.shape

    def test_tensor_creation_scalar_like(self):
        """Test creating a scalar-like Tensor (0-d or 1 element)."""
        data = np.array(5.0)
        t = create_tensor(data)
        np.testing.assert_array_equal(get_tensor_data(t), data)


class TestTensorElementwise:
    """Test element-wise tensor operations (3 points)."""

    def test_tensor_add_same_shape(self):
        """Test adding two tensors of same shape."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = run_tensor_add(a, b)
        expected = np.array([[6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_tensor_add_backward(self):
        """Test addition backward pass."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = run_tensor_add(a, b)
        # Manually set output gradient (simulates gradient from upstream)
        c.grad = np.ones((2, 2))
        c._backward()

        # All gradients should be 1
        np.testing.assert_array_equal(get_tensor_grad(a), np.ones((2, 2)))
        np.testing.assert_array_equal(get_tensor_grad(b), np.ones((2, 2)))

    def test_tensor_mul_same_shape(self):
        """Test multiplying two tensors of same shape."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = run_tensor_mul(a, b)
        expected = np.array([[5.0, 12.0], [21.0, 32.0]])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_tensor_mul_backward(self):
        """Test multiplication backward pass."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = run_tensor_mul(a, b)
        # Manually set output gradient (simulates gradient from upstream)
        c.grad = np.ones((2, 2))
        c._backward()

        # grad_a = upstream_grad * b, grad_b = upstream_grad * a
        np.testing.assert_array_equal(get_tensor_grad(a), get_tensor_data(b))
        np.testing.assert_array_equal(get_tensor_grad(b), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_tensor_elementwise_chain(self):
        """Test chain of element-wise operations."""
        a = create_tensor(np.array([1.0, 2.0, 3.0]))
        b = create_tensor(np.array([2.0, 3.0, 4.0]))
        c = create_tensor(np.array([1.0, 1.0, 1.0]))
        # d = (a + b) * c
        d = run_tensor_mul(run_tensor_add(a, b), c)
        expected = np.array([3.0, 5.0, 7.0])
        np.testing.assert_array_equal(get_tensor_data(d), expected)

    def test_tensor_rsub_forward(self):
        """Test scalar - tensor (calls __rsub__)."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        c = 10.0 - a
        expected = np.array([[9.0, 8.0], [7.0, 6.0]])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_tensor_rsub_backward(self):
        """Test backward pass through tensor __rsub__."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        c = 10.0 - a
        s = run_tensor_sum(c)
        run_tensor_backward(s)
        # d(10 - a)/da = -1 for each element
        expected_grad = -np.ones((2, 2))
        np.testing.assert_array_equal(get_tensor_grad(a), expected_grad)

    def test_tensor_rtruediv_forward(self):
        """Test scalar / tensor (calls __rtruediv__)."""
        a = create_tensor(np.array([[1.0, 2.0], [4.0, 5.0]]))
        c = 20.0 / a
        expected = np.array([[20.0, 10.0], [5.0, 4.0]])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_tensor_rtruediv_backward(self):
        """Test backward pass through tensor __rtruediv__."""
        a_data = np.array([[1.0, 2.0], [4.0, 5.0]])

        def f(x):
            a = create_tensor(x.copy())
            c = 20.0 / a
            s = run_tensor_sum(c)
            return float(get_tensor_data(s))

        # Analytical
        a = create_tensor(a_data.copy())
        c = 20.0 / a
        s = run_tensor_sum(c)
        run_tensor_backward(s)
        grad_ana = get_tensor_grad(a).copy()

        # Numerical
        grad_num = numerical_gradient(f, a_data.copy())

        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-6)

    def test_tensor_neg_backward(self):
        """Test backward through tensor negation."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = -a
        run_tensor_backward(b)
        # d(-a)/da = -1 for each element
        expected = -np.ones((2, 2))
        np.testing.assert_array_equal(get_tensor_grad(a), expected)


class TestTensorBackward:
    """Test Tensor backward pass (2 points)."""

    def test_tensor_backward_simple(self):
        """Test backward on a simple tensor computation."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = run_tensor_add(a, b)
        run_tensor_backward(c)
        # For addition, gradients are all ones
        np.testing.assert_array_equal(get_tensor_grad(a), np.ones((2, 2)))
        np.testing.assert_array_equal(get_tensor_grad(b), np.ones((2, 2)))

    def test_tensor_backward_chain(self):
        """Test backward through a chain of operations."""
        a = create_tensor(np.array([1.0, 2.0, 3.0]))
        b = create_tensor(np.array([2.0, 3.0, 4.0]))
        c = run_tensor_mul(a, b)  # [2, 6, 12]
        d = run_tensor_add(c, a)  # [3, 8, 15]
        run_tensor_backward(d)
        # d = a*b + a, so dd/da = b + 1, dd/db = a
        np.testing.assert_array_equal(get_tensor_grad(a), np.array([3.0, 4.0, 5.0]))
        np.testing.assert_array_equal(get_tensor_grad(b), np.array([1.0, 2.0, 3.0]))

    def test_tensor_backward_grad_shape(self):
        """Test that backward initializes gradient with correct shape."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array([[1.0, 1.0], [1.0, 1.0]]))
        c = run_tensor_mul(a, b)
        run_tensor_backward(c)
        assert get_tensor_grad(c).shape == (2, 2)
        np.testing.assert_array_equal(get_tensor_grad(c), np.ones((2, 2)))

    def test_tensor_div_backward(self):
        """Test tensor / tensor backward pass."""
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[2.0, 4.0], [1.0, 2.0]])

        def f(a_np, b_np):
            a = create_tensor(a_np.copy())
            b = create_tensor(b_np.copy())
            c = a / b
            s = run_tensor_sum(c)
            return float(get_tensor_data(s))

        a = create_tensor(a_data.copy())
        b = create_tensor(b_data.copy())
        c = a / b
        s = run_tensor_sum(c)
        run_tensor_backward(s)
        grad_a_ana = get_tensor_grad(a).copy()
        grad_b_ana = get_tensor_grad(b).copy()

        grad_a_num = numerical_gradient(lambda x: f(x, b_data), a_data.copy())
        grad_b_num = numerical_gradient(lambda x: f(a_data, x), b_data.copy())

        np.testing.assert_allclose(grad_a_ana, grad_a_num, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(grad_b_ana, grad_b_num, rtol=1e-4, atol=1e-6)


class TestTensorBroadcast:
    """Test broadcasting in tensor operations (4 points)."""

    def test_broadcast_add_row_vector(self):
        """Test broadcasting a row vector."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # (2, 3)
        b = create_tensor(np.array([10.0, 20.0, 30.0]))  # (3,)
        c = run_tensor_add(a, b)
        expected = np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_broadcast_add_row_vector_backward(self):
        """Test backward pass with row vector broadcasting."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # (2, 3)
        b = create_tensor(np.array([10.0, 20.0, 30.0]))  # (3,)
        c = run_tensor_add(a, b)
        run_tensor_backward(c)

        # grad_a has shape (2, 3), all ones
        np.testing.assert_array_equal(get_tensor_grad(a), np.ones((2, 3)))
        # grad_b has shape (3,), summed over axis 0 -> [2, 2, 2]
        np.testing.assert_array_equal(get_tensor_grad(b), np.array([2.0, 2.0, 2.0]))

    def test_broadcast_add_column_vector(self):
        """Test broadcasting a column vector."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # (2, 3)
        b = create_tensor(np.array([[10.0], [20.0]]))  # (2, 1)
        c = run_tensor_add(a, b)
        expected = np.array([[11.0, 12.0, 13.0], [24.0, 25.0, 26.0]])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_broadcast_add_column_vector_backward(self):
        """Test backward pass with column vector broadcasting."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # (2, 3)
        b = create_tensor(np.array([[10.0], [20.0]]))  # (2, 1)
        c = run_tensor_add(a, b)
        run_tensor_backward(c)

        np.testing.assert_array_equal(get_tensor_grad(a), np.ones((2, 3)))
        # grad_b summed over axis 1 -> [[3], [3]]
        np.testing.assert_array_equal(get_tensor_grad(b), np.array([[3.0], [3.0]]))

    def test_broadcast_mul_backward(self):
        """Test multiplication backward with broadcasting."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))  # (2, 2)
        b = create_tensor(np.array([2.0, 3.0]))  # (2,)
        c = run_tensor_mul(a, b)
        run_tensor_backward(c)

        # grad_a = b broadcasted = [[2, 3], [2, 3]]
        np.testing.assert_array_equal(get_tensor_grad(a), np.array([[2.0, 3.0], [2.0, 3.0]]))
        # grad_b = sum of a over axis 0 = [4, 6]
        np.testing.assert_array_equal(get_tensor_grad(b), np.array([4.0, 6.0]))

    def test_broadcast_scalar(self):
        """Test broadcasting a scalar."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array(5.0))  # scalar
        c = run_tensor_add(a, b)
        expected = np.array([[6.0, 7.0], [8.0, 9.0]])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_broadcast_complex(self):
        """Test complex broadcasting pattern."""
        a = create_tensor(np.ones((2, 3, 4)))
        b = create_tensor(np.ones((3, 1)))
        c = run_tensor_add(a, b)
        assert get_tensor_data(c).shape == (2, 3, 4)

        run_tensor_backward(c)

        assert get_tensor_grad(a).shape == (2, 3, 4)
        assert get_tensor_grad(b).shape == (3, 1)
        # Each element of b contributes to 2*4=8 elements
        np.testing.assert_array_equal(get_tensor_grad(b), np.full((3, 1), 8.0))


class TestTensorSum:
    """Test sum reduction (3 points)."""

    def test_sum_all(self):
        """Test summing all elements."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_sum(a)
        assert_close(get_tensor_data(b), 10.0)

    def test_sum_axis0(self):
        """Test summing along axis 0."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_sum(a, axis=0)
        np.testing.assert_array_equal(get_tensor_data(b), np.array([4.0, 6.0]))

    def test_sum_axis1(self):
        """Test summing along axis 1."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_sum(a, axis=1)
        np.testing.assert_array_equal(get_tensor_data(b), np.array([3.0, 7.0]))

    def test_sum_keepdims(self):
        """Test sum with keepdims=True."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_sum(a, axis=1, keepdims=True)
        assert get_tensor_data(b).shape == (2, 1)
        np.testing.assert_array_equal(get_tensor_data(b), np.array([[3.0], [7.0]]))

    def test_sum_backward_all(self):
        """Test sum backward (all elements)."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_sum(a)
        run_tensor_backward(b)
        # Check both value and shape - gradient must be explicitly shaped
        assert get_tensor_grad(a).shape == (2, 2), "Gradient shape must match input shape"
        np.testing.assert_array_equal(get_tensor_grad(a), np.ones((2, 2)))

    def test_sum_backward_axis(self):
        """Test sum backward along axis."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = run_tensor_sum(a, axis=1)  # [6, 15]
        run_tensor_backward(b)
        # Check both value and shape
        assert get_tensor_grad(a).shape == (2, 3), "Gradient shape must match input shape"
        np.testing.assert_array_equal(get_tensor_grad(a), np.ones((2, 3)))

    def test_sum_backward_all_through_chain(self):
        """Test sum backward flows correctly through operations."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array([[2.0, 1.0], [1.0, 2.0]]))
        c = run_tensor_mul(a, b)  # [[2, 2], [3, 8]]
        s = run_tensor_sum(c)     # scalar sum
        run_tensor_backward(s)
        # grad_a = b (element-wise), grad_b = a
        np.testing.assert_array_equal(get_tensor_grad(a), get_tensor_data(b))
        np.testing.assert_array_equal(get_tensor_grad(b), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_sum_keepdims_backward(self):
        """Test sum backward with keepdims=True."""
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        def f(x):
            a = create_tensor(x.copy())
            b = run_tensor_sum(a, axis=1, keepdims=True)
            s = run_tensor_sum(b)
            return float(get_tensor_data(s))

        a = create_tensor(a_data.copy())
        b = run_tensor_sum(a, axis=1, keepdims=True)
        s = run_tensor_sum(b)
        run_tensor_backward(s)
        grad_ana = get_tensor_grad(a).copy()

        grad_num = numerical_gradient(f, a_data.copy())
        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-6)


class TestTensorMean:
    """Test mean reduction (2 points)."""

    def test_mean_all(self):
        """Test mean of all elements."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_mean(a)
        assert_close(get_tensor_data(b), 2.5)

    def test_mean_axis(self):
        """Test mean along axis."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_mean(a, axis=0)
        np.testing.assert_array_equal(get_tensor_data(b), np.array([2.0, 3.0]))

    def test_mean_backward_all(self):
        """Test mean backward (all elements)."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_mean(a)
        run_tensor_backward(b)
        # Check shape and value - gradient is 1/n for each element
        assert get_tensor_grad(a).shape == (2, 2), "Gradient shape must match input shape"
        np.testing.assert_array_equal(get_tensor_grad(a), np.full((2, 2), 0.25))

    def test_mean_backward_axis(self):
        """Test mean backward along axis (direct, no sum)."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = run_tensor_mean(a, axis=1)  # [2, 5]
        run_tensor_backward(b)
        # Check shape - gradient must be broadcast back to original shape
        assert get_tensor_grad(a).shape == (2, 3), "Gradient shape must match input shape"
        # Each element contributes 1/3 to its row's mean
        np.testing.assert_allclose(get_tensor_grad(a), np.full((2, 3), 1.0/3.0))

    def test_mean_keepdims_backward(self):
        """Test mean backward with keepdims=True."""
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        def f(x):
            a = create_tensor(x.copy())
            b = run_tensor_mean(a, axis=1, keepdims=True)
            s = run_tensor_sum(b)
            return float(get_tensor_data(s))

        a = create_tensor(a_data.copy())
        b = run_tensor_mean(a, axis=1, keepdims=True)
        s = run_tensor_sum(b)
        run_tensor_backward(s)
        grad_ana = get_tensor_grad(a).copy()

        grad_num = numerical_gradient(f, a_data.copy())
        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-6)


class TestTensorMax:
    """Test max reduction (3 points)."""

    def test_max_all(self):
        """Test max of all elements."""
        a = create_tensor(np.array([[1.0, 4.0], [3.0, 2.0]]))
        b = run_tensor_max(a)
        assert_close(get_tensor_data(b), 4.0)

    def test_max_axis(self):
        """Test max along axis."""
        a = create_tensor(np.array([[1.0, 4.0], [3.0, 2.0]]))
        b = run_tensor_max(a, axis=0)
        np.testing.assert_array_equal(get_tensor_data(b), np.array([3.0, 4.0]))

    def test_max_backward_all(self):
        """Test max backward (all elements)."""
        a = create_tensor(np.array([[1.0, 4.0], [3.0, 2.0]]))
        b = run_tensor_max(a)
        run_tensor_backward(b)
        # Check shape and value - only max element gets gradient
        assert get_tensor_grad(a).shape == (2, 2), "Gradient shape must match input shape"
        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_array_equal(get_tensor_grad(a), expected)

    def test_max_backward_axis(self):
        """Test max backward along axis (direct, no sum)."""
        a = create_tensor(np.array([[1.0, 4.0], [3.0, 2.0]]))
        b = run_tensor_max(a, axis=0)  # [3, 4]
        run_tensor_backward(b)
        # Check shape - gradient must be broadcast back to original shape
        assert get_tensor_grad(a).shape == (2, 2), "Gradient shape must match input shape"
        # Max of column 0 is at row 1, max of column 1 is at row 0
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(get_tensor_grad(a), expected)

    def test_max_backward_ties(self):
        """Test max backward with ties (multiple max values)."""
        a = create_tensor(np.array([[4.0, 4.0], [3.0, 2.0]]))
        b = run_tensor_max(a)
        run_tensor_backward(b)
        # Both 4s are max, gradient should be split
        expected = np.array([[0.5, 0.5], [0.0, 0.0]])
        np.testing.assert_array_equal(get_tensor_grad(a), expected)

    def test_max_keepdims_backward(self):
        """Test max backward with keepdims=True."""
        a_data = np.array([[1.0, 4.0, 2.0], [3.0, 2.0, 5.0]])

        def f(x):
            a = create_tensor(x.copy())
            b = run_tensor_max(a, axis=1, keepdims=True)
            s = run_tensor_sum(b)
            return float(get_tensor_data(s))

        a = create_tensor(a_data.copy())
        b = run_tensor_max(a, axis=1, keepdims=True)
        s = run_tensor_sum(b)
        run_tensor_backward(s)
        grad_ana = get_tensor_grad(a).copy()

        grad_num = numerical_gradient(f, a_data.copy())
        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-6)


class TestTensorReshape:
    """Test reshape operation (2 points)."""

    def test_reshape_flatten(self):
        """Test reshaping to 1D."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_reshape(a, (4,))
        np.testing.assert_array_equal(get_tensor_data(b), np.array([1.0, 2.0, 3.0, 4.0]))

    def test_reshape_2d(self):
        """Test reshaping to different 2D shape."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
        b = run_tensor_reshape(a, (2, 2))
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(get_tensor_data(b), expected)

    def test_reshape_backward(self):
        """Test reshape backward pass."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = run_tensor_reshape(a, (4,))
        c = run_tensor_sum(b)
        run_tensor_backward(c)
        # Gradient should be reshaped back to original shape
        np.testing.assert_array_equal(get_tensor_grad(a), np.ones((2, 2)))

    def test_reshape_with_neg_one(self):
        """Test reshape with -1 dimension."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = run_tensor_reshape(a, (-1,))
        assert get_tensor_data(b).shape == (6,)

    def test_reshape_backward_in_chain(self):
        """Test reshape backward in complex computation chain."""
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([1.0, 2.0, 3.0, 4.0])

        def f(x, y):
            a = create_tensor(x.copy())
            b = create_tensor(y.copy())
            # Reshape a to 1D, multiply with b, sum
            a_flat = run_tensor_reshape(a, (4,))
            c = run_tensor_mul(a_flat, b)
            s = run_tensor_sum(c)
            return float(get_tensor_data(s))

        a = create_tensor(a_data.copy())
        b = create_tensor(b_data.copy())
        a_flat = run_tensor_reshape(a, (4,))
        c = run_tensor_mul(a_flat, b)
        s = run_tensor_sum(c)
        run_tensor_backward(s)
        grad_a_ana = get_tensor_grad(a).copy()

        grad_a_num = numerical_gradient(lambda x: f(x, b_data), a_data.copy())
        np.testing.assert_allclose(grad_a_ana, grad_a_num, rtol=1e-4, atol=1e-6)

    def test_reshape_gradient_accumulation(self):
        """Test reshape backward accumulates gradients when tensor used in multiple paths."""
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])

        def f(x):
            a = create_tensor(x.copy())
            # Use original tensor in TWO different reshape paths
            a_flat1 = run_tensor_reshape(a, (4,))
            s1 = run_tensor_sum(a_flat1)
            a_flat2 = run_tensor_reshape(a, (1, 4))
            s2 = run_tensor_sum(a_flat2)
            total = run_tensor_add(s1, s2)
            return float(get_tensor_data(total))

        a = create_tensor(a_data.copy())
        a_flat1 = run_tensor_reshape(a, (4,))
        s1 = run_tensor_sum(a_flat1)
        a_flat2 = run_tensor_reshape(a, (1, 4))
        s2 = run_tensor_sum(a_flat2)
        total = run_tensor_add(s1, s2)
        run_tensor_backward(total)
        grad_ana = get_tensor_grad(a).copy()

        grad_num = numerical_gradient(f, a_data.copy())
        # Gradient should be 2 for each element (used in two paths)
        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-6)


class TestTensorTranspose:
    """Test transpose operation (1 point)."""

    def test_transpose_2d(self):
        """Test 2D transpose."""
        a = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = run_tensor_transpose(a)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(get_tensor_data(b), expected)

    def test_transpose_backward(self):
        """Test transpose backward pass."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # (3, 2)
        b = run_tensor_transpose(a)  # (2, 3)
        c = run_tensor_sum(b)
        run_tensor_backward(c)
        # Gradient should be transposed back to original shape
        np.testing.assert_array_equal(get_tensor_grad(a), np.ones((3, 2)))

    def test_transpose_gradient_accumulation(self):
        """Test transpose backward accumulates gradients when tensor used in multiple paths."""
        a_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def f(x):
            a = create_tensor(x.copy())
            # Use original tensor in TWO different transpose paths
            a_t1 = run_tensor_transpose(a)
            s1 = run_tensor_sum(a_t1)
            a_t2 = run_tensor_transpose(a)
            s2 = run_tensor_sum(a_t2)
            total = run_tensor_add(s1, s2)
            return float(get_tensor_data(total))

        a = create_tensor(a_data.copy())
        a_t1 = run_tensor_transpose(a)
        s1 = run_tensor_sum(a_t1)
        a_t2 = run_tensor_transpose(a)
        s2 = run_tensor_sum(a_t2)
        total = run_tensor_add(s1, s2)
        run_tensor_backward(total)
        grad_ana = get_tensor_grad(a).copy()

        grad_num = numerical_gradient(f, a_data.copy())
        # Gradient should be 2 for each element (used in two paths)
        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-6)


class TestTensorNumericalGradient:
    """Verify tensor gradients numerically."""

    def test_numerical_gradient_add_broadcast(self):
        """Verify gradient of broadcasted addition numerically."""
        np.random.seed(42)
        a_data = np.random.randn(2, 3)
        b_data = np.random.randn(3)

        def f(a_np, b_np):
            a = create_tensor(a_np.copy())
            b = create_tensor(b_np.copy())
            c = run_tensor_add(a, b)
            s = run_tensor_sum(c)
            return float(get_tensor_data(s))

        # Analytical gradient
        a = create_tensor(a_data.copy())
        b = create_tensor(b_data.copy())
        c = run_tensor_add(a, b)
        s = run_tensor_sum(c)
        run_tensor_backward(s)
        grad_a_ana = get_tensor_grad(a).copy()
        grad_b_ana = get_tensor_grad(b).copy()

        # Numerical gradient
        grad_a_num = numerical_gradient(lambda x: f(x, b_data), a_data.copy())
        grad_b_num = numerical_gradient(lambda x: f(a_data, x), b_data.copy())

        np.testing.assert_allclose(grad_a_ana, grad_a_num, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(grad_b_ana, grad_b_num, rtol=1e-4, atol=1e-6)

    def test_numerical_gradient_mul_broadcast(self):
        """Verify gradient of broadcasted multiplication numerically."""
        np.random.seed(42)
        a_data = np.random.randn(2, 3) + 2  # Avoid near-zero values
        b_data = np.random.randn(3) + 2

        def f(a_np, b_np):
            a = create_tensor(a_np.copy())
            b = create_tensor(b_np.copy())
            c = run_tensor_mul(a, b)
            s = run_tensor_sum(c)
            return float(get_tensor_data(s))

        # Analytical gradient
        a = create_tensor(a_data.copy())
        b = create_tensor(b_data.copy())
        c = run_tensor_mul(a, b)
        s = run_tensor_sum(c)
        run_tensor_backward(s)
        grad_a_ana = get_tensor_grad(a).copy()
        grad_b_ana = get_tensor_grad(b).copy()

        # Numerical gradient
        grad_a_num = numerical_gradient(lambda x: f(x, b_data), a_data.copy())
        grad_b_num = numerical_gradient(lambda x: f(a_data, x), b_data.copy())

        np.testing.assert_allclose(grad_a_ana, grad_a_num, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(grad_b_ana, grad_b_num, rtol=1e-4, atol=1e-5)


