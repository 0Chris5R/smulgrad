"""
Part 5: Matrix Operations (15 points)

These tests verify matrix multiplication and related operations.
"""

import pytest
import numpy as np
from .conftest import assert_close, numerical_gradient
from tests.adapters import (
    create_tensor,
    get_tensor_data,
    get_tensor_grad,
    run_tensor_sum,
    run_tensor_backward,
    run_matmul,
    run_tensor_add,
    run_tensor_relu,
    run_tensor_exp,
    run_tensor_log,
)


class TestMatmul:
    """Test matrix multiplication (5 points)."""

    def test_matmul_2d_basic(self):
        """Test basic 2D matrix multiplication."""
        a = create_tensor(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # (3, 2)
        b = create_tensor(
            np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))  # (2, 3)
        c = run_matmul(a, b)

        expected = np.array([
            [27.0, 30.0, 33.0],
            [61.0, 68.0, 75.0],
            [95.0, 106.0, 117.0]
        ])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_matmul_backward_left(self):
        """Test matmul backward for left operand."""
        a = create_tensor(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # (3, 2)
        b = create_tensor(
            np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))  # (2, 3)
        c = run_matmul(a, b)
        s = run_tensor_sum(c)
        run_tensor_backward(s)

        # dL/dA = dL/dC @ B.T
        # dL/dC is all ones (3, 3)
        # B.T is (3, 2)
        # Expected: ones(3,3) @ B.T = sum of B along axis 1
        expected_grad_a = np.ones((3, 3)) @ get_tensor_data(b).T
        np.testing.assert_array_equal(get_tensor_grad(a), expected_grad_a)

    def test_matmul_backward_right(self):
        """Test matmul backward for right operand."""
        a = create_tensor(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # (3, 2)
        b = create_tensor(
            np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))  # (2, 3)
        c = run_matmul(a, b)
        s = run_tensor_sum(c)
        run_tensor_backward(s)

        # dL/dB = A.T @ dL/dC
        # A.T is (2, 3)
        # dL/dC is all ones (3, 3)
        a_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected_grad_b = a_data.T @ np.ones((3, 3))
        np.testing.assert_array_equal(get_tensor_grad(b), expected_grad_b)

    def test_matmul_square(self):
        """Test square matrix multiplication."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = create_tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = run_matmul(a, b)

        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_equal(get_tensor_data(c), expected)

    def test_matmul_numerical_gradient(self):
        """Verify matmul gradient numerically."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4)
        b_data = np.random.randn(4, 2)

        def f(a_np, b_np):
            a = create_tensor(a_np.copy())
            b = create_tensor(b_np.copy())
            c = run_matmul(a, b)
            s = run_tensor_sum(c)
            return float(get_tensor_data(s))

        # Analytical gradient
        a = create_tensor(a_data.copy())
        b = create_tensor(b_data.copy())
        c = run_matmul(a, b)
        s = run_tensor_sum(c)
        run_tensor_backward(s)
        grad_a_ana = get_tensor_grad(a).copy()
        grad_b_ana = get_tensor_grad(b).copy()

        # Numerical gradient
        grad_a_num = numerical_gradient(lambda x: f(x, b_data), a_data.copy())
        grad_b_num = numerical_gradient(lambda x: f(a_data, x), b_data.copy())

        np.testing.assert_allclose(
            grad_a_ana, grad_a_num, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            grad_b_ana, grad_b_num, rtol=1e-4, atol=1e-6)


class TestBatchedMatmul:
    """Test batched matrix multiplication (4 points)."""

    def test_batched_matmul_3d(self):
        """Test 3D batched matmul."""
        np.random.seed(42)
        a = create_tensor(np.random.randn(2, 3, 4))  # batch=2, (3, 4)
        b = create_tensor(np.random.randn(2, 4, 5))  # batch=2, (4, 5)
        c = run_matmul(a, b)

        assert get_tensor_data(c).shape == (2, 3, 5)

        # Verify against numpy
        a_data = get_tensor_data(create_tensor(np.random.randn(2, 3, 4)))
        np.random.seed(42)
        a_data = np.random.randn(2, 3, 4)
        b_data = np.random.randn(2, 4, 5)
        a = create_tensor(a_data)
        b = create_tensor(b_data)
        c = run_matmul(a, b)
        expected = a_data @ b_data
        np.testing.assert_allclose(get_tensor_data(c), expected)

    def test_batched_matmul_backward(self):
        """Test batched matmul backward pass."""
        np.random.seed(42)
        a_data = np.random.randn(2, 3, 4)
        b_data = np.random.randn(2, 4, 5)

        def f(a_np, b_np):
            a = create_tensor(a_np.copy())
            b = create_tensor(b_np.copy())
            c = run_matmul(a, b)
            s = run_tensor_sum(c)
            return float(get_tensor_data(s))

        # Analytical gradient
        a = create_tensor(a_data.copy())
        b = create_tensor(b_data.copy())
        c = run_matmul(a, b)
        s = run_tensor_sum(c)
        run_tensor_backward(s)
        grad_a_ana = get_tensor_grad(a).copy()
        grad_b_ana = get_tensor_grad(b).copy()

        # Numerical gradient
        grad_a_num = numerical_gradient(lambda x: f(x, b_data), a_data.copy())
        grad_b_num = numerical_gradient(lambda x: f(a_data, x), b_data.copy())

        np.testing.assert_allclose(
            grad_a_ana, grad_a_num, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            grad_b_ana, grad_b_num, rtol=1e-4, atol=1e-5)

    def test_batched_matmul_broadcast(self):
        """Test batched matmul with broadcasting."""
        np.random.seed(42)
        # a has batch dimension, b doesn't
        a_data = np.random.randn(2, 3, 4)
        b_data = np.random.randn(4, 5)

        a = create_tensor(a_data)
        b = create_tensor(b_data)
        c = run_matmul(a, b)

        # Should broadcast b across batch dimension
        expected = a_data @ b_data
        assert get_tensor_data(c).shape == (2, 3, 5)
        np.testing.assert_allclose(get_tensor_data(c), expected)


class TestMatvec:
    """Test matrix-vector products (3 points)."""

    def test_matvec_basic(self):
        """Test matrix-vector multiplication."""
        a = create_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))  # (2, 2)
        x = create_tensor(np.array([1.0, 2.0]))  # (2,)
        y = run_matmul(a, x)

        expected = np.array([5.0, 11.0])
        np.testing.assert_array_equal(get_tensor_data(y), expected)

    def test_matvec_backward(self):
        """Test matrix-vector multiplication backward."""
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_data = np.array([1.0, 2.0])

        def f(a_np, x_np):
            a = create_tensor(a_np.copy())
            x = create_tensor(x_np.copy())
            y = run_matmul(a, x)
            s = run_tensor_sum(y)
            return float(get_tensor_data(s))

        # Analytical gradient
        a = create_tensor(a_data.copy())
        x = create_tensor(x_data.copy())
        y = run_matmul(a, x)
        s = run_tensor_sum(y)
        run_tensor_backward(s)
        grad_a_ana = get_tensor_grad(a).copy()
        grad_x_ana = get_tensor_grad(x).copy()

        # Numerical gradient
        grad_a_num = numerical_gradient(lambda m: f(m, x_data), a_data.copy())
        grad_x_num = numerical_gradient(lambda v: f(a_data, v), x_data.copy())

        np.testing.assert_allclose(
            grad_a_ana, grad_a_num, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            grad_x_ana, grad_x_num, rtol=1e-4, atol=1e-6)

    def test_vecmat_basic(self):
        """Test vector-matrix multiplication."""
        x = create_tensor(np.array([1.0, 2.0, 3.0]))  # (3,)
        a = create_tensor(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # (3, 2)
        y = run_matmul(x, a)

        expected = np.array([22.0, 28.0])
        np.testing.assert_array_equal(get_tensor_data(y), expected)

    def test_vecmat_backward(self):
        """Test vector-matrix multiplication backward."""
        x_data = np.array([1.0, 2.0, 3.0])
        a_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def f(x_np, a_np):
            x = create_tensor(x_np.copy())
            a = create_tensor(a_np.copy())
            y = run_matmul(x, a)
            s = run_tensor_sum(y)
            return float(get_tensor_data(s))

        # Analytical gradient
        x = create_tensor(x_data.copy())
        a = create_tensor(a_data.copy())
        y = run_matmul(x, a)
        s = run_tensor_sum(y)
        run_tensor_backward(s)
        grad_x_ana = get_tensor_grad(x).copy()
        grad_a_ana = get_tensor_grad(a).copy()

        # Numerical gradient
        grad_x_num = numerical_gradient(lambda v: f(v, a_data), x_data.copy())
        grad_a_num = numerical_gradient(lambda m: f(x_data, m), a_data.copy())

        np.testing.assert_allclose(
            grad_x_ana, grad_x_num, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            grad_a_ana, grad_a_num, rtol=1e-4, atol=1e-6)


class TestTensorActivations:
    """Test tensor activation functions (2 points)."""

    def test_tensor_relu(self):
        """Test ReLU on tensor."""
        a = create_tensor(np.array([[-1.0, 2.0], [0.0, -3.0]]))
        b = run_tensor_relu(a)
        expected = np.array([[0.0, 2.0], [0.0, 0.0]])
        np.testing.assert_array_equal(get_tensor_data(b), expected)

    def test_tensor_relu_backward(self):
        """Test ReLU backward on tensor."""
        a = create_tensor(np.array([[-1.0, 2.0], [3.0, -3.0]]))
        b = run_tensor_relu(a)
        run_tensor_backward(b)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(get_tensor_grad(a), expected)

    def test_tensor_exp(self):
        """Test exp on tensor."""
        a = create_tensor(np.array([[0.0, 1.0], [2.0, -1.0]]))
        b = run_tensor_exp(a)
        expected = np.exp(np.array([[0.0, 1.0], [2.0, -1.0]]))
        np.testing.assert_allclose(get_tensor_data(b), expected)

    def test_tensor_exp_backward(self):
        """Test exp backward on tensor."""
        a = create_tensor(np.array([[0.0, 1.0], [2.0, -1.0]]))
        b = run_tensor_exp(a)
        run_tensor_backward(b)
        # Gradient of exp(x) is exp(x)
        expected = np.exp(np.array([[0.0, 1.0], [2.0, -1.0]]))
        np.testing.assert_allclose(get_tensor_grad(a), expected)

    def test_tensor_log(self):
        """Test log on tensor."""
        a = create_tensor(np.array([[1.0, 2.0], [np.e, 4.0]]))
        b = run_tensor_log(a)
        expected = np.log(np.array([[1.0, 2.0], [np.e, 4.0]]))
        np.testing.assert_allclose(get_tensor_data(b), expected)

    def test_tensor_log_backward(self):
        """Test log backward on tensor."""
        a = create_tensor(np.array([[1.0, 2.0], [4.0, 5.0]]))
        b = run_tensor_log(a)
        run_tensor_backward(b)
        # Gradient of log(x) is 1/x
        expected = 1.0 / np.array([[1.0, 2.0], [4.0, 5.0]])
        np.testing.assert_allclose(get_tensor_grad(a), expected)


class TestMatrixChain:
    """Test chains of matrix operations (3 points)."""

    def test_matrix_chain_simple(self):
        """Test chain: X @ W1 @ W2"""
        np.random.seed(42)
        X = create_tensor(np.random.randn(4, 3))
        W1 = create_tensor(np.random.randn(3, 5))
        W2 = create_tensor(np.random.randn(5, 2))

        h = run_matmul(X, W1)
        y = run_matmul(h, W2)
        s = run_tensor_sum(y)
        run_tensor_backward(s)

        # Verify shapes
        assert get_tensor_grad(X).shape == (4, 3)
        assert get_tensor_grad(W1).shape == (3, 5)
        assert get_tensor_grad(W2).shape == (5, 2)

    def test_matrix_chain_with_relu(self):
        """Test chain: relu(X @ W1) @ W2"""
        np.random.seed(42)
        X_data = np.random.randn(4, 3)
        W1_data = np.random.randn(3, 5)
        W2_data = np.random.randn(5, 2)

        def f(x_np, w1_np, w2_np):
            X = create_tensor(x_np.copy())
            W1 = create_tensor(w1_np.copy())
            W2 = create_tensor(w2_np.copy())
            h = run_tensor_relu(run_matmul(X, W1))
            y = run_matmul(h, W2)
            s = run_tensor_sum(y)
            return float(get_tensor_data(s))

        # Analytical gradient
        X = create_tensor(X_data.copy())
        W1 = create_tensor(W1_data.copy())
        W2 = create_tensor(W2_data.copy())
        h = run_tensor_relu(run_matmul(X, W1))
        y = run_matmul(h, W2)
        s = run_tensor_sum(y)
        run_tensor_backward(s)
        grad_X_ana = get_tensor_grad(X).copy()
        grad_W1_ana = get_tensor_grad(W1).copy()
        grad_W2_ana = get_tensor_grad(W2).copy()

        # Numerical gradient
        grad_X_num = numerical_gradient(
            lambda x: f(x, W1_data, W2_data), X_data.copy())
        grad_W1_num = numerical_gradient(
            lambda w: f(X_data, w, W2_data), W1_data.copy())
        grad_W2_num = numerical_gradient(
            lambda w: f(X_data, W1_data, w), W2_data.copy())

        np.testing.assert_allclose(
            grad_X_ana, grad_X_num, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            grad_W1_ana, grad_W1_num, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            grad_W2_ana, grad_W2_num, rtol=1e-4, atol=1e-5)

    def test_matrix_chain_with_bias(self):
        """Test chain: relu(X @ W1 + b1) @ W2 + b2"""
        np.random.seed(42)
        X_data = np.random.randn(4, 3)
        W1_data = np.random.randn(3, 5)
        b1_data = np.random.randn(5)
        W2_data = np.random.randn(5, 2)
        b2_data = np.random.randn(2)

        def f(x_np, w1_np, b1_np, w2_np, b2_np):
            X = create_tensor(x_np.copy())
            W1 = create_tensor(w1_np.copy())
            b1 = create_tensor(b1_np.copy())
            W2 = create_tensor(w2_np.copy())
            b2 = create_tensor(b2_np.copy())

            h = run_tensor_relu(run_tensor_add(run_matmul(X, W1), b1))
            y = run_tensor_add(run_matmul(h, W2), b2)
            s = run_tensor_sum(y)
            return float(get_tensor_data(s))

        # Analytical gradient
        X = create_tensor(X_data.copy())
        W1 = create_tensor(W1_data.copy())
        b1 = create_tensor(b1_data.copy())
        W2 = create_tensor(W2_data.copy())
        b2 = create_tensor(b2_data.copy())

        h = run_tensor_relu(run_tensor_add(run_matmul(X, W1), b1))
        y = run_tensor_add(run_matmul(h, W2), b2)
        s = run_tensor_sum(y)
        run_tensor_backward(s)

        grad_W1_ana = get_tensor_grad(W1).copy()
        grad_b1_ana = get_tensor_grad(b1).copy()
        grad_W2_ana = get_tensor_grad(W2).copy()
        grad_b2_ana = get_tensor_grad(b2).copy()

        # Numerical gradient
        grad_W1_num = numerical_gradient(
            lambda w: f(X_data, w, b1_data, W2_data, b2_data), W1_data.copy()
        )
        grad_b1_num = numerical_gradient(
            lambda b: f(X_data, W1_data, b, W2_data, b2_data), b1_data.copy()
        )
        grad_W2_num = numerical_gradient(
            lambda w: f(X_data, W1_data, b1_data, w, b2_data), W2_data.copy()
        )
        grad_b2_num = numerical_gradient(
            lambda b: f(X_data, W1_data, b1_data, W2_data, b), b2_data.copy()
        )

        np.testing.assert_allclose(
            grad_W1_ana, grad_W1_num, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            grad_b1_ana, grad_b1_num, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            grad_W2_ana, grad_W2_num, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            grad_b2_ana, grad_b2_num, rtol=1e-4, atol=1e-5)
