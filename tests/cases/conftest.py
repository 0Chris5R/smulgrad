"""Pytest configuration and fixtures for SmulGrad tests."""

import numpy as np
import pytest


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def rtol():
    """Relative tolerance for floating point comparisons."""
    return 1e-5


@pytest.fixture
def atol():
    """Absolute tolerance for floating point comparisons."""
    return 1e-6


@pytest.fixture
def grad_check_eps():
    """Epsilon for numerical gradient checking."""
    return 1e-5


@pytest.fixture
def sample_scalar_a():
    """Sample scalar value."""
    return 2.0


@pytest.fixture
def sample_scalar_b():
    """Sample scalar value."""
    return 3.0


@pytest.fixture
def sample_vector():
    """Sample 1D array."""
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def sample_matrix():
    """Sample 2D array."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def sample_matrix_square():
    """Sample square 2D array."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def sample_batch_matrix(seed):
    """Sample batched 3D array."""
    np.random.seed(seed)
    return np.random.randn(2, 3, 4)


def assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=""):
    """Assert that two arrays are close."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=msg)


def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient of f at x.

    Args:
        f: Function that takes x and returns a scalar
        x: numpy array
        eps: Epsilon for finite differences

    Returns:
        Numerical gradient with same shape as x
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        pos = float(f(x))

        x[idx] = old_val - eps
        neg = float(f(x))

        x[idx] = old_val
        grad[idx] = (pos - neg) / (2 * eps)
        it.iternext()

    return grad
