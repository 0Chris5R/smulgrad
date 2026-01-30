"""
Part 6: Neural Network Training (15 points)

These tests verify neural network components and training capabilities.
"""

import pytest
import numpy as np
from .conftest import assert_close, numerical_gradient
from tests.adapters import (
    create_tensor,
    get_tensor_data,
    get_tensor_grad,
    run_tensor_sum,
    run_tensor_mean,
    run_tensor_backward,
    run_matmul,
    run_tensor_add,
    run_tensor_mul,
    run_tensor_relu,
    run_tensor_exp,
    run_tensor_log,
    run_tensor_max,
    create_linear_layer,
    run_linear_layer,
    get_linear_parameters,
    run_softmax,
    run_cross_entropy,
    run_sgd_step,
    zero_grad,
    run_gradient_check,
)


class TestLinearLayer:
    """Test linear (fully connected) layer (2 points)."""

    def test_linear_layer_creation(self):
        """Test creating a Linear layer."""
        layer = create_linear_layer(in_features=3, out_features=2)
        params = get_linear_parameters(layer)

        # Should have weight and bias
        assert len(params) == 2
        weight, bias = params

        # Check shapes
        assert get_tensor_data(weight).shape == (3, 2)
        assert get_tensor_data(bias).shape == (2,)

        # Bias should be initialized to zeros
        np.testing.assert_array_equal(get_tensor_data(bias), np.zeros(2))

    def test_linear_layer_forward(self):
        """Test linear layer forward pass."""
        layer = create_linear_layer(in_features=3, out_features=2)
        params = get_linear_parameters(layer)
        weight, bias = params

        # Set known weights for testing
        weight.data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        bias.data = np.array([0.1, 0.2])

        x = create_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # (2, 3)
        y = run_linear_layer(layer, x)

        # y = x @ weight + bias
        x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        w_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        b_data = np.array([0.1, 0.2])
        expected = x_data @ w_data + b_data

        np.testing.assert_allclose(get_tensor_data(y), expected)

    def test_linear_layer_backward(self):
        """Test linear layer backward pass."""
        np.random.seed(42)

        layer = create_linear_layer(in_features=3, out_features=5)
        params = get_linear_parameters(layer)
        weight, bias = params

        # Set known weights
        w_data = np.random.randn(3, 5)
        b_data = np.random.randn(5)
        weight.data = w_data.copy()
        bias.data = b_data.copy()

        x_data = np.random.randn(4, 3)

        def f(x_np, w_np, b_np):
            # Create fresh layer and set weights
            layer_f = create_linear_layer(in_features=3, out_features=5)
            params_f = get_linear_parameters(layer_f)
            params_f[0].data = w_np.copy()
            params_f[1].data = b_np.copy()
            x = create_tensor(x_np.copy())
            y = run_linear_layer(layer_f, x)
            s = run_tensor_sum(y)
            return float(get_tensor_data(s))

        # Analytical gradient
        x = create_tensor(x_data.copy())
        weight.data = w_data.copy()
        bias.data = b_data.copy()
        weight.grad = np.zeros_like(weight.data)
        bias.grad = np.zeros_like(bias.data)

        y = run_linear_layer(layer, x)
        s = run_tensor_sum(y)
        run_tensor_backward(s)

        grad_w_ana = get_tensor_grad(weight).copy()
        grad_b_ana = get_tensor_grad(bias).copy()

        # Numerical gradient
        grad_w_num = numerical_gradient(lambda ww: f(x_data, ww, b_data), w_data.copy())
        grad_b_num = numerical_gradient(lambda bb: f(x_data, w_data, bb), b_data.copy())

        np.testing.assert_allclose(grad_w_ana, grad_w_num, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(grad_b_ana, grad_b_num, rtol=1e-4, atol=1e-5)


class TestSoftmax:
    """Test softmax function (3 points)."""

    def test_softmax_basic(self):
        """Test softmax on simple input."""
        logits = create_tensor(np.array([[1.0, 2.0, 3.0]]))
        probs = run_softmax(logits, axis=-1)

        # softmax(x)_i = exp(x_i) / sum(exp(x_j))
        expected = np.exp([1.0, 2.0, 3.0]) / np.sum(np.exp([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(get_tensor_data(probs).flatten(), expected, rtol=1e-5)

    def test_softmax_batch(self):
        """Test softmax on batch of inputs."""
        logits = create_tensor(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]))
        probs = run_softmax(logits, axis=-1)

        # First row
        row0 = np.exp([1.0, 2.0, 3.0]) / np.sum(np.exp([1.0, 2.0, 3.0]))
        # Second row - uniform
        row1 = np.array([1/3, 1/3, 1/3])

        expected = np.stack([row0, row1])
        np.testing.assert_allclose(get_tensor_data(probs), expected, rtol=1e-5)

    def test_softmax_numerical_stability(self):
        """Test softmax doesn't overflow with large inputs."""
        logits = create_tensor(np.array([[1000.0, 1001.0, 1002.0]]))
        probs = run_softmax(logits, axis=-1)

        # Should still be valid probabilities
        probs_data = get_tensor_data(probs)
        assert not np.any(np.isnan(probs_data))
        assert not np.any(np.isinf(probs_data))
        np.testing.assert_allclose(np.sum(probs_data), 1.0, rtol=1e-5)

    def test_softmax_sums_to_one(self):
        """Test softmax outputs sum to 1."""
        np.random.seed(42)
        logits = create_tensor(np.random.randn(3, 5))
        probs = run_softmax(logits, axis=-1)

        sums = np.sum(get_tensor_data(probs), axis=-1)
        np.testing.assert_allclose(sums, np.ones(3), rtol=1e-5)

    def test_softmax_backward(self):
        """Test softmax backward pass via cross-entropy (the typical use case)."""
        np.random.seed(42)
        logits_data = np.random.randn(2, 4)
        targets_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

        def f(x):
            logits = create_tensor(x.copy())
            targets = create_tensor(targets_data.copy())
            loss = run_cross_entropy(logits, targets)
            return float(get_tensor_data(loss))

        # Analytical
        logits = create_tensor(logits_data.copy())
        targets = create_tensor(targets_data.copy())
        loss = run_cross_entropy(logits, targets)
        run_tensor_backward(loss)
        grad_ana = get_tensor_grad(logits).copy()

        # Numerical
        grad_num = numerical_gradient(f, logits_data.copy())

        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-5)


class TestCrossEntropy:
    """Test cross-entropy loss (3 points)."""

    def test_cross_entropy_perfect(self):
        """Test cross-entropy with perfect predictions."""
        # Logits that strongly predict correct class
        logits = create_tensor(np.array([[10.0, -10.0, -10.0]]))
        targets = create_tensor(np.array([[1.0, 0.0, 0.0]]))

        loss = run_cross_entropy(logits, targets)
        # Should be very close to 0
        assert get_tensor_data(loss) < 0.01

    def test_cross_entropy_wrong(self):
        """Test cross-entropy with wrong predictions."""
        # Logits that strongly predict wrong class
        logits = create_tensor(np.array([[-10.0, 10.0, -10.0]]))
        targets = create_tensor(np.array([[1.0, 0.0, 0.0]]))

        loss = run_cross_entropy(logits, targets)
        # Should be large (around 20)
        assert get_tensor_data(loss) > 10.0

    def test_cross_entropy_batch(self):
        """Test cross-entropy on batch."""
        logits = create_tensor(np.array([
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0]
        ]))
        targets = create_tensor(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]))

        loss = run_cross_entropy(logits, targets)
        # Loss should be positive and reasonable
        loss_val = get_tensor_data(loss)
        assert loss_val > 0
        assert loss_val < 5.0

    def test_cross_entropy_gradient(self):
        """Test cross-entropy gradient (softmax - targets)."""
        np.random.seed(42)
        logits_data = np.random.randn(2, 3)
        targets_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        logits = create_tensor(logits_data.copy())
        targets = create_tensor(targets_data.copy())

        loss = run_cross_entropy(logits, targets)
        run_tensor_backward(loss)

        # Gradient should be (softmax(logits) - targets) / batch_size
        softmax_out = np.exp(logits_data - logits_data.max(axis=-1, keepdims=True))
        softmax_out = softmax_out / softmax_out.sum(axis=-1, keepdims=True)
        expected_grad = (softmax_out - targets_data) / 2  # batch_size = 2

        np.testing.assert_allclose(
            get_tensor_grad(logits),
            expected_grad,
            rtol=1e-4,
            atol=1e-5
        )

    def test_cross_entropy_numerical_gradient(self):
        """Verify cross-entropy gradient numerically."""
        np.random.seed(42)
        logits_data = np.random.randn(2, 3)
        targets_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        def f(x):
            logits = create_tensor(x.copy())
            targets = create_tensor(targets_data.copy())
            loss = run_cross_entropy(logits, targets)
            return float(get_tensor_data(loss))

        logits = create_tensor(logits_data.copy())
        targets = create_tensor(targets_data.copy())
        loss = run_cross_entropy(logits, targets)
        run_tensor_backward(loss)
        grad_ana = get_tensor_grad(logits).copy()

        grad_num = numerical_gradient(f, logits_data.copy())

        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-5)


class TestSGD:
    """Test SGD optimizer (2 points)."""

    def test_sgd_step(self):
        """Test single SGD update step."""
        w = create_tensor(np.array([1.0, 2.0, 3.0]))
        # Manually set gradient
        w_data = get_tensor_data(w)
        w.grad = np.array([0.1, 0.2, 0.3])  # Assuming direct access or setter

        # Try to use the adapter - it may need the gradient to already be set
        # This test assumes gradients are computed via backward first
        x = create_tensor(np.array([1.0, 1.0, 1.0]))
        y = run_tensor_mul(w, x)
        s = run_tensor_sum(y)
        run_tensor_backward(s)

        w_before = get_tensor_data(w).copy()
        grad_before = get_tensor_grad(w).copy()

        run_sgd_step([w], lr=0.1)

        # w_new = w_old - lr * grad
        expected = w_before - 0.1 * grad_before
        np.testing.assert_allclose(get_tensor_data(w), expected)

    def test_sgd_multiple_params(self):
        """Test SGD with multiple parameters."""
        np.random.seed(42)
        w1 = create_tensor(np.random.randn(3, 4))
        w2 = create_tensor(np.random.randn(4, 2))

        x = create_tensor(np.random.randn(2, 3))
        h = run_matmul(x, w1)
        y = run_matmul(h, w2)
        s = run_tensor_sum(y)
        run_tensor_backward(s)

        w1_before = get_tensor_data(w1).copy()
        w2_before = get_tensor_data(w2).copy()
        grad_w1 = get_tensor_grad(w1).copy()
        grad_w2 = get_tensor_grad(w2).copy()

        run_sgd_step([w1, w2], lr=0.01)

        np.testing.assert_allclose(get_tensor_data(w1), w1_before - 0.01 * grad_w1)
        np.testing.assert_allclose(get_tensor_data(w2), w2_before - 0.01 * grad_w2)

    def test_sgd_custom_learning_rate(self):
        """Test SGD with different learning rates."""
        w = create_tensor(np.array([1.0, 2.0, 3.0]))
        x = create_tensor(np.array([1.0, 1.0, 1.0]))
        y = run_tensor_mul(w, x)
        s = run_tensor_sum(y)
        run_tensor_backward(s)

        w_before = get_tensor_data(w).copy()
        grad_before = get_tensor_grad(w).copy()

        # Test with larger learning rate
        run_sgd_step([w], lr=0.5)

        expected = w_before - 0.5 * grad_before
        np.testing.assert_allclose(get_tensor_data(w), expected)

    def test_zero_grad(self):
        """Test zeroing gradients."""
        w = create_tensor(np.array([1.0, 2.0, 3.0]))
        x = create_tensor(np.array([1.0, 1.0, 1.0]))
        y = run_tensor_mul(w, x)
        s = run_tensor_sum(y)
        run_tensor_backward(s)

        # Gradients should be non-zero
        assert np.any(get_tensor_grad(w) != 0)

        zero_grad([w])

        # Gradients should be zero
        np.testing.assert_array_equal(get_tensor_grad(w), np.zeros(3))


class TestMLPTraining:
    """Test MLP training loop (3 points)."""

    def test_mlp_training_loss_decreases(self):
        """Test that training decreases loss."""
        np.random.seed(42)

        # Simple dataset: XOR-like
        X_data = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        y_data = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])

        # Initialize weights
        W1 = create_tensor(np.random.randn(2, 8) * 0.5)
        b1 = create_tensor(np.zeros(8))
        W2 = create_tensor(np.random.randn(8, 2) * 0.5)
        b2 = create_tensor(np.zeros(2))

        params = [W1, b1, W2, b2]

        def forward_and_loss(X_np, y_np):
            X = create_tensor(X_np)
            y = create_tensor(y_np)

            # Forward pass
            h = run_tensor_relu(run_tensor_add(run_matmul(X, W1), b1))
            logits = run_tensor_add(run_matmul(h, W2), b2)

            # Loss
            loss = run_cross_entropy(logits, y)
            return loss

        # Initial loss
        initial_loss = get_tensor_data(forward_and_loss(X_data, y_data))

        # Training loop
        lr = 0.5
        for _ in range(200):
            zero_grad(params)

            loss = forward_and_loss(X_data, y_data)
            run_tensor_backward(loss)

            run_sgd_step(params, lr=lr)

        # Final loss
        final_loss = get_tensor_data(forward_and_loss(X_data, y_data))

        # Loss should have decreased significantly
        assert final_loss < initial_loss * 0.5, f"Loss did not decrease enough: {initial_loss} -> {final_loss}"

    def test_mlp_overfits_small_data(self):
        """Test that MLP can overfit a small dataset."""
        np.random.seed(42)

        # Small random dataset
        X_data = np.random.randn(8, 4)
        y_data = np.zeros((8, 3))
        for i in range(8):
            y_data[i, i % 3] = 1.0

        # Initialize weights
        W1 = create_tensor(np.random.randn(4, 16) * 0.5)
        b1 = create_tensor(np.zeros(16))
        W2 = create_tensor(np.random.randn(16, 3) * 0.5)
        b2 = create_tensor(np.zeros(3))

        params = [W1, b1, W2, b2]

        def forward_and_loss(X_np, y_np):
            X = create_tensor(X_np)
            y = create_tensor(y_np)

            h = run_tensor_relu(run_tensor_add(run_matmul(X, W1), b1))
            logits = run_tensor_add(run_matmul(h, W2), b2)

            loss = run_cross_entropy(logits, y)
            return loss

        # Training loop
        lr = 0.1
        for _ in range(200):
            zero_grad(params)

            loss = forward_and_loss(X_data, y_data)
            run_tensor_backward(loss)

            run_sgd_step(params, lr=lr)

        # Final loss should be very low (overfitting)
        final_loss = get_tensor_data(forward_and_loss(X_data, y_data))
        assert final_loss < 0.5, f"MLP should overfit small dataset, loss: {final_loss}"


class TestGradientCheck:
    """Test gradient checking (2 points)."""

    def test_gradient_check_simple(self):
        """Test gradient check on simple function."""
        def f(x):
            return run_tensor_sum(run_tensor_mul(x, x))

        np.random.seed(42)
        x = create_tensor(np.random.randn(3, 4))

        result = run_gradient_check(f, x)
        assert result, "Gradient check failed for x * x"

    def test_gradient_check_matmul(self):
        """Test gradient check on matmul."""
        np.random.seed(42)
        W = np.random.randn(4, 5)

        def f(x):
            w = create_tensor(W)
            return run_tensor_sum(run_matmul(x, w))

        x = create_tensor(np.random.randn(3, 4))

        result = run_gradient_check(f, x)
        assert result, "Gradient check failed for matmul"

    def test_gradient_check_complex(self):
        """Test gradient check on complex expression."""
        np.random.seed(42)
        W = np.random.randn(4, 5)
        B = np.random.randn(5)

        def f(x):
            w = create_tensor(W)
            b = create_tensor(B)
            h = run_tensor_relu(run_tensor_add(run_matmul(x, w), b))
            return run_tensor_sum(h)

        x = create_tensor(np.random.randn(3, 4))

        result = run_gradient_check(f, x)
        assert result, "Gradient check failed for relu(x @ w + b)"


class TestEndToEnd:
    """End-to-end tests combining everything."""

    def test_full_forward_backward(self):
        """Test complete forward and backward pass."""
        np.random.seed(42)

        # Input
        X = create_tensor(np.random.randn(2, 4))

        # Layer 1
        W1 = create_tensor(np.random.randn(4, 8) * 0.5)
        b1 = create_tensor(np.zeros(8))

        # Layer 2
        W2 = create_tensor(np.random.randn(8, 3) * 0.5)
        b2 = create_tensor(np.zeros(3))

        # Targets
        targets = create_tensor(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

        # Forward
        h = run_tensor_relu(run_tensor_add(run_matmul(X, W1), b1))
        logits = run_tensor_add(run_matmul(h, W2), b2)
        loss = run_cross_entropy(logits, targets)

        # Backward
        run_tensor_backward(loss)

        # Verify gradients exist and have correct shapes
        assert get_tensor_grad(W1).shape == (4, 8)
        assert get_tensor_grad(b1).shape == (8,)
        assert get_tensor_grad(W2).shape == (8, 3)
        assert get_tensor_grad(b2).shape == (3,)

        # Verify gradients are not all zeros
        assert np.any(get_tensor_grad(W1) != 0)
        assert np.any(get_tensor_grad(W2) != 0)
