"""
Part 7: MLP - The Grand Finale (10 points)

Train your MLP on real patterns and watch it learn!
"""

import pytest
import numpy as np
from tests.adapters import (
    create_tensor,
    get_tensor_data,
    get_tensor_grad,
    run_tensor_backward,
    create_mlp,
    run_mlp,
    get_mlp_parameters,
    run_cross_entropy,
    run_softmax,
    run_sgd_step,
    zero_grad,
)


def compute_accuracy(mlp, X_data, y_data):
    """Compute classification accuracy."""
    X = create_tensor(X_data)
    logits = run_mlp(mlp, X)
    probs = run_softmax(logits, axis=-1)
    preds = np.argmax(get_tensor_data(probs), axis=1)
    targets = np.argmax(y_data, axis=1)
    return np.mean(preds == targets)


class TestMLP:
    """Test your MLP implementation (4 points)."""

    def test_mlp_creation(self):
        """Test MLP can be created."""
        mlp = create_mlp(4, 8, 2)
        params = get_mlp_parameters(mlp)
        # MLP has 2 layers: 4 params (2 weights + 2 biases)
        assert len(params) == 4, f"MLP should have 4 parameters (2 weights + 2 biases), got {len(params)}"

    def test_mlp_forward(self):
        """Test MLP forward pass produces correct output shape."""
        mlp = create_mlp(4, 8, 3)
        X = create_tensor(np.random.randn(5, 4))  # batch of 5, 4 features

        logits = run_mlp(mlp, X)
        output = get_tensor_data(logits)

        assert output.shape == (5, 3), f"Expected (5, 3), got {output.shape}"

    def test_mlp_backward(self):
        """Test MLP backward pass computes gradients."""
        np.random.seed(42)
        mlp = create_mlp(4, 8, 3)
        params = get_mlp_parameters(mlp)

        X = create_tensor(np.random.randn(5, 4))
        y = create_tensor(np.eye(3)[np.random.randint(0, 3, 5)])  # random one-hot labels

        logits = run_mlp(mlp, X)
        loss = run_cross_entropy(logits, y)
        run_tensor_backward(loss)

        # All parameters should have non-zero gradients
        for i, p in enumerate(params):
            grad = get_tensor_grad(p)
            assert grad is not None, f"Parameter {i} has no gradient"
            assert grad.shape == get_tensor_data(p).shape, f"Gradient shape mismatch for param {i}"

    def test_mlp_parameters_method(self):
        """Test that parameters() returns all trainable tensors."""
        mlp = create_mlp(2, 4, 2)
        params = get_mlp_parameters(mlp)

        # Should have weight + bias for each of 2 layers = 4 parameters
        assert len(params) == 4, f"Expected 4 parameters, got {len(params)}"

        # Each param should be a tensor with data
        for p in params:
            data = get_tensor_data(p)
            assert data is not None
            assert data.size > 0


class TestMLPTrainingXOR:
    """Test MLP on XOR problem (3 points)."""

    def test_xor_training(self, capsys):
        """Train MLP on XOR - the classic neural network benchmark."""
        print("\n" + "=" * 60)
        print("       TRAINING MLP ON XOR PROBLEM")
        print("=" * 60)

        # XOR dataset
        X_data = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        y_data = np.array([
            [1.0, 0.0],  # 0 XOR 0 = 0 (class 0)
            [0.0, 1.0],  # 0 XOR 1 = 1 (class 1)
            [0.0, 1.0],  # 1 XOR 0 = 1 (class 1)
            [1.0, 0.0]   # 1 XOR 1 = 0 (class 0)
        ])

        print("\nDataset:")
        print("  Input    -> Expected Output")
        for i in range(4):
            inp = X_data[i]
            out = "0" if y_data[i, 0] == 1.0 else "1"
            print(f"  {int(inp[0])} XOR {int(inp[1])} -> {out}")

        # Create MLP: 2 inputs -> 16 hidden -> 2 outputs
        mlp = create_mlp(2, 16, 2)
        params = get_mlp_parameters(mlp)

        print(f"\nMLP Architecture: 2 -> 16 -> 2")
        print(f"Total parameters: {sum(get_tensor_data(p).size for p in params)}")

        # Training
        lr = 0.5
        epochs = 500
        losses = []

        print(f"\nTraining for {epochs} epochs with lr={lr}...")
        print("-" * 40)

        for epoch in range(epochs):
            zero_grad(params)

            X = create_tensor(X_data)
            y = create_tensor(y_data)

            logits = run_mlp(mlp, X)
            loss = run_cross_entropy(logits, y)
            run_tensor_backward(loss)
            run_sgd_step(params, lr=lr)

            loss_val = float(get_tensor_data(loss))
            losses.append(loss_val)

            if epoch % 100 == 0 or epoch == epochs - 1:
                acc = compute_accuracy(mlp, X_data, y_data)
                print(f"  Epoch {epoch:4d} | Loss: {loss_val:.4f} | Accuracy: {acc*100:.1f}%")

        # Final evaluation
        print("-" * 40)
        print("\nFinal Predictions:")
        X = create_tensor(X_data)
        logits = run_mlp(mlp, X)
        probs = run_softmax(logits, axis=-1)
        probs_data = get_tensor_data(probs)

        all_correct = True
        for i in range(4):
            inp = X_data[i]
            pred_class = np.argmax(probs_data[i])
            true_class = np.argmax(y_data[i])
            confidence = probs_data[i, pred_class] * 100
            status = "OK" if pred_class == true_class else "WRONG"
            if pred_class != true_class:
                all_correct = False
            print(f"  {int(inp[0])} XOR {int(inp[1])} -> pred: {pred_class} (conf: {confidence:.1f}%) [{status}]")

        final_loss = losses[-1]
        final_acc = compute_accuracy(mlp, X_data, y_data)

        print("\n" + "=" * 60)
        if all_correct and final_acc == 1.0:
            print("  SUCCESS! Your MLP learned XOR!")
        else:
            print("  Keep training - XOR is tricky!")
        print("=" * 60 + "\n")

        # Assertions
        assert final_loss < 0.5, f"Loss should be < 0.5, got {final_loss:.4f}"
        assert final_acc >= 0.75, f"Accuracy should be >= 75%, got {final_acc*100:.1f}%"


class TestMLPVisualization:
    """Test MLP with loss visualization (3 points)."""

    def test_mlp_loss_visualization(self, capsys):
        """Train MLP and visualize the learning progress."""
        print("\n" + "=" * 60)
        print("       MLP TRAINING VISUALIZATION")
        print("=" * 60)

        # Create a slightly harder dataset - 2D classification
        np.random.seed(42)
        n_samples = 50

        # Class 0: cluster around (-1, -1)
        X0 = np.random.randn(n_samples, 2) * 0.5 + np.array([-1, -1])
        # Class 1: cluster around (1, 1)
        X1 = np.random.randn(n_samples, 2) * 0.5 + np.array([1, 1])

        X_data = np.vstack([X0, X1])
        y_data = np.zeros((2 * n_samples, 2))
        y_data[:n_samples, 0] = 1.0
        y_data[n_samples:, 1] = 1.0

        # Shuffle
        idx = np.random.permutation(2 * n_samples)
        X_data = X_data[idx]
        y_data = y_data[idx]

        print(f"\nDataset: 2D binary classification")
        print(f"  Samples: {len(X_data)}")
        print(f"  Features: 2")
        print(f"  Classes: 2")

        # Create MLP
        mlp = create_mlp(2, 16, 2)
        params = get_mlp_parameters(mlp)

        total_params = sum(get_tensor_data(p).size for p in params)
        print(f"\nModel: MLP(2 -> 16 -> 2)")
        print(f"  Parameters: {total_params}")

        # Training
        lr = 0.5
        epochs = 200
        losses = []

        initial_acc = compute_accuracy(mlp, X_data, y_data)
        print(f"\nInitial accuracy: {initial_acc*100:.1f}%")

        print(f"\nTraining for {epochs} epochs...")
        print("-" * 50)

        for epoch in range(epochs):
            zero_grad(params)

            X = create_tensor(X_data)
            y = create_tensor(y_data)

            logits = run_mlp(mlp, X)
            loss = run_cross_entropy(logits, y)
            run_tensor_backward(loss)
            run_sgd_step(params, lr=lr)

            loss_val = float(get_tensor_data(loss))
            losses.append(loss_val)

            if epoch % 40 == 0 or epoch == epochs - 1:
                acc = compute_accuracy(mlp, X_data, y_data)
                bar_len = int(acc * 30)
                bar = "#" * bar_len + "-" * (30 - bar_len)
                print(f"  Epoch {epoch:4d} | Loss: {loss_val:.4f} | Acc: [{bar}] {acc*100:.1f}%")

        # Final stats
        final_loss = losses[-1]
        final_acc = compute_accuracy(mlp, X_data, y_data)

        print("-" * 50)
        print("\n  TRAINING SUMMARY")
        print(f"  ----------------")
        print(f"  Initial loss:     {losses[0]:.4f}")
        print(f"  Final loss:       {final_loss:.4f}")
        print(f"  Loss reduction:   {((losses[0] - final_loss) / losses[0] * 100):.1f}%")
        print()
        print(f"  Initial accuracy: {initial_acc*100:.1f}%")
        print(f"  Final accuracy:   {final_acc*100:.1f}%")

        # ASCII loss curve
        print("\n  LOSS CURVE:")
        print("  " + "-" * 45)

        n_rows = 8
        n_cols = 40
        max_loss = max(losses)
        min_loss = min(losses)
        loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

        # Sample losses for display
        step = max(1, len(losses) // n_cols)
        sampled = [losses[i] for i in range(0, len(losses), step)][:n_cols]

        for row in range(n_rows):
            threshold = max_loss - (row + 1) * loss_range / n_rows
            line = "  "
            if row == 0:
                line += f"{max_loss:.2f} |"
            elif row == n_rows - 1:
                line += f"{min_loss:.2f} |"
            else:
                line += "      |"

            for val in sampled:
                if val >= threshold:
                    line += "*"
                else:
                    line += " "
            print(line)

        print("  " + "      +" + "-" * n_cols)
        print("  " + "       0" + " " * (n_cols - 10) + f"{epochs}")
        print("  " + "              epochs")
        print("  " + "-" * 45)

        print("\n" + "=" * 60)
        if final_acc >= 0.95:
            print("  EXCELLENT! Near-perfect classification!")
        elif final_acc >= 0.85:
            print("  GREAT! Your MLP learned the pattern well.")
        elif final_acc >= 0.70:
            print("  GOOD progress - the model is learning.")
        else:
            print("  Keep working on it!")
        print("=" * 60 + "\n")

        # Assertions
        assert final_loss < losses[0], "Loss should decrease"
        assert final_acc > initial_acc, "Accuracy should improve"
        assert final_acc >= 0.80, f"Expected >= 80% accuracy, got {final_acc*100:.1f}%"
