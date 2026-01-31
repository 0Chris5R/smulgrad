"""
Tests for Bonus: MNIST Handwritten Digit Recognition

Run with: uv run pytest -k TestMNIST
"""

import pytest
import numpy as np


class TestMNIST:
    """Test MNIST training bonus exercise."""

    def test_mnist_training_runs(self):
        """Test that MNIST training completes without errors (3 points)."""
        from tests.adapters import run_mnist_training

        # Train with fewer epochs for faster testing
        mlp, accuracy = run_mnist_training(
            hidden_size=64,
            learning_rate=0.1,
            batch_size=256,
            epochs=2
        )

        # Basic checks
        assert mlp is not None, "train_mnist should return a model"
        assert isinstance(accuracy, float), "train_mnist should return accuracy as float"
        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"

        # Should learn something (better than random 10%)
        assert accuracy > 0.5, (
            f"Model should learn something after 2 epochs. "
            f"Got {accuracy*100:.1f}% (random would be 10%)"
        )

    def test_mnist_accuracy_threshold(self):
        """Test that MNIST achieves >95% accuracy (2 points)."""
        from tests.adapters import run_mnist_training

        # Uses student's hyperparameters from run_mnist_training defaults
        mlp, accuracy = run_mnist_training()

        assert accuracy > 0.95, (
            f"Model should achieve >95% accuracy. "
            f"Got {accuracy*100:.1f}%. "
            f"Set hyperparameter defaults in run_mnist_training in adapters.py."
        )

    def test_mnist_learns_over_epochs(self):
        """Test that model improves with more training epochs."""
        from tests.adapters import run_mnist_training

        # Train for 1 epoch
        _, accuracy_1 = run_mnist_training(
            hidden_size=64,
            learning_rate=0.1,
            batch_size=256,
            epochs=1
        )

        # Train for 3 epochs
        _, accuracy_3 = run_mnist_training(
            hidden_size=64,
            learning_rate=0.1,
            batch_size=256,
            epochs=3
        )

        # More epochs should yield better accuracy
        assert accuracy_3 > accuracy_1, (
            f"Model should improve with more epochs. "
            f"1 epoch: {accuracy_1*100:.1f}%, 3 epochs: {accuracy_3*100:.1f}%"
        )
