"""
MNIST training implementation.

This module implements the train_mnist function for the bonus exercise.
"""

import numpy as np
from smulgrad.engine import Tensor
from smulgrad.nn import MLP, softmax, cross_entropy, SGD
from tests.mnist_utils import load_mnist, mini_batch_iterator, compute_accuracy


def train_mnist(
    hidden_size: int = 128,
    learning_rate: float = 0.1,
    batch_size: int = 128,
    epochs: int = 10
) -> tuple:
    """
    Train an MLP on the MNIST dataset.

    Args:
        hidden_size: Number of neurons in hidden layer
        learning_rate: SGD learning rate
        batch_size: Mini-batch size
        epochs: Number of training epochs

    Returns:
        Tuple of (trained_mlp, test_accuracy)
    """
    train_X, train_y, test_X, test_y = load_mnist()

    mlp = MLP(784, hidden_size, 10)
    optimizer = SGD(mlp.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_x, batch_y in mini_batch_iterator(train_X, train_y, batch_size):
            optimizer.zero_grad()
            logits = mlp(Tensor(batch_x))
            loss = cross_entropy(logits, Tensor(batch_y))
            loss.backward()
            optimizer.step()

    # Compute test accuracy
    logits = mlp(Tensor(test_X))
    probs = softmax(logits, axis=-1)
    accuracy = compute_accuracy(probs.data, test_y)

    return mlp, accuracy
