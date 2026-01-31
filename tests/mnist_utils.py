"""
MNIST Utilities - Data loading and batching for the MNIST bonus exercise.

This module provides utility functions for loading the MNIST dataset
and iterating over mini-batches. Students should use these functions
in their train_mnist implementation.
"""

import numpy as np
import gzip
from pathlib import Path
from urllib.request import urlretrieve


# MNIST URLs
MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_mnist(data_dir: Path) -> None:
    """Download MNIST dataset if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, filename in MNIST_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            url = MNIST_BASE_URL + filename
            print(f"Downloading {filename}...")
            urlretrieve(url, filepath)


def _load_mnist_images(filepath: Path) -> np.ndarray:
    """Load MNIST images from gzipped file."""
    with gzip.open(filepath, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")  # magic
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
    return data.astype(np.float32) / 255.0


def _load_mnist_labels(filepath: Path) -> np.ndarray:
    """Load MNIST labels from gzipped file."""
    with gzip.open(filepath, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")  # magic
        num_labels = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def _one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert labels to one-hot encoding."""
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def load_mnist(data_dir: str = None):
    """
    Load the MNIST dataset.

    Downloads the data if not present. Returns normalized images (0-1)
    and one-hot encoded labels.

    Args:
        data_dir: Directory to store/load data from. Defaults to tests/../data/mnist

    Returns:
        train_X: Training images, shape (60000, 784), float32 in [0, 1]
        train_y: Training labels, shape (60000, 10), one-hot encoded
        test_X: Test images, shape (10000, 784), float32 in [0, 1]
        test_y: Test labels, shape (10000, 10), one-hot encoded

    Example:
        >>> train_X, train_y, test_X, test_y = load_mnist()
        >>> print(train_X.shape)  # (60000, 784)
        >>> print(train_y.shape)  # (60000, 10)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "mnist"
    else:
        data_dir = Path(data_dir)

    _download_mnist(data_dir)

    train_images = _load_mnist_images(data_dir / MNIST_FILES["train_images"])
    train_labels = _load_mnist_labels(data_dir / MNIST_FILES["train_labels"])
    test_images = _load_mnist_images(data_dir / MNIST_FILES["test_images"])
    test_labels = _load_mnist_labels(data_dir / MNIST_FILES["test_labels"])

    train_labels_oh = _one_hot_encode(train_labels)
    test_labels_oh = _one_hot_encode(test_labels)

    return train_images, train_labels_oh, test_images, test_labels_oh


def load_mnist_raw(data_dir: str = None):
    """
    Load MNIST with raw integer labels (not one-hot encoded).

    Useful for computing accuracy.

    Returns:
        train_X: Training images, shape (60000, 784)
        train_y: Training labels, shape (60000,), integers 0-9
        test_X: Test images, shape (10000, 784)
        test_y: Test labels, shape (10000,), integers 0-9
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "mnist"
    else:
        data_dir = Path(data_dir)

    _download_mnist(data_dir)

    train_images = _load_mnist_images(data_dir / MNIST_FILES["train_images"])
    train_labels = _load_mnist_labels(data_dir / MNIST_FILES["train_labels"])
    test_images = _load_mnist_images(data_dir / MNIST_FILES["test_images"])
    test_labels = _load_mnist_labels(data_dir / MNIST_FILES["test_labels"])

    return train_images, train_labels, test_images, test_labels


def mini_batch_iterator(X, y, batch_size, shuffle=True):
    """
    Generate mini-batches from data.

    Args:
        X: Input data, shape (n_samples, n_features)
        y: Labels, shape (n_samples, ...) - can be one-hot or integers
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data before iterating (default True)

    Yields:
        Tuples of (batch_X, batch_y) numpy arrays

    Example:
        >>> for batch_x, batch_y in mini_batch_iterator(train_X, train_y, 128):
        ...     # batch_x.shape = (128, 784)
        ...     # batch_y.shape = (128, 10)
        ...     pass
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions, shape (n_samples, n_classes) - logits or probabilities
        labels: True labels, either:
            - shape (n_samples,) integers, or
            - shape (n_samples, n_classes) one-hot encoded

    Returns:
        Accuracy as a float between 0 and 1

    Example:
        >>> acc = compute_accuracy(model_output, test_labels)
        >>> print(f"Accuracy: {acc * 100:.1f}%")
    """
    pred_classes = np.argmax(predictions, axis=1)

    if labels.ndim == 2:
        # One-hot encoded
        true_classes = np.argmax(labels, axis=1)
    else:
        # Integer labels
        true_classes = labels

    return np.mean(pred_classes == true_classes)
