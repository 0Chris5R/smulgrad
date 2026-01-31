"""
Adapter functions for connecting your implementation to tests.

Each function here should call your implementation and return results
in a format the tests expect. You need to implement these functions
to make the tests work with your code.

Example:
    If you implement a Value class, the create_value adapter would be:

    def create_value(data: float) -> "YourValueClass":
        from smulgrad import Value
        return Value(data)
"""

import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union


# =============================================================================
# Part 1: Scalar Values and Basic Operations (15 points)
# =============================================================================

def create_value(data: float) -> Any:
    """
    Create a scalar Value object.

    Args:
        data: The scalar value to wrap

    Returns:
        A Value object with:
            - .data attribute containing the value
            - .grad attribute initialized to 0.0
    """
    # TODO: Import your Value class and create an instance
    # from smulgrad.engine import Value
    # return Value(data)
    raise NotImplementedError("Implement create_value adapter")


def get_value_data(v: Any) -> float:
    """Extract the data from a Value object."""
    # TODO: Return v.data
    raise NotImplementedError("Implement get_value_data adapter")


def get_value_grad(v: Any) -> float:
    """Extract the gradient from a Value object."""
    # TODO: Return v.grad
    raise NotImplementedError("Implement get_value_grad adapter")


def run_add(a: Any, b: Any) -> Any:
    """
    Add two Value objects or a Value and a scalar.

    Args:
        a: Value object or float
        b: Value object or float

    Returns:
        Value object representing a + b
    """
    # TODO: Return a + b (your __add__ implementation)
    raise NotImplementedError("Implement run_add adapter")


def run_mul(a: Any, b: Any) -> Any:
    """
    Multiply two Value objects or a Value and a scalar.

    Args:
        a: Value object or float
        b: Value object or float

    Returns:
        Value object representing a * b
    """
    # TODO: Return a * b (your __mul__ implementation)
    raise NotImplementedError("Implement run_mul adapter")


def run_neg(a: Any) -> Any:
    """
    Negate a Value object.

    Args:
        a: Value object

    Returns:
        Value object representing -a
    """
    # TODO: Return -a (your __neg__ implementation)
    raise NotImplementedError("Implement run_neg adapter")


def run_sub(a: Any, b: Any) -> Any:
    """
    Subtract two Value objects or a Value and a scalar.

    Args:
        a: Value object or float
        b: Value object or float

    Returns:
        Value object representing a - b
    """
    # TODO: Return a - b (your __sub__ implementation)
    raise NotImplementedError("Implement run_sub adapter")


def run_pow(a: Any, n: Union[int, float]) -> Any:
    """
    Raise a Value object to a power.

    Args:
        a: Value object
        n: Exponent (int or float, not a Value)

    Returns:
        Value object representing a ** n
    """
    # TODO: Return a ** n (your __pow__ implementation)
    raise NotImplementedError("Implement run_pow adapter")


def run_div(a: Any, b: Any) -> Any:
    """
    Divide two Value objects or a Value and a scalar.

    Args:
        a: Value object or float
        b: Value object or float

    Returns:
        Value object representing a / b
    """
    # TODO: Return a / b (your __truediv__ implementation)
    raise NotImplementedError("Implement run_div adapter")


# =============================================================================
# Part 2: The Backward Pass (20 points)
# =============================================================================

def run_backward(v: Any) -> None:
    """
    Run backpropagation from a Value object.

    This should compute gradients for all Values in the computational graph.

    Args:
        v: Value object to backpropagate from
    """
    # TODO: Call v.backward()
    raise NotImplementedError("Implement run_backward adapter")


# =============================================================================
# Part 3: More Operations (15 points)
# =============================================================================

def run_relu(a: Any) -> Any:
    """
    Apply ReLU activation to a Value object.

    Args:
        a: Value object

    Returns:
        Value object representing relu(a) = max(0, a)
    """
    # TODO: Return a.relu()
    raise NotImplementedError("Implement run_relu adapter")


def run_exp(a: Any) -> Any:
    """
    Compute exponential of a Value object.

    Args:
        a: Value object

    Returns:
        Value object representing exp(a)
    """
    # TODO: Return a.exp()
    raise NotImplementedError("Implement run_exp adapter")


def run_log(a: Any) -> Any:
    """
    Compute natural logarithm of a Value object.

    Args:
        a: Value object

    Returns:
        Value object representing log(a)
    """
    # TODO: Return a.log()
    raise NotImplementedError("Implement run_log adapter")


def run_tanh(a: Any) -> Any:
    """
    Compute hyperbolic tangent of a Value object.

    Args:
        a: Value object

    Returns:
        Value object representing tanh(a)
    """
    # TODO: Return a.tanh()
    raise NotImplementedError("Implement run_tanh adapter")


# =============================================================================
# Part 4: Tensor Support (22 points)
# =============================================================================

def create_tensor(data: np.ndarray) -> Any:
    """
    Create a Tensor object.

    Args:
        data: numpy array

    Returns:
        A Tensor object with:
            - .data attribute containing the numpy array
            - .grad attribute initialized to zeros with same shape
    """
    # TODO: Import your Tensor class and create an instance
    # from smulgrad.engine import Tensor
    # return Tensor(data)
    raise NotImplementedError("Implement create_tensor adapter")


def get_tensor_data(t: Any) -> np.ndarray:
    """Extract the data from a Tensor object."""
    # TODO: Return t.data
    raise NotImplementedError("Implement get_tensor_data adapter")


def get_tensor_grad(t: Any) -> np.ndarray:
    """Extract the gradient from a Tensor object."""
    # TODO: Return t.grad
    raise NotImplementedError("Implement get_tensor_grad adapter")


def run_tensor_add(a: Any, b: Any) -> Any:
    """
    Add two Tensor objects.

    Args:
        a: Tensor object
        b: Tensor object

    Returns:
        Tensor object representing a + b
    """
    # TODO: Return a + b
    raise NotImplementedError("Implement run_tensor_add adapter")


def run_tensor_mul(a: Any, b: Any) -> Any:
    """
    Element-wise multiply two Tensor objects.

    Args:
        a: Tensor object
        b: Tensor object

    Returns:
        Tensor object representing a * b (element-wise)
    """
    # TODO: Return a * b
    raise NotImplementedError("Implement run_tensor_mul adapter")


def run_tensor_backward(t: Any) -> None:
    """
    Run backpropagation from a Tensor object.

    Args:
        t: Tensor object to backpropagate from
    """
    # TODO: Call t.backward()
    raise NotImplementedError("Implement run_tensor_backward adapter")


def run_tensor_sum(
    a: Any,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
) -> Any:
    """
    Sum a Tensor along specified axes.

    Args:
        a: Tensor object
        axis: Axis or axes along which to sum
        keepdims: Whether to keep reduced dimensions

    Returns:
        Tensor object representing sum of a
    """
    # TODO: Return a.sum(axis=axis, keepdims=keepdims)
    raise NotImplementedError("Implement run_tensor_sum adapter")


def run_tensor_mean(
    a: Any,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
) -> Any:
    """
    Compute mean of a Tensor along specified axes.

    Args:
        a: Tensor object
        axis: Axis or axes along which to compute mean
        keepdims: Whether to keep reduced dimensions

    Returns:
        Tensor object representing mean of a
    """
    # TODO: Return a.mean(axis=axis, keepdims=keepdims)
    raise NotImplementedError("Implement run_tensor_mean adapter")


def run_tensor_max(
    a: Any,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
) -> Any:
    """
    Compute max of a Tensor along specified axes.

    Args:
        a: Tensor object
        axis: Axis or axes along which to compute max
        keepdims: Whether to keep reduced dimensions

    Returns:
        Tensor object representing max of a
    """
    # TODO: Return a.max(axis=axis, keepdims=keepdims)
    raise NotImplementedError("Implement run_tensor_max adapter")


def run_tensor_reshape(a: Any, shape: Tuple[int, ...]) -> Any:
    """
    Reshape a Tensor.

    Args:
        a: Tensor object
        shape: New shape

    Returns:
        Tensor object with new shape
    """
    # TODO: Return a.reshape(shape)
    raise NotImplementedError("Implement run_tensor_reshape adapter")


def run_tensor_transpose(a: Any) -> Any:
    """
    Transpose a Tensor.

    Args:
        a: Tensor object

    Returns:
        Transposed Tensor object
    """
    # TODO: Return a.T or a.transpose()
    raise NotImplementedError("Implement run_tensor_transpose adapter")


# =============================================================================
# Part 5: Matrix Operations (17 points)
# =============================================================================

def run_matmul(a: Any, b: Any) -> Any:
    """
    Matrix multiply two Tensor objects.

    Args:
        a: Tensor object of shape (..., n, m)
        b: Tensor object of shape (..., m, k)

    Returns:
        Tensor object of shape (..., n, k)
    """
    # TODO: Return a @ b or matmul(a, b)
    raise NotImplementedError("Implement run_matmul adapter")


def run_tensor_relu(a: Any) -> Any:
    """Apply ReLU to a Tensor."""
    # TODO: Return a.relu() or relu(a)
    raise NotImplementedError("Implement run_tensor_relu adapter")


def run_tensor_exp(a: Any) -> Any:
    """Compute exp of a Tensor."""
    # TODO: Return a.exp() or exp(a)
    raise NotImplementedError("Implement run_tensor_exp adapter")


def run_tensor_log(a: Any) -> Any:
    """Compute log of a Tensor."""
    # TODO: Return a.log() or log(a)
    raise NotImplementedError("Implement run_tensor_log adapter")


# =============================================================================
# Part 6: Neural Network Training (15 points)
# =============================================================================

def create_linear_layer(in_features: int, out_features: int) -> Any:
    """
    Create a Linear layer instance.

    Args:
        in_features: Number of input features
        out_features: Number of output features

    Returns:
        Linear layer object with weight and bias initialized
    """
    # TODO: Import your Linear class and create an instance
    # from smulgrad.nn import Linear
    # return Linear(in_features, out_features)
    raise NotImplementedError("Implement create_linear_layer adapter")


def run_linear_layer(layer: Any, x: Any) -> Any:
    """
    Apply a linear layer to input.

    Args:
        layer: Linear layer object
        x: Input Tensor of shape (batch, in_features)

    Returns:
        Output Tensor of shape (batch, out_features)
    """
    # TODO: Return layer(x)
    raise NotImplementedError("Implement run_linear_layer adapter")


def get_linear_parameters(layer: Any) -> List[Any]:
    """
    Get the parameters (weight and bias) from a Linear layer.

    Args:
        layer: Linear layer object

    Returns:
        List of Tensor objects [weight, bias]
    """
    # TODO: Return [layer.weight, layer.bias]
    raise NotImplementedError("Implement get_linear_parameters adapter")


def run_softmax(a: Any, axis: int = -1) -> Any:
    """
    Compute softmax along specified axis.

    Args:
        a: Tensor object
        axis: Axis along which to compute softmax

    Returns:
        Tensor with softmax applied
    """
    # TODO: Implement or call your softmax function
    # from smulgrad.nn import softmax
    # return softmax(a, axis)
    raise NotImplementedError("Implement run_softmax adapter")


def run_cross_entropy(logits: Any, targets: Any) -> Any:
    """
    Compute cross-entropy loss from logits.

    Args:
        logits: Tensor of shape (batch, num_classes) - raw scores
        targets: Tensor of shape (batch, num_classes) - one-hot labels

    Returns:
        Scalar Tensor representing the mean cross-entropy loss
    """
    # TODO: Implement or call your cross-entropy function
    # from smulgrad.nn import cross_entropy
    # return cross_entropy(logits, targets)
    raise NotImplementedError("Implement run_cross_entropy adapter")


def run_sgd_step(parameters: List[Any], lr: float) -> None:
    """
    Perform one SGD update step.

    Args:
        parameters: List of Tensor objects to update
        lr: Learning rate
    """
    # TODO: Create SGD optimizer and call step()
    # from smulgrad.nn import SGD
    # optimizer = SGD(parameters, lr)
    # optimizer.step()
    raise NotImplementedError("Implement run_sgd_step adapter")


def zero_grad(parameters: List[Any]) -> None:
    """
    Zero out gradients for all parameters.

    Args:
        parameters: List of Tensor objects
    """
    # TODO: Create SGD optimizer and call zero_grad()
    # from smulgrad.nn import SGD
    # optimizer = SGD(parameters)
    # optimizer.zero_grad()
    raise NotImplementedError("Implement zero_grad adapter")


def create_mlp(input_size: int, hidden_size: int, output_size: int) -> Any:
    """
    Create an MLP (Multi-Layer Perceptron) with ReLU activation.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output classes

    Returns:
        An MLP object that:
            - Is callable: mlp(x) returns the output logits
            - Has a .parameters() method returning list of all weight/bias Tensors
            - Architecture: Linear(input->hidden) -> ReLU -> Linear(hidden->output)

    Example:
        mlp = create_mlp(2, 8, 2)  # 2 inputs, 8 hidden, 2 outputs
        logits = mlp(x)  # x is (batch, 2), logits is (batch, 2)
        params = mlp.parameters()  # List of Tensors
    """
    # TODO: Import your MLP class and create an instance
    # from smulgrad.nn import MLP
    # return MLP(input_size, hidden_size, output_size)
    raise NotImplementedError("Implement create_mlp adapter")


def run_mlp(mlp: Any, x: Any) -> Any:
    """
    Run forward pass through the MLP.

    Args:
        mlp: MLP object
        x: Input Tensor of shape (batch, input_features)

    Returns:
        Output Tensor of shape (batch, output_features) - raw logits
    """
    # TODO: Return mlp(x)
    raise NotImplementedError("Implement run_mlp adapter")


def get_mlp_parameters(mlp: Any) -> List[Any]:
    """
    Get all parameters from the MLP.

    Args:
        mlp: MLP object

    Returns:
        List of all weight and bias Tensors
    """
    # TODO: Return mlp.parameters()
    raise NotImplementedError("Implement get_mlp_parameters adapter")


def run_gradient_check(
    f: Callable,
    input: Any,
    eps: float = 1e-5,
    tol: float = 1e-4
) -> bool:
    """
    Check if analytical gradients match numerical gradients.

    Args:
        f: Function that takes a single Tensor and returns a scalar Tensor
        input: A single Tensor object to check gradients for
        eps: Epsilon for finite differences
        tol: Tolerance for gradient comparison

    Returns:
        True if all gradients match within tolerance
    """
    # TODO: Implement gradient checking
    # from smulgrad.nn import check_gradients
    # return check_gradients(f, input, eps, tol)
    raise NotImplementedError("Implement run_gradient_check adapter")


# =============================================================================
# Bonus: MNIST Training
# =============================================================================


def run_mnist_training(
    # TODO: Change these None values to achieve >95% test accuracy
    hidden_size: int = None,
    learning_rate: float = None,
    batch_size: int = None,
    epochs: int = None
) -> tuple:
    """
    Train an MLP on the MNIST dataset.

    Returns:
        Tuple of (trained_mlp, test_accuracy)
        - trained_mlp: The trained MLP model
        - test_accuracy: Float between 0 and 1
    """
    # from smulgrad.mnist import train_mnist
    # return train_mnist(hidden_size, learning_rate, batch_size, epochs)

    raise NotImplementedError(
        "Implement train_mnist in smulgrad/mnist.py and connect it here."
    )
