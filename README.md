# SmulGrad

A hands-on learning repository for implementing automatic differentiation from scratch. Inspired by Karpathy's micrograd and Stanford CS336's assignment style.

## Overview

This repository provides a structured learning path to understand and implement reverse-mode automatic differentiation (backpropagation). You will build everything from scratch - no scaffolding, no starter code - just comprehensive tests and clear instructions.

**What you will build:**
- A `Value` class for scalar autodiff (like micrograd)
- A `Tensor` class for multi-dimensional autodiff (like PyTorch's autograd)
- Support for broadcasting, reductions, and matrix operations
- A working neural network that you can train

## Quick Start

```bash
# Clone the repository
cd SmulGrad

# Install dependencies with uv
uv sync

# Run all tests (they will fail - that's expected!)
uv run pytest

# Run tests for a specific part
uv run pytest -k "part1"

# Run a single test
uv run pytest -k "test_value_creation"
```

## Structure

```
SmulGrad/
    assignment.pdf       # Detailed instructions (START HERE)
    assignment.tex       # LaTeX source
    pyproject.toml       # Project configuration
    README.md            # This file
    smulgrad/
        engine.py        # Value and Tensor classes (Parts 1-5)
        nn.py            # Neural network utilities (Part 6)
    tests/
        adapters.py      # Connect your code to tests (edit this)
        cases/           # Test files (do not modify)
            conftest.py
            test_part1_scalar.py
            test_part2_backward.py
            test_part3_ops.py
            test_part4_tensor.py
            test_part5_matrix.py
            test_part6_nn.py
```

## Assignment Parts

| Part | Topic | Points | Description |
|------|-------|--------|-------------|
| 1 | Scalars | 15 | Value class, basic arithmetic |
| 2 | Backward | 20 | Gradient computation, topological sort |
| 3 | Operations | 15 | ReLU, exp, log, tanh |
| 4 | Tensors | 20 | Multi-dimensional arrays, broadcasting |
| 5 | Matrix Ops | 15 | matmul, batched operations |
| 6 | Neural Networks | 15 | softmax, cross-entropy, SGD, training |

## How to Work

1. **Read `assignment.pdf`** - Detailed instructions for each part
2. **Implement in `smulgrad/engine.py`** - Value and Tensor classes (Parts 1-5)
3. **Implement in `smulgrad/nn.py`** - Neural network utilities (Part 6)
4. **Connect via `tests/adapters.py`** - Wire your implementation to tests
5. **Run tests to verify** - Each test corresponds to a problem

## Example Workflow

After implementing the `Value` class in `smulgrad/engine.py`:

```python
# In tests/adapters.py
def create_value(data: float):
    from smulgrad.engine import Value
    return Value(data)

def get_value_data(v):
    return v.data

def get_value_grad(v):
    return v.grad

def run_add(a, b):
    return a + b
# ... etc
```

Then run:
```bash
uv run pytest -k "test_value_creation"
# PASSED!
```

## Tips

- Start with Part 1 - get scalar operations working first
- Use numerical gradient checking to verify your implementation
- Read the test code - it shows exactly what behavior is expected
- The assignment.pdf has detailed explanations of the math

## Grading

Run all tests to see your score:
```bash
uv run pytest --tb=no -q
```

Each passing test contributes to your total score (100 points).

## References

- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [PyTorch autograd documentation](https://pytorch.org/docs/stable/autograd.html)
- [Stanford CS231n backprop notes](https://cs231n.github.io/optimization-2/)
- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey"

## License

MIT - Feel free to use this for learning!
