import numpy as np
import math


class Value:

    def __init__(self, data, _op="", _children=()):
        self.data = data
        self.grad = 0
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, _op="+", _children=(self, other))

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, _op="*", _children=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other - self

    def __neg__(self):
        out = Value(-self.data, _op='-', _children=(self, ))

        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (float, int)
                          ), "only supports int/float exponents"
        out = Value(self.data**other, _op='**', _children=(self, ))

        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**(-1)

    def __rtruediv__(self, other):
        return (self/other) ** -1

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):

            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def relu(self):

        value = max(self.data, 0)

        out = Value(value, _op="relu", _children=(self,))

        def _backward():
            if value > 0:
                self.grad += out.grad

        out._backward = _backward

        return out

    def exp(self):

        value = math.exp(self.data)
        out = Value(value, _op="exp", _children=(self,))

        def _backward():
            self.grad += value * out.grad
        out._backward = _backward

        return out

    def log(self):

        value = math.log(self.data)
        out = Value(value, _op="log", _children=(self,))

        def _backward():
            self.grad += (1/self.data) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        value = math.tanh(self.data)
        out = Value(value, _op="tanh", _children=(self,))

        def _backward():
            self.grad += (1-value**2) * out.grad

        out._backward = _backward

        return out


class Tensor():

    def __init__(self, data, _op="", _children=()):
        self.data = data
        self.grad = np.zeros_like(data)
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    @staticmethod
    def _unbroadcast(grad, child):
        out_shape = grad.shape
        child_data = np.asarray(child.data)
        in_shape = child_data.shape
        ndim_diff = len(out_shape) - len(in_shape)
        in_shape = (1,) * ndim_diff + in_shape

        axes = [i for i, (g, s) in enumerate(zip(out_shape, in_shape))
                if s == 1 and g != 1]

        if axes:
            grad = grad.sum(axis=tuple(axes), keepdims=True)

        return grad.reshape(in_shape[ndim_diff:])

    def _broadcast(self, grad, axes):
        if axes:
            grad = np.expand_dims(grad, axes)
        else:
            additional_dims = self.data.ndim - grad.ndim
            grad = grad.reshape(grad.shape + (1,)*additional_dims)

        return grad

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data + other.data, _op="+", _children=(self, other))

        def _backward():
            self.grad += self._unbroadcast(1 * out.grad, self)
            other.grad += self._unbroadcast(1 * out.grad, other)
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data * other.data, _op="*", _children=(self, other))

        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self)
            other.grad += self._unbroadcast(self.data * out.grad, other)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other - self

    def __neg__(self):
        out = Tensor(-self.data, _op='-', _children=(self, ))

        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (float, int)
                          ), "only supports int/float exponents"
        out = Tensor(self.data**other, _op='**', _children=(self, ))

        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**(-1)

    def __rtruediv__(self, other):
        return (self/other) ** -1

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):

            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def relu(self):

        value = np.maximum(self.data, 0)

        out = Tensor(value, _op="relu", _children=(self,))

        def _backward():
            self.grad += np.where(value > 0, out.grad, 0)

        out._backward = _backward

        return out

    def exp(self):

        value = np.exp(self.data)
        out = Tensor(value, _op="exp", _children=(self,))

        def _backward():
            self.grad += value * out.grad
        out._backward = _backward

        return out

    def log(self):

        value = np.log(self.data)
        out = Tensor(value, _op="log", _children=(self,))

        def _backward():
            self.grad += (1/self.data) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, _op="tanh", _children=(self,))

        def _backward():
            self.grad += (1-value**2) * out.grad

        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        value = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(value, _op="sum", _children=(self,))

        def _backward():
            if keepdims is False:
                self.grad += self._broadcast(out.grad, axis)
            else:
                self.grad += out.grad
        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        value = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(value, _op="mean", _children=(self,))

        def _backward():
            if keepdims is False:
                self.grad += (1/np.size(self.data, axis=axis)) * \
                    self._broadcast(out.grad, axis)
            else:
                self.grad += (1/np.size(self.data, axis=axis)) * out.grad

        out._backward = _backward

        return out

    def max(self, axis=None, keepdims=False):
        value = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(value, _op="max", _children=(self,))

        def _backward():
            mask = self.data == value
            count = mask.sum(axis=axis, keepdims=True)
            if keepdims is False:
                gradient = self._broadcast(out.grad, axis)
            else:
                gradient = out.grad
            self.grad += gradient * (mask/count)
        out._backward = _backward

        return out

    def reshape(self, shape):
        value = np.reshape(self.data, shape)
        out = Tensor(value, _op="reshape", _children=(self,))

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward

        return out

    def transpose(self, axes=None):
        value = np.transpose(self.data, axes)
        out = Tensor(value, _op="transpose", _children=(self,))

        def _backward():
            self.grad += out.grad.transpose(axes)
        out._backward = _backward

        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data @ other.data, _op="matmul",
                     _children=(self, other))

        # C = A @ x: (n,m @ m, -> n,
        # for dC/dA: gradient needs shape (n,m)
        # So we need to calculate n,1 @ 1,m
        # for dC/dx: gradient needs shape m,
        # So we need to calculate m,n @ n,1

        if other.data.ndim == 1:

            def _backward():
                self.grad += out.grad[:, None] @ other.data[None, :]
                other.grad += np.squeeze(self.data.swapaxes(-1, -2)
                                         @ out.grad[:, None], axis=-1)

            out._backward = _backward

        # C =  x @ A: (n, @ n,m -> m, )
        # for dC/dx: gradient needs shape (n,)
        # So we need to calculate 1,m @ m, n
        # for dC/dA: gradient needs shape (n,m)
        # So we need to calulcate n,1 @ 1,m

        elif self.data.ndim == 1:

            def _backward():
                self.grad += np.squeeze(out.grad[None, :]
                                        @ other.data.swapaxes(-1, -2), axis=-2)
                other.grad += self.data[:, None] @ out.grad[None, :]

            out._backward = _backward

        # A @ A
        else:

            def _backward():
                self.grad += self._unbroadcast(out.grad @
                                               other.data.swapaxes(-1, -2), self)
                other.grad += self._unbroadcast(
                    self.data.swapaxes(-1, -2) @ out.grad, other)

            out._backward = _backward

        return out

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return other @ self
