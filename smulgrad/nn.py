from smulgrad.engine import Tensor
import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(
            (np.random.rand(in_features, out_features))*0.01,)
        self.bias = Tensor(np.zeros(out_features))

    def __call__(self, x):

        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight.data, self.bias.data]


def softmax(x, axis):
    x = (x-x.max(axis=axis, keepdims=True)).exp()

    return x/x.sum(axis=axis, keepdims=True)


def cross_entropy(logits, targets):
    loss = (logits.exp().sum(axis=-1).log() -
            (targets * logits).sum(axis=-1)).mean()
    return loss


class SGD:

    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)


class MLP:

    def __init__(self, input_size, hidden_size, output_size):

        self.linear1 = Linear(input_size, hidden_size)
        self.linear2 = Linear(hidden_size, output_size)

    def __call__(self, x):

        return self.linear2(self.linear1(x).relu())

    def parameters(self):
        params_layer1 = [self.linear1.weight, self.linear1.bias]
        params_layer2 = [self.linear2.weight, self.linear2.bias]

        return params_layer1 + params_layer2


def check_gradients(f, inputs, epsilon=1e-5, tolerance=1e-4):

    x = f(inputs)
    x.backward()

    analytical = inputs.grad

    numerical = np.zeros_like(inputs.data)
    for i in range(inputs.data.size):
        idx = np.unravel_index(i, inputs.data.shape)
        inputs.data[idx] += epsilon
        f_plus = f(inputs).data
        inputs.data[idx] -= epsilon

        inputs.data[idx] -= epsilon
        f_minus = f(inputs).data
        inputs.data[idx] += epsilon

        numerical[idx] = (f_plus - f_minus) / (2*epsilon)

    rel_error = np.abs(analytical - numerical) / \
        (np.maximum(analytical, numerical) + epsilon)

    return np.all(rel_error <= tolerance)
