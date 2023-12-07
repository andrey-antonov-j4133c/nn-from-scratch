import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0.0, x)


def relu_prime(x):
    return np.heaviside(x, 0.0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
