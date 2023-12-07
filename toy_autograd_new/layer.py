from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Layer(ABC):
    """Abstract layer class"""
    def __init__(self, input_shape: Tuple = None):
        self.__input_shape = input_shape
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_data):
        """
        Computes a forward pass
        :param input_data:
        :return: layer output
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, learning_rate):
        """
        Computes a backward pass, updates parameters
        :param output_error: de/dY - error with respect to the outputs
        :param learning_rate: learning rate scale
        :return: de/dX - error with respect to the inputs
        """
        raise NotImplementedError

    @property
    def input_shape(self):
        """Input shape of the layer"""
        return self.__input_shape

    @property
    @abstractmethod
    def output_shape(self):
        """Output shape of the layer"""
        raise NotImplementedError


class FC(Layer):
    """A fully connected layer"""
    def __init__(self, input_shape: Tuple):
        super().__init__(input_shape)
        self.weights = np.random.randn(*self.input_shape)
        self.bias = np.random.randn(1, self.input_shape[1])

    def forward(self, input_data):
        """
        Computes a forward pass
        :param input_data:
        :return: layer output
        """
        self.input = input_data
        # Y = X dot W + B
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Computes a backward pass, updates parameters
        :param output_error: de/dY - error with respect to the outputs
        :param learning_rate: learning rate scale
        :return: de/dX - error with respect to the inputs
        """
        # dE/dW = X.T dot dE/dY
        de_dw = np.dot(self.input.T, output_error)
        # dE/dX = dE/dY dot W.T
        de_dx = np.dot(output_error, self.weights.T)
        # dE/dB = dE/dY
        de_db = output_error
        # parameters update
        self.weights -= de_dw * learning_rate
        self.bias -= de_db * learning_rate
        # Returns input error (output error for previous layer)
        return de_dx

    @property
    def output_shape(self):
        """Output shape of the layer"""
        return 1, self.input_shape[1]


class Activation(Layer):
    """Activation layer"""
    def __init__(self, activation, activation_prime, input_shape):
        super().__init__(input_shape)
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        """
        Computes a forward pass
        :param input_data:
        :return: layer output
        """
        self.input = input_data
        # Y = [f(x1), f(x2), ..., f(xi)]
        self.output = self.activation(input_data)
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Computes a backward pass
        :param output_error: de/dY - error with respect to the outputs
        :param learning_rate: learning rate scale
        :return: de/dX - error with respect to the inputs
        """
        # dE/dX = dE/dY * [f'(x1), f'(x2), ..., f'(xi)]
        return output_error * self.activation_prime(self.input)

    @property
    def output_shape(self):
        """Output shape of the layer"""
        return self.input_shape


class FlattenLayer(Layer):
    """A flatten layer"""
    def forward(self, input_data):
        """
        Computes a forward pass
        (flattens the inputs)
        :param input_data:
        :return: layer output
        """
        return np.reshape(input_data, (1, -1))

    def backward(self, output_error, learning_rate):
        """
        Computes a backward pass
        (re-shapes the output to the original shape)
        :param output_error: de/dY - error with respect to the outputs
        :param learning_rate: learning rate scale
        :return: de/dX - error with respect to the inputs
        """
        return np.reshape(output_error, self.input_shape)

    @property
    def output_shape(self):
        """Output shape of the layer"""
        return 1, np.prod(self.input_shape)


class SoftmaxLayer(Layer):
    """A flatten layer"""
    def forward(self, input_data):
        """
        Computes a forward pass
        (flattens the inputs)
        :param input_data:
        :return: layer output
        """
        self.input = input_data

        assert len(self.input.shape) == 2
        s = np.max(self.input, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(self.input - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        self.output = e_x / div

        return self.output

    def backward(self, output_error, learning_rate):
        """
        Computes a backward pass
        (re-shapes the output to the original shape)
        :param output_error: de/dY - error with respect to the outputs
        :param learning_rate: learning rate scale
        :return: de/dX - error with respect to the inputs
        """
        out = np.tile(self.output.T, self.input_shape[0])
        return self.output * np.dot(output_error, np.identity(self.input_shape[0]) - out)

    @property
    def output_shape(self):
        """Output shape of the layer"""
        return 1, self.input_shape[0]
