import enum
from typing import List, Tuple

import numpy as np
import sklearn

from toy_autograd_new import layer, activations, errors


class LayersEnum(enum.Enum):
    """Layers for the model"""
    flatten = layer.FlattenLayer
    fully_connected = layer.FC
    softmax = layer.SoftmaxLayer


class ActivationEnum(enum.Enum):
    """Layer activation functions"""
    relu = (activations.relu, activations.relu_prime)
    sigmoid = (activations.sigmoid, activations.sigmoid_prime)
    tanh = (activations.tanh, activations.tanh_prime)


class ErrorEnum(enum.Enum):
    """Model error"""
    mean_square = (errors.mse, errors.mse_prime)


class Model:
    """A simple sequential model"""
    def __init__(self, error: ErrorEnum = ErrorEnum.mean_square):
        self.__layers: List[layer.Layer] = []
        self.__error, self.__error_prime = error.value

    def add_layer(
            self,
            layer_cls: LayersEnum,
            input_shape: Tuple = None,
            num_units: int = None,
            activation: ActivationEnum = None):
        """
        Add layer to the end of the model
        :param layer_cls: Layer
        :param input_shape: input shape (excludes num_units)
        :param num_units: number of layer units (takes input shape from previous layer)
        :param activation: activation function fo the layer
        :return: None
        """
        # Check rules
        if input_shape and num_units:
            raise ValueError("You have to specify ether 'input_shape' or 'num_units', not both")
        layer_cls = layer_cls.value
        if layer_cls not in [itm.value for itm in LayersEnum]:
            raise ValueError(f"Unsupported layer type {layer_cls.__name__}")
        if not self.__layers and not input_shape:
            raise ValueError("Input shape not specified for the first layer")
        # Calculate input shape
        if num_units:
            input_shape = (self.__layers[-1].output_shape[1], num_units)
        # Add the specified layer
        self.__layers.append(layer_cls(input_shape=input_shape))
        # Add an activation layer with the specified activation function
        if activation:
            a_forward, a_prime = activation.value
            self.__layers.append(
                layer.Activation(
                    a_forward,
                    a_prime,
                    input_shape=self.__layers[-1].output_shape)
            )

    def forward(self, x):
        """
        Computes a forward pass of all layers
        :param x: Input
        :return: Model outputs
        """
        output = x
        for l in self.__layers:
            output = l.forward(output)
        return output

    def backward(self, y_true, y_pred, learning_rate=0.01):
        """
        Computes a backward pass of all layers.
        :param y_true: Ground truth.
        :param y_pred: Model outputs.
        :param learning_rate: Learning rate.
        :return: none.
        """
        error = self.__error_prime(y_true, y_pred)
        for l in reversed(self.__layers):
            error = l.backward(error, learning_rate)

    def fit(self, X, Y, num_epochs=10, learning_rate=0.01):
        """
        Fit function for the model.
        Updates nodel weights and biases for the specified number of epochs.
        :param X: Training inputs.
        :param Y: Training outputs.
        :param num_epochs: Number of epochs.
        :param learning_rate: Learning rate.
        :return: none.
        """
        for e in range(num_epochs):
            x_shuffled, y_shuffled = sklearn.utils.shuffle(X, Y, random_state=np.random.randint(100))
            err = .0
            for x, y in zip(x_shuffled, y_shuffled):
                output = self.forward(x)
                err += errors.mse(y, output)
                self.backward(y, output, learning_rate)
            err /= len(X)
            if (e + 1) % 10 == 0:
                print(f'{e + 1}/{num_epochs}, error={err}')

    def summary(self):
        """Prints basic model info"""
        for l in self.__layers:
            print(f"{l.__class__.__name__}: input - {l.input_shape}; output - {l.output_shape}")

    @property
    def layers(self):
        return self.__layers
