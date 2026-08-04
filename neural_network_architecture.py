from abc import abstractmethod

import numpy as np

#####################################


class Layer:
    """
    Parent class for all types of layers.
    """

    @abstractmethod
    def forward(self, input_data):
        ...

    def backward(self, output_error: float, learning_rate: float):
        ...



class Dense(Layer):
    """
    Fully Connected Layer (basic NN architecture)
    """
    def __init__(self, input_size, output_size):

        super().__init__()

        # He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):

        self.input = input_data
        # Y = X.W + b
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    def backward(self, output_error, learning_rate):

        # Calculate gradients
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # Calculate error to pass to previous layer
        input_error = np.dot(output_error, self.weights.T)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        return input_error
