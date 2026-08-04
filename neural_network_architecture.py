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



class ReLU(Layer):
    """
    Activation Layer
    """

    def forward(self, input_data):

        return np.maximum(0, input_data)

    def backward(self, output_error, learning_rate):
        pass                                                # TO DO



class Convolution(Layer):
    pass                                                    # TO DO



class Flatten(Layer):
    pass                                                    # TO DO



class Model:
    """
    Neural Network composed of a sequence of Layers
    """
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        pass                                                          # TO DO

    def train(self, x_train, y_train, epochs, learning_rate):
        pass                                                          # TO DO



def loss(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))



def loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size



if __name__ == "__main__":
    # Initialize network
    model = Model()
