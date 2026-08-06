from abc import abstractmethod

import numpy as np

#####################################


class Layer:
    """

    Parent class for all types of layers.

    Attributes:
        input_dim: int
        output_dim: int
    """

    input_dim: int
    output_dim: int

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        ...



class Dense(Layer):
    """
    Fully Connected Layer (basic NN architecture)
    """
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, input_dim: int, output_dim: int) -> None:

        super().__init__(input_dim, output_dim)

        # He initialization
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((1, output_dim))

    def forward(self, input_data: np.ndarray) -> np.ndarray:

        self.input = input_data
        # Y = X.W + b
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:

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
    def __init__(self, dim: int) -> None:
        super().__init__(dim, dim) # ReLU goes from n dimensions to n dimensions

    def forward(self, input_data: np.ndarray) -> np.ndarray:

        return np.maximum(0, input_data)

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        return np.zeros(0)                                                                 # TO DO



class Convolution(Layer):
    pass                                                                                   # TO DO



class Flatten(Layer):
    pass                                                                                    # TO DO



class Model:
    """
    Neural Network composed of a sequence of Layers
    """
    layers: list[Layer]

    def __init__(self) -> None:
        self.layers = []

    def add(self, layer : Layer) -> None:

        # check input dimensions match output dimensions of previous layer
        if self.layers and (self.layers[-1].output_dim != layer.input_dim):
                raise ValueError

        self.layers.append(layer)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        result = np.array([])
        for i in range(input_data.shape[0]):
            output = np.array([input_data[i]])

            for layer in self.layers:
                output = layer.forward(output)
                print(output)

            result =np.append(result, output)
        return result

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float) -> None:
        pass                                                                                                      # TO DO



def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.power(y_true - y_pred, 2))



def loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size



if __name__ == "__main__":
    # Initialize network
    model = Model()
    model.add(Dense(2, 3))
    model.add(ReLU(3))
    model.add(Dense(3, 1))


    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    test_labels = np.array([0, 1, 1, 0])

    model.predict(test_data)
