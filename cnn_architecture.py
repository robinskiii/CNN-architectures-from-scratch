from abc import abstractmethod
from collections.abc import Callable

import numpy as np

#####################################


class Layer:
    """
    Parent class for all types of layers.

    Attributes:
        input_shape: int | tuple[int, ...]
        output_shape: int | tuple[int, ...]

    """
    input_shape: int | tuple[int, ...]
    output_shape: int | tuple[int, ...]

    def __init__(self, input_shape: int | tuple, output_shape: int | tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        ...



class Dense(Layer):
    """
    Fully Connected Layer (basic NN architecture)

    Attributes:
        weights: np.ndarray
        bias: np.ndarray

    """
    input: np.ndarray

    weights: np.ndarray
    bias: np.ndarray

    output: np.ndarray

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
    input: np.ndarray
    output: np.ndarray

    def __init__(self, dim: int | tuple) -> None:
        super().__init__(dim, dim) # ReLU goes from n dimensions to n dimensions

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        # Derivative of ReLU is 1 if input > 0, else 0
        derivative = self.input > 0
        return output_error * derivative



class Sigmoid(Layer):
    """
    Sigmoid Activation Layer (binary classification)
    """
    input: np.ndarray
    output: np.ndarray

    def __init__(self, dim: int | tuple) -> None:
        super().__init__(dim, dim)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        clipped_input = np.clip(input_data, -500, 500)
        self.output = 1 / (1 + np.exp(-clipped_input))
        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        derivative = self.output * (1 - self.output)
        return output_error * derivative



class Convolution(Layer):
    """
    2D Convolutional Layer

    Data Shape: (channels, height, width)
    Applies a given amount of matrix filters accross 2D data (treats RGB (channels data) independantly in image)

    Attributes:
        weights: np.ndarray   (4D data: (filters, channels, filter_height, filter_width))
        bias: np.ndarray      (1D data: one bias value per filter)
        filters: int
        filter_size: int      (default = 3)
        stride: int           (default = 1)
        padding: int          (default = 1)

    """
    input : np.ndarray
    input_padded: np.ndarray
    output: np.ndarray

    weights: np.ndarray
    bias: np.ndarray

    filters: int
    filter_size: int
    stride: int
    padding: int

    def __init__(self, input_shape: tuple[int, int, int], filters: int, filter_size: int = 3, stride: int = 1) -> None:

        if filters < 1:
            raise ValueError("Convolution layer must contain at least one filter")

        if (filter_size % 2 == 0) or (filter_size < 3):
            raise ValueError("Convolution filter size must be odd (3x3, 5x5, etc...)")

        if stride < 1:
            raise ValueError("Convolution stride must be at least 1")

        # Calculate output dimensions
        channels, in_height, in_width = input_shape
        out_height = ((in_height - 1) // stride) + 1 # euclidian division to see how many strides fit in the input dimension
        out_width = ((in_width - 1) // stride) + 1
        output_shape = (filters, out_height, out_width)

        super().__init__(input_shape, output_shape)

        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = int((filter_size - 1) / 2) # int since filter size is uneven

        # He initialization for weights: (filters, channels, filter_height, filter_width)
        num_input_variables = channels * (filter_size ** 2)
        self.weights = np.random.randn(filters, channels, filter_size, filter_size) * np.sqrt(2 / num_input_variables)
        self.bias = np.zeros((filters, 1, 1))


    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return #todo

    def backwards(self, #to do



class Flatten(Layer):
    pass                                                                                    # TO DO




class Model:
    """
    Neural Network composed of a sequence of Layers
    """
    layers: list[Layer]
    loss_function: Callable
    loss_derivative: Callable

    def __init__(self) -> None:
        self.layers = []

    def add(self, layer : Layer) -> None:

        # check input dimensions match output dimensions of previous layer
        if self.layers and (self.layers[-1].output_shape != layer.input_shape):
                raise ValueError

        self.layers.append(layer)

    def set_loss_function(self, loss_function: Callable, loss_derivative: Callable) -> None:
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative

    def predict(self, input_data: np.ndarray) -> np.ndarray:

        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float) -> None:

        for i in range(epochs):

            # forward prop
            output = x_train
            for layer in self.layers:
                output = layer.forward(output)

            # track the loss
            cost = self.loss_function(y_train, output)

            # back prop
            error = self.loss_derivative(y_train, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)

            print("Epoch: ",(6-len(str(i+1)))*" " ,f"{i+1} out of {epochs}",10*" ",f"cost: {cost} ") # i aknowledge this is somewhat overkill



def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error loss function
    """
    return np.mean(np.power(y_true - y_pred, 2))

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size



def bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary cross entropy loss function
    """
    epsilon = 1e-15 # to prevent undefined log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)



if __name__ == "__main__":
    np.random.seed(0)

    batch_size = 32
    input_dim = 4
    output_dim = 1

    # Random test data
    X_train = np.random.randn(batch_size, input_dim)
    y_train = np.random.randint(2, size=(batch_size, 1))

    # Initialize network
    model = Model()
    model.add(Dense(input_dim, 10))
    model.add(ReLU(10))
    model.add(Dense(10, 6))
    model.add(ReLU(6))
    model.add(Dense(6, 5))
    model.add(ReLU(5))
    model.add(Dense(5, 1))
    model.add(Sigmoid(output_dim))

    model.set_loss_function(bce, bce_derivative)

    print("\n",65*"=")
    model.train(X_train, y_train, epochs=10000, learning_rate=0.01)
    print(65*"=","\n")
    # print(model.predict(X_train))
