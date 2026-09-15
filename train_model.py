import time
from collections.abc import Callable

import numpy as np

from layer_architecture import (
    Convolution,
    Dense,
    Flatten,
    Layer,
    MaxPooling,
    ReLU,
    Sigmoid,
    Softmax,  # noqa: F401
)


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
                raise ValueError("Dimensions do not match between layers.")

        self.layers.append(layer)

    def set_loss_function(self, loss_function: Callable, loss_derivative: Callable) -> None:
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative

    def predict(self, input_data: np.ndarray) -> np.ndarray:

        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float, batch_size: int = 32) -> None:
        """
        Updating the model parameters using mini-batch gradient descent
        """
        num_samples = x_train.shape[0]

        for i in range(epochs):

            # shuffle data before each epoch
            indices = np.random.permutation(num_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_cost = 0
            num_batches = 0

            for j in range(0, num_samples, batch_size): # loop over the dataset in chunks of batch_size
                x_batch = x_shuffled[j : j+batch_size]
                y_batch = y_shuffled[j : j+batch_size]

                # Data augmentation for image batches (during training to reduce memory usage (no need to store mirror images))
                # Expected image layout is (batch, channels, height, width).
                if x_batch.ndim == 4:
                    flip_mask = np.random.rand(x_batch.shape[0]) < 0.5
                    if np.any(flip_mask):
                        x_batch = x_batch.copy() # shallow copy
                        x_batch[flip_mask] = x_batch[flip_mask, :, :, ::-1] # horizontal flip

                # forward prop
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # track the loss
                batch_cost = self.loss_function(y_batch, output)
                epoch_cost += batch_cost
                num_batches += 1

                # back prop
                error = self.loss_derivative(y_batch, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            epoch_cost /= num_batches

            print("Epoch: ",(6-len(str(i+1)))*" " ,f"{i+1} out of {epochs}",10*" ",f"cost: {epoch_cost} ") # i aknowledge this is somewhat overkill



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



def cce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Categorical cross entropy loss function (for Multi-Class Classification)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # sum over classes, mean average over batch
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def cce_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # return error matrix scaled by batch size
    return - (y_true / y_pred) / y_true.shape[0] # derivative of log(y_pred) is 1/y_pred

# ADD REGULARIZATION !!!

if __name__ == "__main__":
    np.random.seed(0)

    # Data dimensions
    total_samples = 128
    channels = 3         # RGB
    image_height = 218
    image_width = 178

    input_shape = (channels, image_height, image_width)

    # Random training data
    X_train = np.random.randn(total_samples, channels, image_height, image_width)
    # Random binary labels
    y_train = np.random.randint(2, size=(total_samples, 1))


    # ---- MODEL ----

    model = Model()

    model.add(Convolution(input_shape, filters=8))          # (3,218,178) -> (8,218,178)
    model.add(ReLU((8, image_height, image_width)))
    model.add(MaxPooling((8, image_height, image_width)))   # (8,218,178) -> (8,109,89)

    model.add(Convolution((8, 109, 89), filters=16))        # (8,109,89) -> (16,109,89)
    model.add(ReLU((16, 109, 89)))
    model.add(MaxPooling((16, 109, 89)))                    # (16,109,89) -> (16,55,45)

    model.add(Convolution((16, 55, 45), filters=32))        # (16,55,45) -> (32,55,45)
    model.add(ReLU((32, 55, 45)))
    model.add(MaxPooling((32, 55, 45)))                     # (32,55,45) -> (32,28,23)

    model.add(Convolution((32, 28, 23), filters=32))        # (32,28,23) -> (32,28,23)
    model.add(ReLU((32, 28, 23)))
    model.add(MaxPooling((32, 28, 23)))                     # (32,28,23) -> (32,14,12)

    model.add(Flatten((32, 14, 12)))                        # (32,14,12) -> (5376)

    model.add(Dense(5376, 128))                             # (5376) -> (128)
    model.add(ReLU(128))
    model.add(Dense(128, 1))                                # (128) -> (1)

    model.add(Sigmoid(1))

    model.set_loss_function(bce, bce_derivative)

    print("\n")
    print(67*"=")
    print("Training Model...")

    start = time.perf_counter()

    model.train(X_train, y_train, epochs=100, learning_rate=0.0001)
    print(67*"=", "\n")

    end = time.perf_counter()

    print(f"Time spent training model: {end-start} seconds\n")

    print(67*"=", "\n")

    print("Predictions vs true labels:")

    y_pred = model.predict(X_train[:6])

    print(np.column_stack((y_train[:6],y_pred)).T)

    print("\n")
    print(67*"=", "\n")
