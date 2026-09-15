from abc import abstractmethod

import numpy as np


class Layer:
    """
    Parent class for all types of layers.

    Attributes:
        input_shape: int | tuple[int, ...]
        output_shape: int | tuple[int, ...]

    """
    input_shape: int | tuple[int, ...]
    output_shape: int | tuple[int, ...]

    input: np.ndarray
    output: np.ndarray

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
        weights: np.ndarray (2D data)
        bias: np.ndarray    (1D data)

    """
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, input_dim: int, output_dim: int) -> None:

        super().__init__(input_dim, output_dim)

        # He initialisation
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



class Softmax(Layer):
    """
    Softmax Activation Layer (multiclass classification)
    """
    def __init__(self, dim: int | tuple) -> None:
        super().__init__(dim, dim)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data

        # shifting all values by a constant (max value in this case)
        # result will stay the same but with less exponential compute (smaller values)
        shifted_input = input_data - np.max(input_data, axis=-1, keepdims=True)

        exps = np.exp(shifted_input)
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        input_error = np.empty_like(output_error)
        # softmax derivative requires a Jacobian matrix calculated per sample in the batch
        for i in range(self.output.shape[0]):
            # Flatten to 1D, then create column vector
            s = self.output[i].reshape(-1, 1)
            # Jacobian matrix calculation: diag(s) - s * s.T
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            # Multiply Jacobian by the error to pass backwards
            input_error[i] = np.dot(jacobian, output_error[i])
        return input_error



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
        padding: int          (calculated such that output has same dimensions as input)

    """
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]

    input: np.ndarray
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
        self.padding = int((filter_size - 1) / 2) # int since filter size is odd

        # He initialisation for weights: (filters, channels, filter_height, filter_width)
        num_weights = channels * (filter_size ** 2)
        self.weights = np.random.randn(filters, channels, filter_size, filter_size) * np.sqrt(2 / num_weights)
        self.bias = np.zeros((filters, 1, 1))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        batch_size = input_data.shape[0]

        # Applying padding to the spatial dimensions (height and width)
        self.input_padded = np.pad(
            input_data,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant'
        )

        # Initialising output tensor to reduce compute (avoiding .append())
        _, out_height, out_width = self.output_shape
        self.output = np.zeros((batch_size, self.filters, out_height, out_width))

        # Convolution: iterating over each pixel
        for i in range(out_height):
            for j in range(out_width):
                height_start = i * self.stride
                height_end = height_start + self.filter_size
                width_start = j * self.stride
                width_end = width_start + self.filter_size

                # Shape: (batch_size, channels, filter_size, filter_size)
                patch = self.input_padded[:, :, height_start : height_end, width_start : width_end]

                # Vectorised dot product over the batch using tensordot
                # Result shape: (batch_size, num_filters)
                self.output[:, :, i, j] = np.tensordot(patch, self.weights, axes=([1, 2, 3], [1, 2, 3])) + self.bias.flatten()

        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        _, out_height, out_width = self.output_shape

        # Initialising gradients
        weights_error = np.zeros_like(self.weights) # same shape as weights tensor
        input_error_padded = np.zeros_like(self.input_padded)

        # Computing gradients
        for i in range(out_height):
            for j in range(out_width):
                height_start = i * self.stride
                height_end = height_start + self.filter_size
                width_start = j * self.stride
                width_end = width_start + self.filter_size

                # The patch used during the forward pass
                patch = self.input_padded[:, :, height_start:height_end, width_start:width_end]

                # Error for this specific spatial location: (batch_size, num_filters)
                err = output_error[:, :, i, j]

                # Accumulate weight gradients
                weights_error += np.tensordot(err, patch, axes=([0], [0]))

                input_error_padded[:, :, height_start : height_end, width_start : width_end] += np.tensordot(err, self.weights, axes=([1], [0]))

        # Bias error: sum of output errors across the batch and height/width dimensions
        bias_error = np.sum(output_error, axis=(0, 2, 3)).reshape(self.filters, 1, 1)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        # Removing padding from the input error to match input shape
        input_error = input_error_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return input_error



class MaxPooling(Layer):
    """
    2D Max Pooling Layer

    Reduces spatial dimensions (height and width) by taking the maximum
    value over a sliding window.

    Attributes:
        pool_size: int   (default = 2 (for 2x2 windows))
    """
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]

    pool_size: int

    input: np.ndarray
    input_padded: np.ndarray
    output: np.ndarray


    def __init__(self, input_shape: tuple[int, int, int], pool_size: int = 2) -> None:
        self.input_shape = input_shape
        self.pool_size = pool_size

        channels, in_height, in_width = input_shape

        # Output dimensions (rounded up to include incomplete patches)
        out_height = ((in_height-1) // pool_size) + 1
        out_width = ((in_width-1)  // pool_size) +1
        self.output_shape = channels, out_height, out_width

        super().__init__(self.input_shape, self.output_shape)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        batch_size, channels, in_height, in_width = input_data.shape
        _, out_height, out_width = self.output_shape

        # if padding is needed (when pool size doesn't divide input dimension)
        pad_height = (out_height * self.pool_size) - in_height
        pad_width = (out_width * self.pool_size) - in_width

        self.input_padded = np.pad(
            input_data,
            pad_width=((0, 0), (0, 0), (0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=-np.inf # to be sure not to interfere in the max calculation
        )

        self.output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                height_start = i * self.pool_size
                height_end = height_start + self.pool_size
                width_start = j * self.pool_size
                width_end = width_start + self.pool_size

                # Extract the patch and find the maximum value across spatial dimensions
                patch = self.input_padded[:, :, height_start : height_end, width_start : width_end]
                self.output[:, :, i, j] = np.max(patch, axis=(2, 3))

        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        _, out_height, out_width = self.output_shape
        _, in_height, in_width = self.input_shape

        input_padded_error = np.zeros_like(self.input_padded)

        for i in range(out_height):
            for j in range(out_width):
                height_start = i * self.pool_size
                height_end = height_start + self.pool_size
                width_start = j * self.pool_size
                width_end = width_start + self.pool_size

                patch = self.input_padded[:, :, height_start : height_end, width_start : width_end]

                # Create a mask of the maximum values in the patch
                # We keepdims to allow comparing against the original patch
                max_val = np.max(patch, axis=(2, 3), keepdims=True)
                mask = (patch == max_val) # Only True in the location of max value

                # Route the error back only to the pixels that were the maximum
                # We use keepdims=True on output_error slice to broadcast properly
                err = output_error[:, :, i:i+1, j:j+1]
                # We patch in the error
                input_padded_error[:, :, height_start : height_end, width_start : width_end] += mask * err

        input_error = input_padded_error[:, :, :in_height, :in_width]

        return input_error



class Flatten(Layer):
    """
    Flattens 4D image data into a 2D matrix (batch size, features)
    To link convolution layer to dense layer
    """
    input_batch_shape: tuple

    def __init__(self, input_shape: tuple) -> None:
        output_shape = int(np.prod(input_shape))
        super().__init__(input_shape, output_shape)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # Store original batch shape to reshape during backprop
        self.input_batch_shape = input_data.shape

        # Reshape into (batch_size, total_features)
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        # Reshape the 2D error matrix back into the 4D volume expected by convolution layers
        return output_error.reshape(self.input_batch_shape)
