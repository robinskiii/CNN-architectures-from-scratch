import numpy as np
import pytest

from cnn_architecture import Convolution, Dense, Model, ReLU, Sigmoid, bce, mse


def test_mse_loss():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.5, 0.5])
    # MSE loss should be ((0.5^2) + (0.5^2)) / 2 = (0.25 + 0.25) / 2 = 0.25
    assert mse(y_true, y_pred) == [0.25]


def test_bce_loss():
    y_true = np.array([1.0, 0.0])

    y_good_pred = np.array([0.99, 0.01])
    loss_accurate = bce(y_true, y_good_pred)
    assert loss_accurate > 0
    assert loss_accurate < 0.05 # Loss should be very small for accurate predictions

    y_bad_pred = np.array([0.01, 0.99])
    loss_inaccurate = bce(y_true, y_bad_pred)
    assert loss_inaccurate > 4  # Loss should be very big for inaccurate predictions


def test_dense_layer_forward_shape():
    """
    Test that the Dense layer outputs the correct tensor shape
    """
    batch_size = 32
    input_dimension = 4
    output_dimension = 10

    layer = Dense(input_dimension, output_dimension)
    x = np.random.randn(batch_size, input_dimension)

    output = layer.forward(x)
    assert output.shape == (batch_size, output_dimension)


def test_relu_activation():
    """
    Test that ReLU correctly zeros out negative values
    """
    layer = ReLU(5)
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    output = layer.forward(x)
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])

    assert np.array_equal(output, expected)


def test_sigmoid_activation():
    """
    Test that Sigmoid bounds outputs between 0 and 1
    """
    layer = Sigmoid(3)
    x = np.array([[-1000.0, 0.0, 1000.0]])

    output = layer.forward(x)

    assert output[0, 0] < 0.01      # Sigmoid(-100) approaches 0
    assert output[0, 1] == 0.5      # Sigmoid(0) exactly 0.5
    assert output[0, 2] > 0.99      # Sigmoid(100) approaches 1


def test_convolution_forward_shape():
    """
    Test the Convolution layer calculates output dimensions correctly
    """
    batch_size = 8
    channels = 3
    in_height, in_width = 28, 28
    filters = 16
    filter_size = 3
    stride = 1

    layer = Convolution(
        input_shape=(channels, in_height, in_width),
        filters=filters,
        filter_size=filter_size,
        stride=stride
    )

    x = np.random.randn(batch_size, channels, in_height, in_width)
    output = layer.forward(x)

    # Because stride is 1 and logic uses ((in - 1)//stride) + 1, out_height and out_width should remain 28.
    assert output.shape == (batch_size, filters, 28, 28)


def test_convolution_invalid_initialisation():
    """
    Test that Convolution raises errors on bad filter shapes
    """
    with pytest.raises(ValueError, match="Convolution filter size must be odd"):
        Convolution(input_shape=(3, 28, 28), filters=10, filter_size=4)

    with pytest.raises(ValueError, match="Convolution layer must contain at least one filter"):
        Convolution(input_shape=(3, 28, 28), filters=0)


def test_model_layer_mismatch():
    """
    Test that the Model catches dimension mismatches when adding layers
    """
    model = Model()
    model.add(Dense(4, 10))

    with pytest.raises(ValueError):
        model.add(Dense(5, 2))

def test_model_forward_pass():
    """
    Test a full forward pass through a simple Model
    """
    model = Model()
    model.add(Dense(4, 10))
    model.add(ReLU(10))
    model.add(Dense(10, 1))
    model.add(Sigmoid(1))

    x = np.random.randn(32, 4)
    output = model.predict(x)

    # Output should match the expected batch size and final output dim
    assert output.shape == (32, 1)

    # Sigmoid bounds should apply
    assert np.all((output >= 0) & (output <= 1))
