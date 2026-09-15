import numpy as np
import pytest

from layer_architecture import (
    Convolution,
    Dense,
    Flatten,
    MaxPooling,
    ReLU,
    Sigmoid,
    Softmax,
)


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


def test_dense_layer_forward_uses_weights_and_bias():
    layer = Dense(2, 2)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.bias = np.array([[0.5, -0.5]])

    output = layer.forward(np.array([[1.0, 2.0]]))

    np.testing.assert_allclose(output, np.array([[7.5, 9.5]]))


def test_dense_layer_backward_returns_input_error_and_updates_parameters():
    layer = Dense(2, 2)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.bias = np.zeros((1, 2))
    layer.forward(np.array([[1.0, 2.0]]))

    input_error = layer.backward(np.array([[0.5, -1.0]]), learning_rate=0.1)

    np.testing.assert_allclose(input_error, np.array([[-1.5, -2.5]]))
    np.testing.assert_allclose(layer.weights, np.array([[0.95, 2.1], [2.9, 4.2]]))
    np.testing.assert_allclose(layer.bias, np.array([[-0.05, 0.1]]))


def test_relu_activation():
    """
    Test that ReLU correctly zeros out negative values
    """
    layer = ReLU(5)
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    output = layer.forward(x)
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])

    assert np.array_equal(output, expected)


def test_relu_backward_only_passes_error_for_positive_inputs():
    layer = ReLU(3)
    layer.forward(np.array([[-1.0, 0.0, 2.0]]))

    input_error = layer.backward(np.array([[4.0, 5.0, 6.0]]), learning_rate=0.1)

    np.testing.assert_array_equal(input_error, np.array([[0.0, 0.0, 6.0]]))


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


def test_sigmoid_backward_uses_sigmoid_derivative():
    layer = Sigmoid(1)
    layer.forward(np.array([[0.0]]))

    input_error = layer.backward(np.array([[2.0]]), learning_rate=0.1)

    np.testing.assert_allclose(input_error, np.array([[0.5]]))


def test_softmax_outputs_probabilities_and_is_numerically_stable():
    layer = Softmax(3)
    output = layer.forward(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))

    assert output.shape == (2, 3)
    np.testing.assert_allclose(np.sum(output, axis=1), np.ones(2))
    np.testing.assert_allclose(
        output[0], np.exp([1.0, 2.0, 3.0]) / np.sum(np.exp([1.0, 2.0, 3.0]))
    )
    np.testing.assert_allclose(output[1], np.full(3, 1 / 3))


def test_softmax_backward_preserves_zero_gradient_for_constant_error():
    layer = Softmax(3)
    layer.forward(np.array([[1.0, 2.0, 3.0]]))

    input_error = layer.backward(np.ones((1, 3)), learning_rate=0.1)

    np.testing.assert_allclose(input_error, np.zeros((1, 3)), atol=1e-12)


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


def test_convolution_forward_applies_filter_and_same_padding():
    layer = Convolution(input_shape=(1, 3, 3), filters=1, filter_size=3)
    layer.weights.fill(1.0)
    layer.bias.fill(0.0)
    x = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)

    output = layer.forward(x)

    expected = np.array(
        [[[[12.0, 21.0, 16.0], [27.0, 45.0, 33.0], [24.0, 39.0, 28.0]]]]
    )
    np.testing.assert_allclose(output, expected)


def test_convolution_backward_returns_input_error_and_updates_parameters():
    layer = Convolution(input_shape=(1, 3, 3), filters=1, filter_size=3)
    layer.weights.fill(1.0)
    layer.bias.fill(0.0)
    x = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    layer.forward(x)

    input_error = layer.backward(np.ones((1, 1, 3, 3)), learning_rate=0.1)

    expected_gradient = np.array(
        [[[[12.0, 21.0, 16.0], [27.0, 45.0, 33.0], [24.0, 39.0, 28.0]]]]
    )
    expected_input_error = np.array(
        [[[[4.0, 6.0, 4.0], [6.0, 9.0, 6.0], [4.0, 6.0, 4.0]]]]
    )

    np.testing.assert_allclose(input_error, expected_input_error)
    np.testing.assert_allclose(layer.weights, 1.0 - 0.1 * expected_gradient)
    np.testing.assert_allclose(layer.bias, np.array([[[-0.9]]]))


def test_convolution_invalid_initialisation():
    """
    Test that Convolution raises errors on bad filter shapes
    """
    with pytest.raises(ValueError, match="Convolution filter size must be odd"):
        Convolution(input_shape=(3, 28, 28), filters=10, filter_size=4)

    with pytest.raises(ValueError, match="Convolution layer must contain at least one filter"):
        Convolution(input_shape=(3, 28, 28), filters=0)

    with pytest.raises(ValueError, match="Convolution stride must be at least 1"):
        Convolution(input_shape=(3, 28, 28), filters=1, stride=0)


def test_max_pooling_forward_handles_odd_spatial_dimensions():
    layer = MaxPooling(input_shape=(1, 3, 5), pool_size=2)
    x = np.array(
        [[[[1.0, 3.0, 2.0, 4.0, 5.0],
           [0.0, 2.0, 6.0, 1.0, 7.0],
           [8.0, 1.0, 9.0, 0.0, 2.0]]]]
    )

    output = layer.forward(x)

    assert output.shape == (1, 1, 2, 3)
    np.testing.assert_array_equal(output, np.array([[[[3.0, 6.0, 7.0], [8.0, 9.0, 2.0]]]]))


def test_max_pooling_backward_routes_error_to_maximum_values():
    layer = MaxPooling(input_shape=(1, 3, 5), pool_size=2)
    x = np.array(
        [[[[1.0, 3.0, 2.0, 4.0, 5.0],
           [0.0, 2.0, 6.0, 1.0, 7.0],
           [8.0, 1.0, 9.0, 0.0, 2.0]]]]
    )
    layer.forward(x)

    input_error = layer.backward(np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]), learning_rate=0.1)

    expected = np.array(
        [[[[0.0, 1.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 2.0, 0.0, 3.0],
           [4.0, 0.0, 5.0, 0.0, 6.0]]]]
    )
    np.testing.assert_array_equal(input_error, expected)


def test_flatten_forward_and_backward_restore_batch_shape():
    layer = Flatten(input_shape=(2, 2, 2))
    x = np.arange(16.0).reshape(2, 2, 2, 2)

    output = layer.forward(x)
    input_error = layer.backward(np.ones((2, 8)), learning_rate=0.1)

    assert layer.output_shape == 8
    assert output.shape == (2, 8)
    np.testing.assert_array_equal(output[0], np.arange(8.0))
    np.testing.assert_array_equal(input_error, np.ones_like(x))
