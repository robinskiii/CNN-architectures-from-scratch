import numpy as np
import pytest

import train_model
from layer_architecture import Dense, Layer, ReLU, Sigmoid
from train_model import (
    Model,
    bce,
    bce_derivative,
    cce,
    cce_derivative,
    mse,
    mse_derivative,
)


def test_mse_loss():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.5, 0.5])
    # MSE loss should be ((0.5^2) + (0.5^2)) / 2 = (0.25 + 0.25) / 2 = 0.25
    assert mse(y_true, y_pred) == pytest.approx(0.25)


def test_mse_derivative():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.5, 0.5], [0.25, 0.75]])

    derivative = mse_derivative(y_true, y_pred)

    expected = np.array([[-0.25, 0.25], [0.125, -0.125]])
    np.testing.assert_allclose(derivative, expected)


def test_bce_loss():
    y_true = np.array([1.0, 0.0])

    y_good_pred = np.array([0.99, 0.01])
    loss_accurate = bce(y_true, y_good_pred)
    assert loss_accurate > 0
    assert loss_accurate < 0.05 # Loss should be very small for accurate predictions

    y_bad_pred = np.array([0.01, 0.99])
    loss_inaccurate = bce(y_true, y_bad_pred)
    assert loss_inaccurate > 4  # Loss should be very big for inaccurate predictions


def test_bce_derivative():
    y_true = np.array([[1.0], [0.0]])
    y_pred = np.array([[0.8], [0.2]])

    derivative = bce_derivative(y_true, y_pred)

    np.testing.assert_allclose(derivative, np.array([[-0.625], [0.625]]))


def test_probability_losses_clip_predictions_at_boundaries():
    y_binary = np.array([1.0, 0.0])
    y_one_hot = np.array([[1.0, 0.0], [0.0, 1.0]])
    predictions = np.array([0.0, 1.0])
    multiclass_predictions = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert np.isfinite(bce(y_binary, predictions))
    assert np.all(np.isfinite(bce_derivative(y_binary, predictions)))
    assert np.isfinite(cce(y_one_hot, multiclass_predictions))
    assert np.all(np.isfinite(cce_derivative(y_one_hot, multiclass_predictions)))


def test_cce_loss_and_derivative():
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])

    assert cce(y_true, y_pred) == pytest.approx(-np.log(0.7))

    derivative = cce_derivative(y_true, y_pred)
    expected = np.array(
        [[-1 / (2 * 0.7), 0.0, 0.0], [0.0, 0.0, -1 / (2 * 0.7)]]
    )
    np.testing.assert_allclose(derivative, expected)


def test_model_starts_empty_and_stores_loss_functions():
    model = Model()

    assert model.layers == []
    model.set_loss_function(mse, mse_derivative)

    assert model.loss_function is mse
    assert model.loss_derivative is mse_derivative


def test_model_layer_mismatch():
    """
    Test that the Model catches dimension mismatches when adding layers
    """
    model = Model()
    model.add(Dense(4, 10))

    with pytest.raises(ValueError):
        model.add(Dense(5, 2))


def test_model_add_accepts_matching_layer_dimensions():
    model = Model()
    first_layer = Dense(4, 10)
    second_layer = ReLU(10)

    model.add(first_layer)
    model.add(second_layer)

    assert model.layers == [first_layer, second_layer]


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


def test_model_predict_returns_input_when_no_layers_are_present():
    model = Model()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    output = model.predict(x)

    np.testing.assert_array_equal(output, x)


def test_model_evaluate_records_batch_weighted_losses_and_binary_accuracy():
    model = Model()
    layer = Dense(1, 1)
    layer.weights.fill(2.0)
    layer.bias.fill(0.0)
    model.add(layer)
    model.add(Sigmoid(1))
    model.set_loss_function(bce, bce_derivative)

    x_train = np.array([[-2.0], [0.0], [2.0]])
    y_train = np.array([[0.0], [1.0], [1.0]])
    x_val = np.array([[-2.0], [0.0], [2.0]])
    y_val = np.array([[0.0], [0.0], [1.0]])

    model.evaluate(x_train, y_train, x_val, y_val, batch_size=2)

    assert model.training_data.train_loss == pytest.approx([
        bce(y_train, model.predict(x_train))
    ])
    assert model.training_data.val_loss == pytest.approx([
        bce(y_val, model.predict(x_val))
    ])
    assert model.training_data.val_accuracy == pytest.approx([2 / 3])
    np.testing.assert_array_equal(layer.weights, [[2.0]])
    np.testing.assert_array_equal(layer.bias, [[0.0]])


def test_model_calculates_binary_confusion_matrix_across_batches():
    model = Model()
    layer = Dense(1, 1)
    layer.weights.fill(2.0)
    layer.bias.fill(0.0)
    model.add(layer)
    model.add(Sigmoid(1))

    x = np.array([[-2.0], [0.0], [2.0], [1.0], [-1.0]])
    y = np.array([[0.0], [0.0], [1.0], [0.0], [1.0]])

    matrix = model.calculate_confusion_matrix(x, y, batch_size=2)

    np.testing.assert_array_equal(matrix, [[1, 2], [1, 1]])
    assert matrix.dtype == np.int64


def test_model_calculates_multiclass_confusion_matrix():
    model = Model()
    predictions = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2],
    ])
    labels = np.eye(3)[[0, 1, 2, 2]]

    matrix = model.calculate_confusion_matrix(predictions, labels, batch_size=2)

    np.testing.assert_array_equal(matrix, [[1, 0, 0], [0, 1, 0], [0, 1, 1]])


def test_model_train_records_one_validation_result_per_epoch():
    model = Model()
    layer = Dense(1, 1)
    layer.weights.fill(0.0)
    layer.bias.fill(0.0)
    model.add(layer)
    model.add(Sigmoid(1))
    model.set_loss_function(bce, bce_derivative)

    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[0.0], [1.0], [1.0]])
    history = model.train(x, y, epochs=2, learning_rate=0.0,
                          batch_size=2, x_val=x, y_val=y)

    assert history is model.training_data
    assert len(history.train_loss) == 2
    assert history.train_loss == pytest.approx([np.log(2), np.log(2)])
    assert history.val_loss == pytest.approx([np.log(2), np.log(2)])
    assert history.val_accuracy == pytest.approx([2 / 3, 2 / 3])
    np.testing.assert_array_equal(history.confusion_matrix, [[0, 1], [0, 2]])


def test_model_train_updates_parameters_and_reduces_loss(capsys):
    np.random.seed(0)
    model = Model()
    layer = Dense(1, 1)
    layer.weights.fill(0.0)
    layer.bias.fill(0.0)
    model.add(layer)
    model.set_loss_function(mse, mse_derivative)

    x_train = np.arange(4.0).reshape(-1, 1)
    y_train = x_train.copy()
    initial_loss = mse(y_train, model.predict(x_train))

    model.train(
        x_train,
        y_train,
        epochs=3,
        learning_rate=0.01,
        batch_size=2,
    )

    captured = capsys.readouterr().out
    final_loss = mse(y_train, model.predict(x_train))

    assert "Epoch:" in captured
    assert "3 out of 3" in captured
    assert final_loss < initial_loss
    assert not np.array_equal(layer.weights, np.zeros((1, 1)))


def test_model_train_processes_final_incomplete_batch():
    batch_sizes = []

    def recording_loss(y_true, y_pred):
        batch_sizes.append(y_true.shape[0])
        return mse(y_true, y_pred)

    model = Model()
    model.add(Dense(1, 1))
    model.set_loss_function(recording_loss, mse_derivative)

    x_train = np.arange(5.0).reshape(-1, 1)
    y_train = np.zeros_like(x_train)
    model.train(x_train, y_train, epochs=2, learning_rate=0.001, batch_size=2)

    assert batch_sizes == [2, 2, 1, 2, 2, 1]


def test_model_train_data_augmentation(monkeypatch):
    """
    Tests that the model randomly flips data during the batch training
    Monkeypatch is used to disable the randomness
    """
    class TestLayer(Layer):
        input_shape = (1, 2, 2)
        output_shape = (1, 2, 2)

        def __init__(self):
            self.seen_batches = []

        def forward(self, input_data):
            self.seen_batches.append(input_data.copy())
            return input_data

        def backward(self, output_error, learning_rate):
            return output_error

    layer = TestLayer()
    model = Model()
    model.add(layer)
    model.set_loss_function(mse, mse_derivative)

    x_train = np.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ]
    )
    y_train = np.zeros((2, 1))

    monkeypatch.setattr(train_model.np.random, "permutation", lambda n: np.arange(n)) # disabling the shuffling
    monkeypatch.setattr(
        train_model.np.random,
        "rand",
        lambda n: np.array([0.25, 0.75]),
    ) # forces the first training array to be flipped and the second to not be flipped

    model.train(x_train, y_train, epochs=1, learning_rate=0.001, batch_size=2)

    expected_batch = np.array(
        [
            [[[2.0, 1.0], [4.0, 3.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ]
    )

    assert len(layer.seen_batches) == 1
    np.testing.assert_array_equal(layer.seen_batches[0], expected_batch)
    np.testing.assert_array_equal(x_train, np.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ]
    ))


def test_model_train_can_disable_data_augmentation(monkeypatch):
    class TestLayer(Layer):
        input_shape = (1, 2, 2)
        output_shape = (1, 2, 2)

        def __init__(self):
            self.seen_batches = []

        def forward(self, input_data):
            self.seen_batches.append(input_data.copy())
            return input_data

        def backward(self, output_error, learning_rate):
            return output_error

    layer = TestLayer()
    model = Model()
    model.add(layer)
    model.set_loss_function(mse, mse_derivative)

    x_train = np.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ]
    )
    y_train = np.zeros((2, 1))

    monkeypatch.setattr(train_model.np.random, "permutation", lambda n: np.arange(n))
    monkeypatch.setattr(
        train_model.np.random,
        "rand",
        lambda n: pytest.fail("random flipping should not run when augmentation is disabled"),
    )

    model.train(
        x_train,
        y_train,
        epochs=1,
        learning_rate=0.001,
        batch_size=2,
        augment=False,
    )

    assert len(layer.seen_batches) == 1
    np.testing.assert_array_equal(layer.seen_batches[0], x_train)
