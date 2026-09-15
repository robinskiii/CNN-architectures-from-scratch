import numpy as np
import pytest

from layer_architecture import Dense, ReLU, Sigmoid
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

    np.testing.assert_allclose(derivative, np.array([[-1.25], [1.25]]))


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
