import json
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from layer_architecture import (
    Convolution,
    Dense,
    Flatten,
    Layer,
    MaxPooling,
    ReLU,
    Sigmoid,
    Softmax,
)
from load_data import process_data


@dataclass
class training_data:
    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    confusion_matrix: np.ndarray



class Model:
    """
    Neural Network composed of a sequence of Layers
    """
    layers: list[Layer]
    loss_function: Callable
    loss_derivative: Callable
    training_data: training_data


    def __init__(self) -> None:
        self.layers = []
        self.training_data = training_data([], [], [], [], np.zeros((2, 2), dtype=np.int64))


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


    def save(self, title: str) -> Path:
        """Save the model architecture and learned parameters as an NPZ archive.

        The archive stores tensors as native NumPy arrays. Its ``architecture``
        entry is JSON metadata that preserves layer order, shapes, configuration,
        and the array keys belonging to each layer.
        """
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty file name")

        requested_name = Path(title)
        if requested_name.name != title or title in {".", ".."}:
            raise ValueError("title must be a file name, not a path")

        filename = title if title.endswith(".npz") else f"{title}.npz"
        models_dir = Path.cwd() / "models"
        models_dir.mkdir(exist_ok=True)
        save_path = models_dir / filename

        arrays: dict[str, np.ndarray] = {}
        layer_metadata = []
        for index, layer in enumerate(self.layers):
            config = {
                name: getattr(layer, name)
                for name in ("filters", "filter_size", "stride", "pool_size")
                if hasattr(layer, name)
            }
            layer_info = {
                "index": index,
                "type": type(layer).__name__,
                "input_shape": layer.input_shape,
                "output_shape": layer.output_shape,
                "config": config,
            }
            for parameter_name in ("weights", "bias"):
                parameter = getattr(layer, parameter_name, None)
                if parameter is not None:
                    array_key = f"layer_{index}_{parameter_name}"
                    arrays[array_key] = parameter
                    layer_info[f"{parameter_name}_key"] = array_key
            layer_metadata.append(layer_info)

        architecture = {"format_version": 1, "layers": layer_metadata}
        np.savez_compressed(
            save_path,
            architecture=np.array(json.dumps(architecture)),
            **typing.cast(dict[str, typing.Any], arrays)
        )

        return save_path


    def evaluate(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            batch_size: int = 1024,
    ) -> None:
        """Record training loss, validation loss, and validation accuracy.

        Predictions are made in batches to limit memory use. Labels must be
        binary (one column) or one-hot encoded (one column per class).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        def measure_perf(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
            if len(x) == 0 or len(x) != len(y):
                raise ValueError("Evaluation inputs and labels must have matching, nonzero lengths")

            total_loss = 0.0
            total_correct = 0
            for start in range(0, len(x), batch_size):
                x_batch = x[start:start + batch_size]
                y_batch = y[start:start + batch_size]
                predictions = self.predict(x_batch)
                if predictions.shape != y_batch.shape:
                    raise ValueError("Evaluation predictions and labels must have matching shapes")

                total_loss += float(self.loss_function(y_batch, predictions)) * len(x_batch)

                if predictions.shape[1] == 1: # for binary class.
                    total_correct += int(np.sum((predictions >= 0.5) == y_batch))
                else: # for multiclass
                    total_correct += int(np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1)))

            return total_loss / len(x), total_correct / len(x)

        train_loss, train_accuracy = measure_perf(X_train, y_train)
        val_loss, val_accuracy = measure_perf(X_val, y_val)

        self.training_data.train_loss.append(train_loss)
        self.training_data.train_accuracy.append(train_accuracy)
        self.training_data.val_loss.append(val_loss)
        self.training_data.val_accuracy.append(val_accuracy)


    def calculate_confusion_matrix(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 1024
            ) -> np.ndarray:
        """
        Functions for binary and multiclass.
        Count predictions by true class (rows) and predicted class (columns).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(x) == 0 or len(x) != len(y) or y.ndim != 2:
            raise ValueError("Inputs and labels must have matching, nonzero lengths and 2D labels")

        num_classes = 2 if y.shape[1] == 1 else y.shape[1]
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        for start in range(0, len(x), batch_size):
            y_batch = y[start:start + batch_size]
            predictions = self.predict(x[start:start + batch_size])
            if predictions.shape != y_batch.shape:
                raise ValueError("Predictions and labels must have matching shapes")

            if predictions.shape[1] == 1:
                true_labels = y_batch[:, 0]
                if not np.all((true_labels == 0) | (true_labels == 1)):
                    raise ValueError("Binary labels must be 0 or 1")
                predicted_labels = (predictions[:, 0] >= 0.5).astype(np.int64)
                true_labels = true_labels.astype(np.int64)
            else:
                true_labels = np.argmax(y_batch, axis=1)
                predicted_labels = np.argmax(predictions, axis=1)

            np.add.at(matrix, (true_labels, predicted_labels), 1)

        return matrix


    def find_misclassified_indices(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 1024,
            ) -> np.ndarray:
        """Return dataset indices where the predicted class is wrong."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(x) == 0 or len(x) != len(y) or y.ndim != 2:
            raise ValueError("Inputs and labels must have matching, nonzero lengths and 2D labels")

        mistake_indices: list[int] = []
        for start in range(0, len(x), batch_size):
            y_batch = y[start:start + batch_size]
            predictions = self.predict(x[start:start + batch_size])
            if predictions.shape != y_batch.shape:
                raise ValueError("Predictions and labels must have matching shapes")

            if predictions.shape[1] == 1:
                true_labels = y_batch[:, 0]
                if not np.all((true_labels == 0) | (true_labels == 1)):
                    raise ValueError("Binary labels must be 0 or 1")
                predicted_labels = (predictions[:, 0] >= 0.5).astype(np.int64)
                true_labels = true_labels.astype(np.int64)
            else:
                true_labels = np.argmax(y_batch, axis=1)
                predicted_labels = np.argmax(predictions, axis=1)

            batch_mistakes = np.flatnonzero(predicted_labels != true_labels)
            mistake_indices.extend((start + batch_mistakes).tolist())

        return np.asarray(mistake_indices, dtype=np.int64)


    def show_misclassified_image(
            self,
            x: np.ndarray,
            y: np.ndarray,
            image_ids: np.ndarray | None = None,
            batch_size: int = 1024,
            mistake_number: int = 0,
            ) -> int | None:
        """Display one misclassified image and return its dataset index."""
        mistake_indices = self.find_misclassified_indices(x, y, batch_size=batch_size)
        if len(mistake_indices) == 0:
            print("No misclassified images found.")
            return None
        if not 0 <= mistake_number < len(mistake_indices):
            raise ValueError("mistake_number must select one of the misclassified images")

        sample_index = int(mistake_indices[mistake_number])
        prediction = self.predict(x[sample_index:sample_index + 1])

        if prediction.shape[1] == 1:
            true_label = int(y[sample_index, 0])
            predicted_label = int(prediction[0, 0] >= 0.5)
            confidence = float(prediction[0, 0])
            label_text = f"true={true_label}, predicted={predicted_label}, p={confidence:.3f}"
        else:
            true_label = int(np.argmax(y[sample_index]))
            predicted_label = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))
            label_text = f"true={true_label}, predicted={predicted_label}, confidence={confidence:.3f}"

        image = x[sample_index]
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        import matplotlib.pyplot as plt

        plt.imshow(np.squeeze(image), cmap="gray" if image.shape[-1] == 1 else None)
        title = f"index={sample_index}, {label_text}"
        if image_ids is not None:
            title = f"{image_ids[sample_index]} | {title}"
        plt.title(title)
        plt.axis("off")
        plt.show()

        return sample_index


    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            learning_rate: float,
            batch_size: int = 32,
            x_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            augment: bool = True
        ) -> training_data :
        """
        Updating the model parameters using mini-batch gradient descent
        (and calling the evaluate method)
        """
        num_samples = x_train.shape[0]

        if (x_val is None) != (y_val is None):
            raise ValueError("x_val and y_val must be provided together")

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
                if augment and x_batch.ndim == 4:
                    flip_mask = np.random.rand(x_batch.shape[0]) < 0.5
                    if np.any(flip_mask):
                        x_batch = x_batch.copy() # shallow copy
                        x_batch[flip_mask] = x_batch[flip_mask, :, :, ::-1] # horizontal flip

                # forward prop
                output = self.predict(x_batch)

                # track the loss
                batch_cost = self.loss_function(y_batch, output)
                epoch_cost += batch_cost
                num_batches += 1

                # back prop
                error = self.loss_derivative(y_batch, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            epoch_cost /= num_batches

            if x_val is not None and y_val is not None:
                self.evaluate(x_train, y_train, x_val, y_val, batch_size=batch_size)
            else:
                self.training_data.train_loss.append(float(epoch_cost))

            print("Epoch: ",(6-len(str(i+1)))*" " ,f"{i+1} out of {epochs}",10*" ",f"cost: {epoch_cost} ") # i aknowledge this is somewhat overkill

        # training data only for models with a classification output layer.
        if x_val is not None and y_val is not None:
            self.training_data.confusion_matrix = self.calculate_confusion_matrix(
                x_val, y_val, batch_size=batch_size
            )
        elif self.layers and isinstance(self.layers[-1], (Sigmoid, Softmax)):
            self.training_data.confusion_matrix = self.calculate_confusion_matrix(
                x_train, y_train, batch_size=batch_size
            )

        return self.training_data



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
    return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.size


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
    np.random.seed(1)

    X_train, y_train, image_ids = process_data(max_samples=32, shuffle=True)

    channels = 3
    image_height = 218
    image_width = 178
    input_shape = (channels, image_height, image_width)

    print(f"Loaded {len(X_train)} CelebA images ({int(y_train.sum())} smiling, "
          f"{len(y_train) - int(y_train.sum())} not smiling).")

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

    data = model.train(
        X_train,
        y_train,
        epochs=30,
        learning_rate=0.01,
        batch_size=8,
        augment=False,
    )

    model_save = model.save("test")
    print(67*"=", "\n")

    end = time.perf_counter()

    print(f"Time spent training model: {end-start} seconds\n")

    print(67*"=", "\n")

    print("Predictions vs true labels:")

    y_pred = model.predict(X_train[:6])

    print(np.column_stack((y_train[:6],y_pred)).T)

    print(67*"=", "\n")

    print(data.confusion_matrix)

    print(67*"=", "\n")

    bad_index = model.show_misclassified_image(X_train, y_train, image_ids, batch_size=8)
    if bad_index is not None:
        print(f"First misclassified image index: {bad_index}")
