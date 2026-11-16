# Implementation design

This document is a developer reference for where behavior lives and how
components connect. The project's motivation and mathematical explanation are
in the [README](README.md).

# Module responsibilities

## [`layer_architecture.py`](layer_architecture.py)

Defines the `Layer` interface and concrete layer implementations. Each layer
stores its input/output shape without a batch dimension. `forward(input_data)`
computes and caches values needed by its matching `backward(output_error,
learning_rate)` call. Parameter updates currently happen inside backward methods;
there is no separate optimizer object.

Core classes:

- `Layer`: shared shape metadata and forward/backward interface.
- `Dense`, `Convolution`: trainable transformations with weights and biases.
- `ReLU`, `Sigmoid`, `Softmax`: elementwise or class-probability activations.
- `MaxPooling`: spatial downsampling.
- `Flatten`: reshapes image features for dense layers and restores them during
  backpropagation.

## [`train_model.py`](train_model.py)

Defines the sequential container, training utilities, loss functions, and the
current executable experiment.

- `training_data`: loss/accuracy histories and a confusion matrix.
- `Model.add(layer)`: verifies shape continuity when appending a layer.
- `Model.predict(input_data)`: applies each layer in order.
- `Model.train(...)`: shuffles and batches training samples, runs forward and
  backward passes, and optionally evaluates against a validation set. Image
  batches may be horizontally flipped for augmentation.
- `Model.evaluate(...)`: records train/validation loss and accuracy.
- `Model.calculate_confusion_matrix(...)`: evaluates binary or one-hot labels.
- `Model.find_misclassified_indices(...)` and
  `Model.show_misclassified_image(...)`: locate and display classification
  mistakes.
- `Model.save(title)`: stores ordered layer metadata and parameter arrays in a
  compressed NPZ archive under `models/`.
- `mse`, `bce`, `cce` and their derivative functions: objectives used by the
  training loop.

The sequential model contract is that each layer's output shape equals the next
layer's input shape. Forward methods cache intermediate values on their layer
instances, so a backward pass relies on the most recent forward pass through the
same model instance.

## [`load_data.py`](load_data.py)

`download_data()` obtains the CelebA dataset through KaggleHub, falling back to
the local KaggleHub cache if access fails. `process_data()` reads attribute and
partition metadata, loads images, optionally resizes them, normalizes pixels,
and returns image tensors, binary labels, and image IDs. When partition metadata
is available, it currently selects partition `0`.

## [`visualise_data.py`](visualise_data.py)

`Sample` pairs an image ID with a binary label. `load_attribute_pools()` builds
positive and negative sample lists for a selected attribute.
`open_visualisation_stream()` displays samples interactively and tracks the
navigation history.

## [`main.py`](main.py)

Contains the project-template entry point. The current CNN training example is
the `__main__` block in `train_model.py`.

# Tests (vibecoded)

- [`tests/test_layer_architecture.py`](tests/test_layer_architecture.py) covers layer outputs, shape behavior,
  parameter updates, and numerical gradient checks.
- [`tests/test_train_model.py`](tests/test_train_model.py) covers losses, sequential composition, saving,
  evaluation, classification metrics, mini-batches, and augmentation.

# Current implementation boundaries

- `Model` is sequential and does not represent branched graphs or skip
  connections.
- `Model.save()` exists; a corresponding model-loading API does not yet exist.
- Gradients and parameter updates use plain mini-batch gradient descent. There
  are no separate optimizer, regularization, or scheduler abstractions.
- Convolution and pooling operate on channels-first tensors with shape
  `(batch, channels, height, width)`; dense layers operate on
  `(batch, features)`.
