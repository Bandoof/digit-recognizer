"""Reproducible MNIST training with an untouched official test partition."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def split_training_data(images, labels, validation_fraction=0.1, seed=42):
    """Split only the official training partition, retaining class proportions."""
    if len(images) != len(labels) or not 0 < validation_fraction < 1:
        raise ValueError("Provide matching images/labels and a validation fraction in (0, 1).")
    return train_test_split(
        images,
        labels,
        test_size=validation_fraction,
        stratify=labels,
        random_state=seed,
    )


def load_mnist(data_path=None):
    if data_path is None:
        from keras.datasets import mnist

        return mnist.load_data()
    with np.load(data_path, allow_pickle=False) as data:
        return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])


def normalize(images):
    return np.asarray(images, dtype=np.float32)[..., None] / 255


def build_model():
    """The original CNN architecture, unchanged (93,322 parameters)."""
    from keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_reports(history, labels, probabilities, output_dir, metadata):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(10))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        epochs = np.arange(1, len(history[metric]) + 1)
        axis.plot(epochs, history[metric], label="Train (augmented, dropout on)")
        axis.plot(epochs, history[f"val_{metric}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), xticks=epochs)
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=np.arange(10)).plot(ax=axis, cmap="Blues")
    axis.set_title("Official MNIST test set — no TTA")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    report = dict(metadata)
    report["history"] = {key: [float(x) for x in values] for key, values in history.items()}
    report["confusion_matrix"] = matrix.tolist()
    report["classification_report"] = classification_report(
        labels,
        predictions,
        labels=np.arange(10),
        output_dict=True,
        zero_division=0,
    )
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def train(output_dir, *, data_path=None, epochs=5, batch_size=64, seed=42, validation_fraction=0.1):
    import tensorflow as tf
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import set_random_seed, to_categorical

    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    output_dir = Path(output_dir)
    # A new directory prevents mixing histories/checkpoints from different runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    (images, labels), (test_images, test_labels) = load_mnist(data_path)
    x_train, x_val, y_train, y_val = split_training_data(images, labels, validation_fraction, seed)
    model = build_model()
    augmentation = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    checkpoint = output_dir / "model.keras"
    history = model.fit(
        augmentation.flow(normalize(x_train), to_categorical(y_train, 10), batch_size=batch_size, seed=seed),
        epochs=epochs,
        validation_data=(normalize(x_val), to_categorical(y_val, 10)),
        callbacks=[ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True)],
        verbose=2,
    )
    # Selection depends only on validation loss. Test is first evaluated after fit.
    selected = load_model(checkpoint)
    test_metrics = selected.evaluate(
        normalize(test_images),
        to_categorical(test_labels, 10),
        batch_size=batch_size,
        verbose=0,
        return_dict=True,
    )
    probabilities = selected.predict(normalize(test_images), batch_size=batch_size, verbose=0)
    metadata = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_fraction": validation_fraction,
        "split_sizes": {"train": len(x_train), "validation": len(x_val), "test": len(test_images)},
        "selected_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "selection_metric": "val_loss",
        "test_metrics": test_metrics,
        "tensorflow_version": tf.__version__,
        "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "data_source": "keras.datasets.mnist" if data_path is None else str(Path(data_path).resolve()),
        "data_sha256": None
        if data_path is None
        else hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
    }
    save_reports(history.history, test_labels, probabilities, output_dir, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--data-path", type=Path, help="Local MNIST npz; otherwise Keras downloads it.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(train(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
