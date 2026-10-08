import json

import numpy as np
import pytest

from digit_recognizer.training import build_model, normalize, split_training_data, train


def test_split_is_stratified_disjoint_and_repeatable():
    identifiers = np.arange(1000)
    labels = identifiers % 10
    first = split_training_data(identifiers, labels)
    second = split_training_data(identifiers, labels)
    x_train, x_val, y_train, y_val = first
    assert len(x_train) == 900 and len(x_val) == 100
    assert not set(x_train) & set(x_val)
    assert set(x_train) | set(x_val) == set(identifiers)
    np.testing.assert_array_equal(np.bincount(y_train), [90] * 10)
    np.testing.assert_array_equal(np.bincount(y_val), [10] * 10)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(x_val, split_training_data(identifiers, labels, seed=7)[1])


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_invalid_validation_fraction(fraction):
    with pytest.raises(ValueError):
        split_training_data(np.arange(100), np.arange(100) % 10, fraction)


def test_architecture_is_unchanged():
    model = build_model()
    assert model.count_params() == 93322
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)
    assert [layer.__class__.__name__ for layer in model.layers] == [
        "Conv2D",
        "MaxPooling2D",
        "Conv2D",
        "MaxPooling2D",
        "Conv2D",
        "Flatten",
        "Dense",
        "Dropout",
        "Dense",
    ]
    assert model.layers[-2].rate == 0.5


def test_normalization():
    images = np.array([np.full((28, 28), 255, dtype=np.uint8)])
    output = normalize(images)
    assert output.shape == (1, 28, 28, 1)
    assert output.dtype == np.float32 and output.min() == output.max() == 1


def test_training_pipeline_uses_validation_only_and_writes_reports(tmp_path, monkeypatch):
    import digit_recognizer.training as training
    from keras.models import load_model

    rng = np.random.default_rng(7)
    # Synthetic data validates plumbing, not recognition accuracy.
    images = rng.integers(0, 128, (200, 28, 28), dtype=np.uint8)
    labels = np.tile(np.arange(10), 20)
    test_images = np.full((20, 28, 28), 255, dtype=np.uint8)
    test_labels = np.tile(np.arange(10), 2)
    monkeypatch.setattr(training, "load_mnist", lambda path: ((images, labels), (test_images, test_labels)))
    original_build = training.build_model
    seen = []

    def build_spy():
        model = original_build()
        original_fit = model.fit

        def fit_spy(generator, **kwargs):
            assert generator.x.max() < 0.51
            validation_images, validation_labels = kwargs["validation_data"]
            assert len(validation_images) == 20
            assert validation_images.max() < 0.51  # test pixels must never reach fit
            assert validation_labels.shape == (20, 10)
            seen.append("fit")
            return original_fit(generator, **kwargs)

        model.fit = fit_spy
        return model

    monkeypatch.setattr(training, "build_model", build_spy)
    output = tmp_path / "run"
    result = train(output, epochs=1, batch_size=32)
    assert seen == ["fit"]
    assert result["split_sizes"] == {"train": 180, "validation": 20, "test": 20}
    assert result["selected_epoch"] == 1
    metrics = json.loads((output / "metrics.json").read_text())
    matrix = np.asarray(metrics["confusion_matrix"])
    assert matrix.shape == (10, 10) and matrix.sum() == 20
    assert len(metrics["history"]["val_loss"]) == 1
    for name in ("learning_curves.png", "confusion_matrix.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG")
    restored = load_model(output / "model.keras", compile=False)
    assert restored.output_shape == (None, 10)
    with pytest.raises(FileExistsError):
        train(output, epochs=1)
