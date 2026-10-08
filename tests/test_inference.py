import numpy as np
import pytest

from digit_recognizer.inference import make_variants, predict_probabilities, recognize_digits
from digit_recognizer.preprocessing import extract_digits_from_pil


def test_fast_views_use_zero_padding_without_wraparound():
    digit = np.zeros((28, 28), dtype=np.uint8)
    digit[14, 0] = 255
    views = make_variants(digit)
    assert views.shape == (5, 28, 28, 1)
    assert views.dtype == np.float32
    assert views.min() >= 0 and views.max() <= 1
    assert not views[1].any()  # left shift moves the pixel outside the crop
    assert views[2, 14, 2, 0] == 1
    assert not views[:, :, -2:, :].any()


def test_multiple_digits_share_one_model_call(fake_model, drawn_one):
    digit = extract_digits_from_pil(drawn_one)[0]
    assert recognize_digits(fake_model, [digit, digit]) == [(1, 100.0), (1, 100.0)]
    assert len(fake_model.batches) == 1
    assert fake_model.batches[0].shape == (10, 28, 28, 1)


def test_chunking_and_empty_input(fake_model):
    digit = np.zeros((28, 28), dtype=np.uint8)
    assert predict_probabilities(fake_model, []).shape == (0, 10)
    assert fake_model.batches == []
    result = predict_probabilities(fake_model, [digit] * 5, "none", chunk_size=2)
    assert result.shape == (5, 10)
    assert [len(batch) for batch in fake_model.batches] == [2, 2, 1]


@pytest.mark.parametrize(
    "digit,mode",
    [
        (np.zeros((10, 10), dtype=np.uint8), "fast"),
        (np.zeros((28, 28)), "fast"),
        (np.zeros((28, 28), dtype=np.uint8), "unknown"),
    ],
)
def test_invalid_inference_input(digit, mode):
    with pytest.raises(ValueError):
        make_variants(digit, mode)


def test_invalid_model_output_rejected():
    with pytest.raises(ValueError, match="invalid digit probabilities"):
        predict_probabilities(
            lambda x, **kwargs: np.full((len(x), 10), np.nan), [np.zeros((28, 28), dtype=np.uint8)]
        )


def test_real_bundled_model_recognizes_one(drawn_one):
    from keras.models import load_model
    from digit_recognizer.gui import MODEL_PATH

    model = load_model(MODEL_PATH, compile=False)
    result = recognize_digits(model, extract_digits_from_pil(drawn_one))
    assert result[0][0] == 1
    assert result[0][1] > 90


def test_probabilities_are_averaged_per_digit_without_mixing():
    dark = np.zeros((28, 28), dtype=np.uint8)
    bright = np.full((28, 28), 255, dtype=np.uint8)

    def model(images, **kwargs):
        means = images.mean(axis=(1, 2, 3))
        result = np.zeros((len(images), 10), dtype=np.float32)
        result[:, 0] = 1 - means
        result[:, 1] = means
        return result

    result = predict_probabilities(model, [dark, bright])
    assert result[0, 0] == 1
    assert result[1, 1] == pytest.approx(make_variants(bright).mean())
    np.testing.assert_allclose(result.sum(axis=1), 1)
