"""Bounded, batched test-time augmentation without digit-changing flips."""

import numpy as np
from PIL import Image

TTA_MODES = ("none", "fast")


def make_variants(digit, mode="fast"):
    if mode not in TTA_MODES:
        raise ValueError(f"Unknown TTA mode: {mode}")
    array = np.asarray(digit)
    if array.shape != (28, 28) or array.dtype != np.uint8:
        raise ValueError("Expected a uint8 digit with shape (28, 28).")
    image = Image.fromarray(array)
    variants = [array]
    if mode == "fast":
        # PIL affine translation fills with zero and never wraps edge pixels.
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            shifted = image.transform(
                image.size,
                Image.Transform.AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                fillcolor=0,
            )
            variants.append(np.asarray(shifted))
    return np.stack(variants).astype(np.float32)[..., None] / 255


def predict_probabilities(model, digits, mode="fast", chunk_size=64):
    """Average all variants per digit; bounded batches avoid large allocations."""
    if mode not in TTA_MODES or chunk_size < 1:
        raise ValueError("Invalid TTA mode or chunk size.")
    results = []
    for start in range(0, len(digits), chunk_size):
        variants = np.stack([make_variants(digit, mode) for digit in digits[start : start + chunk_size]])
        count, views = variants.shape[:2]
        # Direct eager inference avoids model.predict's dataset/thread-pool overhead.
        predictions = np.asarray(model(variants.reshape(-1, 28, 28, 1), training=False))
        if predictions.shape != (count * views, 10) or not np.isfinite(predictions).all():
            raise ValueError("The model returned invalid digit probabilities.")
        results.append(predictions.reshape(count, views, 10).mean(axis=1))
    return np.concatenate(results) if results else np.empty((0, 10), dtype=np.float32)


def recognize_digits(model, digits, mode="fast"):
    probabilities = predict_probabilities(model, digits, mode)
    return [(int(np.argmax(row)), float(np.max(row)) * 100) for row in probabilities]
