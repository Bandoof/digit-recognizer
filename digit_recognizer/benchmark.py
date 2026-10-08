"""Compare fixed TTA policies on validation data, never tune on the test set."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps

from .inference import predict_probabilities
from .training import load_mnist, split_training_data


def legacy_variants(digit):
    """Reproduce the original 17-view GUI policy for comparison only."""
    image = Image.fromarray(digit)
    variants = [image]
    for dx in (-2, 0, 2):
        for dy in (-2, 0, 2):
            if dx == dy == 0:
                continue
            shifted = ImageChops.offset(image, dx, dy)
            if dx > 0:
                shifted.paste(0, (0, 0, dx, image.height))
            elif dx < 0:
                shifted.paste(0, (image.width + dx, 0, image.width, image.height))
            if dy > 0:
                shifted.paste(0, (0, 0, image.width, dy))
            elif dy < 0:
                shifted.paste(0, (0, image.height + dy, image.width, image.height))
            variants.append(shifted)
    variants.extend(image.rotate(angle, fillcolor=0) for angle in (-10, -5, 5, 10))
    variants.extend((ImageOps.mirror(image), ImageOps.flip(image)))
    enhancer = ImageEnhance.Contrast(image)
    variants.extend((enhancer.enhance(0.9), enhancer.enhance(1.1)))
    return np.stack([np.asarray(view) for view in variants]).astype(np.float32)[..., None] / 255


def benchmark(model, images, labels, repeats=3):
    if len(images) == 0 or len(images) != len(labels) or repeats < 1:
        raise ValueError("Provide matching nonempty images/labels and positive repeats.")

    def predict_one(digit, mode):
        if mode == "legacy17":
            return model.predict(legacy_variants(digit), verbose=0).mean(axis=0)
        return predict_probabilities(model, [digit], mode)[0]

    report = {}
    for mode in ("legacy17", "fast", "none"):
        predict_one(images[0], mode)  # Exclude one warmup from timing.
        durations = []
        predictions = None
        for _ in range(repeats):
            start = perf_counter()
            predictions = np.array([predict_one(image, mode).argmax() for image in images])
            durations.append((perf_counter() - start) * 1000 / len(images))
        report[mode] = {
            "accuracy": float(np.mean(predictions == labels)),
            "median_ms_per_digit": float(np.median(durations)),
            "ms_per_digit_runs": durations,
            "views": {"legacy17": 17, "fast": 5, "none": 1}[mode],
        }
    report["fast_vs_legacy_accuracy_delta_pp"] = 100 * (
        report["fast"]["accuracy"] - report["legacy17"]["accuracy"]
    )
    report["fast_vs_legacy_speedup"] = (
        report["legacy17"]["median_ms_per_digit"] / report["fast"]["median_ms_per_digit"]
    )
    return report


def main():
    from keras.models import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tta-benchmark.json"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    (images, labels), _ = load_mnist(args.data_path)
    _, validation, _, targets = split_training_data(
        images, labels, validation_fraction=args.validation_fraction, seed=args.seed
    )
    if not 1 <= args.samples <= len(validation):
        parser.error(f"samples must be between 1 and {len(validation)}")
    indices = np.random.default_rng(args.seed).permutation(len(validation))[: args.samples]
    model = load_model(args.model, compile=False)
    report = benchmark(model, validation[indices], targets[indices], args.repeats)
    report.update(
        {
            "samples": args.samples,
            "repeats": args.repeats,
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "partition": "validation (stratified split of official MNIST training data)",
            "input": "original 28x28 MNIST images, no GUI preprocessing",
            "timing": "serial single-digit calls including augmentation; warmed model; median of repeats",
            "platform": platform.platform(),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
