# Recorded experiment

Run date: 2026-10-08. Python 3.11.16, TensorFlow/Keras 2.15.1/2.15.0, Linux x86_64 CPU. TensorFlow used four intra-op and two inter-op threads. Dependencies are in `requirements.txt`.

The architecture is unchanged. A new model was initialized with seed 42 and trained for five epochs with batch size 64. A stratified 6,000-image validation subset was removed from the official 60,000 training images **before augmentation**. The checkpoint with minimum validation loss (epoch 5) was evaluated on all 10,000 official test images. Confusion counts sum to 10,000, with 9,909 correct predictions. No TTA was applied to these final test metrics.

`metrics.json`, `learning_curves.png`, and `confusion_matrix.png` were copied directly from the completed training run. No historical accuracy is inferred for `models/mnist.h5`. The new checkpoint remains at `artifacts/mnist-seed42/model.keras` in the working environment and is not tracked. Its SHA-256 is recorded in both JSON reports, allowing readers to check that training and benchmark used the same artifact. Re-run training to create your own checkpoint; exact binary hashes are not guaranteed across runs/platforms.

## Dataset provenance

The dataset was obtained through the public [fgnt/mnist mirror](https://github.com/fgnt/mnist) and the decompressed arrays were stored locally in an NPZ archive with the standard Keras keys. Before loading, all four compressed IDX files were checked against the standard MNIST checksums also recorded by [Torchvision's MNIST loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py):

| File | MD5 |
| --- | --- |
| train-images-idx3-ubyte.gz | f68b3c2dcbeaaa9fbdd348bbdeb94873 |
| train-labels-idx1-ubyte.gz | d53e105ee54ea40749a09fcbcd1e9432 |
| t10k-images-idx3-ubyte.gz | 9fb629c4189551a2d022fa330f9573f3 |
| t10k-labels-idx1-ubyte.gz | ec29112dd5afa0611ce80d1b7f02629c |

The NPZ SHA-256 and original local input path are recorded in `metrics.json`. The path describes this run, not a required path on other machines. Default training uses Keras's standard MNIST download; `--data-path` accepts a local archive retaining the same partitions.

```bash
TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 \
python -m digit_recognizer.training \
  --data-path /path/to/mnist.npz --output-dir artifacts/mnist-seed42 \
  --epochs 5 --batch-size 64 --seed 42

TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 \
python -m digit_recognizer.benchmark \
  --model artifacts/mnist-seed42/model.keras --data-path /path/to/mnist.npz \
  --samples 1000 --repeats 3 --seed 42 --output artifacts/tta-benchmark.json
```

## TTA measurement limits

The fast policy was fixed before benchmarking. The 1,000 benchmark examples are selected deterministically from validation, not test. The measured accuracy is 994/1,000 for both legacy and fast TTA and 990/1,000 without TTA. A single sample does not establish statistical equivalence. The checkpoint itself was selected using this validation partition, so these are development measurements, not independent final test results.

Each mode performs serial, single-digit calls, with one warmup excluded and three complete repetitions. `median_ms_per_digit` is the median of the three per-run average times. Timing includes view construction and inference. Legacy mode reproduces the original 17-view augmentation and per-digit `model.predict`; fast/no-TTA use the production direct model call. The comparison therefore measures the combined optimization. Preprocessing, Tk rendering, webcam I/O, and multi-digit batching benefits are excluded. CPU contention and different hardware/software can change timings.

All physical-webcam checks remain manual. Automated GUI tests exercise actual Tk widgets under Xvfb with simulated model/camera behavior; a separate smoke test verifies the real bundled model on a synthetic drawn digit.

## Bundled checkpoint compatibility check

`tta-bundled-model.json` records an additional comparison using the bundled `models/mnist.h5` on the same 1,000-example subset, with one timing repetition. Both legacy and fast TTA classified 995/1,000 correctly; no TTA classified 994/1,000 correctly. This checks the inference change against the actual default checkpoint. Because the old checkpoint's training provenance is unknown, this subset may have been seen during its training; **these numbers are not an independent generalization estimate**. The new model's clean test results and three-repeat speed comparison remain the primary experiment above.
