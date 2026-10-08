import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/test-matplotlib").resolve()))

import numpy as np
from PIL import Image, ImageDraw
import pytest


@pytest.fixture
def drawn_one():
    image = Image.new("L", (480, 480), 255)
    ImageDraw.Draw(image).line([(240, 100), (240, 380)], fill=0, width=35)
    return image


class FakeModel:
    input_shape = (None, 28, 28, 1)
    output_shape = (None, 10)

    def __init__(self):
        self.batches = []

    def __call__(self, images, training=False):
        assert not training
        self.batches.append(np.asarray(images).copy())
        result = np.zeros((len(images), 10), dtype=np.float32)
        result[:, 1] = 1
        return result


@pytest.fixture
def fake_model():
    return FakeModel()
