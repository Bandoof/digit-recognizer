import threading

import numpy as np
import pytest

from digit_recognizer.workers import CameraStream, PredictionWorker


def test_inference_runs_off_main_thread_and_reports_results(fake_model, drawn_one):
    main_thread = threading.get_ident()
    seen = []

    def model(images, **kwargs):
        seen.append(threading.get_ident())
        return fake_model(images, **kwargs)

    worker = PredictionWorker(model)
    try:
        assert worker.submit(42, drawn_one)
        token, result, error = worker.results.get(timeout=5)
        assert token == 42 and result == [(1, 100.0)] and error is None
        assert seen and all(thread != main_thread for thread in seen)
    finally:
        worker.close()
        worker.thread.join(timeout=2)
    assert not worker.thread.is_alive()
    assert not worker.submit(43, drawn_one)


def test_prediction_errors_are_returned(drawn_one):
    def broken_model(*args, **kwargs):
        raise RuntimeError("model unavailable")

    worker = PredictionWorker(broken_model)
    try:
        assert worker.submit(1, drawn_one)
        _, result, error = worker.results.get(timeout=5)
        assert result is None and str(error) == "model unavailable"
    finally:
        worker.close()
        worker.thread.join(timeout=2)


class FakeCapture:
    def __init__(self, opened=True, readable=True):
        self.opened = opened
        self.readable = readable
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return (True, np.zeros((100, 100, 3), dtype=np.uint8)) if self.readable else (False, None)

    def release(self):
        self.released = True


@pytest.mark.parametrize(
    "opened,readable,message",
    [
        (False, True, "Cannot open camera"),
        (True, False, "Camera disconnected"),
    ],
)
def test_camera_failures_release_device(opened, readable, message):
    capture = FakeCapture(opened, readable)
    stream = CameraStream(capture_factory=lambda index: capture)
    frame, error = stream.frames.get(timeout=2)
    stream.thread.join(timeout=2)
    assert frame is None and message in str(error)
    assert capture.released


def test_camera_closes_and_keeps_only_latest_frame():
    capture = FakeCapture()
    stream = CameraStream(capture_factory=lambda index: capture)
    try:
        frame, error = stream.frames.get(timeout=2)
        assert frame.shape == (100, 100, 3) and error is None
        assert stream.frames.maxsize == 1
    finally:
        stream.close()
        stream.thread.join(timeout=2)
    assert capture.released and not stream.thread.is_alive()
