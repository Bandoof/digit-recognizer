import queue
import os
import threading
import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import digit_recognizer.gui as gui


def pump(root, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(0.01)
    pytest.fail("GUI condition did not complete before timeout")


@pytest.fixture
def app(fake_model, request):
    try:
        root = tk.Tk()
    except tk.TclError as error:
        if os.environ.get("DISPLAY"):
            pytest.fail(f"Configured display is broken: {error}")
        pytest.skip(f"GUI test needs a display: {error}")
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    if getattr(request, "param", None) == "real":
        application = gui.DigitRecognizer(root)
    else:
        root.withdraw()
        application = gui.DigitRecognizer(root, model=fake_model)
    yield application
    application.on_close()
    application.worker.thread.join(timeout=2)
    assert not errors, errors


def test_roi_does_not_contain_preview_border():
    frame = np.full((120, 200, 3), 255, dtype=np.uint8)
    roi, preview = gui.camera_roi(frame, 360)
    assert roi.size == (120, 120)
    assert np.asarray(roi).min() == 255
    assert frame.min() == 255  # caller's frame is unchanged
    assert preview.min() == 0


@pytest.mark.gui
def test_drawing_prediction_and_clear(app, drawn_one):
    assert app.predict_from_image(drawn_one)
    pump(app.root, lambda: not app.prediction_pending)
    assert app.result_label.cget("text").startswith("Result: 1 ")
    app.clear_canvas()
    assert app.last_paint_time is None
    assert not app.canvas.find_all()
    assert app.predict_from_image(Image.new("L", (480, 480), 255))
    pump(app.root, lambda: not app.prediction_pending)
    assert app.result_label.cget("text") == "No digits found"


@pytest.mark.gui
def test_auto_recognition_after_paint(app):
    for y in range(100, 350, 10):
        app.paint(SimpleNamespace(x=240, y=y))
    app.last_paint_time = time.monotonic() - 2
    pump(app.root, lambda: app.result_label.cget("text").startswith("Result: 1 "))


@pytest.mark.gui
def test_linux_and_windows_mouse_wheel(app):
    app.on_mouse_wheel(SimpleNamespace(num=4, delta=0))
    assert app.brush_radius == 13
    app.on_mouse_wheel(SimpleNamespace(num=5, delta=0))
    assert app.brush_radius == 12
    app.on_mouse_wheel(SimpleNamespace(delta=-120))
    assert app.brush_radius == 11


@pytest.mark.gui
def test_clear_discards_stale_predictions(app, drawn_one):
    gate = threading.Event()
    original = app.worker.model

    def slow_model(images, **kwargs):
        assert gate.wait(timeout=2)
        return original(images, **kwargs)

    app.worker.model = slow_model
    try:
        assert app.predict_from_image(drawn_one)
        assert not app.predict_from_image(drawn_one)  # no concurrent inference
        app.clear_canvas()
        gate.set()
        pump(app.root, lambda: not app.prediction_pending)
        assert app.result_label.cget("text") == "Draw digits or choose source"
    finally:
        gate.set()


@pytest.mark.gui
def test_bad_image_is_handled(app, tmp_path, monkeypatch):
    path = tmp_path / "bad.png"
    path.write_text("not an image")
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kwargs: str(path))
    app.load_image()
    assert app.result_label.cget("text").startswith("Cannot load image:")
    assert not app.prediction_pending


@pytest.mark.gui
def test_load_image_and_cancel(app, tmp_path, drawn_one, monkeypatch):
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kwargs: "")
    app.load_image()
    assert not app.prediction_pending
    path = tmp_path / "one.png"
    drawn_one.save(path)
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kwargs: str(path))
    app.load_image()
    pump(app.root, lambda: not app.prediction_pending)
    assert app.result_label.cget("text").startswith("Result: 1 ")


class FakeStream:
    def __init__(self, index):
        self.frames = queue.Queue()
        self.thread = SimpleNamespace(is_alive=lambda: False)
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.gui
def test_camera_canvas_item_reused_and_failure_resets_state(app, monkeypatch):
    monkeypatch.setattr(gui, "CameraStream", FakeStream)
    app.toggle_camera()
    assert app.camera_on
    assert app.recognize_btn.cget("state") == "disabled"
    stream = app.camera
    for _ in range(10):
        stream.frames.put((np.full((240, 320, 3), 255, dtype=np.uint8), None))
        app.update_camera()
    assert len(app.canvas.find_all()) == 1
    stream.frames.put((None, RuntimeError("Camera disconnected")))
    app.update_camera()
    assert stream.closed and not app.camera_on
    assert app.recognize_btn.cget("state") == "normal"
    assert app.result_label.cget("text") == "Camera disconnected"
    assert not app.canvas.find_all()


@pytest.mark.gui
def test_close_cancels_timers_and_stops_workers(app, monkeypatch):
    monkeypatch.setattr(gui, "CameraStream", FakeStream)
    app.toggle_camera()
    callbacks = (app.poll_id, app.auto_id)
    app.on_close()
    assert app.closed and app.camera.closed and app.worker.stopped.is_set()
    assert not set(callbacks) & set(app.root.tk.call("after", "info"))
    app.on_close()  # idempotent


@pytest.mark.gui
def test_inference_exception_is_shown(app, drawn_one):
    def broken_model(*args, **kwargs):
        raise RuntimeError("inference unavailable")

    app.worker.model = broken_model
    app.predict_from_image(drawn_one)
    pump(app.root, lambda: not app.prediction_pending)
    assert app.result_label.cget("text") == "Prediction error: inference unavailable"


@pytest.mark.gui
def test_missing_model_displays_startup_error(app, monkeypatch):
    from keras import models

    messages = []
    monkeypatch.setattr(gui.tk, "Tk", lambda: app.root)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, text, **kwargs: messages.append(text))
    monkeypatch.setattr("sys.argv", ["digit-recognizer", "--model", "missing.keras"])

    def fail_load(*args, **kwargs):
        raise OSError("Model file not found")

    monkeypatch.setattr(models, "load_model", fail_load)
    # Keep the fixture window alive so its worker/timer teardown still runs.
    with monkeypatch.context() as context:
        context.setattr(app.root, "destroy", lambda: None)
        assert gui.main() == 1
    assert messages == ["Model file not found"]


@pytest.mark.gui
@pytest.mark.parametrize("app", ["real"], indirect=True)
def test_real_gui_startup_keeps_first_prediction(app, drawn_one):
    # Initial Scale callbacks must not invalidate a request made before mainloop.
    assert app.predict_from_image(drawn_one)
    pump(app.root, lambda: not app.prediction_pending, timeout=15)
    assert app.result_label.cget("text").startswith("Result: 1 ")
