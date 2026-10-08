"""Background workers never access Tk widgets; queues cross the thread boundary."""

import queue
import threading

import cv2

from .inference import recognize_digits
from .preprocessing import extract_digits_from_pil


class PredictionWorker:
    """One model owner and at most one pending request, avoiding inference races."""

    def __init__(self, model):
        self.model = model
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="digit-inference")
        self.thread.start()

    def submit(self, token, image, mode="fast"):
        if self.stopped.is_set():
            return False
        try:
            self.requests.put_nowait((token, image.copy(), mode))
            return True
        except queue.Full:
            return False

    def _run(self):
        while not self.stopped.is_set():
            try:
                token, image, mode = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                digits = extract_digits_from_pil(image)
                result = recognize_digits(self.model, digits, mode)
                self.results.put((token, result, None))
            except Exception as error:
                self.results.put((token, None, error))

    def close(self):
        self.stopped.set()


class CameraStream:
    """Capture on a daemon thread; keep only the most recent frame or error."""

    def __init__(self, index=0, capture_factory=cv2.VideoCapture):
        self.index = index
        self.capture_factory = capture_factory
        self.frames = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="digit-camera")
        self.thread.start()

    def _publish(self, frame, error=None):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait((frame, error))

    def _run(self):
        capture = None
        try:
            capture = self.capture_factory(self.index)
            if not capture.isOpened():
                raise RuntimeError("Cannot open camera. Check permissions and the camera index.")
            while not self.stopped.is_set():
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    raise RuntimeError("Camera disconnected or returned an empty frame.")
                self._publish(frame)
                self.stopped.wait(0.01)
        except Exception as error:
            self._publish(None, error)
        finally:
            if capture is not None:
                capture.release()

    def close(self):
        # Do not join on the UI thread: a device driver can block inside read().
        self.stopped.set()
