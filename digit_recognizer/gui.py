"""Tkinter front end. All widget updates happen on the main thread."""

import argparse
import logging
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk, UnidentifiedImageError

from .workers import CameraStream, PredictionWorker

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parents[1] / "Model" / "mnist.h5"
CANVAS_SIZE = 480
AUTO_RECOGNIZE_DELAY = 1.5
CAMERA_PRED_INTERVAL = 1.0


def camera_roi(frame, size):
    """Copy the unannotated ROI before drawing any overlay on the preview."""
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Expected a nonempty BGR camera frame.")
    height, width = frame.shape[:2]
    side = min(size, height, width)
    x, y = (width - side) // 2, (height - side) // 2
    roi = frame[y : y + side, x : x + side].copy()
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + side - 1, y + side - 1), (0, 200, 0), 2)
    return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)), preview


class DigitRecognizer:
    def __init__(self, root, model=None, model_path=MODEL_PATH, camera_index=0, tta_mode="fast"):
        self.root = root
        if model is None:
            from keras.models import load_model

            model = load_model(model_path, compile=False)
        if tuple(model.input_shape[1:]) != (28, 28, 1) or model.output_shape[-1] != 10:
            raise ValueError("The model must accept (28, 28, 1) images and predict 10 digits.")
        self.model = model
        self.camera_index = camera_index
        self.tta_mode = tta_mode
        self.roi_size = 360
        self.brush_radius = 12
        self.camera_on = False
        self.camera = None
        self.camera_item = None
        self.photo_image = None
        self.closed = False
        self.generation = 0
        self.prediction_pending = False
        self.last_paint_time = None
        self.last_cam_pred = float("-inf")
        self.poll_id = None
        self.auto_id = None
        self._build_widgets()
        self.worker = PredictionWorker(model)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()
        self.check_auto_recognize()

    def _build_widgets(self):
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("1000x850")
        self.root.configure(bg="#f0f0f0")
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack(pady=12)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)
        for event in ("<Button-1>", "<B1-Motion>"):
            self.canvas.bind(event, self.paint)
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(event, self.on_mouse_wheel)
        self.result_label = tk.Label(
            self.root,
            text="Draw digits or choose source",
            font=("Arial", 16),
            wraplength=900,
        )
        self.result_label.pack(pady=8)
        buttons = tk.Frame(self.root)
        buttons.pack(pady=8)
        self.clear_btn = tk.Button(buttons, text="Clear", command=self.clear_canvas, width=12)
        self.clear_btn.grid(row=0, column=0, padx=8)
        self.recognize_btn = tk.Button(
            buttons,
            text="Recognize",
            command=lambda: self.predict_from_image(self.image),
            width=12,
        )
        self.recognize_btn.grid(row=0, column=1, padx=8)
        self.load_btn = tk.Button(buttons, text="Load Image", command=self.load_image, width=12)
        self.load_btn.grid(row=0, column=2, padx=8)
        self.camera_btn = tk.Button(buttons, text="Camera OFF", command=self.toggle_camera, width=12)
        self.camera_btn.grid(row=0, column=3, padx=8)
        self.roi_scale = tk.Scale(
            self.root,
            from_=160,
            to=640,
            orient=tk.HORIZONTAL,
            length=300,
            label="ROI size",
            command=self.update_roi_size,
        )
        self.roi_scale.set(self.roi_size)
        self.roi_scale.pack()
        tk.Label(self.root, text="Brush size: mouse wheel over canvas").pack(pady=8)

    def _invalidate(self):
        self.generation += 1

    def paint(self, event):
        if self.camera_on or self.closed:
            return
        self._invalidate()
        radius = self.brush_radius
        bounds = (event.x - radius, event.y - radius, event.x + radius, event.y + radius)
        self.canvas.create_oval(*bounds, fill="black", outline="black")
        self.draw.ellipse(bounds, fill=0)
        self.last_paint_time = time.monotonic()
        self.result_label.config(text="Drawing...")

    def on_mouse_wheel(self, event):
        if self.camera_on:
            return
        number = getattr(event, "num", None)
        delta = 120 if number == 4 else -120 if number == 5 else getattr(event, "delta", 0)
        if delta:
            self.brush_radius = max(1, min(50, self.brush_radius + (1 if delta > 0 else -1)))
        self.result_label.config(text=f"Brush radius: {self.brush_radius}")

    def clear_canvas(self):
        if self.camera_on:
            return
        self._invalidate()
        self.last_paint_time = None
        self.canvas.delete("all")
        self.camera_item = None
        self.photo_image = None
        self.draw.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=255)
        self.result_label.config(text="Draw digits or choose source")

    def predict_from_image(self, image, camera_mode=False):
        if self.closed or self.prediction_pending or (self.camera_on and not camera_mode):
            return False
        accepted = self.worker.submit(self.generation, image, self.tta_mode)
        if accepted:
            self.prediction_pending = True
            self.last_paint_time = None
            if not camera_mode:
                self.result_label.config(text="Recognizing...")
        return accepted

    def load_image(self):
        if self.camera_on or self.prediction_pending:
            return
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        self._invalidate()
        self.last_paint_time = None
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                image.load()
                self.predict_from_image(image)
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            self.result_label.config(text=f"Cannot load image: {error}")

    def toggle_camera(self):
        if self.camera_on:
            self._stop_camera("Camera stopped")
            return
        if self.camera is not None and self.camera.thread.is_alive():
            self.result_label.config(text="Camera is still stopping; please retry.")
            return
        self._invalidate()
        self.last_paint_time = None
        self.canvas.delete("all")
        self.camera_item = None
        self.camera_on = True
        self.last_cam_pred = float("-inf")
        self.camera = CameraStream(self.camera_index)
        self.camera_btn.config(text="Camera ON", bg="#27ae60")
        for button in (self.clear_btn, self.load_btn, self.recognize_btn):
            button.config(state=tk.DISABLED)
        self.result_label.config(text="Opening camera...")

    def _stop_camera(self, message):
        self.camera_on = False
        if self.camera is not None:
            self.camera.close()
        self.camera_btn.config(text="Camera OFF", bg="#f0f0f0")
        for button in (self.clear_btn, self.load_btn, self.recognize_btn):
            button.config(state=tk.NORMAL)
        self.clear_canvas()
        self.result_label.config(text=message)

    def update_roi_size(self, value):
        size = max(160, min(640, int(float(value))))
        if size != self.roi_size:
            self.roi_size = size
            if self.camera_on:
                self._invalidate()

    def update_camera(self):
        if not self.camera_on:
            return
        try:
            frame, error = self.camera.frames.get_nowait()
        except queue.Empty:
            return
        if error is not None:
            self._stop_camera(str(error))
            return
        try:
            roi, preview = camera_roi(frame, self.roi_size)
            now = time.monotonic()
            if now - self.last_cam_pred >= CAMERA_PRED_INTERVAL:
                if self.predict_from_image(roi, camera_mode=True):
                    self.last_cam_pred = now
            image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
            self.photo_image = ImageTk.PhotoImage(image.resize((CANVAS_SIZE, CANVAS_SIZE)))
            if self.camera_item is None:
                self.camera_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
            else:
                self.canvas.itemconfig(self.camera_item, image=self.photo_image)
        except Exception as error:
            LOGGER.exception("Camera processing failed")
            self._stop_camera(f"Camera error: {error}")

    def _poll(self):
        if self.closed:
            return
        try:
            token, result, error = self.worker.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.prediction_pending = False
            # Clearing, drawing, or switching camera invalidates older results.
            if token == self.generation:
                if error is not None:
                    self.result_label.config(text=f"Prediction error: {error}")
                elif not result:
                    self.result_label.config(text="No digits found")
                else:
                    digits = "".join(str(label) for label, _ in result)
                    confidence = np.mean([confidence for _, confidence in result])
                    details = ", ".join(f"{label}:{score:.0f}%" for label, score in result)
                    self.result_label.config(text=f"Result: {digits}   (avg {confidence:.1f}%)  [{details}]")
        self.update_camera()
        self.poll_id = self.root.after(40, self._poll)

    def check_auto_recognize(self):
        if self.closed:
            return
        if not self.camera_on and self.last_paint_time is not None:
            if time.monotonic() - self.last_paint_time >= AUTO_RECOGNIZE_DELAY:
                self.predict_from_image(self.image)
        self.auto_id = self.root.after(100, self.check_auto_recognize)

    def on_close(self):
        if self.closed:
            return
        self.closed = True
        self.worker.close()
        if self.camera is not None:
            self.camera.close()
        for callback in (self.poll_id, self.auto_id):
            if callback is not None:
                self.root.after_cancel(callback)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--tta", choices=("none", "fast"), default="fast")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    root = None
    try:
        root = tk.Tk()
        DigitRecognizer(root, model_path=args.model, camera_index=args.camera_index, tta_mode=args.tta)
    except Exception as error:
        LOGGER.error("Cannot start Digit Recognizer: %s", error)
        if root is not None:
            root.withdraw()
            messagebox.showerror("Cannot start Digit Recognizer", str(error), parent=root)
            root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
