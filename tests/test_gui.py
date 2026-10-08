import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw


MODULE_PATH = Path(__file__).resolve().parents[1] / "GUI" / "GUI.py"
SPEC = importlib.util.spec_from_file_location("digit_recognizer_gui", MODULE_PATH)
gui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gui)


class PreprocessingTests(unittest.TestCase):
    def test_blank_images_have_no_digits(self):
        for color in (0, 255):
            with self.subTest(color=color):
                image = Image.new("L", (100, 100), color=color)
                self.assertEqual(gui.extract_digits_from_pil(image), [])

    def test_digits_are_returned_from_left_to_right(self):
        image = Image.new("L", (160, 80), color=255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((15, 10, 35, 65), fill=0)
        draw.rectangle((100, 10, 140, 65), fill=0)

        digits = gui.extract_digits_from_pil(image)

        self.assertEqual(len(digits), 2)
        self.assertTrue(all(digit.shape == (28, 28) for digit in digits))

    def test_centered_roi_is_a_copy(self):
        frame = np.full((80, 120, 3), 255, dtype=np.uint8)
        roi, coordinates = gui.centered_roi(frame, 40)
        x1, y1, x2, y2 = coordinates
        frame[y1:y2, x1:x2] = 0

        self.assertTrue(np.all(roi == 255))


class MouseWheelTests(unittest.TestCase):
    def setUp(self):
        self.app = gui.DigitRecognizer.__new__(gui.DigitRecognizer)
        self.app.camera_on = False
        self.app.brush_radius = 12
        self.app.result_label = SimpleNamespace(config=lambda **kwargs: None)

    def test_linux_scroll_up_increases_brush(self):
        self.app.on_mouse_wheel(SimpleNamespace(num=4, delta=0))
        self.assertEqual(self.app.brush_radius, 13)

    def test_linux_scroll_down_decreases_brush(self):
        self.app.on_mouse_wheel(SimpleNamespace(num=5, delta=0))
        self.assertEqual(self.app.brush_radius, 11)


class CameraDisplayTests(unittest.TestCase):
    def test_camera_reuses_one_canvas_item(self):
        class Canvas:
            def __init__(self):
                self.created = 0
                self.updated = 0

            def create_image(self, *args, **kwargs):
                self.created += 1
                return 42

            def itemconfig(self, *args, **kwargs):
                self.updated += 1

        frame = np.full((80, 120, 3), 255, dtype=np.uint8)
        app = gui.DigitRecognizer.__new__(gui.DigitRecognizer)
        app.camera_on = True
        app.cap = SimpleNamespace(
            isOpened=lambda: True,
            read=lambda: (True, frame.copy()),
        )
        app.root = SimpleNamespace(after=lambda *args: 1)
        app.canvas = Canvas()
        app.camera_image_id = None
        app.last_cam_pred = 0
        app.prediction_in_progress = True
        app.show_prediction_result = lambda: None

        with patch.object(gui.ImageTk, "PhotoImage", side_effect=lambda image: image):
            app.update_camera()
            app.update_camera()

        self.assertEqual(app.canvas.created, 1)
        self.assertEqual(app.canvas.updated, 1)


if __name__ == "__main__":
    unittest.main()
