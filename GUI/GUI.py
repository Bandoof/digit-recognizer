import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageChops, ImageEnhance
import numpy as np
import cv2
from keras.models import load_model
import time
import threading
import os

# ====== CONFIG ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "mnist.h5")
MODEL_PATH = os.path.normpath(MODEL_PATH)
CANVAS_SIZE = 480
ROI_SIZE = 360
WINDOW_SIZE = "1200x900"
FONT_LARGE = ("Arial", 18)
FONT_MEDIUM = ("Arial", 14)
MIN_CONTOUR_AREA = 100
CAMERA_PRED_INTERVAL = 1000  # ms
AUTO_RECOGNIZE_DELAY = 1500  # ms пауза після малювання, щоб автоматично розпізнати

# ====================


def deskew(img28):
    m = cv2.moments(img28)
    if abs(m.get('mu02', 0.0)) < 1e-2:
        return img28
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * 28 * skew],
                    [0, 1, 0]])
    img = cv2.warpAffine(img28, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderValue=0)
    return img


def center_image(img28):
    m = cv2.moments(img28)
    if m.get('m00', 0) == 0:
        return img28
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    shiftx = int(round(img28.shape[1] / 2.0 - cx))
    shifty = int(round(img28.shape[0] / 2.0 - cy))
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    shifted = cv2.warpAffine(img28, M, (28, 28), borderValue=0)
    return shifted


def resize_and_center_digit(digit_img):
    h, w = digit_img.shape
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    canvas = deskew(canvas)
    canvas = center_image(canvas)
    return canvas


def extract_digits_from_pil(img_pil, min_area=MIN_CONTOUR_AREA, separate_touching=True):
    gray = img_pil.convert("L")
    arr = np.array(gray)
    if arr.mean() > 127:
        arr = 255 - arr
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    _, thr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if separate_touching:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thr = cv2.erode(thr, kernel, iterations=1)
        thr = cv2.dilate(thr, kernel, iterations=1)

    contours, _ = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = thr.shape
        side = max(h, w)
        canvas_big = np.zeros((side, side), dtype=np.uint8)
        ly = (side - h) // 2
        lx = (side - w) // 2
        canvas_big[ly:ly + h, lx:lx + w] = thr
        resized = cv2.resize(canvas_big, (20, 20), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[4:24, 4:24] = resized
        canvas = deskew(canvas)
        canvas = center_image(canvas)
        return [canvas]

    filtered = []
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        pad = int(0.12 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(thr.shape[1], x + w + pad)
        y2 = min(thr.shape[0], y + h + pad)
        digit = thr[y1:y2, x1:x2]
        proc = resize_and_center_digit(digit)
        filtered.append(proc)

    if not filtered:
        return extract_digits_from_pil(img_pil, min_area=0, separate_touching=False)
    return filtered


class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer — optimized")
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg="#f0f0f0")

        self.model = load_model(MODEL_PATH)

        # Canvas frame with border and shadow effect
        self.canvas_frame = tk.Frame(root, bg="#444", bd=2, relief="ridge")
        self.canvas_frame.pack(pady=10)
        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.result_frame = tk.Frame(root, bg="white", bd=2, relief="groove")
        self.result_frame.pack(pady=10, fill="x", padx=20)
        self.result_label = tk.Label(self.result_frame, text="Draw digits or choose source",
                                     font=FONT_LARGE, bg="white", fg="#333")
        self.result_label.pack(pady=8)

        btn_frame = tk.Frame(root, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_canvas, width=12)
        self.clear_btn.grid(row=0, column=0, padx=8)
        self.recognize_btn = tk.Button(btn_frame, text="Recognize", command=lambda: self.predict_from_image(self.image),
                                       width=12)
        self.recognize_btn.grid(row=0, column=1, padx=8)
        self.load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image, width=12)
        self.load_btn.grid(row=0, column=2, padx=8)
        self.camera_btn = tk.Button(btn_frame, text="Camera OFF", command=self.toggle_camera, width=12, bg="#e74c3c",
                                    fg="white")
        self.camera_btn.grid(row=0, column=3, padx=8)

        roi_frame = tk.Frame(root, bg="#f0f0f0")
        roi_frame.pack(pady=8)
        tk.Label(roi_frame, text="ROI size:", bg="#f0f0f0", font=FONT_MEDIUM).pack(side="left")
        self.roi_scale = tk.Scale(roi_frame, from_=160, to=640, orient=tk.HORIZONTAL, length=300, bg="#f0f0f0",
                                  command=self.update_roi_size)
        self.roi_scale.set(ROI_SIZE)
        self.roi_scale.pack(side="left", padx=10)

        brush_frame = tk.Frame(root, bg="#f0f0f0")
        brush_frame.pack(pady=6)
        tk.Label(brush_frame, text="Brush size: use mouse wheel over canvas", bg="#f0f0f0", font=FONT_MEDIUM).pack()
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)

        self.camera_on = False
        self.cap = None
        self.last_cam_pred = 0
        self.after_id = None

        # Для автоматичного розпізнавання після паузи малювання
        self.last_paint_time = 0
        self.brush_radius = 12

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Старт перевірки паузи в малюванні
        self.check_auto_recognize()

    def paint(self, event):
        if self.camera_on:
            return  # Забороняємо малювати під час роботи камери
        x, y = event.x, event.y
        r = self.brush_radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        self.last_paint_time = time.time() * 1000  # зберігаємо час останнього малювання
        self.result_label.config(text="Drawing...")

    def on_mouse_wheel(self, event):
        if self.camera_on:
            return  # Заборонити змінювати пензель під час камери
        delta = 0
        if hasattr(event, 'delta'):
            delta = event.delta
        elif event.num == 4:
            delta = 120
        elif event.num == 5:
            delta = -120
        if delta > 0:
            self.brush_radius = min(50, self.brush_radius + 1)
        else:
            self.brush_radius = max(1, self.brush_radius - 1)
        self.result_label.config(text=f"Brush radius: {self.brush_radius}")

    def clear_canvas(self):
        if self.camera_on:
            return  # Не можна чистити, поки камера ввімкнена
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
        self.result_label.config(text="Draw digits or choose source")

    def predict_from_image(self, img_pil, camera_mode=False):
        digits = extract_digits_from_pil(img_pil)
        if not digits:
            self.result_label.config(text="No digits found")
            return
        preds = []
        confs = []
        for d in digits:
            pil_img = Image.fromarray(d)  # Конвертуємо numpy у PIL Image
            lab, conf = self.tta_predict(pil_img, camera_mode=camera_mode)
            preds.append(str(lab))
            confs.append(conf)
        result_str = "".join(preds)
        avg_conf = float(np.mean(confs))
        per = ", ".join([f"{p}:{c:.0f}%" for p, c in zip(preds, confs)])
        self.result_label.config(text=f"Result: {result_str}   (avg {avg_conf:.1f}%)  [{per}]")

    def load_image(self):
        if self.camera_on:
            return  # Не можна завантажувати під час камери
        p = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not p:
            return
        img = Image.open(p)
        img = ImageOps.grayscale(img)
        self.predict_from_image(img, camera_mode=False)

    def toggle_camera(self):
        if not self.camera_on:
            self.camera_on = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.result_label.config(text="Cannot open camera")
                self.camera_on = False
                return
            self.camera_btn.config(text="Camera ON", bg="#27ae60")
            self.last_cam_pred = 0
            self.update_camera()
        else:
            self.camera_on = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.after_id:
                try:
                    self.root.after_cancel(self.after_id)
                except Exception:
                    pass
            self.camera_btn.config(text="Camera OFF", bg="#e74c3c")
            self.result_label.config(text="Camera stopped")
            self.clear_canvas()
            # Повернути можливість малювати (очистити полотно теж можна)

    def update_roi_size(self, val):
        global ROI_SIZE
        try:
            ROI_SIZE = int(val)
        except Exception:
            pass

    def update_camera(self):
        if not (self.camera_on and self.cap is not None and self.cap.isOpened()):
            return
        ret, frame = self.cap.read()
        if ret:
            h, w, _ = frame.shape
            x1, y1 = max(0, w // 2 - ROI_SIZE // 2), max(0, h // 2 - ROI_SIZE // 2)
            x2, y2 = min(w, x1 + ROI_SIZE), min(h, y1 + ROI_SIZE)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            pil = Image.fromarray(gray)

            now = int(time.time() * 1000)
            if now - self.last_cam_pred >= CAMERA_PRED_INTERVAL:
                threading.Thread(target=self.safe_predict, args=(pil,)).start()
                self.last_cam_pred = now

            # Показуємо відео безпосередньо на canvas
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((CANVAS_SIZE, CANVAS_SIZE))
            self.photo_image = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        self.after_id = self.root.after(40, self.update_camera)

    def safe_predict(self, pil_img):
        try:
            self.predict_from_image(pil_img, camera_mode=True)
        except Exception as e:
            print("Prediction error:", e)

    def tta_predict(self, img, camera_mode=False):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        variants = []
        variants.append(img)

        shifts = [-2, 0, 2]
        for dx in shifts:
            for dy in shifts:
                if dx == 0 and dy == 0:
                    continue
                shifted = ImageChops.offset(img, dx, dy)
                if dx > 0:
                    shifted.paste(0, box=(0, 0, dx, img.height))
                elif dx < 0:
                    shifted.paste(0, box=(img.width + dx, 0, img.width, img.height))
                if dy > 0:
                    shifted.paste(0, box=(0, 0, img.width, dy))
                elif dy < 0:
                    shifted.paste(0, box=(0, img.height + dy, img.width, img.height))
                variants.append(shifted)

        for angle in [-10, -5, 5, 10]:
            rotated = img.rotate(angle, fillcolor=0)
            variants.append(rotated)

        flipped_h = ImageOps.mirror(img)
        flipped_v = ImageOps.flip(img)
        variants.append(flipped_h)
        variants.append(flipped_v)

        enhancer = ImageEnhance.Contrast(img)
        variants.append(enhancer.enhance(0.9))
        variants.append(enhancer.enhance(1.1))

        arrs = []
        for v in variants:
            arr = np.array(v).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = arr[..., None]
            arrs.append(arr)

        X = np.array(arrs)

        preds = self.model.predict(X, verbose=0)
        avg_pred = preds.mean(axis=0)
        digit = int(np.argmax(avg_pred))
        confidence = float(avg_pred[digit]) * 100
        return digit, confidence

    def check_auto_recognize(self):
        if not self.camera_on:
            now = time.time() * 1000
            if self.last_paint_time > 0 and now - self.last_paint_time > AUTO_RECOGNIZE_DELAY:
                self.predict_from_image(self.image)
                self.last_paint_time = 0
        self.root.after(500, self.check_auto_recognize)

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
