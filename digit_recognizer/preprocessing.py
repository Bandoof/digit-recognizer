"""Convert canvas, file, or camera images into MNIST-like digit crops."""

import cv2
import numpy as np
from PIL import Image, ImageOps

MIN_CONTOUR_AREA = 100


def deskew(image):
    moments = cv2.moments(image)
    if abs(moments["mu02"]) < 1e-2:
        return image
    skew = moments["mu11"] / moments["mu02"]
    matrix = np.float32([[1, skew, -14 * skew], [0, 1, 0]])
    return cv2.warpAffine(
        image,
        matrix,
        (28, 28),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        borderValue=0,
    )


def center_image(image):
    moments = cv2.moments(image)
    if moments["m00"] == 0:
        return image
    dx = int(round(14 - moments["m10"] / moments["m00"]))
    dy = int(round(14 - moments["m01"] / moments["m00"]))
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))


def resize_and_center_digit(image):
    if image.ndim != 2 or image.size == 0:
        raise ValueError("A digit crop must be a nonempty grayscale image.")
    height, width = image.shape
    scale = 20 / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x, y = (28 - size[0]) // 2, (28 - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = resized
    return center_image(deskew(canvas))


def extract_digits_from_pil(image, min_area=MIN_CONTOUR_AREA, separate_touching=True):
    """Return uint8 28×28 crops, left to right; empty/noisy images return [].

    Opening removes small specks but cannot reliably split touching digits.
    The area threshold scales down for images smaller than the 480px canvas.
    """
    if image.width == 0 or image.height == 0:
        raise ValueError("The image is empty.")
    if min_area < 0:
        raise ValueError("min_area must be nonnegative.")
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    gray = background.convert("L")
    gray.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    array = np.asarray(gray)
    if np.ptp(array) < 8:
        return []
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    if np.median(border) > 127:
        array = 255 - array
    blurred = cv2.GaussianBlur(array, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if separate_touching and min(array.shape) >= 64:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    threshold = min_area * min(1, max(array.shape) / 480) ** 2
    digits = []
    for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        if cv2.contourArea(contour) < max(1, threshold):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        pad = int(0.12 * max(width, height))
        crop = binary[max(0, y - pad) : y + height + pad, max(0, x - pad) : x + width + pad]
        digits.append(resize_and_center_digit(crop))
    return digits
