import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pytest

from digit_recognizer.preprocessing import (
    center_image,
    deskew,
    extract_digits_from_pil,
    resize_and_center_digit,
)


@pytest.mark.parametrize("color", [0, 128, 255])
def test_blank_image_has_no_digits(color):
    assert extract_digits_from_pil(Image.new("L", (480, 480), color)) == []


def test_noise_is_not_promoted_to_a_digit():
    image = Image.new("L", (480, 480), 255)
    ImageDraw.Draw(image).point((200, 200), fill=0)
    assert extract_digits_from_pil(image) == []


def test_polarities_have_same_normalized_digit(drawn_one):
    dark = extract_digits_from_pil(drawn_one)
    light = extract_digits_from_pil(ImageOps.invert(drawn_one))
    assert len(dark) == len(light) == 1
    np.testing.assert_array_equal(dark[0], light[0])
    assert dark[0].shape == (28, 28) and dark[0].dtype == np.uint8
    moments = cv2.moments(dark[0])
    assert abs(moments["m10"] / moments["m00"] - 14) <= 1
    assert abs(moments["m01"] / moments["m00"] - 14) <= 1


def test_transparency_is_composited_on_white():
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    assert extract_digits_from_pil(image) == []
    ImageDraw.Draw(image).rectangle((40, 10, 60, 90), fill=(0, 0, 0, 255))
    assert len(extract_digits_from_pil(image)) == 1


def test_multiple_digits_are_sorted_left_to_right():
    image = Image.new("RGB", (480, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 80, 70, 350), fill="black")
    draw.rectangle((200, 100, 350, 330), fill="black")
    digits = extract_digits_from_pil(image)
    assert len(digits) == 2
    widths = [np.count_nonzero(d.max(axis=0)) for d in digits]
    assert widths[0] < widths[1]


def test_small_input_and_border_digit_are_supported():
    image = Image.new("L", (28, 28), 0)
    ImageDraw.Draw(image).line((0, 0, 0, 27), fill=255, width=3)
    assert len(extract_digits_from_pil(image)) == 1


@pytest.mark.parametrize("shape", [(0, 0), (2, 2, 3)])
def test_invalid_crop_rejected(shape):
    with pytest.raises(ValueError):
        resize_and_center_digit(np.zeros(shape, dtype=np.uint8))


def test_empty_moments_are_safe():
    image = np.zeros((28, 28), dtype=np.uint8)
    np.testing.assert_array_equal(center_image(image), image)
    np.testing.assert_array_equal(deskew(image), image)


def test_min_area_is_respected(drawn_one):
    assert extract_digits_from_pil(drawn_one, min_area=1_000_000) == []
    with pytest.raises(ValueError):
        extract_digits_from_pil(drawn_one, min_area=-1)


def test_exif_orientation_is_applied(drawn_one):
    rotated = drawn_one.transpose(Image.Transpose.ROTATE_90)
    rotated.getexif()[274] = 6  # clockwise rotation to restore portrait orientation
    expected = extract_digits_from_pil(drawn_one)[0]
    actual = extract_digits_from_pil(rotated)[0]
    np.testing.assert_array_equal(actual, expected)
