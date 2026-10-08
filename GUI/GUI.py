"""Compatibility entry point; prefer python -m digit_recognizer.gui."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from digit_recognizer.gui import DigitRecognizer, main  # noqa: E402, F401
from digit_recognizer.preprocessing import (  # noqa: E402, F401
    center_image,
    deskew,
    extract_digits_from_pil,
    resize_and_center_digit,
)

if __name__ == "__main__":
    raise SystemExit(main())
