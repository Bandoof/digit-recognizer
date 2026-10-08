"""Compatibility entry point; prefer python -m digit_recognizer.training."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from digit_recognizer.training import main  # noqa: E402

if __name__ == "__main__":
    main()
