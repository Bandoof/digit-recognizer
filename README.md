# Handwritten Digit Recognizer

A desktop application that recognizes handwritten digits with a convolutional neural network trained on MNIST.

## Features

- Draw one or more digits on the canvas.
- Load a PNG or JPEG image.
- Recognize digits from a webcam feed.
- Adjust the drawing brush and camera recognition area.

## Requirements

- Python 3.10 or 3.11
- A webcam (optional)
- Tkinter, usually included with Python. On Ubuntu/Debian, install it with `sudo apt install python3-tk`.

## Installation

```bash
git clone https://github.com/Bandoof/digit-recognizer.git
cd digit-recognizer
python -m venv .venv
```

Activate the virtual environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies and run the application:

```bash
python -m pip install -r requirements.txt
python GUI/GUI.py
```

Draw a digit with the left mouse button. Use the mouse wheel over the canvas to change the brush size. The application recognizes the drawing after a short pause or when you select **Recognize**.

## Training the model

The repository includes a pretrained model at `Model/mnist.h5`. To retrain it:

```bash
python "Train Model/train_digit_recognizer.py"
```

The training script downloads MNIST, trains for five epochs, evaluates the model, and replaces `Model/mnist.h5`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
GUI/GUI.py                              Desktop application and image preprocessing
Model/mnist.h5                          Pretrained Keras model
Train Model/train_digit_recognizer.py   Model training script
tests/                                  Automated preprocessing and UI-logic tests
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
