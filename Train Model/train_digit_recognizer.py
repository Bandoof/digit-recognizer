from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "Model" / "mnist.h5"


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(train_images)

    model.fit(datagen.flow(train_images, train_labels, batch_size=64),
              epochs=5,
              validation_data=(test_images, test_labels))

    _, test_acc = model.evaluate(test_images, test_labels)
    print(f"Точність на тесті: {test_acc * 100:.2f}%")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Модель збережено: {MODEL_PATH}")


if __name__ == "__main__":
    main()
