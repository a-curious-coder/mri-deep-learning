"""Train a small CNN classifier for the public demo.

Separate from image_data/train_test_models.py, which is built against a
private clinical (ADNI/OASIS-style) dataset this repo doesn't ship. This
script trains against Falah/Alzheimer_MRI (Apache-2.0, public, no auth
needed, ~28MB) so the public demo can do real inference without requiring
gated data. Run once locally; commit the resulting checkpoint.
"""
import numpy as np
import tensorflow as tf
from datasets import load_dataset

IMAGE_SIZE = 128
NUM_CLASSES = 4
CLASS_NAMES = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]
MODEL_PATH = "models/alzheimer_classifier.keras"


def to_arrays(split):
    images = np.stack([
        np.array(row["image"].convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))
        for row in split
    ])
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, -1)
    labels = np.array(split["label"])
    return images, labels


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])


def main():
    print("Downloading dataset (Falah/Alzheimer_MRI, Apache-2.0)...")
    ds = load_dataset("Falah/Alzheimer_MRI")

    print("Preparing arrays...")
    x_train, y_train = to_arrays(ds["train"])
    x_test, y_test = to_arrays(ds["test"])

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )

    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {acc:.3f}, loss: {loss:.3f}")

    import os
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
