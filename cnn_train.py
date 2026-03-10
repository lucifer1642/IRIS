import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = r"C:\FP\newdata\images"
TRAIN_PATH = os.path.join(DATA_DIR, "training")
TEST_PATH = os.path.join(DATA_DIR, "testing")
MODEL_SAVE_PATH = r"C:\Users\kiit\Desktop\x\retina_cnn_binary_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_data_generators():
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    test_gen = datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, test_gen


def plot_training_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Test Acc')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_gen):
    preds = model.predict(test_gen)
    preds_bin = (preds > 0.5).astype(int)
    true_labels = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(true_labels, preds_bin, target_names=class_labels))

    cm = confusion_matrix(true_labels, preds_bin)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    train_gen, test_gen = get_data_generators()
    model = build_model()
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")
    plot_training_curves(history)
    evaluate_model(model, test_gen)
