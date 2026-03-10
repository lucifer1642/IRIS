import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model for retinal disease binary classification")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the images directory containing 'training' and 'testing' subfolders")
    parser.add_argument("--save-path", type=str, default="retina_cnn_binary_model.h5", help="Path to save the trained model .h5 file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    return parser.parse_args()

IMG_SIZE = (224, 224)


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


def get_data_generators(train_path, test_path, batch_size):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary'
    )
    test_gen = datagen.flow_from_directory(
        test_path,
        target_size=IMG_SIZE,
        batch_size=batch_size,
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
    plt.savefig('cnn_training_curves.png')
    print("Saved training curves to 'cnn_training_curves.png'")
    try:
        plt.show()
    except Exception:
        pass


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
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png')
    print("Saved confusion matrix to 'cnn_confusion_matrix.png'")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    args = get_args()
    train_path = os.path.join(args.data_dir, "training")
    test_path = os.path.join(args.data_dir, "testing")
    
    train_gen, test_gen = get_data_generators(train_path, test_path, args.batch_size)
    model = build_model()
    history = model.fit(train_gen, epochs=args.epochs, validation_data=test_gen)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    model.save(args.save_path)
    print(f"Model saved at {args.save_path}")
    
    plot_training_curves(history)
    evaluate_model(model, test_gen)
