import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tensorflow as tf
from torch.utils.data import DataLoader
from transformers import SwinForImageClassification
from tensorflow.keras.models import load_model
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from tqdm import tqdm

TEST_DATA_PATH = "/content/drive/MyDrive/retina_project/images/test"
SWIN_MODEL_PATH = "/content/drive/MyDrive/retina_project/swin_model.pth"
CNN_MODEL_PATH = "/content/drive/MyDrive/retina_project/cnn_model.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

swin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset_swin = datasets.ImageFolder(root=TEST_DATA_PATH, transform=swin_transform)
test_loader_swin = DataLoader(test_dataset_swin, batch_size=16, shuffle=False, num_workers=2)
num_classes = len(test_dataset_swin.classes)
class_names = test_dataset_swin.classes
print(f"Loaded {len(test_dataset_swin)} test images across {num_classes} classes")


def prepare_tf_dataset(path, img_size=(224, 224), batch_size=16):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=img_size, batch_size=batch_size, shuffle=False
    )
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    return ds.map(lambda x, y: (norm(x), y))


def evaluate_swin():
    print("\n===== Swin Transformer Evaluation =====")
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224"
    )
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    model.load_state_dict(torch.load(SWIN_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader_swin, desc="Evaluating Swin"):
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


def evaluate_cnn():
    print("\n===== CNN Evaluation =====")
    cnn_model = load_model(CNN_MODEL_PATH)
    test_ds = prepare_tf_dataset(TEST_DATA_PATH)
    cnn_model.evaluate(test_ds, verbose=1)

    cnn_preds, true_labels = [], []
    for images, labels in test_ds:
        predictions = cnn_model.predict(images, verbose=0)
        cnn_preds.extend(tf.argmax(predictions, axis=1).numpy())
        true_labels.extend(labels.numpy())
    return true_labels, cnn_preds


def plot_confusion_matrices(swin_labels, swin_preds, cnn_labels, cnn_preds):
    swin_cm = confusion_matrix(swin_labels, swin_preds)
    cnn_cm = confusion_matrix(cnn_labels, cnn_preds)

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    sns.heatmap(swin_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Swin Transformer Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.subplot(1, 2, 2)
    sns.heatmap(cnn_cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()


def plot_accuracy_comparison(swin_acc, cnn_acc):
    plt.figure(figsize=(8, 5))
    plt.bar(['Swin Transformer', 'CNN'], [swin_acc, cnn_acc],
            color=['steelblue', 'coral'])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(80, 100)
    for i, v in enumerate([swin_acc, cnn_acc]):
        plt.text(i, v + 0.2, f"{v:.2f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()


if __name__ == "__main__":
    swin_labels, swin_preds = evaluate_swin()
    cnn_labels, cnn_preds = evaluate_cnn()

    swin_acc = accuracy_score(swin_labels, swin_preds) * 100
    cnn_acc = accuracy_score(cnn_labels, cnn_preds) * 100

    print("\nSwin Classification Report:")
    print(classification_report(swin_labels, swin_preds, target_names=class_names))
    print("\nCNN Classification Report:")
    print(classification_report(cnn_labels, cnn_preds, target_names=class_names))

    print(f"\n{'='*50}")
    print(f"Swin Accuracy: {swin_acc:.2f}%")
    print(f"CNN Accuracy : {cnn_acc:.2f}%")
    print(f"Difference   : {abs(swin_acc - cnn_acc):.2f}%")
    print(f"{'='*50}")

    plot_confusion_matrices(swin_labels, swin_preds, cnn_labels, cnn_preds)
    plot_accuracy_comparison(swin_acc, cnn_acc)
