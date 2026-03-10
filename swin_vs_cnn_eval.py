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
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare Swin Transformer vs CNN")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to testing images directory")
    parser.add_argument("--swin-model", type=str, default="swin_model.pth", help="Path to trained Swin model")
    parser.add_argument("--cnn-model", type=str, default="retina_cnn_binary_model.keras", help="Path to trained CNN model")
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

swin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Will be initialized in __main__
test_dataset_swin = None
test_loader_swin = None
num_classes = 2
class_names = ['disease', 'non-disease']


def prepare_tf_dataset(path, img_size=(224, 224), batch_size=16):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=img_size, batch_size=batch_size, shuffle=False
    )
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    return ds.map(lambda x, y: (norm(x), y))


def evaluate_swin(swin_model_path):
    print("\n===== Swin Transformer Evaluation =====")
    
    # Needs to match the initialization during training
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        ignore_mismatched_sizes=True,
        num_labels=num_classes
    )
    # The SwinForImageClassification model initializes the classifier based on num_labels
    model.load_state_dict(torch.load(swin_model_path, map_location=device))
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


def evaluate_cnn(cnn_model_path, test_data_path):
    print("\n===== CNN Evaluation =====")
    cnn_model = load_model(cnn_model_path)
    test_ds = prepare_tf_dataset(test_data_path)
    cnn_model.evaluate(test_ds, verbose=1)

    cnn_preds, true_labels = [], []
    for images, labels in test_ds:
        predictions = cnn_model.predict(images, verbose=0)
        
        # Determine if the output is binary (shape [N, 1]) or multi-class (shape [N, C])
        if predictions.shape[-1] == 1:
            # Binary sigmoid output
            pred_classes = (predictions.flatten() > 0.5).astype(int)
        else:
            # Multi-class softmax output
            pred_classes = tf.argmax(predictions, axis=1).numpy()
            
        cnn_preds.extend(pred_classes)
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
    plt.savefig('comparison_confusion_matrices.png')
    print("Saved confusion matrices to 'comparison_confusion_matrices.png'")
    try:
        plt.show()
    except Exception:
        pass


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
    plt.savefig('comparison_accuracy.png')
    print("Saved accuracy comparison to 'comparison_accuracy.png'")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    args = get_args()
    
    test_dataset_swin = datasets.ImageFolder(root=args.test_dir, transform=swin_transform)
    test_loader_swin = DataLoader(test_dataset_swin, batch_size=16, shuffle=False, num_workers=2)
    num_classes = len(test_dataset_swin.classes)
    class_names = test_dataset_swin.classes
    print(f"Loaded {len(test_dataset_swin)} test images across {num_classes} classes")

    swin_labels, swin_preds = evaluate_swin(args.swin_model)
    cnn_labels, cnn_preds = evaluate_cnn(args.cnn_model, args.test_dir)

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
