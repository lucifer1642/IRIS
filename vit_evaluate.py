import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Vision Transformer (ViT) on Retinal Images")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to testing images directory")
    parser.add_argument("--model-path", type=str, default="vit_model_final.pth", help="Path to trained model weights")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for testing")
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


def load_model(model_path, num_classes):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        ignore_mismatched_sizes=True,
        num_labels=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")
    return model


def get_test_loader(test_dir, batch_size):
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset.classes


def run_inference(model, test_loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


def plot_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("ViT Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('vit_confusion_matrix.png')
    print("Saved confusion matrix to 'vit_confusion_matrix.png'")
    try:
        plt.show()
    except Exception:
        pass


def plot_cumulative_accuracy(model, test_loader):
    batch_accuracies = []
    running_correct, total_seen = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
            batch_accuracies.append(running_correct / total_seen)

    plt.figure(figsize=(8, 4))
    plt.plot(batch_accuracies, marker='o', label="Cumulative Accuracy")
    plt.xlabel("Batch Number")
    plt.ylabel("Accuracy")
    plt.title("ViT Cumulative Accuracy Across Test Batches")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('vit_cumulative_accuracy.png')
    print("Saved cumulative accuracy to 'vit_cumulative_accuracy.png'")
    try:
        plt.show()
    except Exception:
        pass

def main():
    args = get_args()
    
    test_loader, class_names = get_test_loader(args.test_dir, args.batch_size)
    num_classes = len(class_names)
    print(f"Classes found ({num_classes}): {class_names}")

    model = load_model(args.model_path, num_classes)
    all_labels, all_preds = run_inference(model, test_loader)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_cumulative_accuracy(model, test_loader)

if __name__ == "__main__":
    main()
