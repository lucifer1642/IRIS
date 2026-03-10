import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

SAVE_PATH = "/content/drive/MyDrive/retina_project/vit_model_final.pth"
TEST_DIR = "/content/drive/MyDrive/retina_project/images/test"
BATCH_SIZE = 16
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


def load_model():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = nn.Linear(model.config.hidden_size, NUM_CLASSES)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")
    return model


def get_test_loader():
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
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
    plt.show()


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
    plt.show()


if __name__ == "__main__":
    model = load_model()
    test_loader, class_names = get_test_loader()
    print(f"Classes: {class_names}")

    all_labels, all_preds = run_inference(model, test_loader)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_cumulative_accuracy(model, test_loader)
