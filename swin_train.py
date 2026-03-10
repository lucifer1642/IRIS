import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
from transformers import SwinForImageClassification, SwinConfig
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def get_args():
    parser = argparse.ArgumentParser(description="Train Swin Transformer on Retinal Images")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training images directory")
    parser.add_argument("--save-path", type=str, default="swin_model.pth", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def build_model(num_classes, dropout_rate=0.2):
    # Set hidden_dropout_prob in the config so it actually gets applied internally
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        hidden_dropout_prob=dropout_rate,
        ignore_mismatched_sizes=True,
        num_labels=num_classes
    )
    return model.to(device)


def train(args):
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    num_classes = len(train_dataset.classes)

    model = build_model(num_classes, args.dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # verbose=True emits a deprecation warning in PyTorch 2.x
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=2
    )

    losses, accuracies = [], []
    all_precision, all_recall, all_f1 = [], [], []

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_predictions, all_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = (correct / total) * 100
        scheduler.step(avg_loss)

        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        losses.append(avg_loss)
        accuracies.append(accuracy)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, "
              f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved at {args.save_path}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(all_true, all_preds)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, args.epochs + 1), losses, marker='o', color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, args.epochs + 1), accuracies, marker='o', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(1, args.epochs + 1), all_precision, marker='o', label='Precision')
    plt.plot(range(1, args.epochs + 1), all_recall, marker='s', label='Recall')
    plt.plot(range(1, args.epochs + 1), all_f1, marker='^', label='F1')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('swin_training_curves.png')
    print("Saved training curves to 'swin_training_curves.png'")
    try:
        plt.show()
    except Exception:
        pass

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(train_dataset.classes))
    plt.xticks(tick_marks, train_dataset.classes, rotation=45, ha='right')
    plt.yticks(tick_marks, train_dataset.classes)
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('swin_confusion_matrix.png')
    print("Saved confusion matrix to 'swin_confusion_matrix.png'")
    try:
        plt.show()
    except Exception:
        pass

    print(f"\nFinal Metrics: Acc={accuracies[-1]:.2f}%, P={all_precision[-1]:.4f}, "
          f"R={all_recall[-1]:.4f}, F1={all_f1[-1]:.4f}")


if __name__ == "__main__":
    args = get_args()
    train(args)
