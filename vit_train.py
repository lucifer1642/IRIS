import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification
from torch.optim import AdamW
from tqdm import tqdm

TRAIN_DIR = "/content/drive/MyDrive/retina_project/images/train"
VAL_DIR = "/content/drive/MyDrive/retina_project/images/val"
TEST_DIR = "/content/drive/MyDrive/retina_project/images/test"
VAL_CSV = "/content/drive/MyDrive/retina_project/validation_labels.csv"
TEST_CSV = "/content/drive/MyDrive/retina_project/testing_labels.csv"
SAVE_PATH = "/content/drive/MyDrive/retina_project/vit_model_final.pth"

EPOCHS = 10
BATCH_SIZE = 16
LR = 3e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


class TrainDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label_name in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label_name)
            if os.path.isdir(label_path):
                label = 1 if label_name.lower() == 'disease' else 0
                for img_name in os.listdir(label_path):
                    self.samples.append((os.path.join(label_path, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except (FileNotFoundError, UnidentifiedImageError):
            return None


class CSVImageDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_path)
        if 'ID' in self.data.columns and 'Disease_Risk' in self.data.columns:
            self.data['filename'] = self.data['ID'].apply(lambda x: str(x) if str(x).endswith(('.png', '.jpg')) else f"{x}.png")
            self.data['label'] = self.data['Disease_Risk'].astype(int)
        else:
            raise ValueError("CSV must have 'ID' and 'Disease_Risk' columns.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, row['label']
        except (FileNotFoundError, UnidentifiedImageError):
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def build_model():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = nn.Linear(model.config.hidden_size, 2)
    return model.to(device)


def train():
    train_dataset = TrainDataset(TRAIN_DIR, transform=transform)
    val_dataset = CSVImageDataset(VAL_DIR, VAL_CSV, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_skip_none)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    train_losses, val_accuracies = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, total_batches = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            if images.numel() == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_batches += 1

        avg_loss = epoch_loss / max(total_batches, 1)
        train_losses.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images.numel() == 0:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved at {SAVE_PATH}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), val_accuracies, marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
