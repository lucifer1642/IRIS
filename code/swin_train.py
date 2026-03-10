import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
from transformers import SwinForImageClassification
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils.dataset import RFMiD2Dataset, load_pos_weights, CLASS_NAMES
from utils.augmentations import get_train_transforms, get_val_transforms

def get_args():
    parser = argparse.ArgumentParser(description="Train Swin Transformer for Multi-Label Classification")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--train-img-dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--val-img-dir", type=str, required=True, help="Path to validation images")
    parser.add_argument("--weights-json", type=str, required=True, help="Path to rfmid_pos_weights.json")
    parser.add_argument("--save-path", type=str, default="swin_model_multilabel.pth", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def build_model(num_classes=51, dropout_rate=0.3):
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        hidden_dropout_prob=dropout_rate,
        ignore_mismatched_sizes=True,
        num_labels=num_classes
    )
    return model.to(device)

def train(args):
    # Load Datasets
    train_dataset = RFMiD2Dataset(args.train_csv, args.train_img_dir, transform=get_train_transforms())
    val_dataset = RFMiD2Dataset(args.val_csv, args.val_img_dir, transform=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load Pos Weights
    pos_weight_tensor = load_pos_weights(args.weights_json, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Build Model
    model = build_model(num_classes=51, dropout_rate=0.3)
    
    # Differential LRs
    optimizer = torch.optim.AdamW([
        {'params': model.swin.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6, min_lr=1e-7
    )
    
    scaler = torch.cuda.amp.GradScaler()

    best_val_f1 = 0.0
    patience_counter = 0
    patience_limit = 10

    train_losses, val_f1_scores = [], []

    for epoch in range(args.epochs):
        # Linear Warmup (3 epochs)
        if epoch < 3:
            warmup_factor = (epoch + 1) / 3.0
            optimizer.param_groups[0]['lr'] = 1e-5 * warmup_factor
            optimizer.param_groups[1]['lr'] = 1e-4 * warmup_factor

        model.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(images).logits
                    probs = torch.sigmoid(logits)
                
                # Use standard 0.5 threshold ONLY for early stopping metrics
                # Optimal threshold sweeping is done in purely in evaluate_models.py
                preds = (probs > 0.5).int()
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        
        # Macro F1 is our explicit early stopping metric target
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1_scores.append(val_macro_f1)

        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Macro-F1={val_macro_f1:.4f}")

        # Reduce LR on Plateau
        if epoch >= 3:
            scheduler.step(val_macro_f1)

        # Early Stopping Logic
        if val_macro_f1 > best_val_f1 + 0.003:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Best model saved! Macro F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  -> No significant improvement. Patience: {patience_counter}/{patience_limit}")

        if epoch >= 19 and patience_counter >= patience_limit: # Minimum 20 epochs
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    # Plot final curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('BCEWithLogitsLoss')
    plt.grid()
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores, label='Val Macro F1', color='green')
    plt.title('Validation Macro F1')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('swin_multilabel_training.png')
    print("Saved training curves to swin_multilabel_training.png")

if __name__ == "__main__":
    args = get_args()
    train(args)
