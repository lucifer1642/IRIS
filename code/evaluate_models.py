import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve
)
from utils.dataset import RFMiD2Dataset, CLASS_NAMES
from utils.augmentations import get_val_transforms

# Model Builders
from transformers import SwinForImageClassification, ViTForImageClassification
from torchvision.models import resnet50

def get_args():
    parser = argparse.ArgumentParser(description="Multi-Label Model Evaluation with Threshold Sweeping")
    parser.add_argument("--model-type", type=str, required=True, choices=["swin", "vit", "resnet50"])
    parser.add_argument("--model-path", type=str, required=True, help="Path to compiled weights")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to val CSV (for threshold sweeping)")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test CSV (for final evaluation)")
    parser.add_argument("--val-img-dir", type=str, required=True, help="Path to validation images")
    parser.add_argument("--test-img-dir", type=str, required=True, help="Path to testing images")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_type, model_path):
    if model_type == "swin":
        model = SwinForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224", ignore_mismatched_sizes=True, num_labels=51)
    elif model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", ignore_mismatched_sizes=True, num_labels=51)
    elif model_type == "resnet50":
        model = resnet50()
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(in_features, 51))
    else:
        raise ValueError("Invalid model type")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def get_probabilities(model, model_type, dataloader, desc="Inferring"):
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(device)
            if model_type in ["swin", "vit"]:
                logits = model(images).logits
            else:
                logits = model(images)
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.vstack(all_probs), np.vstack(all_labels)

def sweep_thresholds(val_probs, val_labels):
    print("\n--- Sweeping Validation Thresholds ---")
    thresholds = np.arange(0.05, 1.0, 0.05)
    optimal_thresholds = np.zeros(51)
    
    for c in range(51):
        best_f1 = -1
        best_th = 0.5
        
        y_true = val_labels[:, c]
        y_prob = val_probs[:, c]
        
        # If there are NO positive examples in val set for this class, we can't tune it properly.
        if np.sum(y_true) == 0:
            optimal_thresholds[c] = 0.3
            print(f"Class {CLASS_NAMES[c]:<5}: Val Positives=0 -> Defaulting threshold=0.30")
            continue
            
        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                
        # If best F1 is exactly 0.0 (model learned nothing/failed), fallback to 0.3 bias toward recall
        if best_f1 == 0.0:
            optimal_thresholds[c] = 0.3
        else:
            optimal_thresholds[c] = best_th
            
    return optimal_thresholds

def evaluate(args):
    print(f"Loading {args.model_type} from {args.model_path}...")
    model = load_model(args.model_type, args.model_path)
    
    val_dataset = RFMiD2Dataset(args.val_csv, args.val_img_dir, transform=get_val_transforms())
    test_dataset = RFMiD2Dataset(args.test_csv, args.test_img_dir, transform=get_val_transforms())
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 1. Validation Sweep
    val_probs, val_labels = get_probabilities(model, args.model_type, val_loader, "Val Sweep")
    optimal_thresholds = sweep_thresholds(val_probs, val_labels)
    
    # 2. Test Evaluation
    test_probs, test_labels = get_probabilities(model, args.model_type, test_loader, "Test Infer")
    
    # Apply optimal thresholds column-wise
    test_preds = (test_probs >= optimal_thresholds).astype(int)
    
    print("\n--- TEST EVALUATION (After Threshold Tuning) ---")
    
    # Per-Class Metrics
    metrics_list = []
    
    for c in range(51):
        y_true = test_labels[:, c]
        y_prob = test_probs[:, c]
        y_pred = test_preds[:, c]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
        except ValueError:
            # Occurs if a class has entirely 0 labels in the split
            auc = 0.0
            ap = 0.0
            
        metrics_list.append({
            'Class': CLASS_NAMES[c],
            'Threshold': optimal_thresholds[c],
            'Support': np.sum(y_true),
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'AUC-ROC': auc,
            'AP': ap
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    print(df_metrics.to_string(index=False))
    
    # Aggregate Metrics
    macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(test_labels, test_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    map_score = np.mean(df_metrics['AP'].values)
    classes_over_50 = len(df_metrics[df_metrics['F1'] > 0.5])
    
    print("\n--- AGGREGATE METRICS ---")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Micro F1:          {micro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print(f"mAP:               {map_score:.4f}")
    print(f"Classes F1 > 0.5:  {classes_over_50} / 51")
    
    # Visualization 1: Horizontal Bar Chart of F1
    df_metrics_sorted = df_metrics.sort_values('F1')
    plt.figure(figsize=(10, 12))
    plt.barh(df_metrics_sorted['Class'], df_metrics_sorted['F1'], color='skyblue')
    plt.xlabel('F1 Score')
    plt.title(f'{args.model_type.upper()} Per-Class F1 Scores (Test Set)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{args.model_type}_f1_bar_chart.png')
    
    # Visualization 2: AUC-ROC Bar Chart
    df_auc_sorted = df_metrics.sort_values('AUC-ROC')
    plt.figure(figsize=(10, 12))
    plt.barh(df_auc_sorted['Class'], df_auc_sorted['AUC-ROC'], color='lightgreen')
    plt.xlabel('AUC-ROC')
    plt.title(f'{args.model_type.upper()} Per-Class AUC-ROC (Test Set)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{args.model_type}_auc_bar_chart.png')
    
    # Visualization 3: PR Curves for Top 5 Support Classes (Tier 1 representation)
    top_5_classes = df_metrics.sort_values('Support', ascending=False).head(5)['Class'].tolist()
    plt.figure(figsize=(8, 8))
    for c in top_5_classes:
        c_idx = CLASS_NAMES.index(c)
        y_true = test_labels[:, c_idx]
        y_prob = test_probs[:, c_idx]
        if np.sum(y_true) > 0:
            p, r, _ = precision_recall_curve(y_true, y_prob)
            plt.plot(r, p, label=f'{c} (AP={df_metrics.iloc[c_idx]["AP"]:.2f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves (Tier 1) - {args.model_type.upper()}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{args.model_type}_pr_curves.png')
    
    print(f"Saved evaluation visualizations for {args.model_type}.")

if __name__ == "__main__":
    evaluate(get_args())
