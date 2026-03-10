import os
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate YOLOv8n Classification")
    parser.add_argument("--data-yaml", type=str, required=True, help="Path to retinadata.yaml")
    parser.add_argument("--test-folder", type=str, required=True, help="Path to test images folder")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test labels CSV")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    return parser.parse_args()

MODEL_WEIGHTS = 'yolov8n-cls.pt'
BEST_WEIGHTS = 'runs/classify/train/weights/best.pt'
METRICS_CSV = 'runs/classify/train/results.csv'
LABEL_MAP = {'non-disease': 0, 'disease': 1}


def train(args):
    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=0 if torch.cuda.is_available() else 'cpu'
    )


def evaluate(args):
    test_csv = pd.read_csv(args.test_csv)
    
    label_col = 'label' if 'label' in test_csv.columns else 'Disease_Risk'
    image_col = 'image' if 'image' in test_csv.columns else 'ID'
    
    if test_csv[label_col].dtype == object:
        test_csv[label_col] = test_csv[label_col].map(LABEL_MAP)

    if os.path.exists(BEST_WEIGHTS):
        print(f"\nLoading best model from {BEST_WEIGHTS}")
        model = YOLO(BEST_WEIGHTS)
    else:
        print("\nCould not find 'best.pt', evaluating with current weights")
        model = YOLO(MODEL_WEIGHTS)

    y_true, y_pred = [], []

    for _, row in test_csv.iterrows():
        img_name = str(row[image_col])
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.png'
            
        img_path = os.path.join(args.test_folder, img_name)
        if not os.path.exists(img_path):
            continue
            
        results = model(img_path, verbose=False)
        y_pred.append(results[0].probs.top1)
        y_true.append(row[label_col])

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['non-disease', 'disease']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non-disease', 'disease'],
                yticklabels=['non-disease', 'disease'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('yolo_confusion_matrix.png')
    print("Saved confusion matrix to 'yolo_confusion_matrix.png'")
    try:
        plt.show()
    except Exception:
        pass


def plot_training_curves():
    df = pd.read_csv(METRICS_CSV)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df[' epoch'], df[' train/loss'], label='Train Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(df[' epoch'], df[' metrics/accuracy(B)'], label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('yolo_training_curves.png')
    print("Saved training curves to 'yolo_training_curves.png'")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    args = get_args()
    train(args)
    evaluate(args)
    if os.path.exists(METRICS_CSV):
        plot_training_curves()
    else:
        print(f"Metrics file {METRICS_CSV} not found, skipping curve plotting.")
