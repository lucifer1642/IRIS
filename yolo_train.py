import os
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

TRAIN_DATA_PATH = 'C:/FP/newdata/images/training'
TEST_FOLDER = 'C:/FP/newdata/images/testing'
TEST_CSV = 'C:/FP/newdata/test_labels.csv'
MODEL_WEIGHTS = 'yolov8n-cls.pt'
BEST_WEIGHTS = 'runs/classify/train/weights/best.pt'
METRICS_CSV = 'runs/classify/train/results.csv'
EPOCHS = 10
IMG_SIZE = 224
BATCH_SIZE = 16

LABEL_MAP = {'non-disease': 0, 'disease': 1}


def train():
    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=TRAIN_DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0 if torch.cuda.is_available() else 'cpu'
    )


def evaluate():
    test_csv = pd.read_csv(TEST_CSV)
    test_csv['label'] = test_csv['label'].map(LABEL_MAP)

    model = YOLO(BEST_WEIGHTS)
    y_true, y_pred = [], []

    for _, row in test_csv.iterrows():
        img_path = os.path.join(TEST_FOLDER, row['image'])
        results = model(img_path)
        y_pred.append(results[0].probs.top1)
        y_true.append(row['label'])

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
    plt.savefig('confusion_matrix.png')
    plt.show()


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
    plt.savefig('train_plots.png')
    plt.show()


if __name__ == "__main__":
    train()
    evaluate()
    plot_training_curves()
