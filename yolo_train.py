import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8n Classification")
    parser.add_argument("--yolo-data-dir", type=str, required=True, help="Path to formatted YOLO dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    return parser.parse_args()

BEST_WEIGHTS = 'runs/classify/train/weights/best.pt'
METRICS_CSV = 'runs/classify/train/results.csv'

def train(args):
    model = YOLO('yolov8n-cls.pt')
    model.train(
        data=args.yolo_data_dir,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        project='runs/classify',
        name='train',
        exist_ok=True
    )

def plot_training_curves():
    if not os.path.exists(METRICS_CSV): 
        return

    df = pd.read_csv(METRICS_CSV)
    df.columns = df.columns.str.strip() # Strip arbitrary spaces appended by YOLO logger natively

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if 'train/loss' in df.columns:
        plt.plot(df['epoch'], df['train/loss'], label='Train Loss', color='blue')
        plt.legend()
        plt.grid(True)
    
    # Automatically locate the custom accuracy column tag provided by YOLO instance
    acc_key = next((col for col in df.columns if 'accuracy_top1' in col or 'accuracy' in col), None)
    plt.subplot(1, 2, 2)
    if acc_key:
        plt.plot(df['epoch'], df[acc_key], label='Accuracy', color='green')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('yolo_train_plots.png')
    print("Saved training curves to 'yolo_train_plots.png'")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    args = get_args()
    train(args)
    plot_training_curves()
