import os
import yaml
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_FOLDER = "/content/drive/MyDrive/retina_project"
LABELS_FOLDER = os.path.join(PROJECT_FOLDER, "labels")
YAML_PATH = os.path.join(PROJECT_FOLDER, "retinadata.yaml")

CSV_PATHS = {
    "train": os.path.join(PROJECT_FOLDER, "training_labels.csv"),
    "val": os.path.join(PROJECT_FOLDER, "validation_labels.csv"),
    "test": os.path.join(PROJECT_FOLDER, "testing_labels.csv"),
}


def generate_label_files(csv_path, folder_type):
    labels_subfolder = os.path.join(LABELS_FOLDER, folder_type)
    os.makedirs(labels_subfolder, exist_ok=True)

    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    with tqdm(total=len(df), desc=f"Labeling {folder_type}", unit="file") as pbar:
        for img_id, risk_label in zip(df['ID'], df['Disease_Risk']):
            label_file = os.path.join(labels_subfolder, f"{img_id}.txt")
            with open(label_file, 'w') as f:
                f.write('1\n' if risk_label == 1 else '0\n')
            pbar.update(1)

    print(f"Labels written to: {labels_subfolder}")


def save_yaml():
    data = {
        'train': os.path.join(PROJECT_FOLDER, 'images/train'),
        'val': os.path.join(PROJECT_FOLDER, 'images/val'),
        'test': os.path.join(PROJECT_FOLDER, 'images/test'),
        'nc': 2,
        'names': ['non-disease', 'disease']
    }
    with tqdm(total=100, desc="Saving YAML", unit="%") as pbar:
        with open(YAML_PATH, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        for _ in range(100):
            time.sleep(0.01)
            pbar.update(1)
    print(f"YAML saved at: {YAML_PATH}")


def plot_split():
    counts = {'Train': 1920, 'Validation': 640, 'Test': 640}
    plt.figure(figsize=(7, 5))
    plt.bar(counts.keys(), counts.values(), color=['blue', 'green', 'red'])
    plt.xlabel('Dataset')
    plt.ylabel('Number of Images')
    plt.title('Dataset Split Visualization')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for split, path in CSV_PATHS.items():
        generate_label_files(path, split)
    save_yaml()
    plot_split()
