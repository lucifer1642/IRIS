import os
import yaml
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Prepare data labels and YAML for YOLOv8")
    parser.add_argument("--project-dir", type=str, required=True, help="Path to project directory containing CSVs and where to save output")
    parser.add_argument("--dataset-yaml", type=str, default="retinadata.yaml", help="Name of the output YAML file")
    return parser.parse_args()


def generate_label_files(csv_path, folder_type, labels_folder):
    labels_subfolder = os.path.join(labels_folder, folder_type)
    os.makedirs(labels_subfolder, exist_ok=True)

    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    with tqdm(total=len(df), desc=f"Labeling {folder_type}", unit="file") as pbar:
        for img_id, risk_label in zip(df['ID'], df['Disease_Risk']):
            label_file = os.path.join(labels_subfolder, f"{img_id}.txt")
            with open(label_file, 'w') as f:
                f.write('1\n' if risk_label == 1 else '0\n')
            pbar.update(1)

    print(f"Labels written to: {labels_subfolder}")
    return len(df)


def save_yaml(project_folder, yaml_path):
    data = {
        'train': os.path.join(project_folder, 'images', 'train'),
        'val': os.path.join(project_folder, 'images', 'val'),
        'test': os.path.join(project_folder, 'images', 'test'),
        'nc': 2,
        'names': ['non-disease', 'disease']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"YAML saved at: {yaml_path}")


def plot_split(counts, project_folder):
    plt.figure(figsize=(7, 5))
    plt.bar(counts.keys(), counts.values(), color=['blue', 'green', 'red'])
    plt.xlabel('Dataset')
    plt.ylabel('Number of Images')
    plt.title('Dataset Split Visualization')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(project_folder, 'yolo_data_split.png')
    plt.savefig(plot_path)
    print(f"Saved split visualization to {plot_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    args = get_args()
    project_folder = args.project_dir
    labels_folder = os.path.join(project_folder, "labels")
    yaml_path = os.path.join(project_folder, args.dataset_yaml)

    csv_paths = {
        "Train": ("train", os.path.join(project_folder, "training_labels.csv")),
        "Validation": ("val", os.path.join(project_folder, "validation_labels.csv")),
        "Test": ("test", os.path.join(project_folder, "testing_labels.csv")),
    }
    
    counts = {}
    for pretty_name, (split, path) in csv_paths.items():
        if os.path.exists(path):
            counts[pretty_name] = generate_label_files(path, split, labels_folder)
        else:
            print(f"Warning: {path} not found.")

    save_yaml(project_folder, yaml_path)
    if counts:
        plot_split(counts, project_folder)
