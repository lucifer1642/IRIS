import os
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Prepare YOLO classification dataset directory structure")
    parser.add_argument("--project-dir", type=str, required=True, help="Path to project directory containing CSVs")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to source images folder containing train/val/test subfolders")
    parser.add_argument("--output-dir", type=str, default="yolo_dataset", help="Path to output formatted dataset directory")
    return parser.parse_args()

CLASS_NAMES = {0: "non-disease", 1: "disease"}

def prepare_yolo_classification_structure(csv_path, split_name, images_source_folder, yolo_dataset_folder):
    if not os.path.exists(csv_path): 
        print(f"CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    source_img_dir = os.path.join(images_source_folder, split_name)
    
    for class_name in CLASS_NAMES.values():
        os.makedirs(os.path.join(yolo_dataset_folder, split_name, class_name), exist_ok=True)

    with tqdm(total=len(df), desc=f"Structuring {split_name}") as pbar:
        for img_id, risk_label in zip(df['ID'], df['Disease_Risk']):
            img_filename = f"{img_id}.png"
            src_path = os.path.join(source_img_dir, img_filename)
            class_folder = CLASS_NAMES.get(risk_label, "unknown")
            dst_path = os.path.join(yolo_dataset_folder, split_name, class_folder, img_filename)
            
            if os.path.exists(src_path): 
                shutil.copy(src_path, dst_path)
            pbar.update(1)

if __name__ == "__main__":
    args = get_args()
    
    csv_paths = {
        "train": os.path.join(args.project_dir, "training_labels.csv"),
        "val": os.path.join(args.project_dir, "validation_labels.csv"),
        "test": os.path.join(args.project_dir, "testing_labels.csv")
    }
    
    for split, path in csv_paths.items():
        prepare_yolo_classification_structure(path, split, args.images_dir, args.output_dir)
        
    print(f"\nYOLO dataset successfully formatted at: {args.output_dir}")
