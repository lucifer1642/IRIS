import pandas as pd
import os
from tqdm import tqdm

PROJECT_FOLDER = "/content/drive/MyDrive/retina_project"
TRAINING_CSV = os.path.join(PROJECT_FOLDER, "training_labels.csv")
VALIDATION_CSV = os.path.join(PROJECT_FOLDER, "validation_labels.csv")
TESTING_CSV = os.path.join(PROJECT_FOLDER, "testing_labels.csv")


def load_csv(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"Loaded: {file_path}")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    print(f"File not found: {file_path}")
    return None


def analyze_dataset(dataset_name, df):
    if df is None:
        print(f"Skipping {dataset_name}: not loaded.")
        return

    print(f"\n--- {dataset_name} DATASET ---")

    if 'Disease_Risk' in df.columns:
        print(f"\nClass Distribution:\n{df['Disease_Risk'].value_counts()}")

    missing = df.isnull().sum()
    print(f"\nMissing Values:\n{missing}")
    print(f"Total NaN: {missing.sum()}")

    disease_cols = df.columns[2:]
    print(f"\nDisease Activity:")
    for col in disease_cols:
        count = df[col].value_counts().get(1, 0)
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {col}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    datasets = {
        "TRAINING": load_csv(TRAINING_CSV),
        "VALIDATION": load_csv(VALIDATION_CSV),
        "TESTING": load_csv(TESTING_CSV),
    }

    for name, df in tqdm(datasets.items(), desc="Analyzing", unit="dataset"):
        analyze_dataset(name, df)
