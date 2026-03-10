import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import json

# The EXACT 51 classes of RFMiD 2.0. Must match rfmid_data_analysis.py exactly.
CLASS_NAMES = [
    'WNL', 'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO', 'CRS',
    'CRVO', 'CSR', 'CWS', 'CSC', 'DN', 'DR', 'EDN', 'ERM', 'GRT', 'HPED', 'HR', 'LS',
    'MCA', 'ME', 'MH', 'MHL', 'MS', 'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH',
    'RD', 'RHL', 'RTR', 'RP', 'RPEC', 'RS', 'RT', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 
    'VS', 'HTN', 'IIH'
]

class RFMiD2Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with multi-label annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional PyTorch transform to be applied on a sample.
        """
        self.labels_df = pd.read_csv(csv_file, encoding="latin1")
        self.labels_df.rename(columns=lambda x: str(x).strip(), inplace=True)
        self.img_dir = img_dir
        self.transform = transform

        # Validate columns
        missing_cols = [c for c in CLASS_NAMES if c not in self.labels_df.columns]
        if missing_cols:
            raise ValueError(f"CSV is missing required 51 disease columns: {missing_cols}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ID mapping
        img_id = str(self.labels_df.iloc[idx, 0])
        
        # Robust extension checking
        if img_id.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            img_name = os.path.join(self.img_dir, img_id)
        else:
            # Fallback to checking disk if extension is missing from CSV
            jpg_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            png_path = os.path.join(self.img_dir, f"{img_id}.png")
            if os.path.exists(jpg_path):
                img_name = jpg_path
            elif os.path.exists(png_path):
                img_name = png_path
            else:
                raise FileNotFoundError(f"Could not find image for ID {img_id} in {self.img_dir}")

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract the 51 labels securely using predefined order
        labels = self.labels_df.iloc[idx][CLASS_NAMES].values.astype('float32')
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return image, labels_tensor

def load_pos_weights(weights_json_path, device='cpu'):
    """
    Loads the securely calculated pos_weights vector into a PyTorch tensor.
    """
    if not os.path.exists(weights_json_path):
        raise FileNotFoundError(f"Weights file not found at {weights_json_path}. Run rfmid_data_analysis.py first.")
        
    with open(weights_json_path, 'r') as f:
        data = json.load(f)
        
    weights_vector = data['pos_weights_vector']
    return torch.tensor(weights_vector, dtype=torch.float32).to(device)
