import os
import re

base_dir = r"C:\lpulab\IRIS_CODE\code"
files = [
    "rfmid_data_analysis.py",
    "utils/dataset.py",
    "swin_train.py",
    "vit_train.py",
    "cnn_train.py",
    "evaluate_models.py",
    "app.py"
]

# 51 Exact Classes from CSV Header
NEW_LIST = """CLASS_NAMES = [
    'WNL', 'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO', 'CRS',
    'CRVO', 'CSR', 'CWS', 'CSC', 'DN', 'DR', 'EDN', 'ERM', 'GRT', 'HPED', 'HR', 'LS',
    'MCA', 'ME', 'MH', 'MHL', 'MS', 'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH',
    'RD', 'RHL', 'RTR', 'RP', 'RPEC', 'RS', 'RT', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 
    'VS', 'HTN', 'IIH'
]"""

for file in files:
    path = os.path.join(base_dir, file)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Regex to replace existing CLASS_NAMES
    text = re.sub(r'CLASS_NAMES\s*=\s*\[[\s\S]*?\]', NEW_LIST, text)

    # Convert strictly the string "49" to "51" to update nn.Linear and indices
    text = text.replace('49', '51')

    # Specific dataset fixes for robustness (Latin1 + Stripping unprintables like ON\xa0)
    if 'dataset.py' in file:
        text = text.replace('self.labels_df = pd.read_csv(csv_file)', 
                            'self.labels_df = pd.read_csv(csv_file, encoding="latin1")\\n        self.labels_df.rename(columns=lambda x: str(x).strip(), inplace=True)')
    
    if 'rfmid_data_analysis' in file:
        text = text.replace('df = pd.read_csv(csv_path)',
                            'df = pd.read_csv(csv_path, encoding="latin1")\\n    df.rename(columns=lambda x: str(x).strip(), inplace=True)')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

print("SUCCESS: Synced 51-class multi-label dimensions across all pipelines.")
