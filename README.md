# IRIS – Retinal Disease Classification
## File Overview

| File | Purpose | Framework |
|------|---------|-----------|
| `cnn_preprocess.py` | Resize + normalize all retinal images in-place | OpenCV |
| `cnn_train.py` | Train binary CNN, plot curves, save `.h5` | TensorFlow/Keras |
| `yolo_data_analysis.py` | Analyze class distributions in RFMiD CSVs | Pandas |
| `yolo_prepare_data.py` | Generate YOLO `.txt` label files + YAML config | Ultralytics format |
| `yolo_train.py` | Train YOLOv8n-cls, evaluate, plot confusion matrix | Ultralytics |
| `vit_train.py` | Fine-tune ViT-Base on retinal data, save `.pth` | PyTorch + HuggingFace |
| `vit_evaluate.py` | Load saved ViT, run inference, confusion matrix | PyTorch |
| `swin_train.py` | Train Swin-Tiny with augmentation + LR scheduler | PyTorch + HuggingFace |
| `swin_vs_cnn_eval.py` | Side-by-side Swin vs CNN comparison on test set | Mixed (PT + TF) |

## Execution Order

### CNN
```bash
python cnn_preprocess.py   # Run once before training
python cnn_train.py
```

### YOLOv8n
```bash
python yolo_data_analysis.py   # Optional: inspect dataset
python yolo_prepare_data.py    # Generate labels + YAML
python yolo_train.py
```

### ViT
```bash
python vit_train.py     # Saves vit_model_final.pth
python vit_evaluate.py  # Loads saved model and evaluates
```

### Swin Transformer
```bash
python swin_train.py          # Saves swin_model.pth
python swin_vs_cnn_eval.py    # Compares Swin vs CNN on test set
```

## Key Paths to Update
- All scripts use hardcoded paths — update `DATA_DIR`, `SAVE_PATH`, etc. before running
- Colab scripts assume Google Drive mounted at `/content/drive/MyDrive/retina_project/`
- Local CNN scripts use `C:\FP\newdata\images\`

## Results Summary

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| CNN | 90.0% | 91.26% | 89.52% | 90.37% |
| YOLOv8n | 89.06% | 88% | 87% | 88% |
| ViT | 90.0% | 90.5% | 91% | 90% |
| Swin | **95.43%** | **95.38%** | **95.43%** | **95.40%** |
