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

## Setup and Requirements

The project requires Python 3.8+ and the following key libraries:

```bash
pip install torch torchvision tensorflow transformers ultralytics opencv-python pandas matplotlib seaborn tqdm scikit-learn
```

## Execution Order

### CNN

```bash
python cnn_preprocess.py --input-dir <path_to_images> --size 224
python cnn_train.py --data-dir <path_to_train_test_folders> --save-path retina_cnn_binary_model.h5
```

### YOLOv8n

```bash
python yolo_data_analysis.py --project-dir <path_to_project_dir>
python yolo_prepare_data.py --project-dir <path_to_project_dir> --images-dir <path_to_images> --output-dir yolo_dataset
python yolo_train.py --yolo-data-dir yolo_dataset
```

### ViT

```bash
# Example
python vit_train.py --train-dir data/train --val-dir data/val --test-dir data/test --val-csv validation_labels.csv --test-csv testing_labels.csv --save-path vit_model_final.pth
python vit_evaluate.py  --test-dir data/test --model-path vit_model_final.pth
```

### Swin Transformer

```bash
python swin_train.py --data-dir <path_to_data> --save-path swin_model.pth
python swin_vs_cnn_eval.py --test-dir <path_to_test_dir> --swin-model swin_model.pth --cnn-model retina_cnn_binary_model.h5
```

## Results Summary

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| CNN | 90.0% | 91.26% | 89.52% | 90.37% |
| YOLOv8n | 89.06% | 88% | 87% | 88% |
| ViT | 90.0% | 90.5% | 91% | 90% |
| Swin | **95.43%** | **95.38%** | **95.43%** | **95.40%** |
