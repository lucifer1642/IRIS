# IRIS Codebase: Algorithmic Changes and Strategic Benefits

This document provides a highly detailed breakdown of the recent algorithmic fixes, structural refinements, and logical corrections applied across the IRIS Retinal Disease Classification repository.

These changes transition the codebase from "prototype script" quality to a robust, mathematically sound, and deployable Machine Learning pipeline.

---

## 1. `cnn_preprocess.py`: Preventing Lossy Data Degradation

### **What Changed:**

- **Removed destructive normalization loop**: Previously, the script was performing `img = img / 255.0` to normalize pixel values to `[0, 1]`, but it then immediately attempted to save the image to disk using `cv2.imwrite(img_path, img * 255)`.
- **Added extension safety filters**: Implemented an explicit tuple check `('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')` via Python's `os.walk()`.

### **How It Works & Why It's Beneficial:**

OpenCV's `imwrite` naturally expects matrices in `uint8` format with pixel intensities from `0-255`. Floating-point manipulations `(img / 255.0)` cause microscopic loss of precision (quantization error) when scaled back up and cast down to integers, heavily degrading the quality of the retinal fluid/capillary details needed for medical imaging.

Now, the script **purely resizes** the images efficiently and saves them unaltered. The pixel scaling is delegated correctly to the deep learning framework itself via `tf.keras.layers.Rescaling(1.0 / 255)` embedded dynamically in the training pipeline pipeline. Additionally, the `os.walk()` filter prevents the script from randomly crashing if it encounters hidden system files (like `.DS_Store` or `desktop.ini`).

---

## 2. `cnn_train.py`: Eliminating Evaluation Bias and Shape Mismatches

### **What Changed:**

- **Inserted `test_gen.reset()`**: Placed directly before model prediction in the evaluation phase.
- **Fixed Boolean array shapes**: Added `.flatten()` to the prediction matrix `(preds.flatten() > 0.5)` so it matches the 1D structure of `true_labels`.
- **Modernized model serialization**: Upgraded `.h5` extension saving to the new native `.keras` Keras format.

### **How It Works & Why It's Beneficial:**

Keras `ImageDataGenerator` yields batches sequentially but doesn't guarantee the internal pointer starts at index `0` when evaluation starts—meaning the generated output probabilities could completely misalign with the array of `test_gen.classes`, generating functionally random classification reports. `test_gen.reset()` mathematically guarantees the sequences are paired correctly.

Fixing the array `.flatten()` prevents NumPy and Scikit-Learn broadcasting errors where a matrix of `[N, 1]` is improperly subtracted against `[N]`, causing precision/recall logic to fail. Switching to `.keras` natively packages custom layers seamlessly without throwing serialization warnings in TF 2.13+.

---

## 3. `swin_train.py`: Massive Training Loop Speedup

### **What Changed:**

- **Relocated Scikit-Learn Metrics**: Lifted `precision_score`, `recall_score`, and `f1_score` calculations completely out of the inner batch loop to instead execute **once** at the end of the `epoch` loop.
- **Fixed unhandled Dropout parameters**: Passed `hidden_dropout_prob` gracefully into the HuggingFace `SwinConfig` rather than assigning it dynamically via `model.dropout = nn.Dropout()`.

### **How It Works & Why It's Beneficial:**

Scikit-Learn metrics calculate mathematically expensive algorithms on CPU arrays. Running precision/recall formulas thousands of times *per batch* creates massive I/O bottlenecks and cripples GPU performance because the CPU has to halt the DataLoader. By tallying `all_predictions` and calculating the metrics strictly at the end of the epoch, epoch training speed is increased exponentially.

Furthermore, naive variable assignment to `model.dropout` does not inject dropout masks into the HuggingFace `forward()` passes. It's essentially "dead code". By correctly mapping `hidden_dropout_prob` via `ignore_mismatched_sizes=True`, the Vision Transformer layers correctly drop neurons to prevent overfitting on the minor variations of retinal scans.

---

## 4. `swin_vs_cnn_eval.py`: Binary Output Thresholding

### **What Changed:**

- **Rewrote the `argmax()` CNN evaluation**: Prevented TF's `argmax` from attempting to evaluate the CNN model's binary `Sigmoid` output layer.

### **How It Works & Why It's Beneficial:**

Because the CNN uses a Dense output dimension of 1 (`Dense(1, activation='sigmoid')`), outputting a single probability (e.g. `0.85`), calling `tf.argmax(predictions, axis=1)` forces TF to look for the highest array index along an axis that only possesses a *single* element (index 0). Thus, the model would seemingly *always* predict `class 0` (`non-disease`).
By catching the shape correctly (`predictions.shape[-1] == 1`), the pipeline now executes a true mathematical binary threshold `(predictions.flatten() > 0.5).astype(int)` against the sigmoid probability, accurately deriving the true positive/false positive scores without zeroing out.

---

## 5 & 6. PyTorch ViT Ecosystem (`vit_train.py`, `vit_evaluate.py`)

### **What Changed:**

- **Corrected Batch Checks**: Swapped `.nelement()` check algorithms to `.numel()` during `DataLoader` iterations.
- **Robust Label Processing**: Modified Pandas algorithms handling IDs to intelligently process missing extensions instead of blindly appending strings `x + ".png"`.

### **How It Works & Why It's Beneficial:**

In older PyTorch, `.nelement()` was an alias. However, `.numel()` is the highly optimized C++ backend command to count tensor cells. More importantly, real-world data science datasets are messy. If an ID column contains `PatientA_v2.jpg` instead of `PatientA`, the old logic would blindly string-format it to `PatientA_v2.jpg.png`, crashing the loader with a `FileNotFoundError`. The new lambda extraction checks if a valid extension exists before writing to the target directory.

---

## 7. `yolo_data_analysis.py`: Clean Logic Parsing

### **What Changed:**

- **Missing Variable Reporting**: Added condition `missing[missing > 0]` rather than dumping massive strings of initialized zeroes to the terminal log.

### **How It Works & Why It's Beneficial:**

Pandas `.isnull().sum()` lists every single column variable in memory (amounting to potentially dozens of columns). Terminal output clutter hides major developer bugs. By slicing the dataframe explicitly for columns suffering from data sparsity `( > 0 )`, developers can immediately diagnose parsing errors in the RFMiD datasets.

---

## 8. `yolo_prepare_data.py`: Native Classification Directory Routing

### **What Changed:**

- **Rebuilt Script for Folder Scaffolding**: Deleted the script generating bounding box `.txt` layout patterns and designed an automated directory builder that sorts images into `yolo_dataset/train/disease`, `yolo_dataset/train/non-disease`, etc.

### **How It Works & Why It's Beneficial:**

`yolov8n-cls.pt` is an image-classification pipeline, *not* an object detection pipeline. YOLO classification engines natively require inputs organized via literal folder hierarchy (Dataset Folder -> Split Folder -> Class Folder). The previous `.txt` logic was attempting to bind bounding-box architecture to the classifier. By creating the correct `shutil.copy` scaffolding mapped mathematically to `CLASS_NAMES`, YOLO can natively ingest the retinal scans at maximum speed directly into the GPU via Ultralytics built-in PyTorch dataloaders.

---

## 9. `yolo_train.py`: Unlocking Native Ultralytics Metric Reporting

### **What Changed:**

- **Pointed to Directory Structure**: Scrapped the `.yaml` pointer code in favor of directly referencing the directory tree generated above.
- **Fixed Pandas KeyErrors via `.str.strip()`**: Filtered Ultralytics' arbitrarily generated CSV whitespace.

### **How It Works & Why It's Beneficial:**

By structuring the dataset robustly as a folder, YOLO abstracts away validation mapping—automating its own validation loss curves cleanly without generating complex mapping yaml files.
More importantly, when YOLO spits out training logs (`results.csv`), it arbitrarily tabs its columns to make them legible in notepad (e.g., `'     train/loss'`). Pandas crashes if it targets `'train/loss'`. The `.str.strip()` command programmatically erases these spaces on load, dynamically isolating `accuracy_top1` variables so standard `matplotlib` figures can accurately trace the model's convergence over time without crashing randomly based on Ultralytics pip versioning.
