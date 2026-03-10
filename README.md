# IRIS: Intelligent Retinal Image System

**A Multi-Label Classification Architecture for RFMiD 2.0**

The IRIS system is a highly specialized algorithmic ensemble built to accurately map 51 independent retinal disease pathologies simultaneously from fundus imagery. Originally constructed as a naive binary classifier, IRIS has been completely rebuilt to act as a robust **Multi-Label Imbalanced Dataset (MMID)** engine capable of resolving the extreme data scarcities natively found in the optical clinic.

---

## 🏗️ Architectural Paradigm

The training splits provided by the RFMiD 2.0 dataset (60/20/20) contain ultra-rare (Tier 3) diseases that possess fewer than 10 positive occurrences. A traditional CrossEntropy approach fundamentally shatters under this constraint.

**Core Upgrades Include:**

1. **51-Class BCEWithLogitsLoss**: All classification heads scale via `pos_weight = min(N_neg / N_pos, 20.0)` calculated statically to prevent NaN degradation on rare classes.
2. **Backbone Differential LRs**: The core representations (Swin-Tiny, ViT-Base, ResNet50) train at `1e-5`, while the linear projection heads learn actively at `1e-4` to prevent catastrophic forgetting.
3. **Cross-Domain Augmentation**: Dynamic Gaussian Blurs, Color Jittering, and Random Erasing algorithms simulate variations between **TOPCON** and **CARL ZEISS** fundus cameras.
4. **Ensemble Inference Array**: A premium FastAPI backend loads all trained backbones into memory conditionally and outputs a unified average probability threshold algorithm directly to the browser.

---

## 🚀 Running the Production Server

The entire local application system is self-contained. Assuming you have already pushed the data through the training pipelines:

```powershell
cd code
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then navigate to `http://localhost:8000` via your optical tablet or web browser to see the live Glassmorphic Multi-Label interface and perform real-time disease thresholding.

---

## 🔬 Executing Model Pipelines

If you intend on retraining the weights from scratch against a new set of data batches, they must adhere exactly to the defined 51-class CSV headers. Ensure your dataset is split securely inside `c:/lpulab/IRIS_CODE/data/...`.

1. **Generate Core Clinical Bounds**

    ```powershell
    cd code
    python rfmid_data_analysis.py --train-csv "..\data\Training_set\RFMiD_2_Training_labels.csv" --output-dir "utils"
    ```

2. **Sequentially Kickoff Backbone Training**

    ```powershell
    python swin_train.py --train-csv "..\data\Training_set\RFMiD_2_Training_labels.csv" --val-csv "..\data\Validation_set\RFMiD_2_Validation_labels.csv" --train-img-dir "..\data\Training_set" --val-img-dir "..\data\Validation_set" --weights-json "utils\rfmid_pos_weights.json" --epochs 60
    ```

    *(Duplicate the command for `vit_train.py` and `cnn_train.py` respectively).*

3. **Evaluate Pathological Subclasses**

    ```powershell
    python evaluate_models.py --model-type swin --model-path "swin_model_multilabel.pth" --val-csv "..\data\Validation_set\RFMiD_2_Validation_labels.csv" --test-csv "..\data\Test_set\RFMiD_2_Testing_labels.csv" --val-img-dir "..\data\Validation_set" --test-img-dir "..\data\Test_set"
    ```

    This sweeps the validation probabilities between threshold bounds of `0.05 - 0.95` to dynamically lock the perfect threshold per class before deploying to the holdout Test set.

---
📘 See **[CHANGELOG_BENEFITS.md](CHANGELOG_BENEFITS.md)** for a rigorous changelog of legacy fixes regarding float normalization and hidden directory filtration issues.
