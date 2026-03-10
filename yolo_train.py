import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Deprecation Notice: YOLOv8 Classification")
    parser.add_argument("--yolo-data-dir", type=str, default="yolo_dataset", help="Path to YOLO formatted dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    return parser.parse_args()

def main():
    args = get_args()
    print("="*80)
    print("WARNING: ARCHITECTURAL DEPRECATION")
    print("YOLOv8-cls is designed for single-label, multi-class architectures.")
    print("RFMiD 2.0 is a 49-class Multi-Label Imbalanced Dataset (MMID).")
    print("Forcing YOLOv8-cls into a multi-label prediction paradigm violates its native")
    print("loss function (CrossEntropy) and feature extractor design.")
    print("\nAs per architectural constraints, the YOLOv8 pipeline is currently suspended")
    print("for classification. To perform bounding box lesion detection, a separate set")
    print("of YOLO coordinate annotations (.txt) must be provided in the future.")
    print("="*80)

if __name__ == "__main__":
    main()
