import os
import cv2
import numpy as np

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess retinal images for CNN/Swin/ViT")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the images directory (e.g., C:/FP/newdata/images)")
    parser.add_argument("--size", type=int, default=224, help="Target image size (default: 224)")
    return parser.parse_args()


def preprocess_and_save_images(folder_path, target_size):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                img_path = os.path.join(root, file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read: {img_path}")
                        continue
                    
                    # Resize image directly
                    img_resized = cv2.resize(img, target_size)
                    
                    # Save the resized image directly (normalization happens dynamically during training via Rescaling)
                    cv2.imwrite(img_path, img_resized)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    args = get_args()
    target_size = (args.size, args.size)
    print(f"Preprocessing images in: {args.input_dir}")
    print(f"Target size: {target_size}")
    preprocess_and_save_images(args.input_dir, target_size)
    print("All images resized and saved in-place.")
