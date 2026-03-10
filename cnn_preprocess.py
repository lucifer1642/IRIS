import os
import cv2
import numpy as np

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess retinal images for CNN/Swin/ViT")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the images directory (e.g., C:/FP/newdata/images)")
    parser.add_argument("--size", type=int, default=224, help="Target image size (default: 224)")
    return parser.parse_args()


def preprocess_and_save_images(folder_path):
    for subfolder in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(sub_path):
            for image_name in os.listdir(sub_path):
                img_path = os.path.join(sub_path, image_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read: {img_path}")
                        continue
                    img = cv2.resize(img, target_size)
                    cv2.imwrite(img_path, img)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    args = get_args()
    target_size = (args.size, args.size)
    print(f"Preprocessing images in: {args.input_dir}")
    print(f"Target size: {target_size}")
    preprocess_and_save_images(args.input_dir)
    print("All images resized and saved in-place.")
