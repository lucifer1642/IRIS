import os
import cv2
import numpy as np

ROOT_PATH = r"C:\FP\newdata\images"
TARGET_SIZE = (224, 224)


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
                    img = cv2.resize(img, TARGET_SIZE)
                    img = img / 255.0
                    img = (img * 255).astype(np.uint8)
                    cv2.imwrite(img_path, img)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    preprocess_and_save_images(ROOT_PATH)
    print("All images resized, normalized and saved in-place.")
