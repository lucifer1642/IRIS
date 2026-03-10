import os

base_dir = r"C:\lpulab\IRIS_CODE\code"
train_scripts = ["swin_train.py", "vit_train.py", "cnn_train.py"]

for file in train_scripts:
    path = os.path.join(base_dir, file)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Safely swap img-dir into train and val directories
    text = text.replace('parser.add_argument("--img-dir", type=str, required=True, help="Path to dataset images directory")', 
                        'parser.add_argument("--train-img-dir", type=str, required=True, help="Path to training images")\n    parser.add_argument("--val-img-dir", type=str, required=True, help="Path to validation images")')
    text = text.replace('val_dataset = RFMiD2Dataset(args.val_csv, args.img_dir, transform=get_val_transforms())',
                        'val_dataset = RFMiD2Dataset(args.val_csv, args.val_img_dir, transform=get_val_transforms())')
    text = text.replace('train_dataset = RFMiD2Dataset(args.train_csv, args.img_dir, transform=get_train_transforms())',
                        'train_dataset = RFMiD2Dataset(args.train_csv, args.train_img_dir, transform=get_train_transforms())')
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# Patch evaluate_models.py
eval_path = os.path.join(base_dir, "evaluate_models.py")
with open(eval_path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace('parser.add_argument("--img-dir", type=str, required=True, help="Path to images directory")',
                    'parser.add_argument("--val-img-dir", type=str, required=True, help="Path to validation images")\n    parser.add_argument("--test-img-dir", type=str, required=True, help="Path to testing images")')
text = text.replace('val_dataset = RFMiD2Dataset(args.val_csv, args.img_dir, transform=get_val_transforms())',
                    'val_dataset = RFMiD2Dataset(args.val_csv, args.val_img_dir, transform=get_val_transforms())')
text = text.replace('test_dataset = RFMiD2Dataset(args.test_csv, args.img_dir, transform=get_val_transforms())',
                    'test_dataset = RFMiD2Dataset(args.test_csv, args.test_img_dir, transform=get_val_transforms())')

with open(eval_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Patching complete.")
