from torchvision import transforms

# Unified augmentation pipeline tailored explicitly to RFMiD 2.0 
# (simulating TOPCON and CARL ZEISS camera variance)
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # Retinal images are symmetric
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05), # Gentle color shifting
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Emulates camera lens variations
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), value='random'), # Simulates small occluded lesions
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """ Strict validation and testing transforms. No augmentation allowed. """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
