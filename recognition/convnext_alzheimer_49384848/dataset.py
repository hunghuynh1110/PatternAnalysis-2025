"""
dataset.py
Dataset loader for the AD vs NC JPEG dataset
Author: Hung Huynh (s4938484)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from constants import IMAGE_SIZE, CLASSES, MEAN, STD, IMG_EXTS

NUM_WORKER = os.cpu_count() // 2

class MRIDataset2D(Dataset):
    """MRI 2D JPEG slice dataset for Alzheimer’s classification (AD vs NC)."""
    def __init__(self, root_dir, classes=CLASSES, transform=None):
        self.classes = classes
        self.class2idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.samples = []

        img_exts = IMG_EXTS

        for c in classes:
            class_dir = os.path.join(root_dir, c)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Missing folder: {class_dir}")
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(img_exts):
                    self.samples.append((os.path.join(class_dir, fname), self.class2idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No image files found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN,STD)
            ])(img)
        return img, label


def get_loaders(data_root, batch_size=16, val_fraction=0.1, seed=42):
    """
    Build train / val / test DataLoaders assuming:
      data_root/train/{AD, NC}, data_root/test/{AD, NC}
    """

    # Define transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])

    # Load full training dataset
    train_root = os.path.join(data_root, 'train')
    full_train = MRIDataset2D(train_root, CLASSES, transform=train_tf)

    # Split into train / val
    n_total = len(full_train)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    # Load test dataset
    test_root = os.path.join(data_root, 'test')
    if not os.path.isdir(test_root):
        # fallback if test folder missing
        test_root = train_root
    test_ds = MRIDataset2D(test_root, CLASSES, transform=val_tf)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKER, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER, pin_memory=True)

    return train_loader, val_loader, test_loader



# Demo block (runs ONLY when executing this file directly, not on import)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DATA_ROOT = "./AD_NC"  # folder alongside this file
    tl, vl, _ = get_loaders(DATA_ROOT, batch_size=4, val_fraction=0.1)

    xb, yb = next(iter(tl))
    print("Train batch:", xb.shape, yb[:4])

    # visualize first 4
    plt.figure(figsize=(10,3))
    for i in range(min(4, xb.size(0))):
        plt.subplot(1, 4, i+1)
        plt.imshow(xb[i].permute(1,2,0)[:, :, 0], cmap='gray')
        plt.title(CLASSES[yb[i].item()])
        plt.axis('off')
    plt.show()