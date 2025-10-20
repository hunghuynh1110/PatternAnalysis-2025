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
import numpy as np

from constants import IMAGE_SIZE, CLASSES, MEAN, STD, IMG_EXTS, NUM_WORKER

import re
from sklearn.model_selection import GroupShuffleSplit


def compute_mean_std(loader, device='mps', max_batches=200):
    """Compute dataset mean and std (for normalization) entirely on MPS."""
    import torch
    device = torch.device(device)

    n = 0
    ch_sum = torch.zeros(3, device=device)
    ch_sq = torch.zeros(3, device=device)

    for i, (x, _) in enumerate(loader):
        if i == max_batches:
            break
        x = x.to(device)
        b = x.size(0)
        n += b
        ch_sum += x.mean(dim=[0, 2, 3]) * b
        ch_sq += (x ** 2).mean(dim=[0, 2, 3]) * b

    # Keep all math on the same MPS device
    mean = ch_sum / n
    std = (ch_sq / n - mean ** 2).sqrt()

    # Only move to CPU at the end for printing
    return mean.cpu().tolist(), std.cpu().tolist()

def _extract_subject_id(path: str):
    # assumes ".../AD/123456_97.jpeg"
    fname = os.path.basename(path)
    m = re.match(r"(\d+)_", fname)
    return m.group(1) if m else fname  # fallback

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

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
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.01),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        transforms.Normalize(MEAN, STD)
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

    # after building full list in MRIDataset2D(...).samples
    # Build arrays for group split

    all_paths  = np.array([p for p, _ in full_train.samples])
    all_labels = np.array([y for _, y in full_train.samples])
    groups     = np.array([_extract_subject_id(p) for p in all_paths])

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(all_paths, all_labels, groups))

    # Subset datasets by indices
    from torch.utils.data import Subset
    train_ds = Subset(full_train, train_idx)
    val_ds   = Subset(full_train, val_idx)
    

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



# # Demo block (runs ONLY when executing this file directly, not on import)
# if __name__ == "__main__":
#     from torchvision import transforms

# # Temporary transform: resize and convert to tensor ONLY
# raw_tf = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor()
# ])

# # Build an unnormalized dataset
# raw_ds = MRIDataset2D("./AD_NC/train", CLASSES, transform=raw_tf)
# raw_loader = DataLoader(raw_ds, batch_size=32, shuffle=False)

# mean, std = compute_mean_std(raw_loader, device='mps')
# print(mean, std)

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