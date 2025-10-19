import os
print("Current working directory:", os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from dataset import get_loaders
from constants import (
    DEVICE_MPS, DEVICE_CUDA, DEVICE_CPU,
    BATCH_SIZE, EPOCHS, LR,
    DATA_ROOT,
    NUM_CLASSES
)

# Device setup
if torch.backends.mps.is_available():
    DEVICE = DEVICE_MPS
elif torch.cuda.is_available():
    DEVICE = DEVICE_CUDA
else:
    DEVICE = DEVICE_CPU

print(f"Using device: {DEVICE}")

# Load data
train_loader, val_loader, test_loader = get_loaders(
    data_root  = DATA_ROOT,
    batch_size = BATCH_SIZE
)

print(f"Train images: {len(train_loader.dataset)} | "
      f"Val images:   {len(val_loader.dataset)} | "
      f"Test images:  {len(test_loader.dataset)}")

# Model setup
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model   = convnext_tiny(weights=weights)
num_features    = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# (Continue with training loop…)
for epoch in range(1, EPOCHS + 1):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()             # reset gradients
        outputs = model(inputs)           # forward pass
        loss = criterion(outputs, labels) # compute loss
        loss.backward()                   # backpropagation
        optimizer.step()                  # update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            print(f'Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | '
                  f'Loss: {running_loss / batch_idx:.4f} | '
                  f'Acc: {100 * correct / total:.2f}%')

    # end of train epoch  
    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc  = 100 * val_correct / val_total

    print(f'Epoch {epoch} completed: '
          f'Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, '
          f'Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%')

    scheduler.step()
