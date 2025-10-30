"""
train.py
Author: Gia Hung Huynh - 49384848
Date: 16th October 2025
Description:
    This script trains a ConvNeXt-based classifier on the ADNI dataset using PyTorch.
    It includes functions for training, validation, plotting metrics, and the main training loop.
"""



import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from modules import ConvNeXtMRI
import matplotlib.pyplot as plt
from utils import mixup_data, mixup_criterion, rand_bbox, cutmix_data

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


from dataset import get_loaders
from constants import (
    DEVICE_MPS, DEVICE_CUDA, DEVICE_CPU, EPOCHS,
    DATA_ROOT, BATCH_SIZE, LR, SWA_LR,
    WD, DROP_PATH_RATE,
    NUM_CLASSES
)

import os

# === Early Stopping config ===
EARLY_STOP = True
early_patience = 10       # stop if no improvement for N epochs
early_min_delta = 0.0     # required improvement amount
monitor_metric = "val_loss"  # "val_loss" (minimize) or "val_acc" (maximize)

# Internal trackers (no need to touch)
_no_improve_epochs = 0
if monitor_metric == "val_loss":
    _best_monitored = float("inf")  # lower is better
else:
    _best_monitored = float("-inf") # higher is better

# Device setup
if torch.backends.mps.is_available():
    DEVICE = DEVICE_MPS
elif torch.cuda.is_available():
    DEVICE = DEVICE_CUDA
else:
    DEVICE = DEVICE_CPU

from sklearn.metrics import roc_auc_score
import importlib, modules
importlib.reload(modules)
from modules import ConvNeXtMRI
from torch_ema import ExponentialMovingAverage

print("Current working directory:", os.getcwd())
print(f"Using device: {DEVICE}")

# Load data
# NOTE: We still load test_loader here because get_loaders returns 3 items,
# but it will not be used in the training or evaluation loops.
train_loader, val_loader, test_loader = get_loaders(
    data_root  =  DATA_ROOT,
    batch_size = BATCH_SIZE
)
print(f"Train images: {len(train_loader.dataset)} | "
    f"Val images:   {len(val_loader.dataset)}")
# Model setup
model = ConvNeXtMRI(
    in_chans=3,
    num_classes=NUM_CLASSES,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=DROP_PATH_RATE
).to(DEVICE)

# model.freeze_stages(n=2)

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# optimizer
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)  # ↑ LR, WD to 0.05 (ConvNeXt default-ish)

# cosine schedule with warmup (replace plain CosineAnnealingLR)
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
warmup_epochs = 5
main_cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
scheduler = SequentialLR(optimizer, schedulers=[warmup, main_cosine], milestones=[warmup_epochs])

# === SWA setup ===
swa_start_epoch = int(EPOCHS * 0.8)  # last 20% of training
swa_model = AveragedModel(model) #
swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR) #


# Schedule variables for augmentation + drop-path
total_epochs = EPOCHS
explore_end_epoch    = int(total_epochs * 0.5)   # e.g. first 50%
transition_end_epoch = int(total_epochs * 0.75) # next ~33%

mixup_alpha_init      = 0.8
cutmix_alpha_init     = 1.0
mixup_prob_init       = 1.0
drop_path_rate_init   = DROP_PATH_RATE


# (Continue with training loop…)
print("🔹 Training started ...")

best_val_acc = 0.0 #
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []} #
epochs = range(1, EPOCHS + 1) #

lrs = []  # track learning rate per epoch

scaler = torch.amp.GradScaler('cuda')

for epoch in range(1, EPOCHS + 1):

    # === Scheduled MixUp / CutMix / DropPath ===
    if epoch <= explore_end_epoch:
        # strong regularization phase
        mixup_prob   = mixup_prob_init
        mixup_alpha  = mixup_alpha_init
        cutmix_alpha = cutmix_alpha_init
        drop_path_rate_current = drop_path_rate_init

    elif epoch <= transition_end_epoch:
        # gradual fade-out
        progress = (epoch - explore_end_epoch) / float(transition_end_epoch - explore_end_epoch)
        mixup_prob   = mixup_prob_init * (1.0 - 0.5 * progress)       # 1.0 → 0.5
        mixup_alpha  = mixup_alpha_init * (1.0 - 0.8 * progress)      # 0.8 → 0.32
        cutmix_alpha = cutmix_alpha_init * (1.0 - 0.7 * progress)     # 1.0 → 0.4
        drop_path_rate_current = drop_path_rate_init * (1.0 - progress)

    elif epoch < swa_start_epoch:
        # low regularization but still small augmentations
        mixup_prob   = 0.2
        mixup_alpha  = 0.2
        cutmix_alpha = 0.3
        drop_path_rate_current = 0.05

    else:
        # SWA phase — no stochastic regularization
        mixup_prob   = 0.0
        mixup_alpha  = 0.0
        cutmix_alpha = 0.0
        drop_path_rate_current = 0.0
    # Update model's drop path rate if supported
    model.set_drop_path_rate(drop_path_rate_current)


    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # --- Apply MixUp ---
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # Apply MixUp/CutMix based on schedule
            if epoch > 5 and random.random() < mixup_prob:
                if random.random() < 0.5:
                    inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha)
                else:
                    inputs, y_a, y_b, lam = cutmix_data(inputs, labels, cutmix_alpha)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


    # === End of training epoch ===
    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total


    # === Validation ===

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_preds, val_labels_np = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels_np.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc  = 100 * val_correct / val_total


    print(f'\nEpoch {epoch} completed: '
          f'Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, '
          f'Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%')

    # === Additional validation metrics ===
    from sklearn.metrics import precision_score, recall_score, f1_score

    val_precision = precision_score(val_labels_np, val_preds, average='macro')
    val_recall = recall_score(val_labels_np, val_preds, average='macro')
    val_f1 = f1_score(val_labels_np, val_preds, average='macro')

    print(f"Val Precision: {val_precision:.3f}, Recall: {val_recall:.3f}, F1: {val_f1:.3f}")


    # --- OPTIONAL: Validation threshold tuning ---
    if epoch % 5 == 0 or epoch == EPOCHS:
        all_val_probs, all_val_labels = [], []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                probs = torch.softmax(model(x), dim=1)[:, 0]  # prob(AD)
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())

        import numpy as np
        best_th, best_f1 = 0.5, 0
        for th in np.linspace(0.2, 0.8, 61):
            preds = (np.array(all_val_probs) >= th).astype(int)
            f1 = f1_score(all_val_labels, preds, pos_label=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th

        print(f"🔧 Best AD threshold (epoch {epoch}): {best_th:.3f} (Val F1={best_f1:.3f})")
        with open("best_threshold.txt", "w") as f:
            f.write(f"{best_th:.3f}\n")

    # === Record metrics and save ===
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth') #
        print(f'✅ New best model saved! (Val Acc: {val_acc:.2f}%)') #

    with open('training_log.txt', 'a') as f:
        f.write(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, '
                f'Precision={val_precision:.3f}, Recall={val_recall:.3f}, F1={val_f1:.3f}\n')

    # === Learning rate + SWA handling ===
    if epoch >= swa_start_epoch:
        swa_model.update_parameters(model) #
        swa_scheduler.step() #
    else:
        scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    print(f"Current LR after epoch {epoch}: {current_lr:.6f}")
    
    # === Early Stopping check (place at end of epoch) ===
    if EARLY_STOP:
        current = val_loss if monitor_metric == "val_loss" else val_acc

        improved = (
            (monitor_metric == "val_loss" and ( _best_monitored - current ) > early_min_delta) or
            (monitor_metric == "val_acc"  and ( current - _best_monitored ) > early_min_delta)
        )

        if improved:
            _best_monitored = current
            _no_improve_epochs = 0
        else:
            _no_improve_epochs += 1
            print(f"⏳ No {monitor_metric} improvement for {_no_improve_epochs}/{early_patience} epoch(s).")
            if _no_improve_epochs >= early_patience:
                print(f"⏹️ Early stopping at epoch {epoch} "
                    f"(no {monitor_metric} improvement ≥ {early_min_delta} for {early_patience} epochs).")
                # (Optional) log it
                with open('training_log.txt', 'a') as f:
                    f.write(f'EARLY STOP at epoch {epoch} on {monitor_metric}\n')
                break



"""# Plot training curves"""

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, history['train_loss'], label='Train Loss')
plt.plot(epochs, history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs, history['train_acc'], label='Train Acc')
plt.plot(epochs, history['val_acc'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print('📊 Training curves saved as training_curves.png')

import pandas as pd

# Save training metrics as CSV for record-keeping
pd.DataFrame(history).to_csv("training_history.csv", index=False)
print("📁 Training history saved as training_history.csv")

"""Plot LR curve"""

plt.figure()
plt.plot(range(1, len(lrs) + 1), lrs, marker='o')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.savefig('lr_schedule.png', dpi=150)
plt.show()
print('📈 Learning rate schedule saved as lr_schedule.png')



# === Finalize SWA model ===
print("🧮 Updating batch-norm statistics for SWA model...")
update_bn(train_loader, swa_model, device=DEVICE) #
torch.save(swa_model.state_dict(), 'swa_model.pth') #
print("✅ SWA model saved as swa_model.pth") #

