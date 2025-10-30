
"""
predict.py
Author: Gia Hung Huynh — 49384848
Date: 23 Oct 2025

Purpose
-------
Evaluate a trained ConvNeXtMRI classifier on the AD/NC test set and report
Accuracy, AUC, and a full classification report. Optionally evaluate an SWA
checkpoint if present.

What this script does
---------------------
- Loads test dataloader via `dataset.get_loaders(DATA_ROOT, BATCH_SIZE)`.
- Instantiates the ConvNeXtMRI model (BN2d path; depths=[3,3,9,3], dims=[96,192,384,768]).
- Loads weights from `./best_model.pth` (and optionally `./swa_model.pth`), then
  runs evaluation on the test set.
- (Helpers included): functions to collect penultimate-layer embeddings and
  project them (UMAP with t-SNE fallback), but these helpers are **not called** by default.

"""

# --- IMPORTS (Remain at top level) ---
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel
from sklearn.metrics import roc_auc_score, classification_report, balanced_accuracy_score


from constants import (
    DEVICE_MPS, DEVICE_CUDA, DEVICE_CPU, EPOCHS,
    DATA_ROOT, BATCH_SIZE, LR, SWA_LR,
    WD, DROP_PATH_RATE,
    NUM_CLASSES
)
# --- LOCAL IMPORTS (Remain at top level) ---
try:
    from modules import ConvNeXtMRI
    from dataset import get_loaders
except ImportError:
    print("❌ Error: Make sure 'modules.py' and 'dataset.py' are in the same directory.")
    print("   (Or that they are in the 'recognition/convnext_alzheimer_49384848' directory)")
    exit()
    
DATA_ROOT = "/content/data/AD_NC"


# === 4. Define Model Architecture (Remain at top level) ===
# This MUST match the architecture used for training
def create_model():
    """Helper function to create a new model instance."""
    return ConvNeXtMRI(
        in_chans=3,
        num_classes=NUM_CLASSES,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=DROP_PATH_RATE
    ) # .to(DEVICE) will be handled in the main block

# === 5. Helper Function for Evaluation (Remain at top level) ===
def evaluate_model(model_to_test, test_loader, DEVICE, model_name="Model"):
    """Runs a model through the test_loader and prints metrics."""
    print(f"\n--- Evaluating {model_name} ---")
    model_to_test.eval()

    all_labels = []
    all_preds = []
    all_probs = [] # Probabilities for the positive class (class 1)
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model_to_test(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # --- FIX 2: Corrected 'total' to 'test_total' ---
    test_acc = 100 * test_correct / test_total if test_total > 0 else 0
    # ------------------------------------------------

    test_auc = roc_auc_score(all_labels, all_probs)

    print(f"🧠 [{model_name}] Test Accuracy: {test_acc:.2f}%")
    print(f"📈 [{model_name}] Test AUC: {test_auc:.4f}")
    print(f"\nClassification Report ({model_name}):")
    print(classification_report(all_labels, all_preds, target_names=['AD (0)', 'NC (1)']))

    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧩 [{model_name}] Model parameters: {num_params/1e6:.2f}M")
    except Exception:
        pass

from constants import DATA_ROOT
# === FIX 1: Wrap all executable code in __name__ == '__main__' ===
if __name__ == '__main__':

    # === 1. Configuration ===
    MODEL_SUBFOLDER = "recognition/convnext_alzheimer_49384848"
    os.makedirs(MODEL_SUBFOLDER, exist_ok=True)
    
    print(f"Data root set to: {DATA_ROOT}")
    if not os.path.exists(DATA_ROOT):
        print(f"⚠️ Warning: Data directory not found at {DATA_ROOT}")
        print("Please update DATA_ROOTs to the correct path.")

    # === 2. Device Setup ===
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    # === 3. Load Test Data ===
    try:
        _, _, test_loader = get_loaders(
            data_root  =  DATA_ROOT,
            batch_size = BATCH_SIZE
        )
        print(f"Loaded test dataset: {len(test_loader.dataset)} images")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please ensure 'dataset.py' is present and DATA_ROOTs is correct.")
        exit()

    # === 6. Load and Test 'best_model_test_acc.pth' ===
    model_path = os.path.join('best_model.pth')

    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")

    if os.path.exists(model_path):
        try:
            model = create_model().to(DEVICE) # Move model to device
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"\n✅ Loaded standard weights from: {model_path}")
            
            # Pass test_loader and DEVICE as arguments
            evaluate_model(model, test_loader, DEVICE, f"Best Model ({os.path.basename(model_path)})")
            
                
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            print("Ensure 'modules.py' and model definition match the saved weights.")
    else:
        print(f"⚠️ Could not find {model_path}. Skipping standard model test.")
    
    # === 7. SWA  ===
    swa_model_path = os.path.join( 'swa_model.pth')
    if os.path.exists(swa_model_path):
        try:
            base_model_for_swa = create_model() # Don't move to device yet
            swa_model = AveragedModel(base_model_for_swa).to(DEVICE)
            swa_model.load_state_dict(torch.load(swa_model_path, map_location=DEVICE))
            print(f"\n✅ Loaded SWA weights from: {swa_model_path}")
    
            # Pass test_loader and DEVICE as arguments
            evaluate_model(swa_model, test_loader, DEVICE, "SWA Model")
    
        except Exception as e:
            print(f"❌ Error loading {swa_model_path}: {e}")
            print("Ensure 'torch.optim.swa_utils.AveragedModel' is imported.")
    else:
        print(f"⚠️ File not found: {swa_model_path}. Skipping SWA model test.")

    print("\n--- Evaluation Complete ---")
