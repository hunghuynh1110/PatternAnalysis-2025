# ConvNeXt Alzheimer's Classification

PatternAnalysis-2025 | Recognition Project - Hard Difficulty
Author: s4938484 - Gia Hung Huynh

## Problem Description

Alzheimer’s disease (AD) is a progressive neurodegenerative disorder characterized by structural brain changes visible in MRI scans.
The goal of this project is to classify MRI brain slices from the ADNI dataset into Alzheimer’s Disease (AD) or Cognitively Normal (CN) categories.

This project contributes to the open-source PatternAnalysis repository under the recognition branch.
It aims to reproduce a clinically relevant classification model achieving a minimum test accuracy of 0.8, as required in the assessment specification

## Project format

File
Purpose

dataset.py
Handles loading the data: defines dataset class(es), transformations, and DataLoaders.

modules.py
Defines model architectures and custom components (like new layers or blocks) used in training.

train.py
Orchestrates the training process: loads data, builds model, defines loss/optimizer/scheduler, trains & validates, saves checkpoints.

utils.py
Contains supporting utility functions (e.g., plotting, checkpoint saving/loading, metrics calculation).

predict.py
Performs inference: loads a trained model, preprocesses new input(s), predicts class(es), and outputs results.

## Algorithm Overview

The model is based on ConvNext


## Data Pre-processing and Dataset Splits

The dataset used in this project is a subset of the **ADNI (Alzheimer’s Disease Neuroimaging Initiative)** MRI collection.  
Each 3D MRI scan was converted into **2D slice images (JPEG format)** for simplified processing. This enables efficient training of convolutional models such as ConvNeXt while still preserving key spatial features in brain anatomy.

### Pre-processing Steps
1. **Slice Extraction**  
   - From each MRI volume, relevant axial slices were extracted and saved as individual `.jpeg` images.  
   - Each subject’s folder contains multiple slice files named according to their slice index (e.g., `123456_80.jpeg`).
2. **Normalization**  
   - All images were normalized to pixel intensity range `[0, 1]`, then standardized using:  
     \[
     \text{Normalize}(x) = \frac{x - 0.5}{0.5}
     \]  
     which centers data around zero for stable training.
3. **Resizing**  
   - Every slice was resized to **224×224 px** to match the ConvNeXt-Tiny input requirement.
4. **Data Augmentation (Training only)**  
   - Random resized crops, horizontal flips, and small rotations (±10°) were applied during training to improve generalisation.  
   - Validation and test sets use deterministic transforms (resize + center-crop) to ensure consistent evaluation.
5. **Class Labelling**  
   - Two classes are used:  
     - `AD` → Alzheimer’s Disease  
     - `NC` → Normal Control  
   - Labels are assigned based on directory names in the dataset structure.

### Directory Structure
```
AD_NC/
├── train/
│   ├── AD/      # Alzheimer’s slices for training
│   └── NC/      # Control slices for training
└── test/
    ├── AD/      # Alzheimer’s slices for testing
    └── NC/      # Control slices for testing
```

### Data Splits
- **Training:** 90% of images from each class (AD and NC).  
- **Validation:** 10% of images from the training folder, held out for model selection and early stopping.  
- **Test:** A separate unseen folder (`AD_NC/test`) used only for final evaluation.  

This split ensures that the model’s generalisation is evaluated on entirely unseen data while maintaining class balance across subsets.

### Rationale
MRI data often suffers from overfitting due to limited samples.  
By slicing 3D volumes and applying random augmentations, we effectively enlarge the dataset while retaining critical anatomical cues.  
Normalization and resizing align data distribution with the pretrained ConvNeXt backbone (originally trained on ImageNet).



## Training Setup


## Evaluation and Results


## How to run


## Dependencies


## Example Outputs


## Reproducibility and Discussion


## References

