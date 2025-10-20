# ConvNeXt Alzheimer's Classification

PatternAnalysis-2025 | Recognition Project - Hard Difficulty
Author: s4938484 - Gia Hung Huynh

## Introduction

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




## Model Architecture

Our model employs a custom reimplementation of the ConvNeXt architecture, built from the ground up to better suit the characteristics of the ADNI dataset. At the core of this design is the ConvNeXtBlock, which integrates several key components inspired by both convolutional and transformer-based architectures.

### ConvNeXtBlock Components

- **Depthwise Convolution:** Each block begins with a large-kernel (e.g., 7×7) depthwise convolution that captures extensive spatial context while maintaining computational efficiency by applying a separate convolution per input channel.
- **Layer Normalization:** Following the depthwise convolution, LayerNorm is applied in a channel-last format to normalize feature activations, which stabilizes training and improves generalization.
- **MLP (Multi-Layer Perceptron):** A two-layer fully connected feed-forward network with a GELU activation in between acts as a channel-wise MLP, inspired by Vision Transformer (ViT) designs, enabling complex feature transformations.
- **LayerScale:** A learnable scaling parameter is applied to the MLP output to modulate residual branch contributions, helping to stabilize training in deep networks.
- **DropPath (Stochastic Depth):** DropPath regularization randomly drops entire residual branches during training, acting as a form of structured dropout to enhance model robustness.
- **Residual Connection:** The block incorporates a residual connection that sums the input with the transformed features, facilitating gradient flow and enabling deeper architectures.

### Stacking and Stages

Multiple ConvNeXtBlocks are sequentially stacked to form ConvNeXt stages. Between stages, progressive downsampling is performed using strided convolutions or patch merging layers to reduce spatial dimensions while increasing channel depth, enabling hierarchical representation learning similar to conventional CNNs.

### Design Motivations

This reimplementation draws on key design principles:
- **Large-kernel Spatial Filtering:** The use of large depthwise convolutions captures broader spatial dependencies compared to small kernels, improving feature extraction for complex brain MRI slices.
- **ViT-inspired MLP and LayerNorm:** Incorporating MLP blocks and LayerNorm layers introduces non-linear channel mixing and stable normalization, inspired by transformer architectures, enhancing representational capacity.
- **LayerScale and DropPath:** These components improve training stability and generalization by modulating residual contributions and applying structured stochastic regularization.

### Adaptation for ADNI Dataset

By reimplementing ConvNeXt from scratch, we tailor the architecture to the limited size and specific characteristics of the ADNI MRI dataset. This approach aims to improve training efficiency, interpretability, and potentially yield better generalization on the Alzheimer’s classification task compared to off-the-shelf pretrained models.



### Comparison with Pretrained ConvNeXt (TorchVision)

| Aspect                      | Custom ConvNeXt Implementation                   | Pretrained ConvNeXt (TorchVision)                |
|-----------------------------|-------------------------------------------------|--------------------------------------------------|
| **Implementation**           | Built from scratch with custom blocks and layers, allowing fine-grained control over architecture details | Official PyTorch implementation with pretrained weights on ImageNet |
| **Training Objectives**      | Designed specifically for Alzheimer’s classification with tailored loss functions and regularization | General-purpose ImageNet classification pretrained weights |
| **Dataset Adaptation**       | Adapted to MRI brain slices with customized input preprocessing and augmentation strategies | Trained on natural images; requires fine-tuning for medical images |
| **Flexibility**              | Full control over architecture modifications, normalization schemes, and regularization techniques | Limited flexibility; mainly fine-tuning pretrained backbone |
| **Interpretability**         | Transparent design facilitates understanding of model behavior and modifications | Black-box pretrained model with limited interpretability |
| **Optimization**             | Includes domain-specific optimizations like LayerScale and DropPath tailored to dataset characteristics | Standard training pipeline optimized for natural images |

The rationale behind choosing a full reimplementation over using the pretrained ConvNeXt from TorchVision is to gain complete control over the model architecture and training process. This enables better adaptation to the unique characteristics of the ADNI MRI dataset, allows incorporation of domain-specific regularization and normalization techniques, and improves interpretability of the model's inner workings. Such flexibility is critical for medical imaging tasks where pretrained models on natural images may not transfer optimally without extensive modification.




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


       ^
  LR   |     /‾‾‾‾‾‾‾‾‾\
       |    /           \
       |---/-------------\------
            Warm-up   Cosine decay
                → Epochs →

A combined linear warm-up and cosine annealing learning rate schedule was used. The warm-up phase gradually increased the learning rate during the first epoch, stabilizing early optimization, followed by a smooth cosine decay to fine-tune convergence.

Figure X. Learning rate schedule combining linear warm-up and cosine annealing. The initial warm-up stabilizes early training, while the cosine decay fine-tunes convergence.


### Optimizer and Regularization

- Optimizer: AdamW (lr = 1e-4, weight_decay = 1e-4)
- Loss Function: CrossEntropy with label smoothing (ε = 0.1)
- Regularization: Drop Path (stochastic depth) and gradient clipping (max-norm = 1.0)

### Drop Path Regularization

ConvNeXt employs stochastic depth (Drop Path) to randomly drop entire residual branches during training, improving ensemble-like regularization and robustness.

### Gradient Clipping

Gradient clipping was used (max-norm = 1.0) to stabilize training and prevent gradient explosion during backpropagation.

### AMP (Mixed Precision)

To accelerate training and reduce GPU memory usage, automatic mixed precision (AMP) was applied using PyTorch’s torch.cuda.amp. This approach leverages half-precision arithmetic on supported GPUs while maintaining numerical stability through gradient scaling, achieving faster convergence without accuracy degradation.

Training was performed under automatic mixed precision (AMP) using torch.cuda.amp.autocast and GradScaler, achieving faster computation (~1.5–2× speedup) and lower GPU memory usage without accuracy loss.


### Validation Optimization (FP16 inference)


### Checkpointing and Logging

The best model (based on validation accuracy) was saved as best_model.pth.
Training progress and metrics were logged to training_log.txt, and all epoch histories were exported to training_history.csv.

### Learning Rate Visualization

Learning rate progression across epochs was plotted and saved as lr_schedule.png, providing a clear overview of the warm-up and cosine decay behavior.

### Training Stability Enhancements

To further stabilize training, the script enabled CuDNN benchmarking for optimal kernel selection on GPU:





## Evaluation and Results


## How to run


## Dependencies


## Example Outputs


## Reproducibility and Discussion


## References

