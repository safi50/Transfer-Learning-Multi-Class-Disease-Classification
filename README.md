# Multi-Class Retinal Disease Classfication using Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive deep learning project for medical image classification focusing on Diabetic Retinopathy (DR), Glaucoma (G), and Age-related Macular Degeneration (AMD) detection.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodologies](#methodologies)
- [Results](#results)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements state-of-the-art deep learning techniques for automated detection and classification of retinal diseases from fundus images. The system addresses key challenges in medical image analysis including:

- **Class Imbalance**: Handled through specialized loss functions (Focal Loss, Class-Balanced Loss)
- **Transfer Learning**: Leveraging pre-trained models (ResNet, EfficientNet)
- **Attention Mechanisms**: Improving model focus on relevant image regions
- **Ensemble Methods**: Combining multiple models for robust predictions

### Key Statistics

- **Classes**: 3 (Diabetic Retinopathy, Glaucoma, AMD)
- **Class Distribution**:
  - DR (Diabetic Retinopathy): 517 instances
  - G (Glaucoma): 163 instances
  - A (AMD): 142 instances
- **Models**: ResNet18, ResNet50, EfficientNet
- **Techniques**: Transfer Learning, Focal Loss, Attention Mechanisms

---

## Features

-   **Transfer Learning** with three different strategies:
  - No fine-tuning (evaluation only)
  - Frozen backbone + fine-tuned classifier
  - Full fine-tuning (backbone + classifier)

-   **Advanced Loss Functions** for handling class imbalance:
  - Focal Loss for focusing on hard examples
  - Class-Balanced Loss for re-weighting based on class frequency

-   **Attention Mechanisms** with improved training strategies:
  - Stratified data splits for balanced class distribution
  - Class-wise weighted loss
  - Enhanced regularization with dropout layers
  - Diverse augmentation strategies

-   **Comprehensive Evaluation**:
  - Detailed performance metrics
  - Confusion matrices
  - Class-wise accuracy analysis
  - Visualization of predictions

---

## 📁 Project Structure

```
medical-image-classification/
│
├── notebooks/                          # Jupyter notebooks for experiments
│   ├── 01_transfer_learning.ipynb     # Transfer learning experiments
│   ├── 02_class_imbalance.ipynb       # Focal & Class-Balanced Loss
│   ├── 03_attention_mechanisms.ipynb  # Attention-based models
│   └── 04_advanced_techniques.ipynb   # Ensemble & advanced methods
│
├── src/                                # Source code modules
│   ├── __init__.py
│   ├── data_loader.py                  # Data loading utilities
│   ├── models.py                       # Model architectures
│   ├── losses.py                       # Custom loss functions
│   ├── train.py                        # Training loops
│   └── utils.py                        # Helper functions
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
└── README.md                           # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/medical-image-classification.git
cd medical-image-classification
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Dataset

The project uses retinal fundus images with three disease classes:

| Class | Description | Training Samples |
|-------|-------------|------------------|
| **DR** | Diabetic Retinopathy | 517 |
| **G** | Glaucoma | 163 |
| **A** | Age-related Macular Degeneration (AMD) | 142 |

### Data Preprocessing

- **Image Size**: 224x224 pixels
- **Normalization**: ImageNet mean and std
- **Augmentation**:
  - Random horizontal/vertical flips
  - Random rotation (±15°)
  - Color jitter
  - Random affine transformations
  - Normalization

---

## Results

### Performance Summary

| Model | Approach | Accuracy | Notes |
|-------|----------|----------|-------|
| ResNet18 | Transfer Learning (Full Fine-tune) | ~85% | Task 1.3 |
| ResNet18 | Class-Balanced Loss | ~87% | Task 2.2 |
| Custom CNN | Attention Mechanisms | ~88% | Task 3 |
| Ensemble | Multiple Models | **~90%** | Best Performance |


## Usage

### Training a Model

```python
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook

# Open and run desired notebook:
# - 01_transfer_learning.ipynb for transfer learning
# - 02_class_imbalance.ipynb for handling imbalanced data
# - 03_attention_mechanisms.ipynb for attention models
# - 04_advanced_techniques.ipynb for ensemble methods
```
## 📝 Future Work

- [ ] Implement advanced architectures (Vision Transformers, ConvNeXt)
- [ ] Add explainability methods (Grad-CAM, attention visualization)
- [ ] Deploy model as web application
- [ ] Extend to multi-label classification
- [ ] Incorporate additional datasets for improved generalization
- [ ] Add real-time inference pipeline
      
---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
