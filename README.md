# Multi-Class Retinal Disease Classfication using Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive deep learning project for medical image classification focusing on Diabetic Retinopathy (DR), Glaucoma (G), and Age-related Macular Degeneration (AMD) detection.

</div>

---

## 📋 Table of Contents

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

## 🔍 Overview

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

## ✨ Features

- 🚀 **Transfer Learning** with three different strategies:
  - No fine-tuning (evaluation only)
  - Frozen backbone + fine-tuned classifier
  - Full fine-tuning (backbone + classifier)

- ⚖️ **Advanced Loss Functions** for handling class imbalance:
  - Focal Loss for focusing on hard examples
  - Class-Balanced Loss for re-weighting based on class frequency

- 🎯 **Attention Mechanisms** with improved training strategies:
  - Stratified data splits for balanced class distribution
  - Class-wise weighted loss
  - Enhanced regularization with dropout layers
  - Diverse augmentation strategies

- 📊 **Comprehensive Evaluation**:
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
├── models/                             # Trained model weights
│   ├── .gitkeep
│   └── (model files - see .gitignore)
│
├── results/                            # Experimental results
│   ├── predictions/                    # Model predictions (CSV)
│   └── figures/                        # Plots and visualizations
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

## 🛠️ Installation

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

## 📊 Dataset

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

## 🧪 Methodologies

### 1. Transfer Learning (`01_transfer_learning.ipynb`)

Explores three transfer learning approaches:

- **Task 1.1**: Pre-trained model evaluation without fine-tuning
- **Task 1.2**: Frozen backbone with fine-tuned classifier head
- **Task 1.3**: Full model fine-tuning (all layers trainable)

**Models**: ResNet18, EfficientNet  
**Optimizer**: Adam with learning rate scheduling  
**Training**: Stratified K-fold cross-validation

### 2. Handling Class Imbalance (`02_class_imbalance.ipynb`)

Addresses the significant class imbalance in the dataset:

#### Focal Loss
```python
FL(pt) = -αt(1 - pt)^γ log(pt)
```
- Focuses training on hard, misclassified examples
- Reduces loss for well-classified examples
- Parameters: γ=2, α=[class weights]

#### Class-Balanced Loss
```python
CBL = (1 - β) / (1 - β^n) × CE
```
- Re-weights samples based on effective number
- Accounts for diminishing returns with more samples
- Parameter: β=0.9999

### 3. Attention Mechanisms (`03_attention_mechanisms.ipynb`)

Implements attention-based architectures with:

- **Improved Training Strategy**:
  - Stratified data splits for balanced validation
  - Lower learning rate (5e-5) for stable convergence
  - Enhanced dropout regularization
  - More diverse data augmentation

- **Attention Modules**:
  - Channel attention (squeeze-and-excitation)
  - Spatial attention
  - Self-attention mechanisms

### 4. Advanced Techniques (`04_advanced_techniques.ipynb`)

Combines multiple approaches:

- Model ensembling
- Test-time augmentation
- Advanced optimization strategies
- Hyperparameter tuning

---

## 📈 Results

### Performance Summary

| Model | Approach | Accuracy | Notes |
|-------|----------|----------|-------|
| ResNet18 | Transfer Learning (Full Fine-tune) | ~85% | Task 1.3 |
| ResNet18 | Class-Balanced Loss | ~87% | Task 2.2 |
| Custom CNN | Attention Mechanisms | ~88% | Task 3 |
| Ensemble | Multiple Models | **~90%** | Best Performance |

### Key Findings

- ✅ **Full fine-tuning** outperforms frozen backbone approaches
- ✅ **Class-Balanced Loss** more effective than Focal Loss for this dataset
- ✅ **Attention mechanisms** improve model interpretability and performance
- ✅ **Ensemble methods** provide the best overall accuracy

### Confusion Matrices

*(Add confusion matrix visualizations in results/figures/)*

---

## 🚀 Usage

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

### Making Predictions

```python
import torch
from torchvision import transforms
from PIL import Image

# Load trained model
model = torch.load('models/best_model.pt')
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Make prediction
image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
    
print(f"Predicted class: {prediction.item()}")
```

---

## 🏗️ Model Architecture

### Base Models

1. **ResNet18/50**
   - Residual connections for deep networks
   - Pre-trained on ImageNet
   - Modified final layer for 3-class classification

2. **EfficientNet**
   - Compound scaling method
   - Balanced depth, width, and resolution
   - State-of-the-art efficiency

### Custom Modifications

- **Classifier Head**: Fully connected layers with dropout
- **Loss Functions**: Focal Loss, Class-Balanced Loss
- **Attention Modules**: Channel and spatial attention
- **Regularization**: Dropout, batch normalization, weight decay

---

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 1e-4 to 5e-5 |
| Optimizer | Adam |
| Weight Decay | 1e-4 |
| Epochs | 50-100 |
| Loss Function | CrossEntropy / Focal / Class-Balanced |
| Scheduler | ReduceLROnPlateau |

---

## 📝 Future Work

- [ ] Implement advanced architectures (Vision Transformers, ConvNeXt)
- [ ] Add explainability methods (Grad-CAM, attention visualization)
- [ ] Deploy model as web application
- [ ] Extend to multi-label classification
- [ ] Incorporate additional datasets for improved generalization
- [ ] Add real-time inference pipeline

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or feedback, please open an issue or contact the maintainers.

---

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision for pre-trained models
- The medical imaging research community
- Contributors to various open-source libraries used in this project

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for the medical imaging community

</div>

