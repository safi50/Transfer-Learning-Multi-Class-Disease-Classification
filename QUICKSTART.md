# Quick Start Guide

This guide will help you get started with the Medical Image Classification project quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-image-classification.git
   cd medical-image-classification
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Notebooks

### Option 1: Transfer Learning
Experiment with different fine-tuning strategies:
```bash
jupyter notebook notebooks/01_transfer_learning.ipynb
```

### Option 2: Handling Class Imbalance
Learn about Focal Loss and Class-Balanced Loss:
```bash
jupyter notebook notebooks/02_class_imbalance.ipynb
```

### Option 3: Attention Mechanisms
Explore attention-based architectures:
```bash
jupyter notebook notebooks/03_attention_mechanisms.ipynb
```

### Option 4: Advanced Techniques
Try ensemble methods and advanced optimization:
```bash
jupyter notebook notebooks/04_advanced_techniques.ipynb
```

## Using the Python Modules

You can also use the provided Python modules directly:

```python
import torch
from src.models import get_model
from src.data_loader import create_data_loaders
from src.train import train_model
from src.utils import get_device

# Get device
device = get_device()

# Create model
model = get_model('resnet18', num_classes=3, pretrained=True)
model = model.to(device)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_csv='path/to/train.csv',
    val_csv='path/to/val.csv',
    img_dir='path/to/images',
    batch_size=32
)

# Train model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

history = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    device, num_epochs=50,
    save_path='models/my_model.pt'
)
```

## Making Predictions

```python
import torch
from torchvision import transforms
from PIL import Image
from src.models import get_model

# Load model
model = get_model('resnet18', num_classes=3, pretrained=False)
model.load_state_dict(torch.load('models/best_model.pt'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
    
classes = ['DR', 'G', 'A']
print(f"Predicted: {classes[prediction.item()]}")
```

## Project Structure

```
medical-image-classification/
├── notebooks/          # Jupyter notebooks
├── src/               # Python modules
│   ├── data_loader.py # Data loading utilities
│   ├── models.py      # Model architectures
│   ├── losses.py      # Loss functions
│   ├── train.py       # Training utilities
│   └── utils.py       # Helper functions
├── models/            # Saved model weights
├── results/           # Experimental results
├── docs/              # Documentation
└── requirements.txt   # Dependencies
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in the data loader
- Use a smaller model (e.g., ResNet18 instead of ResNet50)

### Slow Training
- Enable GPU acceleration
- Increase batch size (if memory allows)
- Use multiple workers in data loader

### Import Errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

## Next Steps

1. ✅ Explore the notebooks to understand different techniques
2. ✅ Try training models with different hyperparameters
3. ✅ Experiment with custom data augmentation
4. ✅ Compare different loss functions
5. ✅ Build ensemble models for better performance

## Need Help?

- Check the [full README](README.md) for detailed information
- Open an issue on GitHub for bugs or questions
- Refer to the [documentation](docs/) for technical details

Happy coding! 🚀
