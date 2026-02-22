"""
Data loading utilities for medical image classification.

This module provides custom dataset classes and data loaders
for training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MedicalImageDataset(Dataset):
    """
    Custom dataset for medical image classification.
    
    Args:
        csv_file (str): Path to CSV file with annotations
        img_dir (str): Directory with all the images
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.annotations.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(train=True, img_size=224):
    """
    Get image transformations for training or validation.
    
    Args:
        train (bool): Whether to use training augmentations
        img_size (int): Target image size
        
    Returns:
        transforms.Compose: Composition of image transformations
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_data_loaders(train_csv, val_csv, img_dir, batch_size=32, num_workers=4):
    """
    Create training and validation data loaders.
    
    Args:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV
        img_dir (str): Directory with images
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = MedicalImageDataset(
        csv_file=train_csv,
        img_dir=img_dir,
        transform=get_transforms(train=True)
    )
    
    val_dataset = MedicalImageDataset(
        csv_file=val_csv,
        img_dir=img_dir,
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
