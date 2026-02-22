"""
Utility functions for medical image classification.

This module provides helper functions for visualization,
metrics computation, and other utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy over epochs.
    
    Args:
        history (dict): Training history with loss and accuracy
        save_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(y_true, y_scores, class_names, save_path=None):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true (array): True labels
        y_scores (array): Prediction scores (probabilities)
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class Classification')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_class_weights(labels, num_classes):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels (array): Training labels
        num_classes (int): Number of classes
        
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Calculate inverse frequency weights
    weights = total_samples / (num_classes * class_counts)
    
    return torch.FloatTensor(weights)


def save_predictions(predictions, filenames, output_path, class_names=None):
    """
    Save predictions to CSV file.
    
    Args:
        predictions (array): Model predictions
        filenames (list): List of image filenames
        output_path (str): Path to save CSV
        class_names (list, optional): List of class names
    """
    if class_names:
        predictions = [class_names[p] for p in predictions]
    
    df = pd.DataFrame({
        'filename': filenames,
        'prediction': predictions
    })
    
    df.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    
    return total_params, trainable_params


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS (Apple Silicon)')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    return device
