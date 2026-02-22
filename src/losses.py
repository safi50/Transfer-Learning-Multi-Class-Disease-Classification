"""
Custom loss functions for handling class imbalance.

This module implements Focal Loss and Class-Balanced Loss
for training on imbalanced datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    
    Args:
        alpha (tensor): Weights for each class
        gamma (float): Focusing parameter (default: 2)
        reduction (str): Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples"
    
    Args:
        samples_per_class (list): Number of samples for each class
        beta (float): Hyperparameter for class-balanced loss (default: 0.9999)
        gamma (float): Focusing parameter for focal loss variant (default: 2)
    """
    
    def __init__(self, samples_per_class, beta=0.9999, gamma=2):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        if inputs.device != self.weights.device:
            self.weights = self.weights.to(inputs.device)
            
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.
    
    Args:
        weights (tensor): Weight for each class
    """
    
    def __init__(self, weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        
    def forward(self, inputs, targets):
        if self.weights is not None:
            if inputs.device != self.weights.device:
                self.weights = self.weights.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=self.weights)
