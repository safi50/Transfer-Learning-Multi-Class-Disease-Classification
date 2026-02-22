"""
Model architectures for medical image classification.

This module provides custom model architectures and pre-trained
model loaders with modified classifier heads.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier with custom head.
    
    Args:
        model_name (str): 'resnet18' or 'resnet50'
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        freeze_backbone (bool): Freeze backbone weights
    """
    
    def __init__(self, model_name='resnet18', num_classes=3, pretrained=True, freeze_backbone=False):
        super(ResNetClassifier, self).__init__()
        
        if model_name == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier with custom head.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        freeze_backbone (bool): Freeze backbone weights
    """
    
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super(EfficientNetClassifier, self).__init__()
        
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        num_features = self.backbone.classifier[1].in_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class AttentionBlock(nn.Module):
    """
    Channel and Spatial Attention Block.
    
    Args:
        channels (int): Number of input channels
    """
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        
        # Channel attention
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        b, c, h, w = x.size()
        channel_att = self.channel_pool(x).view(b, c)
        channel_att = self.channel_fc(channel_att).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_conv(spatial_att)
        x = x * spatial_att
        
        return x


def get_model(model_name='resnet18', num_classes=3, pretrained=True, freeze_backbone=False):
    """
    Factory function to get model by name.
    
    Args:
        model_name (str): Name of the model
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        freeze_backbone (bool): Freeze backbone weights
        
    Returns:
        nn.Module: Model instance
    """
    if model_name in ['resnet18', 'resnet50']:
        return ResNetClassifier(model_name, num_classes, pretrained, freeze_backbone)
    elif model_name == 'efficientnet':
        return EfficientNetClassifier(num_classes, pretrained, freeze_backbone)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
