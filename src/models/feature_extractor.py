"""
ResNet-based feature extractor for handling Latin and Devanagari scripts.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, Tuple

class FeatureExtractor(nn.Module):
    """ResNet-based feature extractor with script-specific adaptations."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        feature_dim: int = 2048,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the feature extractor.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            feature_dim (int): Dimension of feature vectors
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        
        # Using ResNet50 as the backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Script-specific attention modules
        self.latin_attention = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        self.devanagari_attention = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
    def _apply_attention(
        self,
        x: torch.Tensor,
        attention_module: nn.Module
    ) -> torch.Tensor:
        """
        Apply attention mechanism to features.
        
        Args:
            x (torch.Tensor): Input features
            attention_module (nn.Module): Attention module to apply
            
        Returns:
            torch.Tensor: Attention-weighted features
        """
        attention_weights = attention_module(x)
        return x * attention_weights
    
    def forward(
        self,
        x: torch.Tensor,
        script_type: str
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            script_type (str): Type of script ('latin' or 'devanagari')
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'features': Raw extracted features
                - 'attention': Attention weights
                - 'logits': Classification logits
        """
        # Extract base features
        features = self.features(x)
        
        # Apply script-specific attention
        if script_type == 'latin':
            attended_features = self._apply_attention(features, self.latin_attention)
        else:  # devanagari
            attended_features = self._apply_attention(features, self.devanagari_attention)
        
        # Pool and get logits
        pooled = self.pool(attended_features)
        pooled = torch.flatten(pooled, 1)
        logits = self.fc(pooled)
        
        return {
            'features': features,
            'attention': attended_features,
            'logits': logits
        }
    
    def get_attention_maps(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention maps for both scripts.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Latin and Devanagari attention maps
        """
        features = self.features(x)
        latin_attention = self.latin_attention(features)
        devanagari_attention = self.devanagari_attention(features)
        
        return latin_attention, devanagari_attention