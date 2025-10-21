import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    """ResNet-based feature extractor for handling Latin and Devanagari scripts."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        # Using ResNet50 as the backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        """
        Forward pass through the feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Extracted features
        """
        return self.features(x)