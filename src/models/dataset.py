"""
Dataset handling for training the feature extractor.
"""
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import List, Tuple, Dict, Optional, Any
from sklearn.model_selection import train_test_split
from src.models.config import ModelConfig

class MixupTransform:
    """Mixup augmentation for images and labels."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply mixup to a batch of images."""
        if self.alpha <= 0:
            return batch
        
        images, labels = batch['images'], batch['labels']
        batch_size = len(images)
        
        # Generate mixup weights
        weights = np.random.beta(self.alpha, self.alpha, batch_size)
        weights = torch.from_numpy(weights).float()
        
        # Create shuffled indices
        indices = torch.randperm(batch_size)
        
        # Mix the images
        weights = weights.view(-1, 1, 1, 1)
        mixed_images = weights * images + (1 - weights) * images[indices]
        
        # Mix the labels (one-hot encoded)
        weights = weights.view(-1, 1)
        mixed_labels = weights * labels + (1 - weights) * labels[indices]
        
        return {
            'images': mixed_images,
            'labels': mixed_labels,
            'script_types': batch['script_types']  # Keep original script types
        }

class CustomRandomRotation(transforms.RandomRotation):
    """Custom rotation that keeps text readable (90-degree increments)."""
    
    def __init__(self, degrees: float, p: float = 0.5):
        super().__init__(degrees)
        self.p = p
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            angle = random.choice([0, 90, 180, 270])
            return TF.rotate(img, angle)
        return img

class MathDataset(Dataset):
    """Dataset for training the feature extractor on mathematical expressions."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[str],
        script_types: List[str],
        config: ModelConfig,
        is_training: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths (List[str]): List of paths to image files
            labels (List[str]): List of text labels for each image
            script_types (List[str]): List of script types ('latin' or 'devanagari')
            config (ModelConfig): Configuration object
            is_training (bool): Whether this is for training
        """
        self.image_paths = image_paths
        self.labels = labels
        self.script_types = script_types
        self.config = config
        self.is_training = is_training
        
        # Create transforms
        self.transform = self._create_transforms()
        
    def _create_transforms(self) -> transforms.Compose:
        """Create transformation pipeline."""
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize(self.config.IMAGE_SIZE))
        
        if self.is_training and self.config.USE_AUGMENTATION:
            # Random rotation (90-degree increments)
            transform_list.append(CustomRandomRotation(
                self.config.RANDOM_ROTATE_DEGREES
            ))
            
            # Random scaling
            transform_list.append(transforms.RandomAffine(
                degrees=0,
                scale=self.config.RANDOM_SCALE_RANGE
            ))
            
            # Random cropping
            transform_list.append(transforms.RandomResizedCrop(
                self.config.IMAGE_SIZE,
                scale=self.config.RANDOM_CROP_SCALE
            ))
            
            # Color jittering
            transform_list.append(transforms.ColorJitter(
                **self.config.COLOR_JITTER_PARAMS
            ))
            
            # Random grayscale
            transform_list.append(transforms.RandomGrayscale(p=0.1))
        
        # Always include these transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.NORMALIZE_MEAN,
                std=self.config.NORMALIZE_STD
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'label': self.labels[idx],
            'script_type': self.script_types[idx],
            'path': self.image_paths[idx]
        }

class DataModule:
    """Handles all data-related operations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        # Load all data paths and labels
        image_paths, labels, script_types = self._load_data_from_directory(
            self.config.DATA_DIR
        )
        
        # Split data
        train_idx, val_idx = train_test_split(
            range(len(image_paths)),
            test_size=1-self.config.TRAIN_SPLIT,
            stratify=script_types,
            random_state=42
        )
        
        # Create datasets
        full_dataset = MathDataset(
            image_paths, labels, script_types,
            self.config, is_training=True
        )
        
        self.train_dataset = Subset(full_dataset, train_idx)
        
        # Create validation dataset without augmentations
        val_dataset = MathDataset(
            image_paths, labels, script_types,
            self.config, is_training=False
        )
        self.val_dataset = Subset(val_dataset, val_idx)
    
    def _load_data_from_directory(
        self, 
        data_dir: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """Load data paths and labels from directory."""
        image_paths = []
        labels = []
        script_types = []
        
        # Implement data loading logic here
        # This should scan the data directory and collect:
        # - Paths to image files
        # - Labels for each image
        # - Script type for each image
        
        return image_paths, labels, script_types
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )