"""
Training script for the feature extractor model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from .feature_extractor import FeatureExtractor
from .dataset import create_dataloaders

class Trainer:
    """Handles training of the feature extractor model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): The feature extractor model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            learning_rate (float): Learning rate for optimization
            device (str): Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, script_types) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images, script_types[0])  # Assuming batch has same script type
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, script_types in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, script_types[0])
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100. * correct / total
        }
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 1
    ):
        """
        Train the model.
        
        Args:
            num_epochs (int): Number of epochs to train
            checkpoint_dir (Optional[str]): Directory to save checkpoints
            log_interval (int): How often to log progress
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Log progress
            if (epoch + 1) % log_interval == 0:
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'Train Loss: {train_metrics["loss"]:.4f}, '
                      f'Train Acc: {train_metrics["accuracy"]:.2f}%')
                print(f'Val Loss: {val_metrics["loss"]:.4f}, '
                      f'Val Acc: {val_metrics["accuracy"]:.2f}%')
            
            # Save checkpoint if best model
            if val_metrics['loss'] < best_val_loss and checkpoint_dir:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'history': self.history
                }, checkpoint_dir / 'best_model.pt')
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()