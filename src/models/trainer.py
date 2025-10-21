"""
Training script for the feature extractor model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import numpy as np

from .feature_extractor import FeatureExtractor
from .dataset import DataModule, MixupTransform
from .config import ModelConfig

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

class Trainer:
    """Handles training of the feature extractor model with advanced features."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        data_module: DataModule
    ):
        """
        Initialize trainer with advanced features.
        
        Args:
            model (nn.Module): The feature extractor model
            config (ModelConfig): Training configuration
            data_module (DataModule): Data handling module
        """
        self.config = config
        self.model = model.to(config.DEVICE)
        self.data_module = data_module
        
        # Multi-GPU support
        if config.USE_MULTI_GPU:
            self.model = nn.DataParallel(self.model)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = amp.GradScaler() if config.USE_MIXED_PRECISION else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA
        )
        
        # Mixup augmentation
        self.mixup = MixupTransform(alpha=1.0)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Initialize W&B if enabled
        if config.USE_WANDB:
            wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                config=vars(config)
            )
            wandb.watch(model)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with configuration parameters."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=self.config.BETAS
        )
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler based on config."""
        if self.config.LR_SCHEDULER_TYPE == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
        elif self.config.LR_SCHEDULER_TYPE == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.NUM_EPOCHS // 3,
                T_mult=2,
                eta_min=self.config.LEARNING_RATE * self.config.LR_MIN_FACTOR
            )
        elif self.config.LR_SCHEDULER_TYPE == 'linear':
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.LR_MIN_FACTOR,
                total_iters=self.config.NUM_EPOCHS
            )
        else:  # step
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.NUM_EPOCHS // 3,
                gamma=0.1
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with advanced features.
        
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc='Training')
        
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.config.DEVICE)
            labels = batch['label'].to(self.config.DEVICE)
            script_types = batch['script_type']
            
            # Apply mixup augmentation if enabled
            if self.config.USE_AUGMENTATION:
                batch = self.mixup({
                    'images': images,
                    'labels': labels,
                    'script_types': script_types
                })
                images = batch['images']
                labels = batch['labels']
                script_types = batch['script_types']
            
            # Mixed precision training
            with amp.autocast(enabled=self.config.USE_MIXED_PRECISION):
                # Forward pass
                outputs = self.model(images, script_types[0])
                loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            if self.config.USE_MIXED_PRECISION:
                self.scaler.scale(loss).backward()
                if self.config.USE_GRADIENT_CLIPPING:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRADIENT_CLIP_VALUE
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.USE_GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRADIENT_CLIP_VALUE
                    )
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # Log to W&B if enabled
        if self.config.USE_WANDB:
            wandb.log({
                'train_loss': avg_loss,
                'train_accuracy': accuracy,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model with metrics logging.
        
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        attention_maps = []
        
        val_loader = self.data_module.val_dataloader()
        pbar = tqdm(val_loader, desc='Validating')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.config.DEVICE)
                labels = batch['label'].to(self.config.DEVICE)
                script_types = batch['script_type']
                
                # Mixed precision inference
                with amp.autocast(enabled=self.config.USE_MIXED_PRECISION):
                    outputs = self.model(images, script_types[0])
                    loss = self.criterion(outputs['logits'], labels)
                
                # Collect attention maps for visualization
                if len(attention_maps) < 5:  # Store first 5 batches
                    latin_attn, devanagari_attn = self.model.get_attention_maps(images)
                    attention_maps.append({
                        'image': images[0].cpu(),
                        'latin_attn': latin_attn[0].cpu(),
                        'devanagari_attn': devanagari_attn[0].cpu()
                    })
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Log to W&B if enabled
        if self.config.USE_WANDB and attention_maps:
            # Log attention visualizations
            for i, attn_map in enumerate(attention_maps):
                wandb.log({
                    f'attention_vis_{i}': wandb.Image(
                        self._create_attention_visualization(attn_map)
                    )
                })
            
            # Log metrics
            wandb.log({
                'val_loss': avg_loss,
                'val_accuracy': accuracy,
                'best_val_loss': self.early_stopping.best_loss or avg_loss
            })
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def _create_attention_visualization(self, attn_map: Dict[str, torch.Tensor]) -> np.ndarray:
        """Create visualization of attention maps."""
        image = attn_map['image'].permute(1, 2, 0).numpy()
        latin_attn = attn_map['latin_attn'].squeeze().numpy()
        devanagari_attn = attn_map['devanagari_attn'].squeeze().numpy()
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot attention maps
        ax2.imshow(latin_attn, cmap='hot')
        ax2.set_title('Latin Script Attention')
        ax2.axis('off')
        
        ax3.imshow(devanagari_attn, cmap='hot')
        ax3.set_title('Devanagari Script Attention')
        ax3.axis('off')
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return vis_image
    
    def train(
        self,
        num_epochs: int = None,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 1
    ):
        """
        Train the model with advanced features.
        
        Args:
            num_epochs (int, optional): Number of epochs to train. Defaults to config value.
            checkpoint_dir (Optional[str]): Directory to save checkpoints
            log_interval (int): How often to log progress
        """
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize best checkpoints tracking
        best_checkpoints = []
        
        # Training loop
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Train and validate
            train_metrics = self.train_epoch()
            
            # Validate every N epochs or on last epoch
            if (epoch + 1) % self.config.VAL_CHECK_INTERVAL == 0 or epoch == num_epochs - 1:
                val_metrics = self.validate()
                
                # Update learning rate scheduler
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                
                # Update history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['learning_rates'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                
                # Save checkpoint
                if checkpoint_dir:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'history': self.history,
                        'config': self.config
                    }
                    
                    # Save current checkpoint
                    current_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                    torch.save(checkpoint, current_path)
                    
                    # Update best checkpoints list
                    best_checkpoints.append((val_metrics['loss'], current_path))
                    best_checkpoints.sort(key=lambda x: x[0])
                    
                    # Keep only top K checkpoints
                    while len(best_checkpoints) > self.config.SAVE_TOP_K:
                        _, checkpoint_path = best_checkpoints.pop()
                        if checkpoint_path.exists():
                            checkpoint_path.unlink()
                
                # Early stopping
                if self.early_stopping(val_metrics['loss']):
                    print('\nEarly stopping triggered!')
                    break
            
            # Plot and save training curves periodically
            if checkpoint_dir and (epoch + 1) % log_interval == 0:
                self.plot_history(checkpoint_dir / f'training_history_epoch_{epoch+1}.png')
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot detailed training history.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Loss History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_title('Accuracy History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning rate
        ax3.plot(self.history['learning_rates'])
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Plot validation loss vs learning rate
        if len(self.history['val_loss']) > 1:
            ax4.scatter(self.history['learning_rates'],
                       self.history['val_loss'])
            ax4.set_title('Validation Loss vs Learning Rate')
            ax4.set_xlabel('Learning Rate')
            ax4.set_ylabel('Validation Loss')
            ax4.set_xscale('log')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.config.USE_WANDB:
            wandb.log({'training_curves': wandb.Image(fig)})
        
        plt.close()