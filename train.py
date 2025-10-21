"""
Main script for training the feature extractor model.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.config import ModelConfig
from src.models.feature_extractor import FeatureExtractor
from src.models.dataset import create_dataloaders
from src.models.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train feature extractor model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int, default=ModelConfig.NUM_EPOCHS,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=ModelConfig.BATCH_SIZE,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=ModelConfig.LEARNING_RATE,
                      help='Learning rate')
    args = parser.parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=ModelConfig.NUM_WORKERS,
        train_split=ModelConfig.TRAIN_SPLIT
    )
    
    # Initialize model
    model = FeatureExtractor(
        num_classes=ModelConfig.NUM_CLASSES,
        pretrained=True,
        feature_dim=ModelConfig.FEATURE_DIM,
        dropout_rate=ModelConfig.DROPOUT_RATE
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=ModelConfig.DEVICE
    )
    
    # Train model
    trainer.train(
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=1
    )
    
    # Plot and save training history
    trainer.plot_history(save_path=checkpoint_dir / 'training_history.png')

if __name__ == '__main__':
    main()