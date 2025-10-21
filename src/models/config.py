"""
Configuration for model training and evaluation.
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch

@dataclass
class ModelConfig:
    # Model parameters
    NUM_CLASSES: int = 100  # To be adjusted based on your dataset
    FEATURE_DIM: int = 2048
    DROPOUT_RATE: float = 0.5
    
    # Training parameters
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    TRAIN_SPLIT: float = 0.8
    
    # Advanced training options
    USE_MIXED_PRECISION: bool = True
    USE_GRADIENT_CLIPPING: bool = True
    GRADIENT_CLIP_VALUE: float = 1.0
    USE_MULTI_GPU: bool = torch.cuda.device_count() > 1
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE: int = 7
    EARLY_STOPPING_MIN_DELTA: float = 1e-4
    
    # Learning rate scheduling
    LR_SCHEDULER_TYPE: str = 'cosine'  # 'plateau', 'cosine', 'linear', 'step'
    LR_WARMUP_EPOCHS: int = 5
    LR_MIN_FACTOR: float = 1e-6
    
    # Optimizer parameters
    WEIGHT_DECAY: float = 0.01
    MOMENTUM: float = 0.9
    BETAS: Tuple[float, float] = (0.9, 0.999)
    
    # Paths
    DATA_DIR: str = "data/"
    CHECKPOINT_DIR: str = "checkpoints/"
    LOGS_DIR: str = "logs/"
    
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Image parameters
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    CHANNELS: int = 3
    
    # Augmentation parameters
    USE_AUGMENTATION: bool = True
    RANDOM_ROTATE_DEGREES: float = 15.0
    RANDOM_SCALE_RANGE: Tuple[float, float] = (0.8, 1.2)
    RANDOM_CROP_SCALE: Tuple[float, float] = (0.7, 1.0)
    COLOR_JITTER_PARAMS: dict = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
    
    # Normalization parameters
    NORMALIZE_MEAN: List[float] = [0.485, 0.456, 0.406]
    NORMALIZE_STD: List[float] = [0.229, 0.224, 0.225]
    
    # Experiment tracking
    USE_WANDB: bool = False  # Whether to use Weights & Biases
    WANDB_PROJECT: str = "math-paper-checker"
    WANDB_ENTITY: Optional[str] = None
    
    # Validation
    VAL_CHECK_INTERVAL: int = 1  # Validate every N epochs
    SAVE_TOP_K: int = 3  # Number of best models to save