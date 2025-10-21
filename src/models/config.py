"""
Configuration for model training and evaluation.
"""

class ModelConfig:
    # Model parameters
    NUM_CLASSES = 100  # To be adjusted based on your dataset
    FEATURE_DIM = 2048
    DROPOUT_RATE = 0.5
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.8
    
    # Paths
    DATA_DIR = "data/"
    CHECKPOINT_DIR = "checkpoints/"
    
    # Device
    DEVICE = "cuda"  # or "cpu"
    
    # Image parameters
    IMAGE_SIZE = (224, 224)
    CHANNELS = 3
    
    # Normalization parameters
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]