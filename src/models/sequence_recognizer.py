import torch
import torch.nn as nn

class SequenceRecognizer(nn.Module):
    """CNN-BiLSTM-CTC model for sequence recognition."""
    
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer (multiply hidden_size by 2 for bidirectional)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the sequence recognizer.
        
        Args:
            x (torch.Tensor): Input tensor from feature extractor
            
        Returns:
            torch.Tensor: Logits for CTC loss
        """
        # Process through BiLSTM
        outputs, _ = self.bilstm(x)
        
        # Apply classification layer
        logits = self.classifier(outputs)
        
        return logits