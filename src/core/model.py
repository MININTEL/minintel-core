"""
Machine Intelligence Node - Transformer Model

This module defines the Transformer-based model architecture with enhancements
for training efficiency, stability, and generalization.

Author: Machine Intelligence Node Development Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

class BaseModel(nn.Module):
    """
    Base Model class providing a foundation for all AI architectures in Machine Intelligence Node.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclasses.")

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for input sequences.
    Helps the model retain positional information in self-attention layers.
    """
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerModel(BaseModel):
    """
    Transformer-based model for Machine Intelligence Node.
    """
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, ff_hidden_dim=2048, dropout=0.1, max_seq_len=5000):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_hidden_dim, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass through the Transformer model.
        """
        x = self.embedding(x) * math.sqrt(x.shape[-1])
        x = self.positional_encoding(x)
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x

    def load_pretrained_weights(self, path):
        """
        Load pre-trained model weights with error handling.
        """
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            print(f"Successfully loaded pre-trained weights from {path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def _initialize_weights(self):
        """
        Initialize model weights using Xavier uniform initialization.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

# Example instantiation
if __name__ == "__main__":
    vocab_size = 30000
    model = TransformerModel(vocab_size=vocab_size)
    model.to(model.device)
    print(model)
