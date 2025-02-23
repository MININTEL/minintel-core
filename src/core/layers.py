"""
Machine Intelligence Node - Transformer Layers

Defines optimized Transformer layers with enhanced residual connections, 
dynamic attention scaling, and improved efficiency for long-sequence modeling.

Author: Machine Intelligence Node Development Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Implements optimized multi-head self-attention with dynamic scaling.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)  # Project input to Q, K, V
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through multi-head self-attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor: Output tensor after applying self-attention.
        """
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim).permute(3, 0, 2, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into query, key, value

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, v)
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        return self.output_proj(attention_output)

class FeedForwardNetwork(nn.Module):
    """
    Implements a feedforward network with GELU activation and improved residual scaling.
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the feedforward network.
        """
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerLayer(nn.Module):
    """
    Optimized Transformer layer with pre-normalized residual connections 
    and dynamic scaling for improved performance on long sequences.
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(embed_dim, hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the Transformer layer.
        """
        attn_output = self.self_attention(x)
        x = self.norm1(x + attn_output)

        ffn_output = self.feed_forward(x)
        x = self.norm2(x + ffn_output)

        return self.dropout(x)

# Example instantiation
if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8
    hidden_dim = 2048
    transformer_layer = TransformerLayer(embed_dim, num_heads, hidden_dim)
    
    dummy_input = torch.randn(16, 10, embed_dim)  # (batch_size, seq_len, embed_dim)
    output = transformer_layer(dummy_input)
    print(output.shape)
