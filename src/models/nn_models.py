"""Neural network model architectures for tabular data."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class SimpleMLP(nn.Module):
    """Multi-Layer Perceptron with BatchNorm, Dropout, and Residual Connections.
    
    Inspired by the example architecture but adapted for tabular data.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float = 0.5,
        use_residual: bool = True
    ):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Input block with LayerNorm (inspired by example)
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU()
        )
        
        # Hidden blocks
        self.hidden_blocks = nn.ModuleList()
        for i in range(num_layers - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate),
                nn.LeakyReLU()
            )
            self.hidden_blocks.append(block)
        
        # Output block
        self.output_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input block
        x = self.input_block(x)
        
        # Hidden blocks with residual connections
        for block in self.hidden_blocks:
            if self.use_residual:
                residual = x
                x = block(x)
                x = x + residual  # Residual connection
            else:
                x = block(x)
        
        # Output block
        x = self.output_block(x)
        return x


class TabMModel(nn.Module):
    """TabM model based on Yandex architecture with multiplicative interactions.
    
    Reference: https://arxiv.org/abs/2307.14338
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        multiplicative_units: int = 128,
        dropout_rate: float = 0.5
    ):
        super(TabMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.multiplicative_units = multiplicative_units
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Feature embeddings (for multiplicative interactions)
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(input_dim, multiplicative_units) for _ in range(num_layers)
        ])
        
        # Multiplicative interaction layers
        self.multiplicative_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Multiplicative interaction: element-wise product of feature embeddings
            self.multiplicative_layers.append(
                nn.Linear(multiplicative_units, hidden_dim)
            )
            # Standard linear transformation
            self.linear_layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Output block
        self.output_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input projection
        h = self.input_proj(x)
        
        # Multiplicative interaction layers
        for i in range(self.num_layers):
            # Feature embeddings
            feat_emb = self.feature_embeddings[i](x)
            feat_emb = F.leaky_relu(feat_emb)
            
            # Multiplicative interaction
            mult_out = self.multiplicative_layers[i](feat_emb)
            
            # Combine with linear transformation
            linear_out = self.linear_layers[i](h)
            linear_out = self.bn_layers[i](linear_out)
            
            # Element-wise multiplication (multiplicative interaction)
            h = mult_out * linear_out + h  # Residual connection
            
            # Activation and dropout
            h = F.leaky_relu(h)
            h = self.dropout_layers[i](h)
        
        # Output
        output = self.output_block(h)
        return output



