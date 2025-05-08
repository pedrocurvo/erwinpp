from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from erwin import ErwinTransformer

class ErwinTransolver(nn.Module):
    """Combines Transolver's token slicing with Erwin's hierarchical processing.
    Instead of using attention between slice tokens, uses a Erwin network.
    """
    def __init__(
        self, 
        dim: int,
        slice_num: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dimensionality: int = 3
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.dimensionality = dimensionality
        
        # Input projections for slicing
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        
        # Temperature parameter for slice weights
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        
        # Erwin network for processing slice tokens
        self.erwin = ErwinTransformer(
            c_in=dim_head,
            c_hidden=[dim_head, dim_head*2],  # Two levels of hierarchy
            ball_sizes=[min(32, slice_num), min(16, slice_num//2)],  # Progressive reduction
            enc_num_heads=[heads//2, heads],
            enc_depths=[2, 2],
            dec_num_heads=[heads//2],
            dec_depths=[2],
            strides=[2],
            rotate=1,  # Enable rotation for better cross-token mixing
            decode=True,  # We need the full resolution back
            mlp_ratio=4,
            dimensionality=dimensionality,
            mp_steps=0  # No need for MPNN here
        )
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C) where:
               B = batch size
               N = number of points
               C = number of channels/features
        Returns:
            Reduced tensor of shape (B, slice_num, C)
        """
        B, N, C = x.shape
        
        # Project inputs
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        
        # Compute slice weights with temperature scaling
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )
        
        # Normalize slice weights
        slice_norm = slice_weights.sum(2, keepdim=True)
        
        # Create slice tokens through weighted aggregation
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-5)  # [B, H, G, C]
        
        # Process slice tokens with Erwin
        # Reshape for Erwin: [B*H, G, C]
        B, H, G, C = slice_token.shape
        slice_token = slice_token.reshape(B*H, G, C)
        
        # Create batch indices for Erwin
        batch_idx = torch.arange(B*H, device=x.device).repeat_interleave(G)
        
        # Create artificial positions for slice tokens (uniformly distributed in unit cube)
        pos = torch.rand(B*H*G, self.dimensionality, device=x.device)
        
        # Process through Erwin
        processed_tokens = self.erwin(slice_token, pos, batch_idx)
        processed_tokens = processed_tokens.reshape(B, H, G, C)
        
        # Deslice using the same weights
        out = torch.einsum("bhgc,bhng->bhnc", processed_tokens, slice_weights)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)