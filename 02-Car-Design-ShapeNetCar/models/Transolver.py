import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


from .erwin import ErwinTransformer

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
        # Shape explanation:
        # B: batch size (number of samples in the batch)
        # H: number of attention heads
        # G: number of slice tokens (slice_num)
        # C: channels per head (dim_head)
        B, H, G, C = slice_token.shape
        
        # We use Erwin to compute attention between slice tokens. Each slice token represents
        # a weighted aggregation of the input features. By applying Erwin, we allow these slice tokens
        # to interact with each other through hierarchical ball attention, effectively capturing
        # multi-scale relationships between different regions of the input.
        
        # Reshape for Erwin: [B*H, G, C] -> [B*H*G, C]
        slice_token = slice_token.reshape(B*H*G, C)  # Flatten to [total_points, channels]
        
        # Create artificial positions for slice tokens (uniformly distributed in unit cube)
        pos = torch.rand(B*H*G, self.dimensionality, device=x.device)  # [total_points, dims]
        
        # Create batch indices - each slice token needs its own batch index
        batch_idx = torch.arange(B*H, device=x.device).repeat_interleave(G)
        
        # Add safety checks
        assert slice_token.shape[0] == pos.shape[0] == batch_idx.shape[0], \
            f"Shapes mismatch: features {slice_token.shape}, pos {pos.shape}, batch {batch_idx.shape}"
        
        # Process through Erwin - it expects [num_points, channels] for features
        processed_tokens = self.erwin(slice_token, pos, batch_idx)
        
        # Reshape back to original format [B, H, G, C]
        processed_tokens = processed_tokens.reshape(B, H, G, C)
        
        # Deslice using the same weights
        out = torch.einsum("bhgc,bhng->bhnc", processed_tokens, slice_weights)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = ErwinTransolver(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):
        cfd_data, geom_data = data
        x, fx, T = cfd_data.x, None, None
        x = x[None, :, :]
        if self.unified_pos:
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx[0]
