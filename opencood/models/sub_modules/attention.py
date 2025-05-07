import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class MutiHeadAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim, num_heads = 4):
        super(MutiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.dim = dim
        
        self.sqrt_dim = np.sqrt(dim)
        self.Q_gen = nn.Linear(dim, dim * num_heads)
        self.K_gen = nn.Linear(dim, dim * num_heads)
        self.V_gen = nn.Linear(dim, dim * num_heads)
        
        self.out = nn.Linear(dim * num_heads, dim)

    def forward(self, x):
        
        # x: b c h w
        B, C, H, W = x.shape
        
        # b c h w -> b c h*w -> b h*w c
        x = x.view(B, C, -1).permute(0, 2, 1)
        
        # b h*w c -> b h*w m*c -> b h*w m c
        q = self.Q_gen(x).view(-1, self.num_heads, self.dim).permute(0, 2, 1)
        k = self.K_gen(x).view(-1, self.num_heads, self.dim)
        v = self.V_gen(x).view(-1, self.num_heads, self.dim)
        
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context