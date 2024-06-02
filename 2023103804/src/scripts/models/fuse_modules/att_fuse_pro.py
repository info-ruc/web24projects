"""
Enhanced Implementation of Attn Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.models.sub_modules.torch_transformation_utils import (
    warp_affine_simple
)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with mask support.
    """
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.query_linear = nn.Linear(dim, dim)
        self.key_linear = nn.Linear(dim, dim)
        self.value_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        self.scale = np.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        B, N, _ = query.size()
        query = self.query_linear(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(B, N, self.dim)
        out = self.out_linear(context)
        return out, attn

class FeedForward(nn.Module):
    """
    Feed Forward Network
    """
    def __init__(self, dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class AttFusion(nn.Module):
    def __init__(self, args):
        super(AttFusion, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']
        self.num_heads = args['num_heads']
        self.ff_dim = args['ff_dim']

        self.att = MultiHeadAttention(args['in_channels'], self.num_heads)
        self.ffn = FeedForward(args['in_channels'], self.ff_dim)
        self.norm1 = nn.LayerNorm(args['in_channels'])
        self.norm2 = nn.LayerNorm(args['in_channels'])

    def regroup(self, x, record_len):
        split_x = torch.split(x, record_len.tolist())
        return split_x

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = self.regroup(xx, record_len)

        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
        pairwise_t_matrix[..., 0, 1] *= H / W
        pairwise_t_matrix[..., 1, 0] *= W / H
        pairwise_t_matrix[..., 0, 2] /= (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] /= (self.downsample_rate * self.discrete_ratio * H) * 2

        out = []
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            x = warp_affine_simple(split_x[b], t_matrix[0, :, :, :], (H, W))

            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C)
            x, _ = self.att(x, x, x)
            x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]

            x = self.norm1(x + xx)
            x = self.norm2(self.ffn(x) + x)
            out.append(x)

        out = torch.stack(out)
        return out
