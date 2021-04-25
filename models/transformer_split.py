"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import pdb

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization
        self.avg_x = nn.AdaptiveAvgPool2d((1, None))
        self.avg_y = nn.AdaptiveAvgPool2d((None, 1))
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        #pdb.set_trace()
        B, D, H ,W = x.size()[0], x.size()[1], x.size()[2], x.size()[3]
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        x = x.permute(0,2,3,1).view(B, H*W, D)
        x = self.norm1(x)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        
        #calcluate the x attention
        q_x = self.avg_x(q.transpose(2,1).view(B, D, H, W)).permute(0,2,3,1).view(B, W, D)
        k_x = self.avg_x(k.transpose(2,1).view(B, D, H, W)).permute(0,2,3,1).view(B, W, D)
        v_x = self.avg_x(v.transpose(2,1).view(B, D, H, W)).permute(0,2,3,1).view(B, W, D)
        q_x, k_x, v_x = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q_x, k_x, v_x])
        # (B, head, W, head_dim) @ (B, head, head_dim, W) -> (B, head, W, W)
        scores_x = q_x @ k_x.transpose(-2, -1) / np.sqrt(k_x.size(-1))
        h_x = (scores_x @ v_x).transpose(1, 2).contiguous()
        h_x = merge_last(h_x, 2)#B W D
        
        
        
        
        #calculate the y attention
        q_y = self.avg_y(q.transpose(2,1).view(B, D, H, W)).permute(0,2,3,1).view(B, H, D)
        k_y = self.avg_y(k.transpose(2,1).view(B, D, H, W)).permute(0,2,3,1).view(B, H, D)
        v_y = self.avg_y(v.transpose(2,1).view(B, D, H, W)).permute(0,2,3,1).view(B, H, D)
        q_y, k_y, v_y = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q_y, k_y, v_y])
        # (B, head, H, head_dim) @ (B, head, head_dim, H) -> (B, head, H, H,)
        scores_y = q_y @ k_y.transpose(-2, -1) / np.sqrt(k_y.size(-1))
        h_y = (scores_y @ v_y).transpose(1, 2).contiguous()
        h_y = merge_last(h_y, 2)#B H D
        
        h = (h_y.transpose(2, 1).unsqueeze(3) @ h_x.transpose(2, 1).unsqueeze(2)).reshape(B, D, H*W).transpose(2,1)

        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.proj(self.attn(x)))

        return h


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x)
        return x
