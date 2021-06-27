"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import pdb
from models import MPNCOV

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
    def __init__(self, dim, h,w, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        #pdb.set_trace()
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h
        
def cov_feature(x):
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h*w
    x = x.reshape(batchsize,dim,M)
    I_hat = (-1./M/M)*torch.ones(dim,dim,device = x.device) + (1./M)*torch.eye(dim,dim,device = x.device)
    I_hat = I_hat.view(1,dim,dim).repeat(batchsize,1,1).type(x.dtype)
    y = (x.transpose(1,2)).bmm(I_hat).bmm(x)
    return y
    
class MultiHeadedSelfAttention_v2(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, h,w, num_heads, dropout, ):
        super().__init__()
       
        self.proj_v = nn.Linear(dim, dim)
        self.proj_q = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.relu_normal = nn.ReLU(inplace=False)
        self.scores = None # for visualization
        self.h = h
        self.w = w
        self.bn_for_DR_spatial = nn.BatchNorm2d(h*w)
        self.adppool = nn.AdaptiveAvgPool2d((8, 6))
        self.sigmoid = nn.Sigmoid()
        
     
        #dim  reduction
        self.conv_for_DR_spatial = nn.Conv2d(
                 dim, 128, 
                 kernel_size=1,stride=1, bias=True)
        # bn layer,                  
        self.bn_for_DR_spatial = nn.BatchNorm2d(128)
        
        # down sampling
        self.adppool = nn.AdaptiveAvgPool2d((8,6))
        # bn layer
        self.row_bn_for_spatial = nn.BatchNorm2d(self.h*self.w)
        #row-wise conv is realized by group conv
        #self.row_conv_group_for_spatial = nn.Conv2d( 
        #         self.sp_reso, self.sp_reso*4, kernel_size=(self.sp_reso, 1), 
        #         groups=self.sp_reso, bias=True)
        # dim unreduction
        #self.fc_adapt_channels_for_spatial = nn.Conv2d(
        #         self.sp_reso*4, self.sp_reso, kernel_size=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.adpunpool = F.adaptive_avg_pool2d       
    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        #pdb.set_trace()
        #pdb.set_trace()
        '''
        q, v = self.proj_q(x), self.proj_v(x)
        q = self.relu_normal(q)
        out = self.bn_for_DR_spatial(q.unsqueeze(3))
       
        out = MPNCOV.CovpoolLayer(q.unsqueeze(3))#input:B S D output:B S S 
        #out = MPNCOV.SqrtmLayer(out,5)
        out = ((x.size()[2] // 2) ** -.5) * out
        out = F.softmax(out, dim=-1)        
        '''
        v =  self.proj_v(x)
        pre_att = x.permute(0,2,1).view(x.size(0), -1, self.h, self.w)
         # NxCxHxW
        #pdb.set_trace()
        out = self.relu_normal(pre_att)
        out = self.conv_for_DR_spatial(out)
        out = self.bn_for_DR_spatial(out)

        #out = self.adppool(out) # keep the feature map size to 8x8

        #out = cov_feature(out) # Nx64x64
        out = out.view(pre_att.size()[0], -1, self.h*self.w).permute(0,2,1).unsqueeze(3) 
        #out = out.reshape(pre_att.size()[0], 64, -1,1)
       
        out = MPNCOV.CovpoolLayer(out)#input:B S D output:B S S 
        #pdb.set_trace()
        out = MPNCOV.SqrtmLayer(out,5)
       
        #out = out.view(out.size(0), self.h*self.w, self.h*self.w, 1).contiguous()  # Nx64x64x1
        #out = self.row_bn_for_spatial(out)

        #out = self.row_conv_group_for_spatial(out) # Nx256x1x1
        #out = self.relu(out)

        #out = self.fc_adapt_channels_for_spatial(out) #Nx64x1x1
        #out = self.sigmoid(out) 
        out = out.squeeze().contiguous() 
        out = ((x.size()[2] // 2) ** -.5) * out
        out = F.softmax(out, dim=-1) 
        #out = out.view(out.size(0), 1, self.sp_h, self.sp_w).contiguous()#Nx1x8x8
       
        #out = self.adpunpool(out,(pre_att.size(2), pre_att.size(3))) # unpool Nx1xHxW
        

        h = (out @ v)#.transpose(1, 2).contiguous()

        return h   
        
class MultiHeadedSelfAttention_att(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, spatial, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)      
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = F.softmax(scores, dim=-1)#B H S S

        return scores

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
    def __init__(self, dim, h, w, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, h, w, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)


    def forward(self, x, mask):
        output = self.attn(self.norm1(x), mask)
        h = self.drop(self.proj(output))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x
        

class Block_att(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_att(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.attn(self.norm1(x), mask)  #B H S S 

        return h


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, h,w, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, h, w, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
