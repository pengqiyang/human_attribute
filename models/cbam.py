import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def compute_crow_spatial_weight(X, a=2, b=2):
    """
    Given a tensor of features, compute spatial weights as normalized total activation.
    Normalization parameters default to values determined experimentally to be most effective.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :param int a:
        the p-norm
    :param int b:
        power normalization
    :returns ndarray:
        a spatial weight matrix of size (height, width)
    """
    pdb.set_trace()
    S = X.sum(axis=1)#B W H
    z = ((S**a).sum(axis=(1,2))**(1./a)).unsqueeze(1).unsqueeze(1)#B, 1 1
    return (S / z)**(1./b) if b != 1 else (S / z)


def compute_crow_channel_weight(X):
    """
    Given a tensor of features, compute channel weights as the
    log of inverse channel sparsity.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        a channel weight vector
    """
    pdb.set_trace()
    eps = 1e-7
    B, K, w, h = X.shape
    area = w * h
    nonzeros = np.count_nonzero(X.view(B,K, -1).detach().cpu().numpy(), axis=2)/area
    nonzeros = torch.from_numpy(nonzeros).float().cuda() # B K 

    nzsum = (nonzeros.sum(dim=1)).unsqueeze(1)#B, 1 
    '''
    for i in range(B):
        for j in range(K):
            nonzeros[i,j] = torch.log(nzsum[i] / nonzeros[i][j]) if nonzeros[i][j] > 0. else 0.
    '''
    nonzeros = torch.log(nzsum / (eps+nonzeros) ) 
    
    return nonzeros


def apply_crow_aggregation(X):
    """
    Given a tensor of activations, compute the aggregate CroW feature, weighted
    spatially and channel-wise.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        CroW aggregated global image feature    
    """
    S = compute_crow_spatial_weight(X).unsqueeze(1)#B 1 W H
    C = compute_crow_channel_weight(X).unsqueeze(2).unsqueeze(3)#B C 1 1 
    
    X = X * S
    X = torch.sum(X,  dim=(2, 3)).unsqueeze(2).unsqueeze(3)#B C 1 1
    #pdb.set_trace()
    #C = C.double()
    return X * C
    
    
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='gem':
                #pdb.set_trace()
                gem_pool = gem(x)
                channel_att_raw = self.mlp( gem_pool )
            if pool_type=='crow':
                #pdb.set_trace()
                crow_pool = apply_crow_aggregation(x)
                channel_att_raw = self.mlp(crow_pool)
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        self.channel_weight = 0
        self.spatial_weight = 0
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        #pdb.set_trace()
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
        
class SBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SBAM, self).__init__()
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.SpatialGate(x)
        return x_out
