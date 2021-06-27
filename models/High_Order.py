import torch.nn as nn
# from .utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet
from .transformer import Transformer
import pdb
from models import MPNCOV

def get_ResNet(net_type):
    if net_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2])
    elif net_type == "senet101":
        model = models.resnet101(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2])
    else:
        assert False, "unknown ResNet type : " + net_type

    return model


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("instantiating " + self.__class__.__name__)

    def forward(self, x, b, t):
        return [x, None, None]


class CosegAttention(nn.Module):
    def __init__(self, attention_types, num_feat_maps, h_w, t):
        super().__init__()
        print("instantiating " + self.__class__.__name__)
        self.attention_modules = nn.ModuleList()

        for i, attention_type in enumerate(attention_types):
            if attention_type in COSEG_ATTENTION:
                self.attention_modules.append(
                    COSEG_ATTENTION[attention_type](
                        num_feat_maps[i], h_w=h_w[i], t=t)
                )
            else:
                assert False, "unknown attention type " + attention_type

    def forward(self, x, i, b, t):
        return self.attention_modules[i](x, b, t)

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        self.spatial_mask_summary = nn.Sequential(
            nn.Conv2d(16*12, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
        self.add = 0            
        self.mode = mode
        self.eps = 1e-4
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size, C, H , W = x.size(0), x.size(1), x.size(2), x.size(3)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            #pdb.set_trace()
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            #normalized
            theta_x = theta_x - torch.mean(theta_x, 2, keepdim=True)
            theta_x = theta_x / (torch.std(theta_x, 2, keepdim=True)+self.eps)    

            phi_x = phi_x-torch.mean(phi_x, 1, keepdim=True)
            phi_x = phi_x/(torch.std(phi_x, 1, keepdim=True)+ self.eps)
            
            
            f = torch.matmul(theta_x, phi_x)/phi_x.size()[1]
            mutual_correlation =f.permute(0, 2, 1)  # #frames x (t-1 * H * W) x (HW)
            mutual_correlation = mutual_correlation.view(batch_size, H*W, H, W)  # #frames x (t-1 * H * W) x H x W
            mutual_correlation_mask = self.spatial_mask_summary(mutual_correlation).sigmoid() # #frames x 1 x H x W
        

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        '''
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x
        #pdb.set_trace()
        self.add = f_div_C
        '''
        #pdb.set_trace()
        x = x + mutual_correlation_mask*x
        return x
class SOMM_v2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, h=11, w=1, stride=1, downsample=None, attention='2', att_dim=1):
        super(SOMM_v2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)
        
        if attention in {'2','+','M','&'}:
            self.sp_d = att_dim
            self.sp_h = h
            self.sp_w = w
            self.sp_reso = self.sp_h * self.sp_w
            self.conv_for_DR_spatial = nn.Conv2d(
                 inplanes, self.sp_d, 
                 kernel_size=1,stride=1, bias=True)
            self.bn_for_DR_spatial = nn.BatchNorm2d(self.sp_d)

            #self.adppool = nn.AdaptiveAvgPool2d((self.sp_h,self.sp_w))
            self.row_bn_for_spatial = nn.BatchNorm2d(self.sp_reso)
            #row-wise conv is realized by group conv
            #self.row_conv_group_for_spatial = nn.Conv2d( 
            #     self.sp_reso, self.sp_reso, kernel_size=(self.sp_reso, 1), 
            #     groups=self.sp_reso, bias=True)
            self.fc_adapt_channels_for_spatial = nn.Linear(
                 self.sp_reso*self.sp_reso, self.sp_reso)
            self.sigmoid = nn.Sigmoid()
            self.adpunpool = F.adaptive_avg_pool2d

       



    def pos_att(self, out):
        pre_att = out # NxCxHxW
        batch_size, C , H , W = pre_att.size()[0], pre_att.size()[1], pre_att.size()[2], pre_att.size()[3]
        #pdb.set_trace()
        
        out = self.conv_for_DR_spatial(out)
        out = self.bn_for_DR_spatial(out)
        out = self.relu_normal(out)
        #out = self.adppool(out) # keep the feature map size to 8x8

        #out = cov_feature(out) # Nx64x64
        out = out.view(pre_att.size()[0], -1, H*W).permute(0,2,1).unsqueeze(3) 
        #out = out.reshape(pre_att.size()[0], 64, -1,1)
       
        out = MPNCOV.CovpoolLayer(out)
        out = MPNCOV.SqrtmLayer(out,5)
       
        out = out.view(out.size(0), H*W, H*W, 1).contiguous()  # Nx64x64x1
        out = self.row_bn_for_spatial(out)

        out = self.row_conv_group_for_spatial(out) # Nx256x1x1
        #out = self.relu(out)
        #pdb.set_trace()

        out = out.view(out.size(0), -1)
        out = self.fc_adapt_channels_for_spatial(out) #Nx64x1x1
        out = self.sigmoid(out) 
        out = out.view(out.size(0), 1, self.sp_h, self.sp_w).contiguous()#Nx1x8x8

        #out = self.adpunpool(out,(pre_att.size(2), pre_att.size(3))) # unpool Nx1xHxW

        return out


    def forward(self, out):
        
        att = self.pos_att(out)
        out = out*att 

        return out
        
        
class SOMM_v3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, h=11, w=1, stride=1, downsample=None, attention='2', att_dim=1):
        super(SOMM_v3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)
        
    
        DR_stride=1
       
        self.ch_dim = 11
        self.conv_for_DR = nn.Conv2d(
                 self.ch_dim, self.ch_dim, 
                 kernel_size=1,stride=DR_stride, bias=True)
        self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
        self.row_bn = nn.BatchNorm2d(self.ch_dim)
        #row-wise conv is realized by group conv
        self.row_conv_group = nn.Conv2d(
                 self.ch_dim, 4*self.ch_dim, 
                 kernel_size=(self.ch_dim, 1), 
                 groups = self.ch_dim, bias=True)
        self.fc_adapt_channels = nn.Conv2d(
                 4*self.ch_dim, self.ch_dim, 
                 kernel_size=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()

       

    def chan_att(self, out):
        # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR(out)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        out = MPNCOV.CovpoolLayer(out) # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous() # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out) # Nx512x1x1

        out = self.fc_adapt_channels(out) #NxCx1x1
        out = self.sigmoid(out) #NxCx1x1

        return out


    def forward(self, out):
        
        att = self.chan_att(out)
        out = out*att 

        return out        
        
        
        
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
    
class SOMM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, stride=1, downsample=None, attention='2', att_dim=22):
        super(SOMM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)
        
        if attention in {'2','+','M','&'}:
            self.sp_d = att_dim
            self.sp_h = 11
            self.sp_w = 1
            self.sp_reso = self.sp_h * self.sp_w
            #dim  reduction
            self.conv_for_DR_spatial = nn.Conv2d(
                 inplanes, self.sp_d, 
                 kernel_size=1,stride=1, bias=True)
            # bn layer,                  
            self.bn_for_DR_spatial = nn.BatchNorm2d(self.sp_d)
            # down sampling
            self.adppool = nn.AdaptiveAvgPool2d((self.sp_h,self.sp_w))
            # bn layer
            self.row_bn_for_spatial = nn.BatchNorm2d(self.sp_reso)
            #row-wise conv is realized by group conv
            self.row_conv_group_for_spatial = nn.Conv2d( 
                 self.sp_reso, self.sp_reso*4, kernel_size=(self.sp_reso, 1), 
                 groups=self.sp_reso, bias=True)
            # dim unreduction
            self.fc_adapt_channels_for_spatial = nn.Conv2d(
                 self.sp_reso*4, self.sp_reso, kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()
            self.adpunpool = F.adaptive_avg_pool2d

       



    def pos_att(self, out):
        pre_att = out # NxCxHxW
        #pdb.set_trace()
        out = self.relu_normal(out)
        out = self.conv_for_DR_spatial(out)
        out = self.bn_for_DR_spatial(out)

        out = self.adppool(out) # keep the feature map size to 8x8

        #out = cov_feature(out) # Nx64x64
        out = out.view(pre_att.size()[0], -1, 192).permute(0,2,1).unsqueeze(3) 
        #out = out.reshape(pre_att.size()[0], 64, -1,1)
       
        out = MPNCOV.CovpoolLayer(out)#input:B S D output:B S S 
        #pdb.set_trace()
        out = MPNCOV.SqrtmLayer(out,5)
       
        out = out.view(out.size(0), 192, 192, 1).contiguous()  # Nx64x64x1
        out = self.row_bn_for_spatial(out)

        out = self.row_conv_group_for_spatial(out) # Nx256x1x1
        out = self.relu(out)

        out = self.fc_adapt_channels_for_spatial(out) #Nx64x1x1
        out = self.sigmoid(out) 
        out = out.view(out.size(0), 1, self.sp_h, self.sp_w).contiguous()#Nx1x8x8

        out = self.adpunpool(out,(pre_att.size(2), pre_att.size(3))) # unpool Nx1xHxW

        return out


    def forward(self, out):

        
        att = self.pos_att(out)
        #out = out*att 

      

        return att
        
        
        
class SONL(nn.Module):
    def __init__(self, channel):
        super(SONL, self).__init__()
        self.channels = channel
        self.key = nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(channel //2, channel, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        batch, c, h, w = x.shape
        key = self.key(x)
        #value = self.value(x).reshape(batch, -1, h*w)
        value = self.value(x).reshape(batch, h*w, -1)
        # print('value',value.shape)
        #pdb.set_trace()
        cov_mat = MPNCOV.CovpoolLayer(key)
        cov_mat = cov_feature(key)
        cov_mat = ((self.channels // 2) ** -.5) * cov_mat
        cov_mat = F.softmax(cov_mat, dim=-1)
        # print('cov_mat',cov_mat.shape)
        out = torch.matmul(cov_mat,value).view(batch, c // 2, *x.size()[2:])
        out = self.conv(out)
        out += x
        return out

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        # subsample
        # subsample_scale = 2
        # subsample = nn.Upsample(size=(h // subsample_scale, w // subsample_scale), mode='nearest')
        # x_sub = subsample(x)
        # max_pool = nn.MaxPool2d(kernel_size=2)
        # max_pool = nn.AvgPool2d(kernel_size=2)
        # x_sub = self.max_pool(x)
        ##
        ## MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
        # y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_cov = self.conv_du(cov_mat_sum)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_cov*x
class SOSA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOSA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(

            nn.Conv2d(1, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x_pre):
        batch_size_pre, C_pre, h_pre, w_pre = x_pre.shape  # x: NxCxHxW
        x = x_pre.view(batch_size_pre, C_pre , h_pre*w_pre, 1).permute(0,2,1,3)
        batch_size, C, h, w = x.shape 
        
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        # subsample
        # subsample_scale = 2
        # subsample = nn.Upsample(size=(h // subsample_scale, w // subsample_scale), mode='nearest')
        # x_sub = subsample(x)
        # max_pool = nn.MaxPool2d(kernel_size=2)
        # max_pool = nn.AvgPool2d(kernel_size=2)
        # x_sub = self.max_pool(x)
        ##
        ## MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size, 1, h_pre, w_pre )
        #pdb.set_trace()
        # y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        cov_mat_sum = cov_mat_sum.squeeze().view(batch_size, h_pre*w_pre)
        y_cov = (cov_mat_sum-torch.min(cov_mat_sum, 1)[0].unsqueeze(1)) / (torch.max(cov_mat_sum, 1)[0].unsqueeze(1)-torch.min(cov_mat_sum, 1)[0].unsqueeze(1))

        y_cov = y_cov.view(batch_size, 1, h_pre, w_pre )
        #y_cov = self.conv_du(cov_mat_sum)
       
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_cov*x_pre