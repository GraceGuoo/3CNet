import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.orthoAttention.OrthoAttention import OrthoAttention
### 3CM
class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights
class CCModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(CCModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s

        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.crossm1 = OrthoAttention(dim, dim)
        self.crossm2 = OrthoAttention(dim, dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):

        spatial_weights = self.spatial_weights(x1, x2)

        x11 = self.crossm1(x1)
        x22 = self.crossm2(x2)

        out_x1 = x1  + self.lambda_s * spatial_weights[1] * x2 + 0.5*x22 #self.lambda_c * channel_weights[1] * x2
        out_x2 = x2 + self.lambda_s * spatial_weights[0] * x1 + 0.5*x11 #+ self.lambda_c * channel_weights[0] * x1
        return out_x1, out_x2 


# Stage 1
### Feature Fuison
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class F1(nn.Module):
    def __init__(self, all_channel=64):
        super(F1, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(all_channel*2,int( all_channel/2), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel*2,int( all_channel/2), kernel_size=3, dilation=4, padding=4) #3
        self.dconv3 = BasicConv2d(all_channel*2,int( all_channel/2), kernel_size=3, dilation=8, padding=8) #5
        self.dconv4 = BasicConv2d(all_channel*2,int( all_channel/2), kernel_size=3, dilation=12, padding=12) #7
        self.fuse_dconv = nn.Conv2d(all_channel*2, all_channel, kernel_size=3,padding=1)

    def forward(self, x, ir, gate):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(gate*x + (1-gate)*ir)
        fusion = torch.cat((summation, multiplication), dim = 1 )
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))
        return out
