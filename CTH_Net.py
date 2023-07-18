from torchvision import models as resnet_model
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from src.c_branch import res2net50_26w_4s
from src.t_branch import *

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int (method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer (torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super (MultiSpectralAttentionLayer, self).__init__ ()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices (freq_sel_method)
        self.num_split = len (mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer (dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential (
            nn.Linear (channel, channel // reduction, bias=False),
            nn.ReLU (inplace=True),
            nn.Linear (channel // reduction, channel, bias=False),
            nn.Sigmoid ()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d (x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer (x_pooled)

        y = self.fc (y).view (n, c, 1, 1)
        return x * y.expand_as (x)

class MultiSpectralDCTLayer (nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super (MultiSpectralDCTLayer, self).__init__ ()

        assert len (mapper_x) == len (mapper_y)
        assert channel % len (mapper_x) == 0

        self.num_freq = len (mapper_x)

        # fixed DCT init
        self.register_buffer ('weight', self.get_dct_filter (height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len (x.shape) == 4, 'x must been 4 dimensions, but got ' + str (len (x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum (x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos (math.pi * freq * (pos + 0.5) / POS) / math.sqrt (POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt (2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros (channel, tile_size_x, tile_size_y)

        c_part = channel // len (mapper_x)

        for i, (u_x, v_y) in enumerate (zip (mapper_x, mapper_y)):
            for t_x in range (tile_size_x):
                for t_y in range (tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter (t_x, u_x,
                                                                                            tile_size_x) * self.build_filter (
                        t_y, v_y, tile_size_y)

        return dct_filter

class CPAM (nn.Module):
    def __init__(self, in_channels, norm_layer):
        super (CPAM, self).__init__ ()
        self.pool1 =  nn.AdaptiveAvgPool2d (1)
        self.pool2 = nn.AdaptiveAvgPool2d (2)
        self.pool3 = nn.AdaptiveAvgPool2d (3)
        self.pool4 = nn.AdaptiveAvgPool2d (6)

        self.conv1 = nn.Sequential (Conv2d (in_channels, in_channels, 1, bias=False),
                                 norm_layer (in_channels),
                                 nn.ReLU (True))
        self.conv2 = nn.Sequential (Conv2d (in_channels, in_channels, 1, bias=False),
                                 norm_layer (in_channels),
                                 nn.ReLU (True))
        self.conv3 = nn.Sequential (Conv2d (in_channels, in_channels, 1, bias=False),
                                 norm_layer (in_channels),
                                 nn.ReLU (True))
        self.conv4 = nn.Sequential (Conv2d (in_channels, in_channels, 1, bias=False),
                                 norm_layer (in_channels),
                                 nn.ReLU (True))

    def forward(self, x):
        b, c, h, w = x.size ()

        feat1 = self.conv1 (self.pool1 (x)).view (b, c, -1)
        feat2 = self.conv2 (self.pool2 (x)).view (b, c, -1)
        feat3 = self.conv3 (self.pool3 (x)).view (b, c, -1)
        feat4 = self.conv4 (self.pool4 (x)).view (b, c, -1)

        return torch.cat ((feat1, feat2, feat3, feat4), 2)

class FFB(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2, expand_ratio=0):
        super(FFB, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])
        self.conv = nn.Sequential(*layers)
        self.conv11 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False)

    def forward(self, x):
        if self.use_shortcut:
            x_11 = self.conv11(x)
            return x_11 + self.conv(x)
        else:
            return self.conv(x)

class MFFM (nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super (MFFM, self).__init__ ()
        c2wh = dict ([(64, 56), (128, 28), (256, 14), (512, 7)])
        reduction = 16
        self.cab = MultiSpectralAttentionLayer(ch_2 * 4, c2wh[r_2], c2wh[r_2],
                                               reduction=reduction, freq_sel_method = 'top16')
        self.sab = CPAM(ch_1,nn.BatchNorm2d)

        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU (inplace=True)

        self.ffb = FFB(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d (drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        W_g = self.W_g (g)
        W_x = self.W_x (x)
        bp = self.W (W_g * W_x)

        g_in = g
        g = CPAM(g)

        x_in = x
        x = x.mean ((2, 3), keepdim=True)
        x = cab(x)

        f = self.ffb (torch.cat ([g, x, bp], 1))
        if self.drop_rate > 0:
            return self.dropout (f)
        else:
            return f

class BRM (nn.Module):
    def __init__(self, channels,s):
        super(BRM, self).__init__()
        self.ra2_conv1 = Conv2d (s, channels, kernel_size=1)
        self.ra2_conv2 = Conv2d (s, channels, kernel_size=3, padding=1)
        self.ra2_conv3 = Conv2d (s, channels, kernel_size=3, padding=1)
        self.ra2_conv4 = Conv2d (channels, kernel_size=3, padding=1)

    def forward(self, x):
        crop = F.interpolate (x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid (crop)) + 1
        x = x.expand (-1, s,-1, -1).mul (x)
        x = self.conv1 (x)
        x = F.relu (self.conv2 (x))
        x = F.relu (self.conv3 (x))
        ra_feat = self.conv4 (x)
        x = ra_feat + crop
        lateral_map = F.interpolate (x, scale_factor=8,mode='bilinear')

        return lateral_map

class SandglassBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(SandglassBlock, self).__init__()
        self.stride = stride
        mid_channels = in_channels // expansion_factor
        self.identity = stride == 1 and in_channels == out_channels

        self.bottleneck = nn.Sequential(
            Conv3x3BNReLU(in_channels, in_channels, 1, groups=in_channels),
            Conv1x1BN(in_channels, mid_channels),
            Conv1x1BNReLU(mid_channels, out_channels),
            Conv3x3BN(out_channels, out_channels, stride, groups=out_channels),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        if self.identity:
            return out + x
        else:
            return

class FAGM (nn.Module):
    def __init__(self, channels):
        super(FAGM, self).__init__()

        self.conv3_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,1), padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1,3), padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3_1 = nn.ReLU(inplace=True)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.relu1_1 = nn.ReLU (inplace=True)

    def forward(self, x):
        x3_1 = self.conv3_1(x)
        x3_1 = self.relu3_1(x3_1)
        x1_3  = self.conv1_3(x3_1)
        x1_3 = self.relu1_3(x1_3)

        x1_1 = self.conv1_1(x)
        x1_1 = self.relu1_1(x1_1)

        out = x1_3 + x1_1
        return out

class CTH_Net(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(CTH_Net, self).__init__()
        self.cnn = res2net50_26w_4s(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.cnn.layer4 = nn.Identity()
        self.transformer = deit_tiny_patch16_LS(pretrained=pretrained)
        self.bo = SandglassBlock(256, 256, 2, expansion_factor=6)
        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)
        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.up_c = nn.Sequential(
            MFFM(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate/2),
            FAGM(256)
        )
        self.up_c_1_1 = MFFM(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate/2)
        self.up_c_1_2 = nn.Sequential(
            Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True),
            BRM(128,128),
            FAGM(128)
            )
        self.up_c_2_1 = MFFM(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate/2)
        self.up_c_2_2 = nn.Sequential(
            Up(128, 64, 64, attn=True),
            BRM (64, 64),
            FAGM (64)
        )
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, imgs, labels=None):
        x_b = self.transformer(imgs)
        x_b = torch.transpose(x_b, 1, 2)
        x_b = x_b.view(x_b.shape[0], -1, 12, 16)
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)
        x_b_1 = self.drop(x_b_1)
        x_b_2 = self.up2(x_b_1)
        x_b_2 = self.drop(x_b_2)

        x_u = self.cnn.conv1(imgs)
        x_u = self.cnn.bn1(x_u)
        x_u = self.cnn.relu(x_u)
        x_u = self.cnn.maxpool(x_u)

        x_u_2 = self.cnn.layer1(x_u)
        x_u_2 = self.drop(x_u_2)
        x_u_1 = self.cnn.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)

        x_u = self.cnn.layer3(x_u_1)
        x_u = self.drop(x_u)
        x_c = self.up_c(x_u, x_b)
        x_c = self.bo(x_u,x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)
        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)

        m0 = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        m1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        m2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)
        return m0, m1, m2

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

def Conv3x3BNReLU(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

def Conv3x3BN(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels)
        )
