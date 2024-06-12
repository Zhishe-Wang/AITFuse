import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Channel_Transformer import Cross_Channel
from Spatial_Transformer import Cross_Spatial
from t2t_vit import Channel,Spatial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
import fusion_strategy
from function import adaptive_instance_normalization
EPSION = 1e-5
import utils

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        x = x.cuda()

        # out = self.conv2d(x)
        # out = self.bn(out)
        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.res = ResidualBlock(out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res(x)
        x = self.conv2(x)
        return x

class Encoder1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.res = ResidualBlock(out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res(x)
        x = self.conv2(x)
        return x

def softmax(a,b):
    out = torch.exp(a) / (torch.exp(a) + torch.exp(b) + EPSION)
    return out


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        kernel_size = 1
        stride = 1

        self.down1 = nn.AvgPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2)

        self.save_feat = utils.save_feat

        self.conv_in1 = ConvLayer(32, 32, kernel_size = 3, stride = 1)
        self.conv_in2 = ConvLayer(32, 32, kernel_size = 3, stride = 1)
        self.conv_in3 = ConvLayer(32, 32, kernel_size = 3, stride = 1)
        self.conv_in4 = ConvLayer(32, 32, kernel_size = 3, stride = 1)
        self.conv_in5 = ConvLayer(64, 64, kernel_size = 1, stride = 1)
        self.conv_in6 = ConvLayer(64, 64, kernel_size = 1, stride = 1)
        self.conv_in7 = ConvLayer(64, 64, kernel_size = 1, stride = 1)

        self.conv_t3 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv_t2 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv_t1 = ConvLayer(128, 1, kernel_size=3, stride=1, is_last=True)

        self.en1 = Encoder(1, 32, kernel_size, stride)
        self.en2 = Encoder(32, 32, kernel_size, stride)
        self.en3 = Encoder(32, 32, kernel_size, stride)
        self.en4 = Encoder(32, 32, kernel_size, stride)

        self.ctrans1 = Channel(embed_dim=128, patch_size=16, channel=32)
        self.ctrans2 = Channel(embed_dim=128, patch_size=16, channel=32)
        self.ctrans3 = Channel(embed_dim=128, patch_size=16, channel=32)
        self.ctrans4 = Channel(embed_dim=128, patch_size=16, channel=32)

        self.strans1 = Spatial(embed_dim=64, patch_size=8, channel=32)
        self.strans2 = Spatial(embed_dim=64, patch_size=8, channel=32)
        self.strans3 = Spatial(embed_dim=64, patch_size=8, channel=32)
        self.strans4 = Spatial(embed_dim=64, patch_size=8, channel=32)

        self.cross_ctrans1 = Cross_Channel(embed_dim=128, patch_size=16, channel=32)
        self.cross_ctrans2 = Cross_Channel(embed_dim=128, patch_size=16, channel=32)
        self.cross_ctrans3 = Cross_Channel(embed_dim=128, patch_size=16, channel=32)
        self.cross_ctrans4 = Cross_Channel(embed_dim=128, patch_size=16, channel=32)

        self.cross_strans1 = Cross_Spatial(embed_dim=64, patch_size=8, channel=32)
        self.cross_strans2 = Cross_Spatial(embed_dim=64, patch_size=8, channel=32)
        self.cross_strans3 = Cross_Spatial(embed_dim=64, patch_size=8, channel=32)
        self.cross_strans4 = Cross_Spatial(embed_dim=64, patch_size=8, channel=32)

    def encoder(self, ir,vi):
        # 256
        ir_level_1_cov1 = self.en1(ir) #
        ir_level_1_cov2 = self.conv_in1(ir_level_1_cov1)
        ir_level_1_token_re = self.strans1(ir_level_1_cov2)


        vi_level_1_cov1 = self.en1(vi)
        vi_level_1_cov2 = self.conv_in1(vi_level_1_cov1)
        vi_level_1_token_re = self.strans1(vi_level_1_cov2)

        ir_level_1_soft_token = softmax(ir_level_1_token_re,vi_level_1_token_re)
        vi_level_1_soft_token = softmax(vi_level_1_token_re,ir_level_1_token_re)

        ir1t = ir_level_1_cov2 * ir_level_1_soft_token
        vi1t = vi_level_1_cov2 * vi_level_1_soft_token

        ir_level_1_channel_re = self.ctrans1(ir1t)
        vi_level_1_channel_re = self.ctrans1(vi1t)

        ir_level_1_soft_channel = softmax(ir_level_1_channel_re,vi_level_1_channel_re)
        vi_level_1_soft_channel = softmax(vi_level_1_channel_re,ir_level_1_channel_re)

        ir_level_1_out = ir_level_1_cov2 * ir_level_1_soft_channel
        vi_level_1_out = vi_level_1_cov2 * vi_level_1_soft_channel
        # self.save_feat(1, 32, ir_level_1_out, vi_level_1_out, './save_feat')

        # 128
        ir_level_2_cov1 = self.en2(self.down1(ir_level_1_out))
        ir_level_2_cov2 = self.conv_in2(ir_level_2_cov1)
        ir_level_2_token_re = self.strans2(ir_level_2_cov2)

        vi_level_2_cov1 = self.en2(self.down1(vi_level_1_out))
        vi_level_2_cov2 = self.conv_in2(vi_level_2_cov1)
        vi_level_2_token_re = self.strans2(vi_level_2_cov2)

        ir_level_2_soft_token = softmax(ir_level_2_token_re,vi_level_2_token_re)
        vi_level_2_soft_token = softmax(vi_level_2_token_re,ir_level_2_token_re)

        ir2t = ir_level_2_cov2 * ir_level_2_soft_token
        vi2t = vi_level_2_cov2 * vi_level_2_soft_token
        ir_level_2_channel_re = self.ctrans2(ir2t)
        vi_level_2_channel_re = self.ctrans2(vi2t)

        ir_level_2_soft_channel = softmax(ir_level_2_channel_re,vi_level_2_channel_re)
        vi_level_2_soft_channel = softmax(vi_level_2_channel_re,ir_level_2_channel_re)

        ir_level_2_out = ir_level_2_cov2 * ir_level_2_soft_channel
        vi_level_2_out = vi_level_2_cov2 * vi_level_2_soft_channel

        # 64
        ir_level_3_cov1 = self.en3(self.down1(ir_level_2_out))
        ir_level_3_cov2 = self.conv_in3(ir_level_3_cov1)
        ir_level_3_token_re = self.strans3(ir_level_3_cov2)

        vi_level_3_cov1 = self.en3(self.down1(vi_level_2_out))
        vi_level_3_cov2 = self.conv_in3(vi_level_3_cov1)
        vi_level_3_token_re = self.strans3(vi_level_3_cov2)

        ir_level_3_soft_token = softmax(ir_level_3_token_re,vi_level_3_token_re)
        vi_level_3_soft_token = softmax(vi_level_3_token_re,ir_level_3_token_re)

        ir3t = ir_level_3_cov2 * ir_level_3_soft_token
        vi3t = vi_level_3_cov2 * vi_level_3_soft_token

        ir_level_3_channel_re = self.ctrans3(ir3t)
        vi_level_3_channel_re = self.ctrans3(vi3t)

        ir_level_3_soft_channel = softmax(ir_level_3_channel_re,vi_level_3_channel_re)
        vi_level_3_soft_channel = softmax(vi_level_3_channel_re,ir_level_3_channel_re)

        ir_level_3_out = ir_level_3_cov2 * ir_level_3_soft_channel
        vi_level_3_out = vi_level_3_cov2 * vi_level_3_soft_channel

        # 32
        ir_level_4_cov1 = self.en4(self.down1(ir_level_3_out))
        ir_level_4_cov2 = self.conv_in4(ir_level_4_cov1)
        ir_level_4_token_re = self.strans4(ir_level_4_cov2)

        vi_level_4_cov1 = self.en4(self.down1(vi_level_3_out))
        vi_level_4_cov2 = self.conv_in4(vi_level_4_cov1)
        vi_level_4_token_re = self.strans4(vi_level_4_cov2)

        ir_level_4_soft_token = softmax(ir_level_4_token_re,vi_level_4_token_re)
        vi_level_4_soft_token = softmax(vi_level_4_token_re,ir_level_4_token_re)

        ir4t = ir_level_4_cov2 * ir_level_4_soft_token
        vi4t = vi_level_4_cov2 * vi_level_4_soft_token


        ir_level_4_channel_re = self.ctrans4(ir4t)
        vi_level_4_channel_re = self.ctrans4(vi4t)

        ir_level_4_soft_channel = softmax(ir_level_4_channel_re,vi_level_4_channel_re)
        vi_level_4_soft_channel = softmax(vi_level_4_channel_re,ir_level_4_channel_re)

        ir_level_4_out = ir_level_4_cov2 * ir_level_4_soft_channel
        vi_level_4_out = vi_level_4_cov2 * vi_level_4_soft_channel

        return ir_level_1_out, vi_level_1_out, ir_level_2_out, vi_level_2_out,\
               ir_level_3_out, vi_level_3_out, ir_level_4_out, vi_level_4_out


    def fusion1(self, ir_feature, vi_feature):

        ir_level_1_c1 = self.cross_strans1(ir_feature,vi_feature)
        vi_level_1_c1 = self.cross_strans1(vi_feature,ir_feature)

        ir_level_1_s1 = self.cross_ctrans1(ir_level_1_c1,vi_level_1_c1)
        vi_level_1_s1 = self.cross_ctrans1(vi_level_1_c1,ir_level_1_c1)

        ir_level_1_out = ir_feature + ir_level_1_s1
        vi_level_1_out = vi_feature + vi_level_1_s1

        out = torch.cat([ir_level_1_out,vi_level_1_out],1)
        return out


    def fusion2(self, ir_feature, vi_feature):

        ir_level_2_c2 = self.cross_strans2(ir_feature,vi_feature)
        vi_level_2_c2 = self.cross_strans2(vi_feature,ir_feature)

        ir_level_2_s2 = self.cross_ctrans2(ir_level_2_c2,vi_level_2_c2)
        vi_level_2_s2 = self.cross_ctrans2(vi_level_2_c2,ir_level_2_c2)

        ir_level_2_out = ir_feature + ir_level_2_s2
        vi_level_2_out = vi_feature + vi_level_2_s2

        out = torch.cat([ir_level_2_out,vi_level_2_out],1)
        return out


    def fusion3(self, ir_feature, vi_feature):

        ir_level_3_c3 = self.cross_strans3(ir_feature,vi_feature)
        vi_level_3_c3 = self.cross_strans3(vi_feature,ir_feature)

        ir_level_3_s3 = self.cross_ctrans3(ir_level_3_c3,vi_level_3_c3)
        vi_level_3_s3 = self.cross_ctrans3(vi_level_3_c3,ir_level_3_c3)

        ir_level_3_out = ir_feature + ir_level_3_s3
        vi_level_3_out = vi_feature + vi_level_3_s3

        out = torch.cat([ir_level_3_out,vi_level_3_out],1)
        return out


    def fusion4(self, ir_feature, vi_feature):

        ir_level_4_c4 = self.cross_strans4(ir_feature,vi_feature)
        vi_level_4_c4 = self.cross_strans4(vi_feature,ir_feature)

        ir_level_4_s4 = self.cross_ctrans4(ir_level_4_c4,vi_level_4_c4)
        vi_level_4_s4 = self.cross_ctrans4(vi_level_4_c4,ir_level_4_c4)

        ir_level_4_out = ir_feature + ir_level_4_s4
        vi_level_4_out = vi_feature + vi_level_4_s4

        out = torch.cat([ir_level_4_out,vi_level_4_out],1)
        return out

    def decoder(self,f1, f2, f3, f4):

        f_3 = self.conv_t3(torch.cat([self.conv_in5(self.up1(f4)),f3],1))
        f_2 = self.conv_t2(torch.cat([self.conv_in6(self.up1(f_3)),f2],1))
        out = self.conv_t1(torch.cat([self.conv_in7(self.up1(f_2)),f1],1))

        return out

