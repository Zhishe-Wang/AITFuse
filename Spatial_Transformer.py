import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class encoder(nn.Module):
    def __init__(self, embed_dim=256, depth=2, num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x, y):
        for blk in self.blocks:
            x = blk(x, y)
        x = self.norm(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Cross_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class S_DePatch(nn.Module):
    def __init__(self, channel=16, embed_dim=128, patch_size=16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size**2*channel),
        )

    def forward(self, x, ori):
        b, c, h, w = ori
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x

class Cross_Spatial(nn.Module):
    def __init__(self, size=256, embed_dim=256, depth=2, channel=16,
                 num_heads=4, mlp_ratio=2., patch_size=16, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size ** 2 * channel, embed_dim),
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.en = encoder(embed_dim, depth,
                          num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                          drop_path_rate, norm_layer)

        self.depatch = S_DePatch(channel=channel, embed_dim=embed_dim, patch_size=patch_size)

    def forward(self, x , y):

        ori = x.shape
        x = self.embedding(x)
        y = self.embedding(y)

        x = self.pos_drop(x)
        y = self.pos_drop(y)

        x1 = self.en(x, y)
        # y1 = self.en(y , x)

        out = self.depatch(x1, ori)
        return out