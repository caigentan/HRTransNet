import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size = 16, in_channels=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, f"img_size {img_size} should be divided by patch_size {patch_size}"
        self.H,self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 第一二维不变，后面的维度压缩
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x,(H,W)
