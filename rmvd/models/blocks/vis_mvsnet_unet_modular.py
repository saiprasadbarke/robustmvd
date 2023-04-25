# Standard library imports
from typing import List
from collections import OrderedDict

# Local imports
from .list_module import ListModule

# External imports
import numpy as np
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dim=2):
        super(BasicBlock, self).__init__()

        self.conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        self.bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        # self.bn_fn = nn.GroupNorm

        self.conv1 = self.conv3x3(inplanes, planes, stride)
        # nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = self.bn_fn(planes)
        # nn.init.constant_(self.bn1.weight, 1)
        # nn.init.constant_(self.bn1.bias, 0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        # nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = self.bn_fn(planes)
        # nn.init.constant_(self.bn2.weight, 0)
        # nn.init.constant_(self.bn2.bias, 0)
        self.downsample = downsample
        self.stride = stride

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return self.conv_fn(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return self.conv_fn(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _make_layer(inplanes, block, planes, blocks, stride=1, dim=2):
    """The _make_layer function creates a sequence of blocks in a Residual Network (ResNet). It takes the following input arguments:

    inplanes: The number of input channels for the first block.
    block: The class implementing the block, typically BasicBlock or Bottleneck for ResNets.
    planes: The number of output channels for the blocks.
    blocks: The number of blocks (layers) to be created in the sequence.
    stride (optional): The stride value for the first block's convolutional layer, default is 1.
    dim (optional): Dimension of the convolution operation, either 2 for 2D convolutions or 3 for 3D convolutions, default is 2."""
    downsample = None
    conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
    bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    # bn_fn = nn.GroupNorm
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv_fn(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            bn_fn(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, dim=dim))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, dim=dim))

    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(
        self,
        inplanes: int,
        enc: int,
        dec: int,
        initial_scale: int,
        bottom_filters: List[int],
        filters: List[int],
        head_filters: List[int],
        prefix: str,
        dim: int = 2,
    ):
        super(UNet, self).__init__()

        conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        # bn_fn = nn.GroupNorm
        deconv_fn = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        current_scale = initial_scale
        idx = 0
        prev_f = inplanes

        self.bottom_blocks = OrderedDict()
        for f in bottom_filters:
            block = _make_layer(
                prev_f, BasicBlock, f, enc, 1 if idx == 0 else 2, dim=dim
            )
            self.bottom_blocks[f"{prefix}{current_scale}_{idx}"] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.bottom_blocks = ListModule(self.bottom_blocks)

        self.enc_blocks = OrderedDict()
        for f in filters:
            block = _make_layer(
                prev_f, BasicBlock, f, enc, 1 if idx == 0 else 2, dim=dim
            )
            self.enc_blocks[f"{prefix}{current_scale}_{idx}"] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.enc_blocks = ListModule(self.enc_blocks)

        self.dec_blocks = OrderedDict()
        for f in filters[-2::-1]:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False),
                conv_fn(2 * f, f, 3, 1, 1, bias=False),
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            # nn.init.xavier_uniform_(block[0].weight)
            # nn.init.xavier_uniform_(block[1].weight)
            self.dec_blocks[f"{prefix}{current_scale}_{idx}"] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.dec_blocks = ListModule(self.dec_blocks)

        self.head_blocks = OrderedDict()
        for f in head_filters:
            block = [deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False)]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            block = nn.Sequential(*block)
            # nn.init.xavier_uniform_(block[0])
            self.head_blocks[f"{prefix}{current_scale}_{idx}"] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.head_blocks = ListModule(self.head_blocks)

    def forward(self, x, multi_scale=1):
        for b in self.bottom_blocks:
            x = b(x)
        enc_out = []
        for b in self.enc_blocks:
            x = b(x)
            enc_out.append(x)
        dec_out = [x]
        for i, b in enumerate(self.dec_blocks):
            if len(b) == 3:
                deconv, post_concat, res = b
            elif len(b) == 2:
                deconv, post_concat = b
            else:
                raise RuntimeError("Invalid number of blocks in Unet dec_blocks")
            x = deconv(x)
            x = torch.cat([x, enc_out[-2 - i]], 1)
            x = post_concat(x)
            if len(b) == 3:
                x = res(x)
            dec_out.append(x)
        for b in self.head_blocks:
            x = b(x)
            dec_out.append(x)
        if multi_scale == 1:
            return x
        else:
            return dec_out[-multi_scale:]
