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
        # The conv_fn and bn_fn variables are used to store the appropriate convolution and batch normalization functions based on the specified dimensionality (dim).
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
        """1x1 convolution with stride"""
        return self.conv_fn(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding and stride"""
        return self.conv_fn(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

    def forward(self, x):
        # Storee the input tensor x as the residual, which will be added back to the output after the convolutions and activations.
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If a downsample layer is provided, it is applied to the input x and the result is stored in the residual variable.
        if self.downsample is not None:
            residual = self.downsample(x)

        # The residual is added to the output from the second batch normalization layer.
        out += residual
        out = self.relu(out)

        return out


def _make_layer(inplanes, block, planes, blocks, stride=1, dim=2):
    """The _make_layer function creates a sequence of blocks. It takes the following input arguments:

    inplanes: The number of input channels for the first block.
    block: The class implementing the block, typically BasicBlock or Bottleneck for ResNets.
    planes: The number of output channels for the blocks.
    blocks: The number of blocks (layers) to be created in the sequence.
    stride (optional): The stride value for the first block's convolutional layer, default is 1.
    dim (optional): Dimension of the convolution operation, either 2 for 2D convolutions or 3 for 3D convolutions, default is 2."""
    # The downsample variable is initialized to None. It will be used to store the downsample layer (if needed) to match the output dimensions of the residual path in the residual blocks.
    downsample = None
    # The conv_fn variable stores the appropriate convolution function based on the specified dimensionality (dim). If dim is 2, it uses nn.Conv2d; if dim is 3, it uses nn.Conv3d.
    conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
    # The bn_fn variable stores the appropriate batch normalization function based on the specified dimensionality (dim). If dim is 2, it uses nn.BatchNorm2d; if dim is 3, it uses nn.BatchNorm3d.
    bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    # bn_fn = nn.GroupNorm

    # This condition checks if the stride is not equal to 1 or if the input channel size doesn't match the output channel size. If true, a downsample layer is required. The downsample layer is a convolutional layer with a kernel size of 1, a stride equal to the stride value, and a number of output channels equal to the number of output channels of the block times the block's expansion factor. This is set to 1 in the BasicBlock class attribute above
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
    # Initialize an empty list to store the layers that will be created.
    layers = []
    # Append the first block to the layers list, which may include a downsample layer if needed.
    layers.append(block(inplanes, planes, stride, downsample, dim=dim))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, dim=dim))

    return nn.Sequential(*layers)


class UNet(nn.Module):
    """
    The UNet class defines the architecture for a U-Net model, a type of convolutional neural network (CNN) widely used for image segmentation tasks. This implementation allows for 2D and 3D convolutions based on the input dim value. The class takes the following arguments:

    Arguments:
                inplanes: The number of input channels for the first block.
                enc: The number of encoding blocks in the U-Net.
                dec: The number of decoding blocks in the U-Net.
                initial_scale: The initial scale factor for naming the blocks.
                bottom_filters: A list of filter sizes for the bottom layers of the U-Net.
                filters: A list of filter sizes for the encoding and decoding layers.
                head_filters: A list of filter sizes for the head layers.
                prefix: A string prefix for naming the blocks.
                dim (optional): Dimension of the convolution operation, either 2 for 2D convolutions or 3 for 3D convolutions, default is 2.

    """

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
        # Initialize the appropriate convolution, batch normalization, and deconvolution functions depending on the input dimension 'dim'.
        conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        # bn_fn = nn.GroupNorm
        deconv_fn = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d

        current_scale = initial_scale
        idx = 0
        prev_f = inplanes

        # Create the bottom blocks of the UNet model. These blocks are stored in an OrderedDict and then converted into a ListModule.
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

        # Create the encoding blocks of the UNet model, similar to the bottom blocks. They are also stored in an OrderedDict and then converted into a ListModule.
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

        # Create the decoding blocks of the UNet model, iterating through the filter sizes in reverse order, and adding deconvolution and residual layers as needed. These blocks are stored in a ListModule.
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

        # Create the head layers of the UNet model, adding deconvolution and residual layers as needed. These layers are stored in a ListModule.
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
        # Pass the input tensor through the bottom blocks, updating the input tensor at each step.
        for b in self.bottom_blocks:
            x = b(x)
        enc_out = []
        # Pass the input tensor through the encoding blocks, storing the output of each block in the enc_out list and updating the input tensor.
        for b in self.enc_blocks:
            x = b(x)
            enc_out.append(x)
        # Initialize a list dec_out to store the output of the decoding blocks.
        dec_out = [x]
        # Iterate through the decoding blocks, deconvolving the input tensor, concatenating it with the corresponding encoding block output, and passing the result through the post-concatenation and residual layers if present. Append the output to the dec_out list.
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
        # Pass the input tensor through the head layers, appending the output to the dec_out list.
        for b in self.head_blocks:
            x = b(x)
            dec_out.append(x)
        # Return the final output tensor if multi_scale is 1, or the last 'multi_scale (3 in the case of vis-mvsnet official implementation)' output tensors from dec_out otherwise.
        if multi_scale == 1:
            return x
        else:
            return dec_out[-multi_scale:]
