# Standard library imports


# Local imports
from .vis_mvsnet_unet_modular import UNet

# External library imports
import torch
import torch.nn as nn


class FeatExt(nn.Module):
    """
    This FeatExt (Feature Extraction) class is a neural network module that combines an initial 2D convolution layer, a 2D UNet, and three final 2D convolution layers to extract features from the input data. The three convolution layers are used to process the output features from the UNet at three different scales. Refer to the 'multi_scale' attribute of the forward function in the UNet class. These extracted features are used later on to construct the cost volumes at three different resolutions. The FeatExt class is used in the VisMVSnetModel class.
    """

    def __init__(self):
        super(FeatExt, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2, bias=False), nn.BatchNorm2d(16), nn.ReLU()
        )
        self.unet = UNet(16, 2, 1, 2, [], [32, 64, 128], [], "2d", 2)
        self.final_conv_1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.init_conv(x)
        out1, out2, out3 = self.unet(out, multi_scale=3)
        return self.final_conv_1(out1), self.final_conv_2(out2), self.final_conv_3(out3)
