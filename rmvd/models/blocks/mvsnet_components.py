# External imports

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(nn.Module): #TODO: mvsnet_pl does not use the relu in the forward method but the cvp_mvsnet implementation does. Add an optional argument to the init method to configure the use of relu or leaky relu.
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FeatureNet(nn.Module):  #TODO: Modify this code to accept arguments for setting the inchannels, outchannels, kernel size, stride, and padding. This can then be used in cvp_mvsnet implementation. Need an additional optional argument for configuring the use of relu or leaky relu on the convolutions.
    def __init__(self):
        super().__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module): #TODO: Modify this code to accept arguments for setting the inchannels, outchannels, kernel size, stride, and padding. This can then be used in cvp_mvsnet implementation. Need an additional optional argument for configuring the use of relu or leaky relu on the convolutions.
    def __init__(self):
        super().__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)[:, :, :, :75, :]
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x
    # def forward(self, x):
    #     model_ip = x

    #     try:
    #         # Check initial input
    #         assert not torch.isnan(model_ip).any(), "Nan values in initial input"
    #         print(f"Initial input min: {model_ip.min()}, max: {model_ip.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     conv0 = self.conv0(model_ip)

    #     try:
    #         # Check conv0
    #         assert not torch.isnan(conv0).any(), "Nan values in conv0"
    #         print(f"conv0 min: {conv0.min()}, max: {conv0.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     conv1_output = self.conv1(conv0)
    #     conv2 = self.conv2(conv1_output)

    #     try:
    #         # Check conv2
    #         assert not torch.isnan(conv2).any(), "Nan values in conv2"
    #         print(f"conv2 min: {conv2.min()}, max: {conv2.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     conv3_output = self.conv3(conv2)
    #     conv4 = self.conv4(conv3_output)

    #     try:
    #         # Check conv4
    #         assert not torch.isnan(conv4).any(), "Nan values in conv4"
    #         print(f"conv4 min: {conv4.min()}, max: {conv4.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     conv5_output = self.conv5(conv4)
    #     conv6_output = self.conv6(conv5_output)
    #     x = conv6_output

    #     try:
    #         # Check x after conv6
    #         assert not torch.isnan(x).any(), "Nan values in output after conv6"
    #         print(f"output after conv6 min: {x.min()}, max: {x.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     x = conv4 + self.conv7(x)

    #     try:
    #         # Check x after conv4+conv7
    #         assert not torch.isnan(x).any(), "Nan values in output after conv4+conv7"
    #         print(f"output after conv4+conv7 min: {x.min()}, max: {x.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     x = conv2 + self.conv9(x)[:, :, :, :conv2.shape[3], :]

    #     try:
    #         # Check x after conv2+conv9
    #         assert not torch.isnan(x).any(), "Nan values in output after conv2+conv9"
    #         print(f"output after conv2+conv9 min: {x.min()}, max: {x.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     x = conv0 + self.conv11(x)

    #     try:
    #         # Check x after conv0+conv11
    #         assert not torch.isnan(x).any(), "Nan values in output after conv0+conv11"
    #         print(f"output after conv0+conv11 min: {x.min()}, max: {x.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     x = self.prob(x)

    #     try:
    #         # Check final output
    #         assert not torch.isnan(x).any(), "Nan values in final output"
    #         print(f"Final output min: {x.min()}, max: {x.max()}")
    #     except AssertionError as e:
    #         print(str(e))

    #     return x



class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined
