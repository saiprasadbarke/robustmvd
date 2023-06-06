# Standard Imports
from typing import List, Union, Tuple

# Local Imports

# External Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.utils import create_meshgrid


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    mod = nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True,
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )

    return mod


class ReLUAndSigmoid(nn.Module):
    def __init__(self, inplace: bool = False, min: float = 0, max: float = 1):
        super().__init__()
        self.inplace = inplace
        self.min = min
        self.max = max
        self.range = max - min

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input0 = F.relu(input[:, :1, :, :], inplace=self.inplace)
        input1 = (
            torch.sigmoid(input[:, 1:, :, :] * (4 / self.range)) * self.range
        ) + self.min
        return torch.cat([input0, input1], 1)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


###################### Utils for Vis-MVSNet #######################


def soft_argmin(volume, dim, keepdim=False, window=None):
    prob_vol = nn.Softmax(dim=dim)(volume)
    length = volume.size()[dim]
    index = torch.arange(0, length, dtype=prob_vol.dtype, device=prob_vol.device)
    index_shape = [length if i == dim else 1 for i in range(len(volume.size()))]
    index = index.reshape(index_shape)
    out = torch.sum(index * prob_vol, dim=dim, keepdim=True)
    out_sq = out.squeeze(dim) if not keepdim else out
    if window is None:
        return prob_vol, out_sq
    else:
        mask = ((index - out).abs() <= window).to(volume.dtype)
        prob_map = torch.sum(prob_vol * mask, dim=dim, keepdim=keepdim)
        return prob_vol, out_sq, prob_map


def entropy(volume, dim, keepdim=False):
    return torch.sum(-volume * volume.clamp(1e-9, 1.0).log(), dim=dim, keepdim=keepdim)


def groupwise_correlation(v1, v2, groups, dim):
    # assert v1.size() == v2.size()
    size = list(v1.size())
    s1 = size[:dim]
    c = size[dim]
    s2 = size[dim + 1 :]
    assert c % groups == 0
    reshaped_size = s1 + [groups, c // groups] + s2
    v1_reshaped = v1.view(*reshaped_size)
    size = list(v2.size())
    s1 = size[:dim]
    c = size[dim]
    s2 = size[dim + 1 :]
    assert c % groups == 0
    reshaped_size = s1 + [groups, c // groups] + s2
    v2_reshaped = v2.view(*reshaped_size)
    vc = (v1_reshaped * v2_reshaped).sum(dim=dim + 1)
    return vc


class NanError(Exception):
    pass


def get_homographies(
    left_cam, right_cam, depth_num, depth_start, depth_interval, inv=False
):
    #                n244      n244       1          n111/n1hw    n111/n1hw
    with torch.no_grad():
        n, _, sh, sw = depth_start.size()
        n, _, ih, iw = depth_interval.size()
        d = depth_num

        # cameras (K, R, t)
        R_left = left_cam[:, 0, :3, :3]  # n33
        R_right = right_cam[:, 0, :3, :3]  # n33
        t_left = left_cam[:, 0, :3, 3:4]  # n31
        t_right = right_cam[:, 0, :3, 3:4]  # n31
        K_left = left_cam[:, 1, :3, :3]  # n33
        K_right = right_cam[:, 1, :3, :3]  # n33

        # depth nd1111/ndhw11
        if not inv:
            depth = depth_start + depth_interval * torch.arange(
                0, depth_num, dtype=left_cam.dtype, device=left_cam.device
            ).view(1, d, 1, 1)
        else:
            depth_end = depth_start + (depth_num - 1) * depth_interval
            inv_interv = (1 / (depth_start + 1e-9) - 1 / (depth_end + 1e-9)) / (
                depth_num - 1 + 1e-9
            )
            depth = 1 / (
                1 / (depth_end + 1e-9)
                + inv_interv
                * torch.arange(
                    0, depth_num, dtype=left_cam.dtype, device=left_cam.device
                ).view(1, d, 1, 1)
            )

        depth = depth.unsqueeze(-1).unsqueeze(-1)

        # preparation
        K_left_inv = K_left.float().inverse().to(left_cam.dtype)
        R_left_trans = R_left.transpose(-2, -1)
        R_right_trans = R_right.transpose(-2, -1)

        fronto_direction = R_left[:, 2:3, :3]  # n13

        c_left = -R_left_trans @ t_left
        c_right = -R_right_trans @ t_right  # n31
        c_relative = c_right - c_left

        # compute
        temp_vec = (c_relative @ fronto_direction).view(n, 1, 1, 1, 3, 3)  # n11133

        middle_mat0 = torch.eye(3, dtype=left_cam.dtype, device=left_cam.device).view(
            1, 1, 1, 1, 3, 3
        ) - temp_vec / (
            depth + 1e-9
        )  # ndhw33
        middle_mat1 = (R_left_trans @ K_left_inv).view(n, 1, 1, 1, 3, 3)  # n11133
        middle_mat2 = middle_mat0 @ middle_mat1  # ndhw33

        homographies = (
            K_right.view(n, 1, 1, 1, 3, 3)
            @ R_right.view(n, 1, 1, 1, 3, 3)
            @ middle_mat2
        )  # ndhw33

    if (homographies != homographies).any():
        raise NanError

    return homographies


# Used in GNRefine class forward function and homography_warping function below
def get_pixel_grids(height, width):
    x_coord = (torch.arange(width, dtype=torch.float32).cuda() + 0.5).repeat(height, 1)
    y_coord = (
        (torch.arange(height, dtype=torch.float32).cuda() + 0.5).repeat(width, 1).t()
    )
    ones = torch.ones_like(x_coord)
    indices_grid = torch.stack([x_coord, y_coord, ones], dim=-1).unsqueeze(-1)  # hw31
    return indices_grid


# Used in GNRefine class forward function and homography_warping function below
def interpolate(image, coord):  # nchw, nhw2 => nchw
    with torch.no_grad():
        warped_coord = coord.clone()
        warped_coord[..., 0] /= warped_coord.size()[2]
        warped_coord[..., 1] /= warped_coord.size()[1]
        warped_coord = (warped_coord * 2 - 1).clamp(-1.1, 1.1)
    warped = F.grid_sample(
        image, warped_coord, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    if (warped != warped).any():
        raise NanError
    return warped


# Used in SingleStage class in the build_cost_volume function
def homography_warping(
    input, H
):  # nchw, n33/nhw33 -> nchw   #TODO: How to unite all homography warping fucntions. This one is used in SingleStage class in the build_cost_volume function
    if len(H.size()) == 3:
        H = H.view(-1, 1, 1, 3, 3)
    with torch.no_grad():
        pixel_grids = get_pixel_grids(*input.size()[-2:]).unsqueeze(0)  # 1hw31
        warped_homo_coord = (H @ pixel_grids).squeeze(-1)  # nhw3
        warped_coord = warped_homo_coord[..., :2] / (
            warped_homo_coord[..., 2:3] + 1e-9
        )  # nhw2
    warped = interpolate(input, warped_coord)
    return warped  # nchw


def scale_camera(cam: Union[np.ndarray, torch.Tensor], scale: Union[Tuple, float] = 1):
    """resize input in order to produce sampled depth map"""
    if type(scale) != tuple:
        scale = (scale, scale)
    if type(cam) == np.ndarray:
        new_cam = np.copy(cam)
        # focal:
        new_cam[1, 0, 0] = cam[1, 0, 0] * scale[0]
        new_cam[1, 1, 1] = cam[1, 1, 1] * scale[1]
        # principle point:
        new_cam[1, 0, 2] = cam[1, 0, 2] * scale[0]
        new_cam[1, 1, 2] = cam[1, 1, 2] * scale[1]
    elif type(cam) == torch.Tensor:
        new_cam = cam.clone()
        # focal:
        new_cam[..., 1, 0, 0] = cam[..., 1, 0, 0] * scale[0]
        new_cam[..., 1, 1, 1] = cam[..., 1, 1, 1] * scale[1]
        # principle point:
        new_cam[..., 1, 0, 2] = cam[..., 1, 0, 2] * scale[0]
        new_cam[..., 1, 1, 2] = cam[..., 1, 1, 2] * scale[1]
    # elif type(cam) == tf.Tensor:
    #     scale_tensor = np.ones((1, 2, 4, 4))
    #     scale_tensor[0, 1, 0, 0] = scale[0]
    #     scale_tensor[0, 1, 1, 1] = scale[1]
    #     scale_tensor[0, 1, 0, 2] = scale[0]
    #     scale_tensor[0, 1, 1, 2] = scale[1]
    #     new_cam = cam * scale_tensor
    else:
        raise TypeError
    return new_cam


#################### Utils for mvsnet ####################
def homo_warp(
    src_feat, src_proj, ref_proj_inv, depth_values
):  # TODO: How to unite all homography warping fucntions.
    # src_feat: (B, C, H, W)
    # src_proj: (B, 4, 4)
    # ref_proj_inv: (B, 4, 4)
    # depth_values: (B, D)
    # out: (B, C, D, H, W)
    """This is a Python function that performs a homography warp on a given source feature map (src_feat) using the provided source and reference camera projection matrices (src_proj and ref_proj_inv), and a set of depth values (depth_values). The function returns a warped source feature map."""
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device
    dtype = src_feat.dtype
    # Compute the transformation matrix by multiplying the source and inverse reference camera projection matrices.
    transform = src_proj @ ref_proj_inv
    # Extract the rotation (R) and translation (T) matrices from the transformation matrix.
    R = transform[:, :3, :3]  # (B, 3, 3)
    T = transform[:, :3, 3:]  # (B, 3, 1)
    # create grid from the ref frame
    # Create a grid of points in the reference frame using create_meshgrid function.
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False)  # (1, H, W, 2)
    ref_grid = ref_grid.to(device).to(dtype)
    # Reshape and expand the reference grid to match the input batch size.
    ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
    ref_grid = ref_grid.reshape(1, 2, H * W)  # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
    # Compute the reference grid points in 3D by multiplying with the depth values.
    ref_grid_d = ref_grid.unsqueeze(2) * depth_values.view(B, 1, D, 1)  # (B, 3, D, H*W)
    ref_grid_d = ref_grid_d.view(B, 3, D * H * W)
    # Transform the 3D reference grid points to the source frame using the rotation and translation matrices.
    src_grid_d = R @ ref_grid_d + T  # (B, 3, D*H*W)
    del ref_grid_d, ref_grid, transform, R, T  # release (GPU) memory
    src_grid = src_grid_d[:, :2] / src_grid_d[:, -1:]  # divide by depth (B, 2, D*H*W)
    del src_grid_d
    # Normalize the 2D source grid points to the range of -1 to 1.
    src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
    src_grid = src_grid.view(B, D, H * W, 2)

    # Perform a bilinear grid sampling on the source feature map using the transformed source grid points.
    warped_src_feat = F.grid_sample(
        src_feat, src_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # (B, C, D, H*W)
    # Reshape the result to match the input dimensions.
    warped_src_feat = warped_src_feat.view(B, C, D, H, W)

    return warped_src_feat


def depth_regression(
    p, depth_values
):  # This is reused inthe implementation for cvp_mvsnet
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # print(f"Min p: {p.min()}, Max p: {p.max()}")
    # print(f"Min prob sum: {p.sum(dim=1).min()}, Max prob sum: {p.sum(dim=1).max()}")
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    # print(f"Min depth: {depth.min()}, Max depth: {depth.max()}")
    return depth
