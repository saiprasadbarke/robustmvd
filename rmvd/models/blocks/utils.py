# Standard Imports
from typing import List

# Local Imports

# External Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super(ReLUAndSigmoid, self).__init__()
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


def bin_op_reduce(lst: List, func):
    result = lst[0]
    for i in range(1, len(lst)):
        result = func(result, lst[i])
    return result


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
def homography_warping(input, H):  # nchw, n33/nhw33 -> nchw
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
