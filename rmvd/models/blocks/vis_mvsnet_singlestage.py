# Standard imports

# Local imports
from .vis_mvsnet_unet_modular import UNet
from .list_module import ListModule
from .utils import (
    scale_camera,
    get_homographies,
    homography_warping,
    groupwise_correlation,
    soft_argmin,
    entropy,
)

# External library imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class Reg(nn.Module):  # Used in class singlestage
    def __init__(self):
        super().__init__()
        self.init_conv = lambda x: x
        self.unet = UNet(8, 1, 0, 4, [], [8, 16], [], "reg1", dim=3)

    def forward(self, x):
        init = self.init_conv(x)
        out = self.unet(init)
        return out


class RegPair(nn.Module):  # Used in class singlestage
    def __init__(self):
        super().__init__()
        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.final_conv(x)
        return out


class RegFuse(nn.Module):  # Used in class singlestage
    def __init__(self):
        super().__init__()
        self.init_conv = lambda x: x
        self.unet = UNet(8, 1, 0, 4, [], [8, 16], [], "reg2", dim=3)
        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        init = self.init_conv(x)
        out = self.unet(init)
        out = self.final_conv(out)
        return out


class UncertNet(nn.Module):  # Used in class singlestage
    def __init__(self, num_heads=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU()
        )
        self.head_convs = ListModule(
            [nn.Conv2d(8, 1, 3, 1, 1, bias=False) for _ in range(num_heads)]
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        outs = [conv(out) for conv in self.head_convs]
        return outs


class SingleStage(nn.Module):  # Used in the class Model
    def __init__(self):
        super().__init__()
        self.reg = Reg()
        self.reg_fuse = RegFuse()
        self.reg_pair = RegPair()  # MVS
        self.uncert_net = UncertNet(2)  # MVS

    def build_cost_volume(
        self,
        ref,
        ref_cam,
        src,
        src_cam,
        depth_num,
        depth_start,
        depth_interval,
        s_scale,
        d_scale,
    ):
        ref_cam_scaled, src_cam_scaled = [
            scale_camera(cam, 1 / s_scale) for cam in [ref_cam, src_cam]
        ]
        Hs = get_homographies(
            ref_cam_scaled,
            src_cam_scaled,
            depth_num // d_scale,
            depth_start,
            depth_interval * d_scale,
        )
        # ndhw33
        src_nd_c_h_w = (
            src.unsqueeze(1)
            .repeat(1, depth_num // d_scale, 1, 1, 1)
            .view(-1, *src.size()[1:])
        )  # n*d chw
        warped_src_nd_c_h_w = homography_warping(
            src_nd_c_h_w, Hs.view(-1, *Hs.size()[2:])
        )  # n*d chw
        warped_src = warped_src_nd_c_h_w.view(
            -1, depth_num // d_scale, *src.size()[1:]
        ).transpose(
            1, 2
        )  # ncdhw
        return warped_src

    def build_cost_maps(
        self,
        ref,
        ref_cam,
        source,
        source_cam,
        depth_num,
        depth_start,
        depth_interval,
        scale,
    ):
        ref_cam_scaled, source_cam_scaled = [
            scale_camera(cam, 1 / scale) for cam in [ref_cam, source_cam]
        ]
        Hs = get_homographies(
            ref_cam_scaled, source_cam_scaled, depth_num, depth_start, depth_interval
        )

        cost_maps = []
        for d in range(depth_num):
            H = Hs[:, d, ...]
            warped_source = homography_warping(source, H)
            cost_maps.append(torch.cat([ref, warped_source], dim=1))
        return cost_maps

    def forward(
        self,
        sample,
        depth_num,
        upsample=False,
        mem=False,
        mode="soft",
        depth_start_override=None,
        depth_interval_override=None,
        s_scale=1,
    ):
        if mem:
            raise NotImplementedError

        ref_feat, ref_cam, srcs_feat, srcs_cam = sample
        depth_start = (
            ref_cam[:, 1:2, 3:4, 0:1]
            if depth_start_override is None
            else depth_start_override
        )  # n111 or n1hw
        depth_interval = (
            ref_cam[:, 1:2, 3:4, 1:2]
            if depth_interval_override is None
            else depth_interval_override
        )  # n111

        upsample_scale = 1
        d_scale = 1
        interm_scale = 1

        ref_ncdhw = ref_feat.unsqueeze(2).repeat(1, 1, depth_num // d_scale, 1, 1)
        pair_results = []  # MVS

        if mode == "soft":
            weight_sum = (
                torch.zeros(
                    (
                        ref_ncdhw.size()[0],
                        1,
                        1,
                        ref_ncdhw.size()[3] // interm_scale,
                        ref_ncdhw.size()[4] // interm_scale,
                    )
                )
                .to(ref_ncdhw.dtype)
                .cuda()
            )
        if mode == "hard":
            weight_sum = (
                torch.zeros(
                    (
                        ref_ncdhw.size()[0],
                        1,
                        1,
                        ref_ncdhw.size()[3] // interm_scale,
                        ref_ncdhw.size()[4] // interm_scale,
                    )
                )
                .to(ref_ncdhw.dtype)
                .cuda()
            )
        if mode == "average":
            pass
        if mode == "uwta":
            min_weight = None
        if mode == "maxpool":
            init = True
        fused_interm = (
            torch.zeros(
                (
                    ref_ncdhw.size()[0],
                    8,
                    ref_ncdhw.size()[2] // interm_scale,
                    ref_ncdhw.size()[3] // interm_scale,
                    ref_ncdhw.size()[4] // interm_scale,
                )
            )
            .to(ref_ncdhw.dtype)
            .cuda()
        )

        for src_feat, src_cam in zip(srcs_feat, srcs_cam):
            warped_src = self.build_cost_volume(
                ref_feat,
                ref_cam,
                src_feat,
                src_cam,
                depth_num,
                depth_start,
                depth_interval,
                s_scale,
                d_scale,
            )
            cost_volume = groupwise_correlation(ref_ncdhw, warped_src, 8, 1)
            interm = self.reg(cost_volume)
            # if not self.training: del cost_volume
            score_volume = self.reg_pair(interm)  # n1dhw
            if d_scale != 1:
                score_volume = F.interpolate(
                    score_volume,
                    scale_factor=(d_scale, 1, 1),
                    mode="trilinear",
                    align_corners=False,
                )
            score_volume = score_volume.squeeze(1)  # ndhw
            prob_volume, est_depth_class = soft_argmin(
                score_volume, dim=1, keepdim=True
            )
            est_depth = est_depth_class * depth_interval * interm_scale + depth_start
            entropy_ = entropy(prob_volume, dim=1, keepdim=True)
            heads = self.uncert_net(entropy_)
            pair_results.append([est_depth, heads])
            # if not self.training: del score_volume, prob_volume

            if mode == "soft":
                weight = (-heads[0]).exp().unsqueeze(2)
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            if mode == "hard":
                weight = (heads[0] < 0).to(interm.dtype).unsqueeze(2) + 1e-4
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            if mode == "average":
                weight = None
                fused_interm = fused_interm + interm
            if mode == "uwta":
                weight = heads[0].unsqueeze(2)
                if min_weight is None:
                    min_weight = weight
                    mask = torch.ones_like(min_weight).to(interm.dtype).cuda()
                else:
                    mask = (weight < min_weight).to(interm.dtype)
                    min_weight = weight * mask + min_weight * (1 - mask)
                fused_interm = interm * mask + fused_interm * (1 - mask)
            if mode == "maxpool":
                weight = None
                if init:
                    fused_interm = fused_interm + interm
                    init = False
                else:
                    fused_interm = torch.max(fused_interm, interm)

            if not self.training:
                del (
                    weight,
                    prob_volume,
                    est_depth_class,
                    score_volume,
                    interm,
                    cost_volume,
                    warped_src,
                )

        if mode == "soft":
            fused_interm /= weight_sum
        if mode == "hard":
            fused_interm /= weight_sum
        if mode == "average":
            fused_interm /= len(srcs_feat)
        if mode == "uwta":
            pass
        if mode == "maxpool":
            pass

        score_volume = self.reg_fuse(fused_interm)  # n1dhw
        if d_scale != 1:
            score_volume = F.interpolate(
                score_volume,
                scale_factor=(d_scale, 1, 1),
                mode="trilinear",
                align_corners=False,
            )
        score_volume = score_volume.squeeze(1)  # ndhw
        if upsample:
            score_volume = F.interpolate(
                score_volume,
                scale_factor=upsample_scale,
                mode="bilinear",
                align_corners=False,
            )

        prob_volume, est_depth_class, prob_map = soft_argmin(
            score_volume, dim=1, keepdim=True, window=2
        )
        est_depth = est_depth_class * depth_interval + depth_start

        # entropy_ = entropy(prob_volume, dim=1, keepdim=True)
        # uncert = self.uncert_net(entropy_)
        # uncert = torch.cuda.FloatTensor(*est_depth.size()).zero_()

        # if upsample and est_depth.size() != gt.size():
        #     final_size = gt.size()
        #     size = est_depth.size()
        #     # est_depth, uncert = [
        #     #     F.interpolate(img, size=(final_size[2], final_size[3]), mode='bilinear', align_corners=False)
        #     #     for img in [est_depth, uncert]
        #     # ]
        #     est_depth = F.interpolate(est_depth, size=(final_size[2], final_size[3]), mode='bilinear', align_corners=False)

        return est_depth, prob_map, pair_results  # MVS
