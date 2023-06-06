import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np

from rmvd.models.blocks.vis_mvsnet_feature_extractor import FeatExt
from rmvd.models.blocks.vis_mvsnet_singlestage import SingleStage
from rmvd.models.wrappers.wrappers import ModelWrappers

from .registry import register_model
from .helpers import build_model_with_cfg
from rmvd.utils import (
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)
from rmvd.data.transforms import ResizeInputs


class VisMvsnet(ModelWrappers):
    def __init__(self, num_sampling_steps=192):
        super().__init__()
        self.feat_ext = FeatExt()
        self.stage1 = SingleStage()
        self.stage2 = SingleStage()
        self.stage3 = SingleStage()
        self.input_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.num_sampling_steps = num_sampling_steps

    def forward(
        self, images, poses, intrinsics, keyview_idx, depth_range, **_
    ):  # this is fixed
        # Extract the minimum and maximum depth from the depth_range input.
        N = images[0].shape[0]  # batch size
        min_depth, max_depth = depth_range
        # Calculate the step size for depth sampling based on the min_depth, max_depth, and the number of sampling steps.
        step_size = (max_depth - min_depth) / self.num_sampling_steps
        # Initialize an empty list named cams for storing camera matrices.
        cams = []
        # for each intrinsic and pose in the input, create a camera matrix
        for intrinsic, pose in zip(intrinsics, poses):
            cam = torch.zeros((N, 2, 4, 4), dtype=torch.float32, device=pose.device)
            cam[:, 0, :, :] = pose
            cam[:, 1, :3, :3] = intrinsic
            cam[:, 1, 3, 0] = min_depth
            cam[:, 1, 3, 1] = step_size
            cam[:, 1, 3, 2] = self.num_sampling_steps
            cam[:, 1, 3, 3] = max_depth

            # cam = cam[np.newaxis, :]  # 1, 2, 4, 4
            # Append it to the cams list.
            cams.append(cam)
        # Convert images, keyview_idx, and cams to PyTorch tensors and move them to the appropriate device.
        images, keyview_idx, cams = to_torch(
            (images, keyview_idx, cams),
            device=get_torch_model_device(self),
        )
        # Extract the reference image and source images based on the keyview_idx.
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        # Extract the reference camera matrix and source camera matrices based on the keyview_idx.
        cam_key = select_by_index(cams, keyview_idx)
        cam_source = exclude_index(cams, keyview_idx)
        # Stack the source images and source camera matrices along a new dimension.
        images_source = torch.stack(images_source, 1)  # N, num_views, 3, H, W
        cam_source = torch.stack(cam_source, 1)  # N, num_views, 4, 4
        sample = {
            "ref": image_key,
            "ref_cam": cam_key,
            "srcs": images_source,
            "srcs_cam": cam_source,
        }
        # Set up the initial parameters for the multi-scale depth estimation, including the number of depth levels at each scale and the scale factors for depth intervals.
        depth_nums = [64, 32, 16]
        interval_scales = [4.0, 2.0, 1.0]
        mode = "soft"
        mem = False
        upsample = False
        # Extract the reference image, reference camera matrix, source images, and source camera matrices from the sample dictionary.
        ref, ref_cam, srcs, srcs_cam = [
            sample[attr] for attr in ["ref", "ref_cam", "srcs", "srcs_cam"]
        ]
        # Compute the starting depth and depth interval from the reference camera matrix.
        depth_start = ref_cam[:, 1:2, 3:4, 0:1]  # n111
        depth_interval = ref_cam[:, 1:2, 3:4, 1:2]  # n111
        srcs_cam = [srcs_cam[:, i, ...] for i in range(srcs_cam.size()[1])]

        n, v, c, h, w = srcs.size()
        img_pack = torch.cat([ref, srcs.transpose(0, 1).reshape(v * n, c, h, w)])
        feat_pack_1, feat_pack_2, feat_pack_3 = self.feat_ext(img_pack)

        ref_feat_1, *srcs_feat_1 = [
            feat_pack_1[i * n : (i + 1) * n] for i in range(v + 1)
        ]
        est_depth_1, prob_map_1, pair_results_1 = self.stage1(
            [ref_feat_1, ref_cam, srcs_feat_1, srcs_cam],
            depth_num=depth_nums[0],
            upsample=False,
            mem=mem,
            mode=mode,
            depth_start_override=None,
            depth_interval_override=depth_interval * interval_scales[0],
            s_scale=8,
        )
        prob_map_1_up = F.interpolate(
            prob_map_1, scale_factor=4, mode="bilinear", align_corners=False
        )

        ref_feat_2, *srcs_feat_2 = [
            feat_pack_2[i * n : (i + 1) * n] for i in range(v + 1)
        ]
        depth_start_2 = (
            F.interpolate(
                est_depth_1.detach(),
                size=(ref_feat_2.size()[2], ref_feat_2.size()[3]),
                mode="bilinear",
                align_corners=False,
            )
            - depth_nums[1] * depth_interval * interval_scales[1] / 2
        )
        est_depth_2, prob_map_2, pair_results_2 = self.stage2(
            [ref_feat_2, ref_cam, srcs_feat_2, srcs_cam],
            depth_num=depth_nums[1],
            upsample=False,
            mem=mem,
            mode=mode,
            depth_start_override=depth_start_2,
            depth_interval_override=depth_interval * interval_scales[1],
            s_scale=4,
        )
        prob_map_2_up = F.interpolate(
            prob_map_2, scale_factor=2, mode="bilinear", align_corners=False
        )

        ref_feat_3, *srcs_feat_3 = [
            feat_pack_3[i * n : (i + 1) * n] for i in range(v + 1)
        ]
        depth_start_3 = (
            F.interpolate(
                est_depth_2.detach(),
                size=(ref_feat_3.size()[2], ref_feat_3.size()[3]),
                mode="bilinear",
                align_corners=False,
            )
            - depth_nums[2] * depth_interval * interval_scales[2] / 2
        )
        est_depth_3, prob_map_3, pair_results_3 = self.stage3(
            [ref_feat_3, ref_cam, srcs_feat_3, srcs_cam],
            depth_num=depth_nums[2],
            upsample=upsample,
            mem=mem,
            mode=mode,
            depth_start_override=depth_start_3,
            depth_interval_override=depth_interval * interval_scales[2],
            s_scale=2,
        )

        # refined_depth = self.refine(est_depth_3, ref_feat_3, ref_cam, srcs_feat_3, srcs_cam, 2)
        refined_depth = est_depth_3

        outputs, pred_depth, prob_maps = (
            [
                [est_depth_1, pair_results_1],
                [est_depth_2, pair_results_2],
                [est_depth_3, pair_results_3],
            ],
            refined_depth,
            [prob_map_1_up, prob_map_2_up, prob_map_3],
        )
        pred_depth_confidence = prob_maps[2]
        pred_depth_uncertainty = 1 - pred_depth_confidence

        pred = {"depth": pred_depth, "depth_uncertainty": pred_depth_uncertainty}
        aux = {"outputs": outputs, "prob_maps": prob_maps, "ref_cam": ref_cam}

        return pred, aux

    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None
    ):
        device = get_torch_model_device(self)

        orig_ht, orig_wd = images[0].shape[-2:]
        ht, wd = int(math.ceil(orig_ht / 64.0) * 64.0), int(
            math.ceil(orig_wd / 64.0) * 64.0
        )
        if (orig_ht != ht) or (orig_wd != wd):
            resized = ResizeInputs(size=(ht, wd))(
                {"images": images, "intrinsics": intrinsics}
            )
            images = resized["images"]
            intrinsics = resized["intrinsics"]

        for idx, image_batch in enumerate(images):
            tmp_images = []
            image_batch = image_batch.transpose(0, 2, 3, 1)
            for image in image_batch:
                image = self.input_transform(image.astype(np.uint8)).float()
                image = torch.flip(image, [0])  # RGB to BGR
                tmp_images.append(image)

            image_batch = torch.stack(tmp_images)
            images[idx] = image_batch
        depth_range = [0.2, 100] if depth_range is None else depth_range
        images, keyview_idx, poses, intrinsics, depth_range = to_torch(
            (images, keyview_idx, poses, intrinsics, depth_range), device=device
        )
        sample = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "keyview_idx": keyview_idx,
            "depth_range": depth_range,
        }
        return sample

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model
def vis_mvsnet(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    pretrained_weights = "https://raw.githubusercontent.com/jzhangbs/Vis-MVSNet/master/pretrained_model/vis/20000.tar"
    weights = pretrained_weights if (pretrained and weights is None) else None
    cfg = {
        "num_sampling_steps": 192,
    }
    model = build_model_with_cfg(
        model_cls=VisMvsnet, cfg=cfg, weights=weights, train=train, num_gpus=num_gpus
    )
    return model
