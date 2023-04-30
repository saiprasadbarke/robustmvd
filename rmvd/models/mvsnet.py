# Local imports
from .blocks.mvsnet_components import FeatureNet, CostRegNet
from .blocks.mvsnet_convbnrelu import homo_warp, depth_regression
from .registry import register_model
from .helpers import build_model_with_cfg
from rmvd.utils import (
    get_path,
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)
from rmvd.data.transforms import Resize

# External Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class MVSNet(nn.Module):
    def __init__(self):
        super(MVSNet, self).__init__()
        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()

    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None
    ):
        pass

    def forward(self, images, poses, intrinsics, keyview_idx, depth_range, **_):
        pass

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)

    def forward(self, imgs, proj_mats, depth_values):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 4, 4)
        # depth_values: (B, D)
        B, V, _, H, W = imgs.shape
        D = depth_values.shape[1]

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        imgs = imgs.reshape(B * V, 3, H, W)
        feats = self.feature(imgs)  # (B*V, F, h, w)
        del imgs
        feats = feats.reshape(B, V, *feats.shape[1:])  # (B, V, F, h, w)
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        ref_proj, src_projs = proj_mats[:, 0], proj_mats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, F, h, w)
        src_projs = src_projs.permute(1, 0, 2, 3)  # (V-1, B, 4, 4)

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, F, D, h, w)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume**2
        del ref_volume

        for src_feat, src_proj in zip(src_feats, src_projs):
            warped_volume = homo_warp(src_feat, src_proj, ref_proj, depth_values)
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume**2
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
        del volume_sq_sum, volume_sum

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance).squeeze(1)
        prob_volume = F.softmax(cost_reg, 1)  # (B, D, h, w)
        depth = depth_regression(prob_volume, depth_values)

        with torch.no_grad():
            # sum probability of 4 consecutive depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(
                F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                (4, 1, 1),
                stride=1,
            ).squeeze(
                1
            )  # (B, D, h, w)
            # find the (rounded) index that is the final prediction
            depth_index = depth_regression(
                prob_volume,
                torch.arange(D, device=prob_volume.device, dtype=prob_volume.dtype),
            ).long()  # (B, h, w)
            # the confidence is the 4-sum probability at this index
            confidence = torch.gather(
                prob_volume_sum4, 1, depth_index.unsqueeze(1)
            ).squeeze(
                1
            )  # (B, h, w)

        return depth, confidence


@register_model(trainable=False)
def mvsnet(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."
    cfg = {"sample_in_inv_depth_space": False, "num_sampling_steps": 192}
    model = build_model_with_cfg(
        model_cls=MVSNet,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model
