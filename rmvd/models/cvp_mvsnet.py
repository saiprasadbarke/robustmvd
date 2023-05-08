import os.path as osp
import math

import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict


from .blocks.cvp_mvsnet_components import FeaturePyramid, CostRegNet, calDepthHypo, calSweepingDepthHypo, conditionIntrinsics, depth_regression, depth_regression_refine, homo_warping, proj_cost
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


class CVPMVSNet(nn.Module):
    def __init__(self, num_sampling_steps=192):
        super().__init__()
        self.featurePyramid = FeaturePyramid()
        self.cost_reg_refine = CostRegNet()
        self.args = EasyDict({  # parameters are taken from original repository when executing eval.sh script
            'nsrc': None,  # will be set in forward()
            'nscale': 5,
            'mode': 'test',
        })

        self.num_sampling_steps = num_sampling_steps



    def forward(self, images,  poses, intrinsics, keyview_idx, depth_range, **_):
        N = images[0].shape[0]
        depth_range = [torch.Tensor([0.2]), torch.Tensor([100])] if depth_range is None else depth_range
        min_depth, max_depth = depth_range
        min_depth = torch.stack([min_depth] * N)
        max_depth = torch.stack([max_depth] * N)
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        self.args.nsrc = len(images_source)
        images_source = torch.stack(images_source, dim=1)  # N, NV, 3, H, W

        intrinsics_key = select_by_index(intrinsics, keyview_idx)  # N, 3, 3
        intrinsics_source = exclude_index(intrinsics, keyview_idx)  # N, NV, 3, 3
        intrinsics_source = torch.stack(intrinsics_source, dim=1)  # N, NV, 3, 3

        pose_key = select_by_index(poses, keyview_idx)  # N, 4, 4
        poses_source = exclude_index(poses, keyview_idx)
        poses_source = torch.stack(poses_source, dim=1)  # N, NV, 4, 4

        inp = {
            "ref_img": image_key,
            "src_imgs": images_source,
            "ref_in": intrinsics_key,
            "src_in": intrinsics_source,
            "ref_ex": pose_key,
            "src_ex": poses_source,
            "depth_min": min_depth,
            "depth_max": max_depth,
        }
        ####################### End of transformations #######################
        ## Initialize output list for loss
        depth_est_list = []
        output = {}

        ## Feature extraction
        ref_feature_pyramid = self.featurePyramid(image_key,self.args.nscale)

        src_feature_pyramids = []
        for i in range(self.args.nsrc):
            src_feature_pyramids.append(self.featurePyramid(images_source[:,i,:,:,:],self.args.nscale))

        # Pre-conditioning corresponding multi-scale intrinsics for the feature:
        ref_in_multiscales = conditionIntrinsics(intrinsics_key,image_key.shape,[feature.shape for feature in ref_feature_pyramid])
        src_in_multiscales = []
        for i in range(self.args.nsrc):
            src_in_multiscales.append(conditionIntrinsics(intrinsics_source[:,i],image_key.shape, [feature.shape for feature in src_feature_pyramids[i]]))
        src_in_multiscales = torch.stack(src_in_multiscales).permute(1,0,2,3,4)

        ## Estimate initial coarse depth map
        depth_hypos = calSweepingDepthHypo(ref_in_multiscales[:,-1],src_in_multiscales[:,0,-1],pose_key,poses_source,min_depth, max_depth)

        ref_volume = ref_feature_pyramid[-1].unsqueeze(2).repeat(1, 1, len(depth_hypos[0]), 1, 1)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume.pow_(2)
        if self.args.mode == "test":
            del ref_volume

        for src_idx in range(self.args.nsrc):
            # warpped features
            warped_volume = homo_warping(src_feature_pyramids[src_idx][-1], ref_in_multiscales[:,-1], src_in_multiscales[:,src_idx,-1,:,:], pose_key, poses_source[:,src_idx], depth_hypos)


            if self.args.mode == "train":
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            elif self.args.mode == "test":
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
                del warped_volume
            else: 
                print("Wrong!")
                #pdb.set_trace()
        
        # Aggregate multiple feature volumes by variance
        cost_volume = volume_sq_sum.div_(self.args.nsrc+1).sub_(volume_sum.div_(self.args.nsrc+1).pow_(2))
        if self.args.mode == "test":
            del volume_sum
            del volume_sq_sum

        # Regularize cost volume
        cost_reg = self.cost_reg_refine(cost_volume)

        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_hypos)
        depth_est_list.append(depth)
                ## Upsample depth map and refine along feature pyramid
        for level in range(self.args.nscale-2,-1,-1):

            # Upsample
            depth_up = nn.functional.interpolate(depth[None,:],size=None,scale_factor=2,mode='bicubic',align_corners=None)
            depth_up = depth_up.squeeze(0)
            # Generate depth hypothesis
            depth_hypos = calDepthHypo(self.args,depth_up,ref_in_multiscales[:,level,:,:], src_in_multiscales[:,:,level,:,:],pose_key,poses_source,min_depth, max_depth,level)

            cost_volume = proj_cost(self.args,ref_feature_pyramid[level],src_feature_pyramids,level,ref_in_multiscales[:,level,:,:], src_in_multiscales[:,:,level,:,:], pose_key, poses_source[:,:],depth_hypos)

            cost_reg2 = self.cost_reg_refine(cost_volume)
            if self.args.mode == "test":
                del cost_volume
            
            prob_volume = F.softmax(cost_reg2, dim=1)
            if self.args.mode == "test":
                del cost_reg2

            # Depth regression
            depth = depth_regression_refine(prob_volume, depth_hypos)

            depth_est_list.append(depth)

        # Photometric confidence
        with torch.no_grad():
            num_depth = prob_volume.shape[1]
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            prob_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        if self.args.mode == "test":
            del prob_volume

        ## Return
        depth_est_list.reverse() # Reverse the list so that depth_est_list[0] is the largest scale.
        ######################## End forward pass ########################
        #outputs = self.model(**inp)
        pred_depth = depth_est_list[0]  # N, H, W
        pred_depth_confidence = prob_confidence
        pred_depth_uncertainty = 1 - pred_depth_confidence  # N, H, W

        pred_depth = pred_depth.unsqueeze(1)
        pred_depth_uncertainty = pred_depth_uncertainty.unsqueeze(1)

        pred = {"depth": pred_depth, "depth_uncertainty": pred_depth_uncertainty}
        aux = {}

        return pred, aux
    
    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None  # TODO: does it make sense that poses etc are set to None?
    ):
        device = get_torch_model_device(self)

        orig_ht, orig_wd = images[0].shape[-2:]
        ht, wd = int(math.ceil(orig_ht / 64.0) * 64.0), int(
            math.ceil(orig_wd / 64.0) * 64.0
        )
        if (orig_ht != ht) or (orig_wd != wd):
            resized = Resize(size=(ht, wd))(
                {"images": images, "intrinsics": intrinsics}
            )
            images = resized["images"]
            intrinsics = resized["intrinsics"]

        # normalize images
        images = [image / 255.0 for image in images]

        depth_range = [np.array([0.2]), np.array([100])] if depth_range is None else depth_range
        min_depth, max_depth = depth_range

        images, keyview_idx, poses, intrinsics, min_depth, max_depth = to_torch(
            (images, keyview_idx, poses, intrinsics, min_depth, max_depth), device=device)

        # TODO: check min_depth, max_depth dtype with given or default depth range

        sample = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "keyview_idx": keyview_idx,
            "min_depth": min_depth,
            "max_depth": max_depth,
        }
        return sample
    
    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def cvp_mvsnet(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):

    cfg = {
        "num_sampling_steps": 192,
    }

    model = build_model_with_cfg(
        model_cls=CVPMVSNet,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model
