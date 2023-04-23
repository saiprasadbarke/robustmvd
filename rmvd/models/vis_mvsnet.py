import math

import torch
import torch.nn as nn
import numpy as np

from .registry import register_model
from .helpers import build_model_with_cfg
from rmvd.utils import (
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)
from rmvd.data.transforms import Resize


class VisMvsnet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, poses, intrinsics, keyview_idx, **_):
        pass

    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None
    ):
        pass

    def output_adapter(self, model_output):
        pass


@register_model
def vis_mvsnet(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    pretrained_weights = "https://lmb.informatik.uni-freiburg.de/people/schroepp/weights/robustmvd_600k.pt"  # TODO: CHnage this
    weights = pretrained_weights if (pretrained and weights is None) else None
    model = build_model_with_cfg(
        model_cls=VisMvsnet, weights=weights, train=train, num_gpus=num_gpus
    )
    return model
