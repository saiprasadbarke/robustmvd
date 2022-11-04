# Standard Library Imports
import os.path as osp

# External Library Imports
import torch
import torch.nn as nn
import numpy as np

# Internal Imports
from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import (
    get_path,
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)
from rmvd.data.transforms import Resize
from wrappers import ModelWrappers


class VisMvsnetWrapped(ModelWrappers):
    def __init__(self, sample_in_inv_depth_space=False, num_sampling_steps=192):
        super().__init__()

        import sys

        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), "paths.toml")
        repo_path = get_path(paths_file, "mvsnet_pl", "root")
        sys.path.insert(0, repo_path)

    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None
    ):
        return None

    def forward(self, images, proj_mats, depth_samples, keyview_idx, **_):
        return None

    def output_adapter(self, model_output):
        return None
