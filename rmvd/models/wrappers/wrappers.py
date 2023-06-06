# Standard library imports
from abc import ABCMeta, abstractmethod

# Third party imports
import torch.nn as nn


class ModelWrappers(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None
    ):
        pass

    @abstractmethod
    def forward(self, images, proj_mats, depth_samples, keyview_idx, **_):
        pass

    @abstractmethod
    def output_adapter(self, model_output):
        pass
