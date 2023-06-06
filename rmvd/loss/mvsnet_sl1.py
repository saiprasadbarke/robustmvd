import torch
import torch.nn as nn
from torch.nn import functional as F
from rmvd.loss.registry import register_loss


@register_loss
class SL1Loss(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction="none")

    @property
    def name(self):
        name = type(self).__name__
        return name

    def forward(self, sample_inputs, sample_gt, pred, aux, iteration):
        inputs = pred["depth"]
        targets = sample_gt["depth"]

        masks = sample_inputs["masks"] if "masks" in sample_inputs else targets > 0
        masks = masks.unsqueeze(1) if masks.dim() == 3 else masks
        with torch.no_grad():
            targets = F.interpolate(
                targets, size=inputs.shape[-2:], mode="bilinear", align_corners=False
            )
            masks = F.interpolate(masks.float(), size=inputs.shape[-2:], mode="nearest")
        masks = masks > 0
        loss = self.loss(inputs[masks], targets[masks])

        return torch.mean(loss), {}, {}
