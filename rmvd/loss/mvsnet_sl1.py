
import torch
import torch.nn as nn
from torch.nn import functional as F
from rmvd.loss.registry import register_loss

@register_loss
class SL1Loss(nn.Module):
    def __init__(self, ohem=False, topk=0.6, model=None): #TODO: ohem can be configured to true
        super().__init__()
        self.ohem = ohem
        self.topk = topk
        self.loss = nn.SmoothL1Loss(reduction='none')
        
    @property
    def name(self):
        name = type(self).__name__
        return name
    
    def forward(self, sample_inputs, sample_gt, pred, aux, iteration):
        inputs = pred['depth']
        targets = sample_gt['depth']
        
        masks =sample_inputs['masks'] if 'masks' in sample_inputs else targets > 0
        masks = masks[0].unsqueeze(1) if isinstance(masks, list) else masks
        with torch.no_grad():
            targets = F.interpolate(targets, size=inputs.shape[-2:], mode='bilinear', align_corners=True)
            masks = F.interpolate(masks.float(), size=inputs.shape[-2:], mode='nearest')
        masks = masks > 0.5
        loss = self.loss(inputs[masks], targets[masks])

        if self.ohem:
            num_hard_samples = int(self.topk * loss.numel())
            loss, _ = torch.topk(loss.flatten(), 
                                 num_hard_samples)

        return torch.mean(loss),{} , {}


        # run the rest of the pipeline in the evening. Start training on the cluster.