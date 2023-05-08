
import torch
import torch.nn as nn

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
    
    def forward(self, inputs, targets, mask):
        loss = self.loss(inputs[mask], targets[mask])

        if self.ohem:
            num_hard_samples = int(self.topk * loss.numel())
            loss, _ = torch.topk(loss.flatten(), 
                                 num_hard_samples)

        return torch.mean(loss)