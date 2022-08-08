from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch import Tensor

class DiceLoss(Module):
    def __init__(self, reduction: str = 'mean', smooth: float = 1. ) -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward( self, preds: Tensor, masks: Tensor ) -> Tuple[Tensor, Tensor]:

        intersection: Tensor = ( preds * masks ).sum( dim=(2,3) )
        union: Tensor = preds.sum( dim=(2,3) ) + masks.sum( dim=(2,3) )
        dice: Tensor = ( 2 * intersection + self.smooth ) / ( union + self.smooth )
        loss: Tensor = 1 - dice
        loss += self.criterion( preds, masks )

        if self.reduction == "mean":
            return loss.mean(), dice.mean()
        elif self.reduction == "sum":
            return loss.sum(), dice.sum()

