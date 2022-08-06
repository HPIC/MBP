from typing import Tuple

import torch.nn.functional as F
from torch.nn.modules import Module
from torch import Tensor


class DiceLoss(Module):
    def __init__(self, reduction: str = 'mean', smooth: float = 1.0 ) -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward( self, inputs: Tensor, masks: Tensor ) -> Tuple[Tensor, Tensor]:
        intersection: Tensor = ( inputs * masks ).sum(  )
        union: Tensor = inputs.sum(  ) + masks.sum(  )
        dice: Tensor = ( 2 * intersection + self.smooth ) / ( union + self.smooth )
        loss: Tensor = 1 - dice

        if self.reduction == 'mean':
            return loss.mean(), dice.mean()
        elif self.reduction == 'sum':
            return loss.sum(), dice.mean()


class DiceBCELoss(DiceLoss):
    def __init__(self, reduction: str = 'mean', smooth: float = 1.0) -> None:
        super().__init__(reduction, smooth)
    
    def forward( self, inputs: Tensor, masks: Tensor ):
        intersection: Tensor = ( inputs * masks ).sum(  )
        union: Tensor = inputs.sum(  ) + masks.sum(  )
        dice: Tensor = ( 2 * intersection + self.smooth ) / ( union + self.smooth )
        loss: Tensor = 1 - dice
        bce: Tensor = F.binary_cross_entropy_with_logits(inputs, masks, reduction=self.reduction)
        loss += bce

        if self.reduction == 'mean':
            return loss.mean(), dice.mean()
        elif self.reduction == 'sum':
            return loss.sum(), dice.mean()
