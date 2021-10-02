from mbs.types import TorchLossType

import math

from torch import Tensor
from torch.nn import Module


class MBSLoss(Module):
    def __init__(
        self,
        mbs_block,
        loss_fn : TorchLossType,
        normalize_factor : float = 1.0
    ) -> None:
        r'''
            MBSLoss, only this calss inherits torch.nn.Module class.

            Args:
                loss_fn : torch.nn.Module (Because loss functions are torch.nn.Module type)
                mbs : Micro Batch Streaming object,
                    this is to share data between MBS subclass like MBSLoss, MBSDataloader, MBSOptimizer.
                normalize_factor : float-type
                    it determines the impact of micro-batch normalize value.
                    - 0 (high impact of micro-batch loss values to mini-batch loss value)
                    - 1 (low impact of micro-batch loss values to mini-batch loss value)
        '''
        super(MBSLoss, self).__init__()
        self.mbs_block = mbs_block
        self._loss_fn = loss_fn
        self._normalize_factor = normalize_factor

    def forward(
        self, input : Tensor, target : Tensor, normalize_rate : float = None
    ):
        r'''
            Args:
                input : torch.Tensor-type, output by model.
                target : torch.Tensor-type, comparison target for calculating the difference.
                normalize_rate : float-type,
                    it adjusts normalize_factor rate for normalizing micro-batch loss value to mini-batch.
            Returns:
                loss_value : torch.Tensor-type,
                    return Tensor to calculate gradients.
        '''
        # Calculate streaming step
        streaming_step = self.mbs_block.remaining_step

        # print(
        #     f"{self.mbs_block._cur_batch_step}/{self.mbs_block.batch_iter}",
        #     f"{self.mbs_block.cur_batch_input_size}",
        #     f"{self.mbs_block.update_timing}"
        # )

        # Calculate loss value.
        loss_value = self._loss_fn(input, target)

        # Calculate degree of normalization for normalizing loss.
        rate = self._normalize_factor if normalize_rate == None else normalize_rate
        degree_of_normalization = streaming_step * rate

        # Apply.
        if degree_of_normalization == 0:
            return loss_value
        else:
            return loss_value / degree_of_normalization

