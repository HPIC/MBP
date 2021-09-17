from mbs.types import TorchLossType

import math

from torch import Tensor
from torch.nn import Module


class MBSLoss(Module):
    def __init__(
        self,
        loss_fn : TorchLossType,
        mbs,
        mini_batch_size : int,
        micro_batch_size : int,
        normalize_factor : float = 1
    ) -> None:
        '''
            MBSLoss, only this calss inherits torch.nn.Module class.

            Args:
                loss_fn : torch.nn.Module (Because loss functions are torch.nn.Module type)
                mbs : Micro Batch Streaming object,
                    this is to share data between MBS subclass like MBSLoss, MBSDataloader, MBSOptimizer.
                *mini_batch_size : int-type
                *micro_batch_size : int-type
                    these two variables find streaming steps to normalize loss.
                    normalize_value = math.ceil( mini_batch_size / micro_batch_size )
                normalize_factor : float-type
                    it determines the impact of micro-batch normalize value.
                    - 0 (high impact of micro-batch normalization value to mini-batch)
                    - 1 (low impact of micro-batch normalization value to mini-batch)
        '''
        super(MBSLoss, self).__init__()
        self._comm_mbs = mbs
        self._loss_fn = loss_fn
        self._normalize_factor = normalize_factor
        self._normalize_value = math.ceil( mini_batch_size / micro_batch_size )

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
        _rate : float = self._normalize_factor
        _normalize : int = self._normalize_value
        loss_value : Tensor = self._loss_fn(input, target)
        if normalize_rate != None:
            _rate = normalize_rate
            _normalize = _normalize * _rate
        loss_value = loss_value / _normalize

        return loss_value

