import math
from typing import Union
import torch
import torch.nn as nn


class MBSLoss(nn.Module):
    def __init__(
        self,
        loss_fn : nn.Module,
        mbs,
        mini_batch_size : int,
        micro_batch_size : int,
        normalize_factor : Union[int, float] = None
    ) -> None:
        super(MBSLoss, self).__init__()
        self._comm_mbs = mbs
        self._loss_fn = loss_fn
        self._normalize_factor = normalize_factor if normalize_factor != None else 1
        self._normalize_value = math.ceil( mini_batch_size / micro_batch_size )

    def forward(
        self, input : torch.Tensor, target : torch.Tensor, normalize_rate : Union[int, float] = None
    ):
        _rate = self._normalize_factor
        _normalize = self._normalize_value
        loss_value = self._loss_fn(input, target)
        if normalize_rate != None:
            _rate = normalize_rate
            _normalize = _normalize * _rate
        loss_value = loss_value / _normalize

        return loss_value

