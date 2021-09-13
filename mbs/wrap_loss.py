import math
import torch
import torch.nn as nn
import torch.nn.modules.loss as loss

class MBSLoss(nn.Module):
    def __init__(
        self,
        loss_fn,
        mbs,
        mini_batch_size : int,
        micro_batch_size : int
    ) -> None:
        super(MBSLoss, self).__init__()
        self._comm_mbs = mbs
        self._loss = loss_fn
        self._step = math.ceil( mini_batch_size / micro_batch_size )

    def forward(
        self, input : torch.Tensor, target : torch.Tensor
    ):
        return self._loss(input, target) / self._step

