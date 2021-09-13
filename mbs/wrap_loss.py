import math
import torch
import torch.nn as nn

from mbs.micro_batch_streaming import MicroBatchStreaming

class MBSLoss(nn.Module):
    def __init__(
        self,
        loss_fn : nn.Module,
        mbs : MicroBatchStreaming,
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

