from typing import List, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from mbs.types import (
    TorchSingleOptimizer,
    TorchMultiOptimizer,
    TorchOptimizer,
    MBSSingleOptimizer,
    MBSOptimizers,
    MBSDataloaders,
    MBSLosses,
)
from mbs.wrap_dataloader import MBSDataloader
from mbs.wrap_optimizer import MBSOptimizer
from mbs.wrap_loss import MBSLoss

class MicroBatchStreaming:
    def __init__( self ) -> None:
        self._zero_grad_timing : bool = False
        self._update_timing : bool = False
        self._optimizers : MBSOptimizers = []
        self._dataloaders : MBSDataloaders = []
        self._losses : MBSLosses = []
        self._mini_batch_size : int = None
        self._micro_batch_size : int = None

    ''' Control dataloader '''
    def set_dataloader(
        self, dataloader : DataLoader, micro_batch_size : int = None
    ):
        self._mini_batch_size = dataloader.batch_size
        self._micro_batch_size = micro_batch_size
        mbs_dataloader = MBSDataloader(
            dataloader=dataloader,
            micro_batch_size=micro_batch_size,
            mbs=self
        )
        self._dataloaders.append( mbs_dataloader )
        # return MBSBlock( dataloader=mbs_dataloader, mbs=self )
        return mbs_dataloader

    ''' Control optimizer '''
    def set_optimizer(
        self, optimizer : Optimizer
    ):
        mbs_optimizer = MBSOptimizer(
            optimizer,
            mbs=self
        )
        self._optimizers.append( mbs_optimizer )
        # return MBSBlock( optimizer=mbs_optimizer, mbs=self )
        return mbs_optimizer

    ''' Control optimizer '''
    def set_loss(
        self, loss_fn : nn.Module
    ):
        mbs_loss = MBSLoss(
            loss_fn=loss_fn,
            mbs=self,
            mini_batch_size=self._mini_batch_size,
            micro_batch_size=self._micro_batch_size
        )
        self._losses.append( mbs_loss )
        return mbs_loss


