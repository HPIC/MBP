from .wrap_model import MBSBatchNorm

import math
from typing import Optional, Tuple

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class _MBSBlock:
    def __init__(
        self,
        debug: Optional[str] = None
    ) -> None:
        self._init = True
        self._bn = False

        self.debug_msg = debug

    def _debug(self):
        if self.debug_msg == 'early stop':
            for name, para in self.module.named_parameters():
                print(name, '*'*30)
                print(para.data)
            print("\n\n\n")
            raise Exception(f"[MBS Debug] early stop")


__all__ = [
    MBSBatchNorm,
    "MicroBatchStreaming",
    "MBSSegmentation"
]


class MicroBatchStreaming(_MBSBlock):
    def __init__(
        self,
        dataloader: DataLoader,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
        warmup_factor: Optional[int] = None,
        device_index: Optional[int] = None,
        batch_size: int = 1,
        micro_batch_size: int = 1,
        bn_factor: bool = False,
        debug: Optional[str] = None
    ) -> None:
        super().__init__(debug=debug)
        self.device = torch.device(f'cuda:{device_index}')

        self.dataloader = dataloader
        if bn_factor:
            print("[MBS] Consider BatchNorm layers")
            self.module = MBSBatchNorm.wrap_batch_norm(model, self).to(self.device)
        else:
            print("[MBS] Does not consider BatchNorm layers")
            self.module = model
        self.criterion = criterion
        self.mean = False
        if self.criterion.reduction == "mean":
            print(f"[MBS] Loss function is based on {criterion.reduction} reduction")
            self.mean = True
        self.optimizer = optimizer

        ''' Warmup arguments '''
        self.scheduler = lr_scheduler
        self.warmup_factor = warmup_factor
        if self.scheduler is None or self.warmup_factor is None:
            print("[MBS] Does not consider Scheduler or Warmup algorithm")
        else:
            print("[MBS] Consider Scheduler or Warmup algorithm")

        self.batch_size = batch_size
        self.micro_batch = micro_batch_size
        self.chunks = math.ceil( self.batch_size / self.micro_batch )

        self.debug_msg = debug

    def get_model(self):
        return self.module

    def get_trainer(self):
        return self, self.module

    def train(self) -> float:
        data0: torch.Tensor
        data1: torch.Tensor
        epoch_loss = 0
        total_size = 0
        self.module.train()
        for idx, (data0, data1) in enumerate( self.dataloader ):
            mini_loss = 0
            total_size += data0.size(0)
            chunks = self.chunks
            if data0.size(0) != self.batch_size:
                chunks = math.ceil( data0.size(0) / self.micro_batch )

            chunk_data0 = data0.chunk( chunks )
            chunk_data1 = data1.chunk( chunks )

            self.optimizer.zero_grad()

            for jdx, (i, l) in enumerate( zip(chunk_data0, chunk_data1) ):
                self._bn = (jdx + 1) == chunks

                input = i.to(self.device)
                label = l.to(self.device)

                output: torch.Tensor = self.module( input )
                loss: torch.Tensor = self.criterion( output, label )
                if self.mean:
                    loss: torch.Tensor = torch.div( loss, chunks )
                loss.backward()
                mini_loss += loss.detach().item()

            self.optimizer.step()
            epoch_loss += mini_loss

            self._init = self._init and False

            if self.warmup_factor is not None:
                if idx <= self.warmup_factor:
                    self.scheduler.step()
            elif self.scheduler is not None:
                self.scheduler.step()

        return epoch_loss / total_size



class MBSSegmentation(MicroBatchStreaming):
    def __init__(
        self, 
        dataloader: DataLoader, 
        model: Module, 
        criterion: Module, 
        optimizer: Optimizer, 
        lr_scheduler: _LRScheduler = None, 
        warmup_factor: Optional[int] = None, 
        device_index: Optional[int] = None, 
        batch_size: int = 1, 
        micro_batch_size: int = 1, 
        bn_factor: bool = False,
        debug: Optional[str] = None
    ) -> None:
        super().__init__(
            dataloader, 
            model, 
            criterion, 
            optimizer, 
            lr_scheduler=lr_scheduler, 
            warmup_factor=warmup_factor, 
            device_index=device_index, 
            batch_size=batch_size, 
            micro_batch_size=micro_batch_size, 
            bn_factor=bn_factor,
            debug=debug
        )

    def train(self) -> Tuple[float, float]:
        data0: torch.Tensor
        data1: torch.Tensor

        pred: torch.Tensor
        loss: torch.Tensor
        dice: torch.Tensor

        epoch_loss = 0
        epoch_dice = 0
        total_size = 0
        self.module.train()
        for idx, (data0, data1) in enumerate( self.dataloader ):
            total_size += data0.size(0)
            mini_loss = 0
            mini_dice = 0
            chunks = self.chunks
            if data0.size(0) != self.batch_size:
                chunks = math.ceil( data0.size(0) / self.micro_batch )

            chunk_data0 = data0.chunk( chunks )
            chunk_data1 = data1.chunk( chunks )

            self.optimizer.zero_grad()

            for jdx, (i, l) in enumerate( zip(chunk_data0, chunk_data1) ):
                self._bn = (jdx + 1) == chunks

                input = i.to(self.device)
                mask = l.to(self.device)

                pred = self.module( input )
                loss, dice = self.criterion( pred, mask )
                if self.mean:
                    loss: torch.Tensor = loss.div(chunks)
                    dice: torch.Tensor = dice.div(chunks)
                loss.backward()
                mini_loss += loss.detach().item()
                mini_dice += dice.detach().item()

            self.optimizer.step()
            epoch_loss += mini_loss
            epoch_dice += mini_dice

            self._init = self._init and False

            if self.warmup_factor is not None:
                if idx <= self.warmup_factor:
                    self.scheduler.step()
            elif self.scheduler is not None:
                self.scheduler.step()

            self._debug()

        return epoch_loss / total_size, epoch_dice / total_size
