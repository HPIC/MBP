import math
from typing import List, Optional, Union
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from torch.cuda import (
    Stream,
    stream,
    current_stream,
    default_stream,
)

from .wrap_model import MBSBatchNorm

class _MBSBlock:
    def __init__(
        self,
    ) -> None:
        self._bn = False

class MicroBatchStreaming(_MBSBlock):
    def __init__(
        self,
        dataloader: DataLoader,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
        device_index: Optional[int] = None,
        batch_size: int = 1,
        micro_batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.device = torch.device(f'cuda:{device_index}')

        self.dataloader = dataloader
        self.module = MBSBatchNorm.wrap_batch_norm(model, self).to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        self.batch_size = batch_size
        self.micro_batch = micro_batch_size
        self.chunks = math.ceil( self.batch_size / self.micro_batch )

        # self.cpy_strm = Stream(self.device)
        # self.cmp_strm = Stream(self.device)

    def get_model(self):
        return self.module

    def get_trainer(self):
        return self, self.module

    def train(self):
        self.epoch_loss = 0
        self.module.train()
        for idx, (data0, data1) in enumerate( self.dataloader ):
            data0: torch.Tensor
            data1: torch.Tensor

            mini_loss = 0
            # size = self.micro_batch
            chunks = self.chunks
            if data0.size(0) != self.batch_size:
                chunks = math.ceil( data0.size(0) / self.micro_batch )
                # size = math.ceil( data0.size(0) / chunks )

            chunk_data0 = data0.chunk( chunks )
            chunk_data1 = data1.chunk( chunks )

            self.optimizer.zero_grad()

            for jdx, (i, l) in enumerate( zip(chunk_data0, chunk_data1) ):
                self._bn = (jdx + 1) == chunks

                input = i.to(self.device)
                label = l.to(self.device)

                output: torch.Tensor = self.module( input )
                loss: torch.Tensor = self.criterion( output, label ) / chunks
                mini_loss += loss.detach().item()
                loss.backward()

            # for cidx in range(chunks):
            #     with stream( self.cpy_strm ):
            #         lower = cidx * size
            #         upper = ( cidx + 1 ) * size
            #         da0: torch.Tensor = data0[lower:upper].to(self.device)
            #         da1: torch.Tensor = data1[lower:upper].to(self.device)

            #     self.cpy_strm.wait_stream( self.cmp_strm )
            #     self.cmp_strm.wait_stream( self.cpy_strm )
            #     with stream( self.cmp_strm ):
            #         da0.record_stream( self.cmp_strm )
            #         da1.record_stream( self.cmp_strm )
            #         output: torch.Tensor = self.module( da0 )
            #         loss: torch.Tensor = self.criterion( output, da1 ) / chunks
            #         loss.backward()
            #         mini_loss += loss.detach().item()

            self.optimizer.step()
            self.epoch_loss += mini_loss

    def get_loss(self):
        return self.epoch_loss / self.dataloader.__len__()


# def micro_batch_streaming(
#     dataloader: DataLoader,
#     model: Module,
#     criterion: Module,
#     optimizer: Optimizer,
#     lr_scheduler: _LRScheduler = None,
#     dev: Union[ torch.device, int ] = 0,
#     micro_batch_size: int = 4,
# ):
#     epoch_loss = 0
#     for _, (data0, data1) in enumerate( dataloader ):
#         data0: torch.Tensor
#         data1: torch.Tensor

#         mini_loss = 0
#         chunks = math.ceil( dataloader.batch_size / micro_batch_size )
#         if data0.size(0) != dataloader.batch_size:
#             chunks = math.ceil( data0.size(0) / micro_batch_size )

#         chunk_data0 = data0.chunk( chunks )
#         chunk_data1 = data1.chunk( chunks )

#         optimizer.zero_grad()

#         for _, (i, l) in enumerate( zip(chunk_data0, chunk_data1) ):
#             input = i.to(dev)
#             label = l.to(dev)
#             output: torch.Tensor = model( input )
#             loss: torch.Tensor = criterion( output, label ) / chunks
#             loss.backward()
#             mini_loss += loss.detach().item()

#         optimizer.step()
#         epoch_loss += mini_loss
#     return epoch_loss / len(dataloader)
