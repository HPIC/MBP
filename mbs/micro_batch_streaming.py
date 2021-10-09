import math
from typing import List, Optional
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from queue import Queue
from threading import Thread
import torch.multiprocessing as mp
from contextlib import contextmanager

from torch.profiler import profile, record_function, ProfilerActivity

from torch.cuda import (
    Stream,
    stream,
    current_stream,
    default_stream,
)

status = {
    'stop': 0,
    'wait': 1,
    'load': 2,
}

class _MBSBlock:
    def __init__(
        self,
        target_dev_idx: Optional[int],
        batch_size: int,
        micro_batch: int,
    ) -> None:
        n_gpu = torch.cuda.device_count()
        if n_gpu < target_dev_idx:
            raise ValueError('error device index')
        if target_dev_idx == None:
            raise ValueError('Only GPU.')
        self.device = torch.device(f'cuda:{target_dev_idx}')

        self.batch_size = batch_size
        self.micro_batch = micro_batch
        self.chunks = math.ceil( batch_size / micro_batch )

class MicroBatchStreaming(_MBSBlock):
    def __init__(
        self,
        dataloader: DataLoader,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
        device_index: Optional[int] = None,
        micro_batch_size: int = 4,
    ) -> None:
        super().__init__(
            target_dev_idx=device_index,
            batch_size=dataloader.batch_size,
            micro_batch=micro_batch_size
        )
        self.dataloader = dataloader
        self.module = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        self.data_queue = Queue()
        self.loss_queue = Queue()

    def train(self):
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        self.epoch_loss = 0
        for _, (data0, data1) in enumerate( self.dataloader ):
            data0: torch.Tensor
            data1: torch.Tensor

            mini_loss = 0
            chunks = self.chunks
            if data0.size(0) != self.batch_size:
                chunks = math.ceil( data0.size(0) / self.micro_batch )

            chunk_data0 = data0.chunk( chunks )
            chunk_data1 = data1.chunk( chunks )

            self.optimizer.zero_grad()

            # with torch.cuda.device(self.device):
            for _, (input, label) in enumerate( zip(chunk_data0, chunk_data1) ):
                input = input.to(self.device)
                label = label.to(self.device)
                output: torch.Tensor = self.module( input )
                loss: torch.Tensor = self.criterion( output, label ) / chunks
                loss.backward()
                mini_loss += loss.item()

            self.optimizer.step()
            self.epoch_loss += mini_loss
                # if idx > 1:
                #     break
        # prof.export_chrome_trace("trace.json")

    def get_loss(self):
        return self.epoch_loss / self.dataloader.__len__()


def wait_stream(source: Stream, target: Stream):
    source.wait_stream(target)

@contextmanager
def use_stream(strm: Stream):
    with stream( strm ):
        yield

class Task:
    def __init__(self) -> None:
        pass

