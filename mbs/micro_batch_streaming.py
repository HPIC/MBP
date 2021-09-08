import math
from typing import List

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from mbs.wrap_dataloader import MBSDataloader
from mbs.wrap_optimizer import MBSOptimizer

class MicroBatchStreaming:
    def __init__(self) -> None:
        # for optimizer
        self.zero_grad_timing = False
        self.update_timing = False
        self.optimizers : List[MBSOptimizer] = []

    ''' Control dataloader '''
    def set_dataloader(
        self, target_dataloader : DataLoader, micro_batch_size : int = None
    ):
        self.dataloader = MBSDataloader(target_dataloader, micro_batch_size)
        return self

    def __iter__(self):
        for (ze, up, data) in self.dataloader:
            self.zero_grad_timing = ze
            self.update_timing = up
            yield data

    def __len__(self):
        r'''
            Return mini-batch-based dataloader size
            ---
            Example::
                >>> dataloader = mbs.set_dataloader(dataloader)
                >>> print( len(dataloader) )
        '''
        return len(self.dataloader)

    def micro_len(self):
        r'''
            Return micro-batch-based dataloader size
            ---
            Example::
                >>> dataloader = mbs.set_dataloader(dataloader)
                >>> print( dataloder.micro_len() )
        '''
        return self.dataloader.micro_len()

    ''' Control optimizer '''
    def set_optimizer(
        self, optimizer : Optimizer
    ):
        self.optimizers.append( MBSOptimizer(optimizer) )
        return self

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad_accu(self.zero_grad_timing)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step_accu(self.update_timing)
