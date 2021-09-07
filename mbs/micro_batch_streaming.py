from typing import List

from mbs.types import ModelList
from mbs.wrap_dataloader import MBSDataloader

from torch.optim.optimizer import Optimizer

class MicroBatchStreaming:
    """
    This class is running like dataloader or enumerate
    """

    def __init__(self) -> None:
        self.models: ModelList = []

        # for optimizer
        self.zero_grad_timing = False
        self.update_timing = False
        self.optimizers : List[Optimizer] = []

    def set_dataloader(self, dataloader, micro_batch_size=16):
        self._dataloder = MBSDataloader(dataloader, micro_batch_size)
        return self

    def set_optimizer(self, _optim):
        self.optimizers.append(_optim)
        return self

    def __iter__(self):
        dataloader_gen = self._dataloder.__iter__()
        for rtn_data in dataloader_gen:
            (self.zero_grad_timing, self.update_timing, output) = rtn_data
            yield output

    def zero_grad(self):
        if self.zero_grad_timing:
            for optim in self.optimizers:
                optim.zero_grad()

    def step(self):
        if self.update_timing:
            for optim in self.optimizers:
                optim.step()


def step_accu(self, _update: bool = False):
    if _update:
        self.step()


def zero_grad_accu(self, _zero: bool = False):
    if _zero:
        self.zero_grad()
