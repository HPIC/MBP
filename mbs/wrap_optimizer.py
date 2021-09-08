import torch
from torch.optim.optimizer import Optimizer


class MBSOptimizer(Optimizer):
    def __init__(self, _optimizer : Optimizer) -> None:
        self.optimizer = _optimizer

    def zero_grad_accu(self, timing : bool = False):
        if timing:
            self.optimizer.zero_grad()

    def step_accu(self, timing : bool = False):
        if timing:
            self.optimizer.step()

