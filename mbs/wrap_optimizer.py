import torch
from torch.optim.optimizer import Optimizer


class MBSOptimizer(Optimizer):
    def __init__(self, optimizer : Optimizer, mbs ) -> None:
        self._comm_mbs = mbs
        self._optimizer = optimizer

    def zero_grad(self):
        if self._comm_mbs._zero_grad_timing:
            # print('step')
            self._optimizer.zero_grad()

    def step(self):
        if self._comm_mbs._update_timing:
            # print('update')
            self._optimizer.step()

