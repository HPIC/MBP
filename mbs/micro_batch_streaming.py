import types

from mbs.types import ModelList
from mbs.wrap_dataloader import MBSDataloader


class MicroBatchStreaming:
    """
    This class is running like dataloader or enumerate
    """

    def __init__(self, micro_batch_size=4) -> None:
        self.models: ModelList = []
        self.micro_batch_size = micro_batch_size

    def set_dataloader(self, *args, **kwargs):
        return MBSDataloader(self.micro_batch_size, *args, **kwargs)

    def set_optimizer(self, _optim):
        _optim.step_accu = types.MethodType(step_accu, _optim)
        _optim.zero_grad_accu = types.MethodType(zero_grad_accu, _optim)
        return _optim


def step_accu(self, _update: bool = False):
    if _update:
        self.step()


def zero_grad_accu(self, _zero: bool = False):
    if _zero:
        self.zero_grad()
