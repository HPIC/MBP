import types

from mbs.types import ModelList
from mbs.wrap_dataloader import wrap_dataset


class MicroBatchStreaming:
    """
    This class is running like dataloader or enumerate
    """

    def __init__(self, micro_batch_size=4) -> None:
        self.models: ModelList = []
        self.optim = {}
        self.micro_batch_size = micro_batch_size

        self.grad_buffer = {}
        for name, _ in enumerate(self.models):
            self.grad_buffer[name] = None
        self.micro_epoch_counter = 0
        self.num_optim = 0

    def dataloader(self, _dataloader):
        return wrap_dataset(_dataloader, self.micro_batch_size)

    def model(self, _model):
        if isinstance(_model, tuple):
            self.models = list(_model)
        else:
            self.models.append(_model)

    def optimizer(self, _optim):
        _optim.step_allreduce = types.MethodType(step_allreduce, _optim)
        _optim.zero_grad_allreduce = types.MethodType(zero_grad_allreduce, _optim)
        return _optim


def step_allreduce(self, _update: bool = False):
    if _update:
        self.step()


def zero_grad_allreduce(self, _zero: bool = False):
    if _zero:
        self.zero_grad()
