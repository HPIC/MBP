from mbs.wrap_dataloader import wdataloader
from typing import List, Tuple, Union
import types
import torch

ModelType = Union[torch.nn.Module, torch.nn.Sequential]

DatasetType = torch.Tensor
DatasetList = Tuple[DatasetType]

OptimsType = torch.optim

class MicroBatchStreaming:
    '''
    This class is running like dataloader or enumerate
    '''
    def __init__(self, micro_batch_size=4) -> None:
        self.models : List[ ModelType ] = []
        self.micro_batch_size = micro_batch_size

        self.grad_buffer = {}
        for name, mod in enumerate(self.models):
            self.grad_buffer[name] = None
        self.micro_epoch_counter = 0

    def dataloader(self, _dataloader):
        return wdataloader(_dataloader, self.micro_batch_size)

    def model(self, _model):
        if isinstance(_model, tuple):
            self.models = list(_model)
        else:
            self.models.append(_model)

    def optimizer(self, _optim):
        _optim.step_allreduce = types.MethodType(step_allreduce, _optim)
        return _optim

    def store_grad(self):
        for name, mod in enumerate(self.models):
            self.grad_buffer[name] = mod.parameters()

def step_allreduce(self, _update : bool = False):
    if _update:
        pass
    else:
        pass


