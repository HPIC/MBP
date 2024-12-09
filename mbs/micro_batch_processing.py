import math
import numpy as np
from typing import Optional, Union, List
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.utils.data import DataLoader

LOSS_T = Union[_Loss, _WeightedLoss]


class MBPSoft:
    '''
        Micro Batch Processing with Soft approach
        ---
        Args:
            model : torch.nn.Module
            loss_fn : torch.nn.modules.loss._Loss or torch.nn.modules.loss._WeightedLoss
            micro_batch_size : int

        Example:
            >>> model = torch.nn.Linear(10, 1)
            >>> loss_fn = torch.nn.MSELoss()
            >>> mbp = MBPSoft(model, loss_fn, micro_batch_size=1)
            >>> for e in range(epochs):
            >>>     for x, y in dataloader:
            >>>         loss = mbp.train(x, y) # Including backward
            >>>     optimizer.step()
            >>>     optimizer.zero_grad()
    '''
    def __init__(
        self, 
        model: nn.Module, 
        loss_fn: LOSS_T, 
        micro_batch_size: int = 1,
        dp: bool = False,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        assert isinstance(model, nn.Module), "model must be an instance of torch.nn.Module"
        # assert isinstance(loss_fn, _Loss) or isinstance(loss_fn, _WeightedLoss), \
        #     "loss_fn must be a subclass of torch.nn.modules.loss._Loss or torch.nn.modules.loss._WeightedLoss"
        if dp:
            model = nn.DataParallel(model, device_ids=device_ids)
        self.model = model
        self.dev = next(model.parameters()).device
        self.loss_fn = loss_fn
        self.loss_fn_reduction = getattr(loss_fn, 'reduction')
        self.micro_batch_size = micro_batch_size
        self.device_ids = device_ids

    @property
    def is_mean(self) -> bool:
        return self.loss_fn_reduction == 'mean'

    @property
    def num_devices(self) -> int:
        return len(self.device_ids) if self.device_ids is not None else 1

    def train(self, x: Tensor, target: Tensor) -> Tensor:
        mini_loss = 0
        if x.size(0) <= self.micro_batch_size:
            chunk_size = 1
            x_list = [x]
            target_list = [target]
        else:
            chunk_size = math.ceil(x.size(0) / self.micro_batch_size)
            chunk_size = math.ceil(chunk_size / self.num_devices)
            x_list = x.chunk(chunk_size)
            target_list = target.chunk(chunk_size)

        for _x, _t in zip(x_list, target_list):
            _x, _t = _x.to(self.dev), _t.to(self.dev)
            _out: Tensor = self.model(_x)
            _loss: Tensor = self.loss_fn(_out, _t)
            if self.is_mean:
                _loss /= chunk_size
            _loss.backward()
            mini_loss += _loss.detach().item()
        return mini_loss


class MBPHard(MBPSoft):
    '''
        Micro Batch Processing with Hard approach
        ---
        Args:
            model : torch.nn.Module
            loss_fn : torch.nn.modules.loss._Loss or torch.nn.modules.loss._WeightedLoss
            micro_batch_size : int
            dataloader : torch.utils.data.DataLoader

        Example:
            >>> model = torch.nn.Linear(10, 1)
            >>> loss_fn = torch.nn.MSELoss()
            >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
            >>> mbp = MBPHard(model, loss_fn, dataloader, micro_batch_size=1)
            >>> for e in range(epochs):
            >>>     epoch_loss, avg_epoch_loss = mbp.train() # Including data loading and backward
            >>>     optimizer.step()
            >>>     optimizer.zero_grad()
    '''
    def __init__(
        self, 
        model: nn.Module, 
        loss_fn: LOSS_T,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        micro_batch_size: int = 1,
        dp: bool = False,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        super().__init__(model, loss_fn, micro_batch_size, dp, device_ids)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.dataloader_iter = iter(self.dataloader)
        self.dataloader_size = len(self.dataloader)
        self.workers = ThreadPoolExecutor()

    def train(self) -> List[Tensor]:
        epoch_loss = []
        mini_batches = []
        for idx in range(self.dataloader_size):
            if idx == 0:
                mini_batches.append( self._dataload() )
            else:
                cur_mini_batch = mini_batches.pop(0)
                worker_args = [ (self._dataload, ), (self._train, *cur_mini_batch) ]
                for i, rtn in enumerate(self.workers.map(lambda x: x[0](*x[1:]), worker_args)):
                    if i == 0:
                        mini_batches.append( rtn )
                    else:
                        epoch_loss.append( rtn )
        if len(mini_batches) != 0:
            epoch_loss.append( self._train(*mini_batches[0]) )
        return epoch_loss, np.mean(epoch_loss)

    def _dataload(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)

    def _train(self, x: Tensor, target: Tensor) -> Tensor:
        with self._update():
            return super().train(x, target)

    @contextmanager
    def _update(self):
        self.optimizer.zero_grad()
        yield
        self.optimizer.step()


class MBP:
    @classmethod
    def soft(
        cls,
        model: nn.Module,
        loss_fn: LOSS_T,
        micro_batch_size: int = 1,
        dp: bool = False,
        device_ids: Optional[List[int]] = None,
    ) -> MBPSoft:
        '''
            Micro Batch Processing with Soft approach
            ---
            Args:
                model : torch.nn.Module
                loss_fn : torch.nn.modules.loss._Loss or torch.nn.modules.loss._WeightedLoss
                micro_batch_size : int

            Example:
                >>> model = torch.nn.Linear(10, 1)
                >>> loss_fn = torch.nn.MSELoss()
                >>> mbp = MBP.soft(model, loss_fn, micro_batch_size=1)
                >>> for e in range(epochs):
                >>>     for x, y in dataloader:
                >>>         loss = mbp.train(x, y)
        '''
        return MBPSoft(model, loss_fn, micro_batch_size, dp, device_ids)

    @classmethod
    def hard(
        cls,
        model: nn.Module,
        loss_fn: LOSS_T,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        micro_batch_size: int = 1,
        dp: bool = False,
        device_ids: Optional[List[int]] = None,
    ) -> MBPHard:
        '''
            Micro Batch Processing with Hard approach
            ---
            Args:
                model : torch.nn.Module
                loss_fn : torch.nn.modules.loss._Loss or torch.nn.modules.loss._WeightedLoss
                micro_batch_size : int
                dataloader : torch.utils.data.DataLoader

            Example:
                >>> model = torch.nn.Linear(10, 1)
                >>> loss_fn = torch.nn.MSELoss()
                >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
                >>> mbp = MBP.hard(model, loss_fn, dataloader, micro_batch_size=1)
                >>> for e in range(epochs):
                >>>     epoch_loss, avg_epoch_loss = mbp.train()
        '''
        return MBPHard(model, loss_fn, dataloader, optimizer, micro_batch_size, dp, device_ids)

