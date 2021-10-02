from typing import List, Union
import torch

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import math

from mbs.types import TorchLossType

from mbs.wrap_dataloader import MBSDataloader, TestMBSDatalaoder
from mbs.wrap_loss import MBSLoss
from mbs.wrap_optimizer import MBSOptimizer
from mbs.wrap_model import MBSBatchNorm

MBSOptimizers = Union[ MBSOptimizer, List[MBSOptimizer] ]
MBSDataloaders = Union[ MBSDataloader, List[MBSDataloader] ]
MBSLosses = Union[ MBSLoss, List[MBSLoss] ]


class _MBSBlock:
    def __init__(self) -> None:
        self.zero_grad_timing = False
        self.update_timing = False
        self.mini_batch_size : int = None
        self.micro_batch_size : int = None

        self.current_iter: int = 0
        self.chunks: int = 0
        self.dataset_size: int = 0
        self.total_iter: int = 0
        self.remaining_step: int = 0

        # check mini-batch iteration
        self.mini_iter: int = 0
        self.total_mini_iter: int = 0
        self.temp: int = 0

    def update(self):
        chunks = self.chunks

        if self.current_iter % chunks == 1:
            self.mini_iter += 1
            self.zero_grad_timing = True
            self.update_timing = False
        elif self.current_iter % chunks == 0 or self.current_iter == self.total_iter:
            self.zero_grad_timing = False
            self.update_timing = True
        else:
            self.zero_grad_timing = False
            self.update_timing = False

        if self.mini_iter == self.total_mini_iter:
            self.temp = self.current_iter if self.temp == 0 else self.temp
            self.remaining_step = self.total_iter - self.temp + 1
        else:
            self.remaining_step = chunks

        # print(
        #     self.mini_iter, self.total_mini_iter,
        #     self.current_iter, self.total_iter, self.remaining_step,
        #     self.zero_grad_timing, self.update_timing, " "*10,
        # )


class MicroBatchStreaming(_MBSBlock):
    r'''
        Micro Batch Stream object
        - It is library(or framework) for training models with a large dataset on small GPU.
        - It is appropriate in several situations.
            1. Using a large (mini)batch size on small GPU.
            2. Using large image size datasets with a large (mini)batch size.

        Warning::
            Micro Batch Stream is running like OOP interface.
            therefore MBS subclass does not inherit torch class.
            (MBSLoss only inherits torch.nn.Module class, because )

        Example::
            >>> mbs = MicroBatchStream()
            >>>     ...
            >>> model = YourModel()
            >>> model : torch.nn.Module = mbs.set_batch_norm( model )
            >>>     ...
            >>> dataloader = torch.utils.data.Dataloader(...)
            >>> dataloader : MBSDataloader = mbs.set_dataloader( dataloader )
            >>>     ...
            >>> criterion = nn.CrossEntropyLoss(...)
            >>> criterion : MBSLoss = mbs.set_loss( criterion )
            >>>     ...
            >>> optimizer = torch.optim.SGD(...)
            >>> optimizer : MBSOptimizer = mbs.set_optimizer( optimizer )
            >>>     ...
            >>> for idx, (input, target) in enumerate(dataloader):
            >>>     optimizer.zero_grad()
            >>>     input = input
            >>>     output = model(input)
            >>>     loss = criterion(output, target)
            >>>     loss.backward()
            >>>     optimizer.step()
    '''
    def __init__( self ) -> None:
        r'''
            Initializes MicroBatchStreaming state.
        '''
        super(MicroBatchStreaming, self).__init__()
        self._optimizers : MBSOptimizers = []
        self._dataloaders : MBSDataloaders = []
        self._losses : MBSLosses = []

    def set_dataloader(
        self, dataloader : DataLoader, device: torch.device, micro_batch_size : int = 4, shuffle: bool = True,
    ) -> TestMBSDatalaoder:
        r'''
            Wrap PyTorch dataloader to MBS dataloader,
            MBS dataloader returns micro-batch-based dataset to user model when user's model is training.

            Args:
                dataloader : torch.utils.data.dataloader.
                micro_batch_size : int-type, set micro batch size for streaming in each iteration.
            Returns:
                MBSDataloader : output MBS(OOP interface-based) dataloader.
            Raises:
                TypeError : input(dataloader) is not torch.utils.data.dataloader format.
                    or input(micro_batch_size) is not int-type format.

            Example::
                >>> dataloader = torch.utils.data.Dataloader(...)
                >>> mbs = MicroBatchStream()
                >>> dataloader = mbs.set_dataloader( dataloader )
                >>> for idx, (input, target) in enumerate(dataloader):
                >>>     input = input.to(device)
                >>>         ...
        '''
        self.mini_batch_size = dataloader.batch_size
        self.micro_batch_size = micro_batch_size
        self.dataset_size = dataloader.dataset.__len__()
        self.total_iter = math.ceil(self.dataset_size / self.micro_batch_size)
        self.total_mini_iter = dataloader.__len__()
        self.chunks = math.ceil( self.mini_batch_size / self.micro_batch_size )
        # mbs_dataloader = MBSDataloader(
        #     dataloader=dataloader,
        #     micro_batch_size=micro_batch_size,
        #     mbs=self
        # )
        # self._dataloaders.append( mbs_dataloader )
        mbs_dataloader = TestMBSDatalaoder.wrap_dataloader(
            mbs_block=self,
            dataloader=dataloader,
            shuffle=shuffle,
            device=device,
            micro_batch_size=micro_batch_size
        )
        return mbs_dataloader

    def set_optimizer(
        self, optimizer : Optimizer
    ) -> MBSOptimizer:
        r'''
            Wrap PyTorch optimizer to MBS optimizer,

            Args:
                optimizer : torch.optim.Optimizer.
            Returns:
                MBSOptimizer : output MBS(OOP interface-based) optimizer.
            Raises:
                TypeError : input(optimizer) type is not torch.optim.Optimizer.

            Example::
                >>> optimizer = torch.optim.SGD(...)
                >>> mbs = MicroBatchStream()
                >>> optimizer = mbs.set_optimizer( optimizer )
        '''
        if not isinstance(optimizer, Optimizer):
            raise TypeError('[MBS Error] input(optimizer) type does not match, please check input(optmizer) type.')

        mbs_optimizer = MBSOptimizer(
            optimizer,
            mbs_block=self
        )
        self._optimizers.append( mbs_optimizer )
        return mbs_optimizer

    def set_loss(
        self, loss_fn : TorchLossType, normalize_factor : float = 1.0
    ) -> MBSLoss:
        r'''
            Wrap PyTorch loss function to MBS loss function,

            Args:
                loss_fn : torch.nn.Module,
                    because loss function is torch.nn.Module-based class in PyTorch.
                normalize_factor : float-type,
                    Determine the degree of normalization of micro-batch.
                    0(do not normalization) --> 1(strongest normalization)
            Returns:
                MBSLoss : output MBS(OOP interface-based) loss function(object).
            Raises:
                TypeError : input(loss_fn) is not torch.nn.Module format.
                    or input(normalize_factor) is not float-type format.

            Example::
                >>> criterion = nn.CrossEntropyLoss(...)
                >>> mbs = MicroBatchStream()
                >>> criterion = mbs.set_loss( criterion )
                >>> for idx, (input, target) in enumerate(dataloader):
                >>>         ...
                >>>     input = input
                >>>     output = model(input)
                >>>     loss = criterion(output, target)
                >>>     loss.backward()
                >>>         ...
        '''
        if not isinstance(loss_fn, TorchLossType):
            raise TypeError(
                '[MBS error] loss function type does not match, please check loss function format.'
            )
        if not isinstance(normalize_factor, float):
            raise TypeError(
                '[MBS error] normalize_factor type does not match, please check normalize factor format'
            )

        mbs_loss = MBSLoss(
            mbs_block=self,
            loss_fn=loss_fn,
            normalize_factor=normalize_factor
        )
        self._losses.append( mbs_loss )
        return mbs_loss

    def set_batch_norm(
        self, module : Module
    ):
        r'''
            wrap PyTorch::BatchNorm to MBS::BatchNorm.
            it means only replace PyTorch::BatchNorm to MBS::BatchNorm.

            Args:
                module : Pytorch::Module

            Returns:
                module_output : Pytorch::Module
                    but Pytorch::BatchNorm layers are replaced MBS::BatchNorm.

            Example::
                >>> model = net()
                >>> mbs = MicroBatchStream()
                >>> model = mbs.set_batch_norm( model )
        '''
        if not isinstance(module, Module):
            raise TypeError(
                '[MBS error] input module type does not match, please check input module type.'
            )
        mbs_model = MBSBatchNorm.wrap_batch_norm(module, self)
        return mbs_model

