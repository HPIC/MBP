import math
from typing import List

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

class MicroBatchStreaming(DataLoader, Optimizer):
    """
    This class is running like dataloader or enumerate
    """

    def __init__(self) -> None:
        # for optimizer
        self.zero_grad_timing = False
        self.update_timing = False
        self.optimizers : List[Optimizer] = []

    def set_dataloader(
        self, target_dataloader : DataLoader, micro_batch_size : int = None
    ):
        self.dataloader = target_dataloader
        self.mini_batch_size = target_dataloader.batch_size

        # setting micro batch size
        if micro_batch_size == None:
            self.micro_batch_size = self.mini_batch_size
        else:
            self.micro_batch_size = micro_batch_size

        return self

    def chunk_dataset(self, dataset):
        rtn_dataset = []
        num_chunk = math.ceil( self.mini_batch_size / self.micro_batch_size )

        for _, data in enumerate(dataset):
            data = dataset.get(data) if isinstance(dataset, dict) else data
            if isinstance(data, torch.Tensor):
                if data.size(0) != self.mini_batch_size:
                    num_chunk = math.ceil( data.size(0) / self.micro_batch_size )
                rtn_dataset.append(data.chunk( num_chunk ))
            else:
                rtn_dataset.append(self.no_tensor_chunk(data, num_chunk))

        return rtn_dataset, num_chunk

    def no_tensor_chunk(dataset, num_chunk):
        temp = []
        rtn_dataset = []
        step = math.ceil( len(dataset) / num_chunk )

        for idx, data in enumerate(dataset):
            temp.append(data)
            if (idx + 1) % step == 0 or (idx + 1) == len(dataset):
                rtn_dataset.append(temp)
                temp = []

        return rtn_dataset

    def __iter__(self):
        for data in self.dataloader:
            micro_dataset, num_micro_batch = self.chunk_dataset(data)
            for midx in range(num_micro_batch):
                self.zero_grad_timing = midx == 0
                self.update_timing = (midx + 1) == num_micro_batch
                rtn_data = [ micro_dataset[cidx][midx] for cidx, _ in enumerate(micro_dataset) ]
                yield rtn_data

    def set_optimizer(self, _optim):
        self.optimizers.append(_optim)
        return self

    def zero_grad(self):
        if self.zero_grad_timing:
            print('zero gradients')
            for optim in self.optimizers:
                optim.zero_grad()

    def step(self):
        if self.update_timing:
            print('update')
            for optim in self.optimizers:
                optim.step()
