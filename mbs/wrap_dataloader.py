import math
from typing import Iterator, List, Tuple, Union
from torch.utils.data import DataLoader
import torch

class MBSDataloader:
    def __init__(
        self, dataloader : DataLoader, micro_batch_size : int = None, mbs = None
    ) -> None:
        self._comm_mbs = mbs
        self._dataloader = dataloader
        self._dataset = dataloader.dataset
        self._mini_batch_size = dataloader.batch_size

        if micro_batch_size == None:
            self._micro_batch_size = self._mini_batch_size
        else:
            self._micro_batch_size = micro_batch_size

    def _check_dataset(self, dataset):
        return torch.is_tensor(dataset)

    def _chunk_tensor(
        self, dataset : torch.Tensor
    ):
        num_chunk = math.ceil( self._mini_batch_size / self._micro_batch_size )
        if dataset.size(0) != self._mini_batch_size:
            num_chunk = math.ceil( dataset.size(0) / self._micro_batch_size )
        dataset = dataset.chunk( chunks=num_chunk )
        return dataset, num_chunk

    def _chunk_tensors(
        self, dataset : Tuple[torch.Tensor]
    ):
        chunk_dataset = []
        num_chunk = math.ceil( self._mini_batch_size / self._micro_batch_size )
        if dataset[0].size(0) != self._mini_batch_size:
            num_chunk = math.ceil( dataset[0].size(0) / self._micro_batch_size )

        for data in dataset:
            chunk_dataset.append(data.chunk(chunks=num_chunk))

        rtn_dataset = []
        for cidx in range(num_chunk):
            bundle = []
            for midx in range(len(dataset)):
                bundle.append( chunk_dataset[midx][cidx] )
            tuple_bundle = tuple(bundle)
            rtn_dataset.append( tuple_bundle )

        return rtn_dataset, num_chunk

    def _chunk(
        self, dataset : Union[ torch.Tensor, Tuple[torch.Tensor] ]
    ):
        is_atomic = self._check_dataset(dataset)
        if is_atomic:
            return self._chunk_tensor( dataset )
        else:
            return self._chunk_tensors( dataset )

    def __iter__(self):
        for data in self._dataloader:
            micro_dataset, num_micro_batch = self._chunk(data)
            for idx in range(num_micro_batch):
                self._comm_mbs._zero_grad_timing = idx == 0
                self._comm_mbs._update_timing = (idx + 1) == num_micro_batch
                yield micro_dataset[idx]

    def __len__(self):
        return len(self._dataloader)

    def micro_len(self):
        total_num_dataset = len(self._dataset)
        std_micro_len = ( total_num_dataset // self._micro_batch_size ) + \
            math.ceil( (total_num_dataset % self._micro_batch_size) / self._micro_batch_size )

        return std_micro_len
