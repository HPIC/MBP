from typing import Tuple, Union

import math

from torch.utils.data import DataLoader
from torch import Tensor
import torch

from mbs.types import (
    MultiTensors,
    ChunkTensor,
    ChunkTensors,
    ChunkTensororTensors
)

class MBSDataloader:
    def __init__(
        self, dataloader : DataLoader, micro_batch_size : int = None, mbs = None
    ) -> None:
        r'''
            MBSDataloader, like OOP interface.
            does not inherit torch.utils.data.Dataloader.

            Warning:
                if you do not define micro_batch_size,
                it will be out of GPU memory when micro-batch size bigger than GPU memory size.
                *** therefore if your (mini batch based) dataloader is bigger than GPU memory size,
                please define micro_batch_size. ***

            Args:
                dataloader : torch.utils.data.Dataloader (PyTorch Dataloader-type)
                    input the user-defined dataloader with PyTorch.
                micro_batch_size : int-type
                    it is optional.
                    define micro batch size for streaming dataloader.
                    if you do not define this variable,
                    the sizes of micro-batch and mini-batch are the same.
                mbs : Micro Batch Streaming object
                    this is to share data between MBS subclass like MBSLoss, MBSDataloader, MBSOptimizer.
        '''
        self._comm_mbs = mbs
        self._dataloader = dataloader
        self._dataset = dataloader.dataset
        self._mini_batch_size = dataloader.batch_size

        if micro_batch_size == None:
            self._micro_batch_size = self._mini_batch_size
        else:
            self._micro_batch_size = micro_batch_size

    def _check_dataset(self, dataset) -> bool:
        r''' Check dataset. if a single Tensor based dataset, return True. '''
        return torch.is_tensor(dataset)

    def _chunk_tensor( self, dataset : Tensor ) -> Tuple[ ChunkTensor, int ]:
        r''' Chunk dataset if dataset is only a single Tensor. '''
        num_chunk = math.ceil( self._mini_batch_size / self._micro_batch_size )
        if dataset.size(0) != self._mini_batch_size:
            num_chunk = math.ceil( dataset.size(0) / self._micro_batch_size )
        dataset = dataset.chunk( chunks=num_chunk )
        return dataset, num_chunk

    def _chunk_tensors( self, dataset : MultiTensors ) -> Tuple[ ChunkTensors, int ]:
        r''' 
            Chunk dataset if dataset is multiple Tensors.

            Args:
                dataset : (Tensor0, Tensor1, ...)
            Returns:
                rtn_datasset: Tuple(Tensor)-type, like below
                    (
                        (Tensor0[0], Tensor1[0], ...),
                        (Tensor0[1], Tensor1[1], ...),
                        ...,
                        (Tensor0[n], Tensor1[n], ...)
                    )
                num_chunk: int-type,
                    return number of chunk.
        '''
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

    def _chunk( self, dataset : Union[ Tensor, MultiTensors ] ) -> Tuple[ ChunkTensororTensors, int ]:
        r''' check dataset type, and then chunk dataset. '''
        is_atomic = self._check_dataset(dataset)
        if is_atomic:
            return self._chunk_tensor( dataset )
        else:
            return self._chunk_tensors( dataset )

    def __iter__(self):
        for data in self._dataloader:
            micro_dataset, num_micro_batch = self._chunk(data)
            self._comm_mbs._num_chunk = num_micro_batch
            for idx in range(num_micro_batch):
                self._comm_mbs._zero_grad_timing = idx == 0
                self._comm_mbs._update_timing = (idx + 1) == num_micro_batch
                yield micro_dataset[idx]

    def __len__(self):
        return len(self._dataloader)

    def micro_len(self) -> int:
        total_num_dataset = len(self._dataset)
        std_micro_len = ( total_num_dataset // self._micro_batch_size ) + \
            math.ceil( (total_num_dataset % self._micro_batch_size) / self._micro_batch_size )

        return std_micro_len
