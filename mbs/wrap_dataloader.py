import math

from typing import List
from torch.utils.data import DataLoader
import torch


class _MBSChunk:
    def  __init__(self) -> None:
        pass

    @classmethod
    def _chunk_dataset(
        cls, dataset : List[torch.Tensor], mini_batch_size: int, micro_batch_size: int
    ):
        chunk_dataset = []
        num_chunk = math.ceil( mini_batch_size / micro_batch_size )

        for data in dataset:
            if data.size(0) != mini_batch_size:
                num_chunk = math.ceil( data.size(0) / micro_batch_size )
            chunk_dataset.append(data.chunk(chunks=num_chunk))

        return chunk_dataset, num_chunk

class MBSDataloader(_MBSChunk):
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

    def __iter__(self):
        for data in self._dataloader:
            micro_dataset, num_micro_batch = self._chunk_dataset(data, self._mini_batch_size, self._micro_batch_size)
            for midx in range(num_micro_batch):
                self._comm_mbs._zero_grad_timing = midx == 0
                self._comm_mbs._update_timing = (midx + 1) == num_micro_batch
                rtn_data = [ micro_dataset[cidx][midx] for cidx, _ in enumerate(micro_dataset) ]
                yield rtn_data

    def __len__(self):
        return len(self._dataloader)

    def micro_len(self):
        total_num_dataset = len(self._dataset)
        std_micro_len = ( total_num_dataset // self._micro_batch_size ) + \
            math.ceil( (total_num_dataset % self._micro_batch_size) / self._micro_batch_size )

        return std_micro_len

