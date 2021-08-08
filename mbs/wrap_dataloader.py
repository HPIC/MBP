from typing import Tuple
import torch
import math

DatasetType = Tuple[torch.Tensor]

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

class streaming_dataloader:
    def __init__(self, mini_batch_dataset : DatasetType, mini_batch_size=4, micro_batch_size=4) -> None:
        self.mini_batch_dataset = mini_batch_dataset
        self.mini_batch_size = mini_batch_size
        self.micro_batch_size = micro_batch_size
        self.num_micro_batch = math.ceil(mini_batch_size / micro_batch_size)

    def __iter__(self):
        chunk_dataset = []
        for _ in range(len(self.mini_batch_dataset)):
            chunk_dataset.append(None)

        for i, data in enumerate(self.mini_batch_dataset):
            chunk_data = data.chunk(self.num_micro_batch, dim=0)
            chunk_dataset[i] = chunk_data

        for rtn_idx in range(self.num_micro_batch):
            rtn_tensors = []
            for _, dataset in enumerate(chunk_dataset):
                rtn_tensors.append( dataset[rtn_idx] )
            yield tuple(rtn_tensors)
