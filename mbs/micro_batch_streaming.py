from contextlib import contextmanager
import math
from typing import List, Tuple
import torch

DatasetType = torch.Tensor
DatasetList = Tuple[DatasetType]

@contextmanager
def MBStreaming(dataset : DatasetList, mini_batch_size=4, micro_batch_size=4):
    num_dataset = len(dataset)
    micro_dataset = []
    micro_batch_slice = 1
    for _ in range(num_dataset):
        micro_dataset.append( None )

    if mini_batch_size > micro_batch_size:
        micro_batch_slice = math.ceil( mini_batch_size / mini_batch_size )
        for idx, data in enumerate(dataset):
            chunk_dataset = data.chunk(micro_batch_slice)
            micro_dataset[idx] = chunk_dataset
    else:
        for idx, data in enumerate(dataset):
            micro_dataset[idx] = [ data ]

    try:
        for i in range(len(micro_dataset[0])):
            streaming_batch = []
            for mb_idx in range(num_dataset):
                streaming_batch.append( micro_dataset[mb_idx][i] )
            yield (tensor for tensor in streaming_batch)
    finally:
        pass

