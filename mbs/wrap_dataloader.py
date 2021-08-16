import math
import torch

from torch.utils.data import DataLoader

class MBSDataloader(DataLoader):
    def __init__(self, micro_batch_size, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs )
        self.mini_batch_size = self.batch_size
        self.micro_batch_size = micro_batch_size

    def chunk_dataset(self, dataset):
        rtn_dataset = []
        num_chunk = math.ceil( self.mini_batch_size / self.micro_batch_size )

        if isinstance( dataset, dict ):
            for i, name in enumerate(dataset):
                if isinstance(dataset[name], torch.Tensor):
                    if dataset[name].size(0) != self.mini_batch_size:
                        num_chunk = math.ceil( dataset[name].size(0) / self.micro_batch_size )
                    rtn_dataset.append(dataset[name].chunk( num_chunk ))
                else:
                    rtn_dataset.append(no_tensor_chunk(dataset[name], num_chunk))
        elif isinstance( dataset, list ):
            for data in dataset:
                if isinstance(data, torch.Tensor):
                    if data.size(0) != self.mini_batch_size:
                        num_chunk = math.ceil( data.size(0) / self.micro_batch_size )
                    rtn_dataset.append(data.chunk( num_chunk ))
                else:
                    rtn_dataset.append(no_tensor_chunk(data, num_chunk))
        return rtn_dataset, num_chunk

    def __iter__(self):
        _dataloader = self._get_iterator()
        for data in _dataloader:
            micro_dataset, num_micro_batch = self.chunk_dataset(data)
            for midx in range(num_micro_batch):
                zero = midx == 0
                update = (midx + 1) == num_micro_batch
                rtn_data = [ micro_dataset[cidx][midx] for cidx, _ in enumerate(micro_dataset) ]
                yield (zero, update, rtn_data)


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