import math
import torch

from torch.utils.data import DataLoader

class MBSDataloader(DataLoader):
    def __init__(
        self,
        target_dataloader : DataLoader,
        micro_batch_size : int = None
    ) -> None:
        self.dataloader = target_dataloader
        self.dataset = target_dataloader.dataset
        self.mini_batch_size = target_dataloader.batch_size

        if micro_batch_size == None:
            self.micro_batch_size = self.mini_batch_size
        else:
            self.micro_batch_size = micro_batch_size

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
                rtn_dataset.append(no_tensor_chunk(data, num_chunk))

        return rtn_dataset, num_chunk

    def __iter__(self):
        for data in self.dataloader:
            micro_dataset, num_micro_batch = self.chunk_dataset(data)
            for midx in range(num_micro_batch):
                zero_grad_timing = midx == 0
                update_timing = (midx + 1) == num_micro_batch
                rtn_data = [ micro_dataset[cidx][midx] for cidx, _ in enumerate(micro_dataset) ]
                yield (zero_grad_timing, update_timing, rtn_data)

    def __len__(self):
        return len(self.dataloader)

    def micro_len(self):
        total_num_dataset = len(self.dataset)
        std_micro_len = ( total_num_dataset // self.micro_batch_size ) + \
            math.ceil( (total_num_dataset % self.micro_batch_size) / self.micro_batch_size )

        return std_micro_len

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