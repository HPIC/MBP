from typing import Tuple
import torch
import math

DatasetType = Tuple[torch.Tensor]

def wdataloader(_dataloader, _micro_batch_size):
    count_iter = 0
    for _, data in enumerate(_dataloader):
        A = data['A']
        B = data['B']

        num_micro_batch = math.ceil(A.size(0) / _micro_batch_size)

        As = A.chunk(num_micro_batch)
        Bs = B.chunk(num_micro_batch)

        for a, b in zip(As, Bs):
            count_iter += 1
            if count_iter == _micro_batch_size:
                count_iter = 0
                update = True
            else:
                update = False
            yield (update, a, b)
