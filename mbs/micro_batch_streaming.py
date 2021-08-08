from contextlib import contextmanager
import math
from mbs.pipeline_allreduce import ModelList
from typing import Tuple
import torch

DatasetType = torch.Tensor
DatasetList = Tuple[DatasetType]

class mbstreaming:
    '''
    This class is running like dataloader or enumerate
    '''
    def __init__(self, models, optims) -> None:
        self.models : ModelList = models
        self.optims = optims

        self.grad_buffer = {}
        self.micro_epoch_counter = 0

    def store_grad(self):
        self.grad_buffer[self.micro_epoch_counter] = {}
        for idx, model in enumerate(self.models):
            model_grad = []
            for para in model.parameters():
                model_grad.append(para.grad.data)
            self.grad_buffer[self.micro_epoch_counter][idx] = model_grad
        self.micro_epoch_counter += 1

    def allreduce(self):
        all_grad = {}
        for index in self.grad_buffer:
            for name in self.grad_buffer[index]:
                if not name in list(all_grad.keys()):
                    all_grad[name] = self.grad_buffer[index][name]
                else:
                    for idx, para in enumerate(self.grad_buffer[index][name]):
                        all_grad[name][idx] += para

        for name in all_grad:
            for idx, _ in enumerate(all_grad[name]):
                all_grad[name][idx] /= self.micro_epoch_counter

        for i, mod in enumerate(self.models):
            for idx, para in enumerate(mod.parameters()):
                para.grad.data = all_grad[i][idx]

        self.micro_epoch_counter = 0

    def streaming(self, dataset, mini_batch_size=4, micro_batch_size=4):
        self.num_dataset = len(dataset)
        self.micro_dataset = []
        micro_batch_slice = 1
        for _ in range(self.num_dataset):
            self.micro_dataset.append( None )

        if mini_batch_size > micro_batch_size:
            micro_batch_slice = math.ceil( mini_batch_size / micro_batch_size )
            for idx, data in enumerate(dataset):
                chunk_dataset = data.chunk(micro_batch_slice)
                self.micro_dataset[idx] = chunk_dataset
        else:
            for idx, data in enumerate(dataset):
                self.micro_dataset[idx] = [ data ]

        for i in range(len(self.micro_dataset[0])):
            streaming_batch = []
            for mb_idx in range(self.num_dataset):
                streaming_batch.append( self.micro_dataset[mb_idx][i] )
            yield (tensor for tensor in streaming_batch)

