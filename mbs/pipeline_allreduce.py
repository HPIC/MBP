from typing import List, Union
import torch

ModelType = Union[torch.nn.Module, torch.nn.Sequential]
ModelList = List[ModelType]

class PipeAllReduce:
    def __init__(self, *model) -> None:
        self.models : ModelList = []
        for mod in model:
            self.models.append(mod)

        self.micro_epoch_counter = 0
        self.grad_buffer = {}

    def store_grad(self):
        self.grad_buffer[self.micro_epoch_counter] = {}
        for idx, model in enumerate(self.models):
            model_grad = []
            for para in model.parameters():
                model_grad.append(para.grad.data)
            self.grad_buffer[self.micro_epoch_counter][idx] = model_grad
        self.micro_epoch_counter += 1

    def exe_grad(self):
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

        for mod in self.models:
            for idx, para in enumerate(mod.parameters()):
                para.grad.data = all_grad[mod][idx]

        self.micro_epoch_counter = 0

