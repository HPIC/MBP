from typing import Dict

from mbs.types import ModelList, ModelTuple


class PipeAllReduce:
    def __init__(self, model: ModelList) -> None:
        self.models = model
        self.grad_buffer = {}
        for model_name, model in enumerate(self.models):
            self.grad_buffer[model_name] = None

    def stack_grad(self):
        for model_name, model in enumerate(self.models):
            if self.grad_buffer[model_name]:
                for idx, para in enumerate(model.parameters()):
                    self.grad_buffer[model_name][idx] += para.grad.data
            else:
                paras = []
                for idx, para in enumerate(model.parameters()):
                    paras.append(para.grad.data)
                self.grad_buffer[model_name] = paras

    def allreduce(self, num_micro_epoch):
        # allreduce
        for model_name, model in enumerate(self.models):
            for idx, para in enumerate(model.parameters()):
                para.grad.data = self.grad_buffer[model_name][idx] / num_micro_epoch

        # reset
        for model_name, model in enumerate(self.models):
            self.grad_buffer[model_name] = None


def stack_grad(models: ModelTuple, grad_buffer: Dict):
    for model_name, model in enumerate(models):
        if grad_buffer[model_name]:
            for idx, para in enumerate(model.parameters()):
                grad_buffer[model_name][idx] += para.grad.data
        else:
            paras = []
            for idx, para in enumerate(model.parameters()):
                paras.append(para.grad.data)
            grad_buffer[model_name] = paras

    return grad_buffer


def allreduce(models, grad_buffer, num_micro_epoch):
    # allreduce
    for model_name, model in enumerate(models):
        for idx, para in enumerate(model.parameters()):
            para.grad.data = grad_buffer[model_name][idx] / num_micro_epoch

    # reset
    for model_name, model in enumerate(models):
        grad_buffer[model_name] = None

    return grad_buffer
