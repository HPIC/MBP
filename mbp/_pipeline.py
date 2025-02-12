from typing import Dict, Tuple

import torch.nn as nn
from torch import Tensor
from torch import device as Device

from ._log import backward_log, forward_log

total_size_submodules = 0
submodules: Dict[int, Tuple[str, nn.Module]] = {}


def apply_pipeline(model: nn.Module, device: Device, method: str = "simple"):
    r"""
    Module Pipeline
    ---
    Example:
        ```python
        >>> import torch
        >>> import mbp
        >>> device = torch.device("cuda:0")
        >>> model = ... # Define your model
        >>> model = mbp.apply_pipeline(model, device)
        ```
    """
    if method == "simple":
        return apply_pipeline_simple(model, device)
    elif method == "preload":
        return apply_pipeline_preload(model, device)
    else:
        raise NotImplementedError(f"Method `{method}` is not implemented!")


def move_to_gpu(name: str, device: Device, next_mod_index: int | None = None):
    def _wrapper(module: nn.Module, *args, **kwargs):
        if next_mod_index is not None:
            if next_mod_index < total_size_submodules:
                next_mod_name, next_module = submodules[next_mod_index]
                next_module.to(device, non_blocking=True)
                forward_log(
                    f">>> [Forward (pre-load)] Move `{next_mod_name}` to {next(module.parameters()).device}!"
                )
            if next_mod_index - 1 == 0:
                module.to(device, non_blocking=True)
                forward_log(
                    f">>> [Forward] Move `{name}` to {next(module.parameters()).device}!"
                )
        else:
            module.to(device, non_blocking=True)
            forward_log(
                f">>> [Forward] Move `{name}` to {next(module.parameters()).device}!"
            )

    return _wrapper


def move_to_cpu(name: str, next_mod_index: int):
    def _auto_onload_offload(
        name: str, module: nn.Module, device: Device, next_mod_index
    ):
        def _wrapper(*args, **kwargs):
            if next_mod_index < total_size_submodules:
                next_mod_name, next_module = submodules[next_mod_index]
                next_module.to("cpu", non_blocking=True)
                backward_log(
                    f"<<< [Backward] Move `{next_mod_name}` to {next(module.parameters()).device}!"
                )
            module.to(device, non_blocking=True)
            backward_log(
                f">>> [Backward] Move `{name}` to {next(module.parameters()).device}!"
            )

        return _wrapper

    def _wrapper(module: nn.Module, input: Tensor, output: Tensor):
        output.grad_fn.register_prehook(
            _auto_onload_offload(name, module, output.device, next_mod_index)
        )
        module.to("cpu", non_blocking=True)
        forward_log(
            f"<<< [Forward] Move `{name}` to {next(module.parameters()).device} (attach backward hooks: {hex(id(output))})!"
        )

    return _wrapper


def apply_pipeline_simple(model: nn.Module, device: Device):
    global submodules, total_size_submodules
    leaf_modules = []
    mod_index = 0
    for name, _ in model.state_dict().items():
        name_list = name.split(".")[:-1]
        mod: nn.Module = model
        for n in name_list:
            mod = getattr(mod, n)

        name = ".".join(name_list)
        if name not in leaf_modules:
            if not mod._parameters:
                continue
            leaf_modules.append(name)
            submodules[mod_index] = (name, mod)
            mod.register_forward_pre_hook(move_to_gpu(name, device))
            mod.register_forward_hook(move_to_cpu(name, mod_index + 1))
            mod_index += 1
    total_size_submodules = len(submodules)
    return model


def apply_pipeline_preload(model: nn.Module, device: Device):
    global submodules, total_size_submodules
    leaf_modules = []
    mod_index = 0
    for name, _ in model.state_dict().items():
        name_list = name.split(".")[:-1]
        mod: nn.Module = model
        for n in name_list:
            mod = getattr(mod, n)

        name = ".".join(name_list)
        if name not in leaf_modules:
            if not mod._parameters:
                continue
            leaf_modules.append(name)
            submodules[mod_index] = (name, mod)
            mod.register_forward_pre_hook(move_to_gpu(name, device, mod_index + 1))
            mod.register_forward_hook(move_to_cpu(name, mod_index + 1))
            mod_index += 1
    total_size_submodules = len(submodules)
    return model
