from typing import List, Tuple

import torch.nn as nn
from torch import Tensor
from torch import device as Device

from ._log import log_message


def autoload(model: nn.Module) -> nn.Module:
    r"""
    Model Auto-load and Auto-offload.
    ---
    This function is used to automatically load and offload the model to the specified device, even when the devices (i.e., GPUs)
    have insufficient memory to allocate even the model to the device memory.

    Example:
        ```python
        >>> import torch
        >>> import mbp
        >>> device = torch.device("cuda:0")
        >>> model = ... # Define your model
        >>> mbp.autoload(model, device)
        ```

    Args:
        model (nn.Module): The model to be loaded and offloaded.
        device (Device): The device to which the model will be loaded.
        device_idx (List[int] | Tuple[int] | None): List of GPU IDs for multi-GPU training. Defaults to None.

    Returns:
        nn.Module: Autoloading functionality-applied model.
    """
    leaf_modules = []
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
            mod.register_forward_pre_hook(_fwd_onload(name))
            mod.register_forward_hook(_fwd_offload(name))

    return model


def _fwd_onload(name: str):
    def _wrapper(module: nn.Module, inputs: List[Tensor] | Tuple[Tensor, ...]):
        device = inputs[0].device
        # Attach backward offload hook
        if hasattr(inputs[0], "grad_fn") and inputs[0].grad_fn is not None:
            inputs[0].grad_fn.register_prehook(_bwd_offload(name, module))
        # Onload forward
        module.to(device, non_blocking=True)
        log_message(
            f">>> [Forward] Move `{name}` to CUDA!",
            color="green",
        )

    return _wrapper


def _fwd_offload(name: str):
    def _wrapper(module: nn.Module, input: Tensor, output: Tensor):
        # Attach backward onload hook
        output.grad_fn.register_prehook(_bwd_onload(name, module, output.device))
        # Offload forward
        module.to("cpu", non_blocking=True)
        log_message(
            f"<<< [Forward] Move `{name}` to CPU!",
            color="green",
        )

    return _wrapper


def _bwd_onload(name: str, module: nn.Module, device: Device):
    def _wrapper(*args, **kwargs):
        module.to(device, non_blocking=True)
        log_message(
            f">>> [Backward] Move `{name}` to CUDA!",
            color="blue",
        )

    return _wrapper


def _bwd_offload(name: str, module: nn.Module):
    def _wrapper(*args, **kwargs):
        module.to("cpu", non_blocking=True)
        log_message(
            f"<<< [Backward] Move `{name}` to CPU!",
            color="blue",
        )

    return _wrapper
