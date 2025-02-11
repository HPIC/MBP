import functools
import gc
import logging
import math
import warnings
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch import device as Device
from torch.nn.modules.loss import _Loss

from ._pipeline import apply_pipeline


def apply(
    batch_names: List[str] | Tuple[str, ...],
    ub_size: int = 1,
    device: str | int | Device = "cpu",
    device_ids: List[int] | Tuple[int, ...] | None = None,
) -> Callable:
    r"""
    MBP Decorator
    ---
    The `apply()` decorator helps in training models with large batches by splitting them into smaller micro-batches.
    This decorator calculates the loss and performs backpropagation to accumulate gradients without updating model parameters.

    Usage:
    1. Specify the tensor names to be split and the micro-batch size.
    2. Ensure the decorated function computes and returns the loss without calling `.backward()`.
    3. Explicitly define the corresponding tensors in `key=value` format (e.g., `x=image`) to apply MBP.
    If not, MBP will not be applied to the tensors.

    Example:
        ```python
        >>> import mbp
        >>> @mbp.apply(["x", "label"], ub_size=16) # 1
        >>> def train_fn(model, criterion, x, label, *args, **kwargs):
        ...     o = model(x)
        ...     loss = criterion(o, label)
        ...     return loss # 2
        >>> for image, label in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = train_fn(model, criterion, x=image, label=label) # 3
        ...     optimizer.step()
        ```

    Args:
        batch_names (List[str] | Tuple[str]): List of batch names to be split.
        ub_size (int): Micro-batch size. Defaults to 1.
        device (str | int | Device, optional): Device to which the micro-batches will be sent. Defaults to "cpu".
        device_ids (List[int] | Tuple[int]): List of GPU IDs for multi-GPU training. Defualts to None.

    Returns:
        loss (float): The loss value.
    """
    assert isinstance(
        batch_names, (list, tuple)
    ), "`batch_names` must be a list or tuple."
    assert isinstance(
        ub_size, int
    ), "`ub_size` must be an integer or string. (Default: 1)"
    assert isinstance(
        device, (str, int, Device)
    ), "`device` must be a string, integer, or torch.device. (Default: 'cpu')"
    if isinstance(device, str):
        dev_ = Device(device)
    elif isinstance(device, int):
        dev_ = Device(f"cuda:{device}")
    else:
        dev_ = device
    if device_ids is not None:
        assert isinstance(
            device_ids, (list, tuple)
        ), "`device_ids` must be a list or tuple. (Default: 'None')"
        device_ids_ = len(device_ids)
    else:
        device_ids_ = 1

    def decorate(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "dev"):
                wrapper.dev, wrapper.num_device = dev_, device_ids_
            loss = 0
            batch_chunker = BatchChunker(
                batch_names, kwargs, ub_size, wrapper.dev, wrapper.num_device
            )
            if batch_chunker.is_valid:
                u_loss: Tensor
                others: List[Any] = []
                for kwargs in batch_chunker:
                    out = func(*args, **kwargs)
                    if isinstance(out, tuple):
                        u_loss, *others_ = out
                        assert isinstance(
                            u_loss, Tensor
                        ), "The first return value must be a loss tensor."
                        assert u_loss.dim() == 0, "The loss tensor must be a scalar."
                        others = _store_others(others_, out=others)
                    else:
                        u_loss = out
                        assert isinstance(
                            u_loss, Tensor
                        ), "The first return value must be a loss tensor."
                        assert u_loss.dim() == 0, "The loss tensor must be a scalar."
                    u_loss /= batch_chunker.chunk_size
                    u_loss.backward()
                    loss += u_loss.detach().item()
                if len(others) > 0:
                    return loss, *_gather_others(others)
                return loss
            else:
                warnings.warn(
                    "No tensors to split. please check the format of arguments in the function (key=value).",
                )
                return func(*args, **kwargs)

        return wrapper

    return decorate


def _store_others(others: Tuple[Any], out: List[List[Any]]) -> List[Any]:
    if len(out) == 0:
        out = [[o] for o in others]
    else:
        for i, o in enumerate(others):
            out[i].append(o)
    return out


def _gather_others(others: List[Any]) -> List[Any]:
    for i, o in enumerate(others):
        if len(o) == 1:
            others[i] = o[0]
        else:
            if isinstance(o[0], Tensor):
                # Only for tensors to concatenate
                others[i] = torch.cat(o)
            else:
                others[i] = o
    return others


def _get_model_device() -> Tuple[Device, int]:
    model = None
    num_device = 1
    while model is None:
        for obj in gc.get_objects():
            if (
                isinstance(obj, nn.Module)
                and hasattr(obj, "parameters")
                and not isinstance(obj, _Loss)
            ):
                model = obj
                break
    if isinstance(model, nn.DataParallel):
        logging.info("DataParallel model detected.")
        num_device = len(model.device_ids)
    assert isinstance(model, nn.Module), "No model found."
    device = next(model.parameters()).device
    return device, num_device


def _chunk(
    batch: Tensor, ub_size: int, num_device: int
) -> Tuple[Tuple[Tensor, ...], int]:
    chunk_size = math.ceil(batch.shape[0] / ub_size) if batch.shape[0] > ub_size else 1
    chunk_size = math.ceil(chunk_size / num_device) if num_device > 1 else chunk_size
    return batch.chunk(chunk_size), chunk_size


class BatchChunker:
    def __init__(
        self,
        batch_names: List[str] | Tuple[str, ...],
        kwargs: Dict[str, Tensor],
        ub_size: int,
        device: Device,
        num_device: int,
    ):
        m_batch = {k: kwargs[k] for k in batch_names if k in kwargs}
        _size = 1
        _chunked = {}
        for k, mb in m_batch.items():
            _chunked[k], _size = _chunk(mb, ub_size, num_device)

        self._chunked = _chunked
        self._stop_index = _size
        self._curr_index = 0
        self.kwargs = kwargs
        self.device = device

    def __len__(self):
        return self._stop_index

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_index < self._stop_index:
            self._curr_index += 1
            for k, ub in self._chunked.items():
                self.kwargs[k] = ub[self._curr_index - 1].to(self.device)
            return self.kwargs
        else:
            self._curr_index = 0
            raise StopIteration

    @property
    def chunk_size(self):
        return self._stop_index

    @property
    def is_valid(self):
        return len(self._chunked) > 0


__all__ = ["apply", apply_pipeline]
__version__ = "0.3.0"
