import functools
import gc
import logging
import math
import warnings
from typing import Callable, Dict, List, Tuple

import torch.nn as nn
from torch import Tensor
from torch import device as Device
from torch.nn.modules.loss import _Loss


def apply(
    target_batch: List[str] | Tuple[str, ...],
    ub_size: int,
    device: str | int | Device = "auto",
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
        target_batch (List[str]): List of tensor names to be split.
        ub_size (int): Micro-batch size.
        device (str | int | Device, optional): Device to use for the micro-batch. Defaults to "auto".

    Returns:
        loss (float): The loss value.
    """
    assert isinstance(
        target_batch, (list, tuple)
    ), "`target_batch` must be a list or tuple."
    assert isinstance(
        ub_size, (int, str)
    ), "`ub_size` must be an integer or string. (Default: 'auto')"
    assert isinstance(
        device, (str, int, Device)
    ), "`device` must be a string, integer, or torch.device. (Default: 'auto')"
    if device == "auto":
        dev_ = device
    else:
        if isinstance(device, str):
            dev_ = Device(device)
        elif isinstance(device, int):
            dev_ = Device(f"cuda:{device}")
        else:
            dev_ = device

    def decorate(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "dev"):
                wrapper.dev, wrapper.num_device = (
                    _get_model_device() if dev_ == "auto" else dev_
                )
            loss = 0
            batch_chunker = BatchChunker(
                target_batch, kwargs, ub_size, wrapper.dev, wrapper.num_device
            )
            if batch_chunker.is_valid:
                for kwargs in batch_chunker:
                    u_loss: Tensor = func(*args, **kwargs)
                    u_loss /= batch_chunker.chunk_size
                    u_loss.backward()
                    loss += u_loss.detach().item()
                return loss
            else:
                warnings.warn(
                    "No tensors to split. please check the format of arguments in the function (key=value).",
                )
                return func(*args, **kwargs)

        return wrapper

    return decorate


def _get_model_device() -> Device:
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
    return next(model.parameters()).device, num_device


def _chunk(
    batch: Tensor, ub_size: int, num_device: int
) -> Tuple[Tuple[Tensor, ...], int]:
    chunk_size = math.ceil(batch.shape[0] / ub_size) if batch.shape[0] > ub_size else 1
    chunk_size = math.ceil(chunk_size / num_device) if num_device > 1 else chunk_size
    return batch.chunk(chunk_size), chunk_size


class BatchChunker:
    def __init__(
        self,
        target_batch: List[str] | Tuple[str, ...],
        kwargs: Dict[str, Tensor],
        ub_size: int,
        device: Device,
        num_device: int,
    ):
        m_batch = {k: kwargs[k] for k in target_batch if k in kwargs}
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


__all__ = ["apply"]
