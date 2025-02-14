import functools
import math
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch import device as Device

from ._autoload import autoload
from ._log import runtime_


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

            try:
                mb_size = kwargs[batch_names[0]].shape[0]
            except KeyError:
                message = "No batches to be split. Please check the format of arguments in the function (key=value)!"
                raise KeyError(message)
            chunk_size = _get_chunk_size(mb_size, ub_size, wrapper.num_device)

            if chunk_size >= 1:
                batch_chunker = BatchChunker(
                    batch_names,
                    kwargs,
                    mb_size,
                    chunk_size,
                    ub_size,
                    wrapper.dev,
                    wrapper.num_device,
                )
                for n in batch_names:
                    del kwargs[n]
                loss = 0.0
                others: List[Any] = []  # Store/Stack others
                for ubatch in batch_chunker:
                    kwargs.update(ubatch)
                    out = _forward_ub(func, args, kwargs)
                    ub_loss, others = _seperate_uloss_and_others(out, others)
                    loss += _backward_ub(ub_loss, batch_chunker.chunk_size)
                if len(others) > 0:
                    return loss, *_gather_others(others)
                else:
                    return loss
            else:
                raise ValueError(
                    "The chunk size must be greater than or equal to 1. Please check the micro-batch size and the number of devices."
                )

        return wrapper

    return decorate


@runtime_
def _forward_ub(func: Callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    return func(*args, **kwargs)


@runtime_
def _backward_ub(ub_loss: Tensor, chunk_size: int) -> Tensor:
    ub_loss /= chunk_size  # Apply Loss Normalization.
    ub_loss.backward()
    return ub_loss


def _seperate_uloss_and_others(
    out: Any, others_list: List[List[Any]] | None = None
) -> Tuple[Tensor, List[List[Any]]] | Tensor:
    # Separate the loss tensor and others from the output of the function.
    if isinstance(out, tuple):
        ub_loss, *others = out
        assert isinstance(
            ub_loss, Tensor
        ), "The first return value must be a loss tensor."
        assert ub_loss.requires_grad, "The loss tensor must require gradients."
        assert ub_loss.dim() == 0, "The loss tensor must be a scalar."
        if others_list is None:
            return ub_loss, *others
        others_list = _store_others(others, out=others_list)
        return ub_loss, others_list
    else:
        ub_loss = out
        assert isinstance(
            ub_loss, Tensor
        ), "The first return value must be a loss tensor."
        assert ub_loss.requires_grad, "The loss tensor must require gradients."
        assert ub_loss.dim() == 0, "The loss tensor must be a scalar."
        if others_list is None:
            return ub_loss
        return ub_loss, others_list


def _store_others(others: Tuple[Any], out: List[List[Any]]) -> List[Any]:
    # Store others in a list.
    if len(out) == 0:
        out = [[o] for o in others]
    else:
        for i, o in enumerate(others):
            out[i].append(o)
    return out


def _gather_others(others: List[Any]) -> List[Any]:
    # Gather others from the list.
    for i, o in enumerate(others):
        if len(o) == 1:
            others[i] = o[0]
        else:
            if isinstance(o[0], Tensor):
                others[i] = torch.cat(o)  # Only for tensors to concatenate
            else:
                others[i] = o
    return others


def _get_chunk_size(mb_size: int, ub_size: int, num_device: int) -> bool:
    chunk_size = math.ceil(mb_size / ub_size) if mb_size > ub_size else 1
    chunk_size = math.ceil(chunk_size / num_device) if num_device > 1 else chunk_size
    return chunk_size


class BatchChunker:
    def __init__(
        self,
        batch_names: List[str] | Tuple[str, ...],
        kwargs: Dict[str, Tensor],
        mb_size: int,
        chunk_size: int,
        ub_size: int,
        device: Device,
        num_device: int = 1,
    ):
        self._stop_index = chunk_size
        self._curr_index = 0
        self.device: Device = device
        self.batch_names = batch_names
        self.micro_batches: List[Dict[str, Tensor]] = []
        for s in range(chunk_size):
            sidx = s * (ub_size * num_device)
            eidx = min(sidx + (ub_size * num_device), mb_size)
            self.micro_batches.append({n: kwargs[n][sidx:eidx] for n in batch_names})

    def __len__(self):
        return self._stop_index

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_index < self._stop_index:
            self._curr_index += 1
            return self._load_ub(self._curr_index - 1)
        else:
            self._curr_index = 0
            raise StopIteration

    @runtime_
    def _load_ub(self, _curr_index: int) -> Dict[str, Tensor]:
        ubatch = self.micro_batches[_curr_index]
        for n in ubatch:
            ubatch[n] = ubatch[n].to(self.device, non_blocking=True)
        return ubatch

    @property
    def chunk_size(self):
        return self._stop_index


__all__ = [apply, autoload]
__version__ = "0.2.6"
