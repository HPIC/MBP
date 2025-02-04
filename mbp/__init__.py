import functools
import math
import warnings
from typing import Callable, Dict, List, Tuple

from torch import Tensor
from torch import device as Device


def autobackward(
    target_batch: List[str] | Tuple[str, ...],
    micro_batch_size: int,
    device_: str | int | Device = None,
) -> Callable:
    r"""
    MBP Decorator
    ---
    The `autobackward()` decorator helps in training models with large batches by splitting them into smaller micro-batches.
    This decorator calculates the loss and performs backpropagation to accumulate gradients without updating model parameters.

    Usage:
    1. Specify the tensor names to be split and the micro-batch size.
    2. Ensure the decorated function computes and returns the loss without calling `.backward()`.
    3. Explicitly define the corresponding tensors in `key=value` format (e.g., `x=image`) to apply MBP.
    If not, MBP will not be applied to the tensors.

    Example:
        ```python
        >>> import mbp
        >>> @mbp.autobackward(["x", "label"], micro_batch_size=16) # 1
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
        micro_batch_size (int): Size of each micro-batch.
        device_ (str | int | Device, optional): Device to move the micro-batch to. Defaults to None.

    Returns:
        loss (float): The loss value.

    ---
    To specify the device for the batch, include the device in the MBP decorator arguments or the function arguments.

    Example:
        ```python
        >>> device = "cuda:0"
        >>> @mbp.autobackward(["x", "label"], micro_batch_size=16, device_=device) # Optional-1
        >>> def train_fn(model, criterion, x, label, *args, **kwargs):
        ...    ...
        ```
        or
        ```python
        >>> device = "cuda:0"
        >>> for image, label in dataloader:
        ...     loss = train_fn(model, criterion, x=image, label=label, device_=device) # Optional-2
        ```
    """
    assert isinstance(
        target_batch, (list, tuple)
    ), "target_batch must be a list or tuple."
    assert isinstance(micro_batch_size, int), "micro_batch_size must be an integer."
    dev_ = device_ if device_ is not None else "cpu"

    def decorate(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            dev = dev_
            if "device" in kwargs:
                dev = kwargs["device_"]
                del kwargs["device_"]

            loss = 0
            batch_chunker = BatchChunker(target_batch, kwargs, micro_batch_size, dev)
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


def _chunk(batch: Tensor, micro_batch_size: int) -> Tuple[Tuple[Tensor, ...], int]:
    chunk_size = (
        math.ceil(batch.shape[0] / micro_batch_size)
        if batch.shape[0] > micro_batch_size
        else 1
    )
    return batch.chunk(chunk_size), chunk_size


class BatchChunker:
    def __init__(
        self,
        target_batch: List[str] | Tuple[str, ...],
        kwargs: Dict[str, Tensor],
        micro_batch_size: int,
        device: str | int | Device,
    ):
        self._chunked = {}
        _size = 1
        for k in target_batch:
            if k in kwargs:
                self._chunked[k], _size = _chunk(kwargs[k], micro_batch_size)
        self._stop_index = _size
        self._curr_index = 0
        self.kwargs = kwargs
        if isinstance(device, str):
            self.device = Device(device)
        elif isinstance(device, int):
            self.device = Device(f"cuda:{device}")
        else:
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
