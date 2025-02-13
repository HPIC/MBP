# Micro-Batch Processing (MBP)
[![pypi](https://img.shields.io/pypi/v/mbp-pytorch.svg)](https://pypi.org/project/mbp-pytorch/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

MBP is a simple method to enable deep learning (DL) models to train in large mini-batches, even when the devices (i.e., GPUs) have limited memory size. This GitHub repository provides an implementation of PyTorch-based MBP. You can easily train your DL models in large-batch training without the need to apply complex techniques, increase the number of GPUs, or GPU memory size.

The MBP implementation was designed with reference to the research presented in:
"[Enabling large batch size training for DNN models beyond the memory limit while maintaining performance](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10242106)"
by XinYu Piao, DoangJoo Synn, Jooyoung Park, and Jong-Kook Kim (IEEE Access 2023).

## Install
> **Requirements:**
> Python 3.10 or higher and a CUDA-enabled version of PyTorch are required.

You can easily install the MBP implementation using `pip`:
```bash
pip install mbp-pytorch
```
If you want to install MBP from source code, please follow the steps below:

```bash
git clone https://github.com/HPIC/MBP.git
cd MBP/
pip install .
```

## Usage
```python
import torch
import mbp

device = torch.device("cuda:0")

@mbp.apply(["x", "label"], 16, device) # Condition-1
def train_fn(model, criterion, x, label, *args, **kwargs):
    output = model(x)
    loss = criterion(output, label)
    return loss, output, ... # Condition-2

for image, label in dataloader:
    optimizer.zero_grad()
    loss, *others = train_fn(model, criterion, x=image, label=label) # Condition-3
    optimizer.step()
```
You can easily apply MBP to your DL training by following three conditions [[see exmaple code](./examples/how_to_use.py)]:

1. **Specify Batch Names, Micro-Batch Size, and Target Device**: Use the `@mbp.apply` decorator to specify the batch names to be split, the desired micro-batch size, and the target device.
2. **Return Loss Without Backpropagation**: Verify that the decorated function returns the loss without calling `.backward()`. There is no need to call `.backward()` because the MBP decorator handles backpropagation automatically. Also, make sure that the loss value is the first item returned by the function decorated with the MBP decorator.
3. **Define Batches Explicitly**: Explicitly define the corresponding batches in `key=value` format when using/calling the function decorated with MBP. If not specified, MBP will not be applied to those batches.

> ⚠️ **Caution:**
> The function decorated with MBP decorator returns the loss value for verification purposes only. Do not call `.backward()` on this loss value, as the MBP decorator will handle the backpropagation automatically. Do not follow the example below:
> ```python
> loss, *others = train_fn(model, criterion, x=image, label=label)
> loss.backward() # Don't try this!
> ```

MBP automatically detects whether the model is allocated to a CPU or GPU. It then splits a large mini-batch into smaller micro-batches and streams them sequentially to the detected device.

> ⚠️ **Caution:** Ensure that the specified device has enough memory to hold the micro-batches and intermediate computations. To apply MBP, ensure that after uploading the model to the GPU memory, there is enough remaining GPU memory capacity to allocate at least one micro-batch size and store intermediate computations computed by each layer of the model.

### Arguments
- `batch_names` (type: List[str] or Tuple[str]): List of batch names to be applied MBP.
- `ub_size` (type: int): The maximum size of a micro-batch. This value represents the maximum size of a micro-batch that can be executed on each device, and the batch size split by MBP cannot exceed this value. (Default: 1)
- `device` (type: str | int | Device, optional): Device to which the micro-batch will be sent. (Default: "cpu")
- `device_ids` (type: List[int] or Tuple[int]): List of GPU IDs for multi-GPU training. When using multiple GPUs with `torch.nn.DataParallel`, you need to specify the IDs of the GPUs to be used. (Defualt: None)

### Multi-GPU
MBP can be applied to both a single GPU and multi-GPUs. One of the key scalability features of MBP is its compatibility with PyTorch's `torch.nn.DataParallel()` without requiring any code modifications. Simply specify the `device_ids` as shown in the example below [[see exmaple code](./examples/how_to_use_multi.py)]:
```python
import torch
import torch.nn as nn
import mbp

device = torch.device("cuda")
device_ids = [0, 1]
model = nn.DataParallel(model, device_ids=device_ids).to(device)

@mbp.apply(["x", "label"], 16, device=device, device_ids=device_ids)
def train_fn(model, criterion, x, label, *args, **kwargs):
    output = model(x)
    loss = criterion(output, label)
    return loss

for image, label in dataloader:
    optimizer.zero_grad()
    loss, *others = train_fn(model, criterion, x=image, label=label)
    optimizer.step()
```

## Citation
If you use this MBP implementation in your research, please cite the following paper:
```bibtex
@article{piao2023enabling,
    title={Enabling large batch size training for DNN models beyond the memory limit while maintaining performance},
    author={Piao, XinYu and Synn, DoangJoo and Park, Jooyoung and Kim, Jong-Kook},
    journal={IEEE Access},
    year={2023},
    publisher={IEEE}
}
```
