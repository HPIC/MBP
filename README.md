# Micro-Batch Processing (MBP)
MBP is a method to enable deep learning (DL) models to train in large mini-batches, even when the devices have limited memory capacity.
The main idea of MBP is to split a large mini-batch into smaller micro-batches and stream them sequentially to the GPU, which sequentially computes and trains the model based on micro-batches.
To ensure the model trains correctly, MBP uses a loss normalization method to compute the gradient accumulated from multiple micro-batches almost the same as the gradient computed from a single mini-batch.
In experiments, the results show that MBP can increase the training batch size to the full size of the training set regardless of the GPU memory size.

This GitHub repository provides an implementation of PyTorch-based MBP.
The MBP provided in this GitHub repository is compatible with the latest version of PyTorch and does not require any additional libraries.
You can easily train your DL models in large-batch training without the need to apply complex techniques, increase the number of GPUs, and GPU memory size 😀.

## Install
You can easily install the MBP implementation compatible with PyTorch using pip:
```python
$ pip install git+https://github.com/HPIC/MBP.git
```
> **Requirements:**
> Python 3.10 or higher and a CUDA-enabled version of PyTorch are required.

## Usage
```python
import mbp

@mbp.apply(["x", "label"], 16) # Condition-1
def train_fn(model, criterion, x, label, *args, **kwargs):
    o = model(x)
    loss = criterion(o, label)
    return loss # Condition-2

for image, label in dataloader:
    optimizer.zero_grad()
    loss = train_fn(model, criterion, x=image, label=label) # Condition-3
    optimizer.step()
```
You can easily apply MBP to your DL training by following three conditions:

1. **Specify Tensor Names and Micro-Batch Size**: Use the `@mbp.apply` decorator to specify the tensor names to be split and the desired micro-batch size.
2. **Return Loss Without Backpropagation**: Ensure the decorated function computes and returns the loss without calling `.backward()`. The MBP decorator will handle the backpropagation automatically.
3. **Define Tensors Explicitly**: Explicitly define the corresponding tensors in `key=value` format when calling the decorated function to apply MBP. If not specified, MBP will not be applied to those tensors.

> **Note:**
> The function decorated with the MBP decorator returns the loss value for verification purposes only. Do not call `.backward()` on this loss value, as the MBP decorator will handle the backpropagation automatically. See the example below:
> ```python
> loss = train_fn(model, criterion, x=image, label=label)
> loss.backward() # Don't do this!
> ```

> **Caution:** Ensure that the specified device has enough memory to handle the micro-batches and intermediate computations.
> To apply MBP, ensure that after uploading the model to the GPU memory, there is enough remaining GPU memory capacity to allocate at least one batch size and store intermediate computations computed by each layer of the model.

MBP automatically detects whether the model is allocated to a CPU or GPU. It then splits a large mini-batch into smaller micro-batches and streams them sequentially to the detected device.
### Arguments
- `target_batch` (type: List[str] or Tuple[str]): List of tensor names to be applied MBP.
- `ub_size` (type: int): Micro-batch size.
- `device` (typ: str | int | Device, optional): Device to use for the micro-batch. (Default: "auto").

### Multi-GPU
MBP can be applied not only to single GPU systems but also to multi-GPU systems.
One of the remarkable scalability features of MBP is that it can be easily applied to PyTorch's `DataParallelism()` without any code modifications, see the example below:

```python
import torch
import torch.nn as nn
import mbp

device = torch.device("cuda")
model = nn.DataParallel(model).to(device) # Just add PyTorch's DataParallel() code!

@mbp.apply(["x", "label"], 16)
def train_fn(model, criterion, x, label, *args, **kwargs):
    o = model(x)
    loss = criterion(o, label)
    return loss
```

## Citation
```bibtex
@article{piao2023enabling,
    title={Enabling large batch size training for DNN models beyond the memory limit while maintaining performance},
    author={Piao, XinYu and Synn, DoangJoo and Park, Jooyoung and Kim, Jong-Kook},
    journal={IEEE Access},
    year={2023},
    publisher={IEEE}
}
```
