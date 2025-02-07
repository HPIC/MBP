# Micro-Batch Processing (MBP)
MBP is a simple method to enable deep learning (DL) models to train in large mini-batches, even when the devices (i.e., GPUs) have limited memory size. This GitHub repository provides an implementation of PyTorch-based MBP. You can easily train your DL models in large-batch training without the need to apply complex techniques, increase the number of GPUs, or GPU memory size.

The MBP implementation in this GitHub repository was designed with reference to the research presented in:
"[Enabling large batch size training for DNN models beyond the memory limit while maintaining performance](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10242106)"
by XinYu Piao, DoangJoo Synn, Jooyoung Park, and Jong-Kook Kim (IEEE Access 2023).

## Install
You can easily install the MBP implementation compatible with PyTorch using pip:
```bash
pip install git+https://github.com/HPIC/MBP.git
```
> **Requirements:**
> Python 3.10 or higher and a CUDA-enabled version of PyTorch are required.

## Usage
```python
import mbp

@mbp.apply(["x", "label"], 16) # Condition-1
def train_fn(model, criterion, x, label, *args, **kwargs):
    output = model(x)
    loss = criterion(output, label)
    return loss, output, ... # Condition-2

for image, label in dataloader:
    optimizer.zero_grad()
    loss, output, ... = train_fn(model, criterion, x=image, label=label) # Condition-3
    optimizer.step()
```
You can easily apply MBP to your DL training by following three conditions:

1. **Specify Tensor Names and Micro-Batch Size**: Use the `@mbp.apply` decorator to specify the tensor names to be split and the desired micro-batch size.
2. **Return Loss Without Backpropagation**: Ensure the decorated function computes and returns the loss without calling `.backward()`. The MBP decorator will handle the backpropagation automatically. Also, make sure that the loss value is the first item returned by the function decorated with the MBP decorator.
3. **Define Tensors Explicitly**: Explicitly define the corresponding tensors in `key=value` format when calling the decorated function to apply MBP. If not specified, MBP will not be applied to those tensors.

> ⚠️ **Caution:**
> The function decorated with the MBP decorator returns the loss value for verification purposes only. Do not call `.backward()` on this loss value, as the MBP decorator will handle the backpropagation automatically. See the example below:
> ```python
> loss = train_fn(model, criterion, x=image, label=label)
> loss.backward() # Don't try this!
> ```

MBP automatically detects whether the model is allocated to a CPU or GPU. It then splits a large mini-batch into smaller micro-batches and streams them sequentially to the detected device.

> ⚠️ **Caution:** Ensure that the specified device has enough memory to hold the micro-batches and intermediate computations. To apply MBP, ensure that after uploading the model to the GPU memory, there is enough remaining GPU memory capacity to allocate at least one micro-batch size and store intermediate computations computed by each layer of the model.

### Arguments
- `target_batch` (type: List[str] or Tuple[str]): List of tensor names to be applied MBP.
- `ub_size` (type: int): The maximum size of a micro-batch. This value represents the maximum size of a micro-batch that can be executed on each device, and the batch size split by MBP or other functions cannot exceed this value.
- `device` (type: str | int | Device, optional): Device to use for the micro-batch. If set to "auto", the MBP will automatically find the device where the model is stored and send the micro-batch to that device. Use only if you need to specify. (Default: "auto")

### Multi-GPU
MBP can be applied not only to single GPU systems but also to multi-GPU systems. One of the remarkable scalability features of MBP is that it can be easily applied to PyTorch's `torch.nn.DataParallel()` without any code modifications, see the example below:

```python
import torch
import torch.nn as nn
import mbp

device = torch.device("cuda")
model = nn.DataParallel(model).to(device)

# No additional arguments or code modifications are necessary.
@mbp.apply(["x", "label"], 16)
def train_fn(model, criterion, x, label, *args, **kwargs):
    output = model(x)
    loss = criterion(output, label)
    return loss
```

## Citation
If you use this GitHub repository or pip package in your research, please cite the following paper:
```bibtex
@article{piao2023enabling,
    title={Enabling large batch size training for DNN models beyond the memory limit while maintaining performance},
    author={Piao, XinYu and Synn, DoangJoo and Park, Jooyoung and Kim, Jong-Kook},
    journal={IEEE Access},
    year={2023},
    publisher={IEEE}
}
```
