from typing import List, Tuple, Union
from torch import Tensor

from torch.optim.optimizer import Optimizer
from torch.nn import Module

__all__ = [
    "TorchLossType",
    "MultiTensors",
    "ChunkTensor",
    "ChunkTensors",
    "ChunkTensororTensors",
]

# PyTorch Loss-type Define
TorchLossType = Module

# PyTorch Tensors-type Define
MultiTensors = Tuple[ Tensor ]
ChunkTensor = MultiTensors
ChunkTensors = List[ MultiTensors ]
ChunkTensororTensors = Union[ ChunkTensor, ChunkTensors ]

