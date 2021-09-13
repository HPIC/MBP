from typing import List, Tuple, Union

from torch.optim.optimizer import Optimizer
from mbs.wrap_dataloader import MBSDataloader
from mbs.wrap_optimizer import MBSOptimizer

__all__ = [
    "TorchSingleOptimizer",
    "TorchMultiOptimizer",
    "TorchOptimizer",
    "MBSSingleOptimizer",
    "MBSOptimizers",
    "MBSDataloaders",
]

# PyTorch Optimizer related custom types
TorchSingleOptimizer = Optimizer
TorchMultiOptimizer = List[ Optimizer ]
TorchOptimizer = Union[ TorchSingleOptimizer, TorchMultiOptimizer ]

# MBS optimizer related custom types
MBSSingleOptimizer = MBSOptimizer
MBSOptimizers = List[ MBSOptimizer ]

# MBS dataloaders related custom types
MBSDataloaders = Union[ MBSDataloader, List[MBSDataloader] ]