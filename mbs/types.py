from typing import List, Tuple, Union

import torch

__all__ = [
    "ModelType",
    "ModelList",
    "ModelTuple",
    "DatasetType",
    "DatasetList",
    "OptimType",
]

# Model related custom types
ModelType = Union[torch.nn.Module, torch.nn.Sequential]
ModelList = List[ModelType]
ModelTuple = Tuple[ModelType]

# Dataset related custom types
DatasetType = torch.Tensor
DatasetList = Tuple[DatasetType]

# Optimizer related custom types
OptimType = torch.optim