from typing import Callable, Optional
from torch.utils.data import DataLoader

from .dataset import get_dataset


def get_dataloader(
    root: str,
    train: bool = True,
    download: bool = False,
    transform: Optional[Callable] = None,
    **kwargs
):
    dataset = get_dataset(root, train, download, transform)
    return DataLoader(dataset=dataset, **kwargs)


if __name__ == "__main__":
    trainloader = get_dataloader("../dataset/voc2012", train=True, batch_size=2, pin_memory=True)
    for i, (d0, d1) in enumerate(trainloader):
        print(d0.shape, d1.shape)
        break