from typing import Callable, Optional
from torch.utils.data import DataLoader

from .dataset import carvana_dataset, get_vocdataset, get_voc_dataset


def get_dataloader(
    root: str,
    image_size: int,
    train_batch: int,
    test_batch: int,
    **kwargs
):
    # dataset = get_dataset(
    #     root=root, 
    #     download=download,
    #     image_size=image_size, 
    #     train=train, 
    #     transform=transform
    # )
    train_dataset, test_dataset = carvana_dataset( root=root, image_size=image_size )
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, **kwargs)

    return train_loader, test_loader


def get_vocloader(
    root: str,
    image_size: int,
    train_batch: int,
    test_batch: int,
    **kwargs
):
    train_dataset, test_dataset = get_vocdataset( image_size=image_size, root=root )
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, **kwargs)

    return train_loader, test_loader


def get_voc_dataloader(
    root: str,
    image_size: int,
    train: bool,
    **kwargs
):
    dataset = get_voc_dataset(root=root, image_size=image_size, train=train)
    return DataLoader(dataset=dataset, **kwargs)


if __name__ == "__main__":
    trainloader = get_dataloader("../dataset/voc2012", train=True, batch_size=2, pin_memory=True)
    for i, (d0, d1) in enumerate(trainloader):
        print(d0.shape, d1.shape)
        if i > 1:
            break