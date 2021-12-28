from typing import List
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

def cifar_dataset( 
    path='./data/cifar10', image_size: int = 32, is_train: bool = False, download: bool = False
):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    dataset = datasets.CIFAR10(
        root=path,
        train=is_train,
        transform=transforms.Compose(
            [
                transforms.Resize( (image_size, image_size) ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        ),
        download=download
    )

    return dataset


if __name__=='__main__':
    ''' Download CIFAR dataset '''
    dataset = cifar_dataset(download=True)