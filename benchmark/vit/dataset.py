from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import numpy as np


def get_dataset(
    cls: int, 
    path: str, 
    is_train: bool, 
    image_size: int,
    transform: transforms = None
):
    if cls == 10:
        return cifar10(path, is_train, image_size)
    elif cls == 100:
        return cifar100(path, is_train, image_size)
    elif cls == 102:
        ''' Flower dataset '''
        normalize = None
        if transform != None:
            normalize = transform
        else:
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize( (image_size, image_size) ),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return FlowerDataset( path, normalize )


def cifar10(
    path: str, is_train: bool, image_size: int,
    transform: transforms = None
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
        )
    )

    return dataset


def cifar100(
    path: str, is_train: bool, image_size: int,
    transform: transforms = None
):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    dataset = datasets.CIFAR100(
        root=path,
        train=is_train,
        transform=transforms.Compose(
            [
                transforms.Resize( (image_size, image_size) ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
    )

    return dataset


class FlowerDataset(Dataset):
    def __init__(self, path, transform=None) -> None:
        super().__init__()
        self.path = path
        self.files = []
        for (dirpath, _, filenames) in os.walk(self.path):
            for f in filenames:
                if f.endswith('.jpg'):
                    p = {}
                    p['image_path'] = dirpath + '/' + f
                    self.files.append(p)
        if transform == None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path: str = self.files[idx]['image_path']
        image_label = img_path.split('/')[-2]
        image = np.array( Image.open( img_path ).convert('RGB') )

        # image_label = self.transform(image_label)
        image_label = int(image_label) - 1
        image = self.transform(image)

        return image, image_label

