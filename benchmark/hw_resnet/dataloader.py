from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os
import numpy as np
from PIL import Image

def get_dataset(path, dataset_type, args, is_train):
    # if dataset_type == 'imagenet2012':
    #     return rtn_imagenet( path, config, args, is_train )
    # elif dataset_type == 'cifar10':
    #     return rtn_cifar10( path, config, args, is_train )
    # elif dataset_type == 'cifar100':
    #     return rtn_cifar100( path, config, args, is_train )
    if dataset_type == 'cifar10':
        return rtn_cifar10(path, args, is_train)
    elif dataset_type == 'cifar100':
        return rtn_cifar100(path, args, is_train)
    elif dataset_type == 'flower102':
        return rtn_flower( path, args)

def rtn_imagenet( path, config, args, is_train: bool = True):
    if is_train:
        path = path + '/train'
    else:
        path = path + '/val'

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    dataset = datasets.ImageFolder(
        path,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
    )

    return dataset


def rtn_cifar10(path, args, is_train=True):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    dataset = datasets.CIFAR10(
        root=path,
        train=is_train,
        transform=transforms.Compose(
            [
                transforms.Resize( (args.image_size, args.image_size) ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(args.image_size, 4),
                transforms.ToTensor(),
                normalize
            ]
        )
    )

    return dataset


def rtn_cifar100(path, args, is_train=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in[0.507, 0.487, 0.441]],
        std=[x / 255.0 for x in [0.267, 0.256, 0.276]]
    )
    dataset = datasets.CIFAR100(
        root=path,
        train=is_train,
        transform=transforms.Compose(
            [
                transforms.Resize( (args.image_size, args.image_size) ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(args.image_size, 4),
                transforms.ToTensor(),
                normalize
            ]
        )
    )
    return dataset


def rtn_flower(path, args):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (args.image_size, args.image_size) ),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return ImageFolder( path, normalize )


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
