from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_dataset(path, dataset_type, config, args, is_train):
    if dataset_type == 'imagenet2012':
        print('imagenet')
        return rtn_imagenet( path, config, args, is_train )
    elif dataset_type == 'cifar10':
        return rtn_cifar10( path, config, args, is_train )
    elif dataset_type == 'cifar100':
        return rtn_cifar100( path, config, args, is_train )

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
                transforms.RandomResizedCrop(config.data.dataset.train.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.dataset.train.batch_size,
        num_workers=config.data.dataset.train.num_worker,
        shuffle=config.data.dataset.train.shuffle,
        pin_memory=config.data.dataset.train.pin_memory,
        prefetch_factor=config.data.dataset.train.prefetch,
    )

    return dataloader


def rtn_cifar10(path, config, args, is_train=True):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    dataset = datasets.CIFAR10(
        root=path,
        train=is_train,
        transform=transforms.Compose(
            [
                transforms.Resize( (config.data.dataset.train.image_size, config.data.dataset.train.image_size) ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(config.data.dataset.train.image_size, 4),
                transforms.ToTensor(),
                normalize
            ]
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.data.dataset.train.batch_size,
        num_workers=config.data.dataset.train.num_worker,
        shuffle=config.data.dataset.train.shuffle,
        pin_memory=config.data.dataset.train.pin_memory,
        prefetch_factor=config.data.dataset.train.prefetch,
    )

    return dataloader


def rtn_cifar100(path, config, args, is_train=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in[0.507, 0.487, 0.441]],
        std=[x / 255.0 for x in [0.267, 0.256, 0.276]]
    )

    if is_train:
        dataset = datasets.CIFAR100(
            root=path,
            train=is_train,
            transform=transforms.Compose(
                [
                    transforms.Resize( (config.data.dataset.train.image_size, config.data.dataset.train.image_size) ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(config.data.dataset.train.image_size, 4),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.dataset.train.batch_size,
            num_workers=config.data.dataset.train.num_worker,
            shuffle=True,
            pin_memory=True,
        )
    else:
        dataset = datasets.CIFAR100(
            root=path,
            train=is_train,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize
                ]
            )
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.dataset.train.batch_size,
            num_workers=config.data.dataset.train.num_worker,
            shuffle=True,
            pin_memory=True,
        )

    return dataloader

