from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Resize

def get_dataset(path, dataset_type, batch_size, image_size):
    if dataset_type == 'imagenet2012':
        return rtn_imagenet( path, batch_size, image_size )
    elif dataset_type == 'cifar10':
        return rtn_cifar10( path, batch_size, image_size )
    elif dataset_type == 'cifar100':
        return rtn_cifar100( path, batch_size, image_size )

def rtn_imagenet( path, batch_size, image_size):
    post_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    imagenet_dataset = datasets.ImageNet(
        root=path,
        split='train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            post_transforms,
        ])
    )

    # imagenet_dataset = ImageFolder(
    #     path,
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(image_size),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     ),
    # )

    imagenet_dataloader = DataLoader(
        imagenet_dataset,
        batch_size=batch_size,
        num_workers=16,
        drop_last=True,
        shuffle=True,
        pin_memory=True
    )

    return imagenet_dataloader


def rtn_cifar10(path, batch_size, image_size=32):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    cifar10_datasets = datasets.CIFAR10(
            root=path,
            train=True,
            transform=transforms.Compose(
                [
                    # transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize
                ]
            )
    )
    dataloader = DataLoader(
        cifar10_datasets,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return dataloader


def rtn_cifar100(path, batch_size, image_size=32):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    cifar10_datasets = datasets.CIFAR100(
            root=path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize
                ]
            )
    )
    dataloader = DataLoader(
        cifar10_datasets,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    return dataloader


if __name__=='__main__':
    rtn_imagenet('./dataset/imagenet2012', image_size=0, batch_size=4)
    print('imagenet OK...')
    rtn_cifar10('./dataset/cifar10', image_size=0, batch_size=4)
    print('cifar10 OK...')
    rtn_cifar100('./dataset/cifar100', image_size=0, batch_size=4)
    print('cifar100 OK...')
    print('All dataloader OK...')