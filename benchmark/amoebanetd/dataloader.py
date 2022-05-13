from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Resize
from torchvision.datasets import ImageFolder

def get_dataset(path, dataset_type, args, is_train):
    if dataset_type == 'imagenet2012':
        return rtn_imagenet( path, args.train_batch, args.image_size )
    elif dataset_type == 'cifar10':
        print("CIFAR10!")
        return rtn_cifar10( path, args.train_batch if is_train else args.test_batch, args.image_size, is_train )
    elif dataset_type == 'cifar100':
        print("CIFAR100!")
        return rtn_cifar100( path, args.train_batch if is_train else args.test_batch, args.image_size, is_train )
    elif dataset_type == 'flower102':
        print("FLOWER!")
        return rtn_flower( path, args, is_train)

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
        num_workers=6,
        drop_last=True,
        shuffle=True,
        pin_memory=True
    )

    return imagenet_dataloader


def rtn_cifar10(path, batch_size, image_size=None, is_train=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    dataset = datasets.CIFAR10( root=path, train=is_train,
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
    dataloader = DataLoader( dataset, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=True )

    return dataloader


def rtn_cifar100(path, batch_size, image_size=None, is_train=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    dataset = datasets.CIFAR100( root=path, train=is_train,
            transform=transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    normalize
                ]
            )
    )
    dataloader = DataLoader( dataset, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=True )

    return dataloader

def rtn_flower(path, args, is_train: bool):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (args.image_size, args.image_size) ),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder( path, normalize )
    return DataLoader(
            dataset,
            batch_size  =args.train_batch if is_train else args.test_batch,
            num_workers =args.num_workers,
            shuffle     =args.shuffle,
            pin_memory  =args.pin_memory,
        )

if __name__=='__main__':
    rtn_imagenet('./dataset/imagenet2012', image_size=0, batch_size=4)
    print('imagenet OK...')
    rtn_cifar10('./dataset/cifar10', image_size=0, batch_size=4)
    print('cifar10 OK...')
    rtn_cifar100('./dataset/cifar100', image_size=0, batch_size=4)
    print('cifar100 OK...')
    print('All dataloader OK...')