from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import sys
import math
sys.path.append('../')

from benchmark.cyclegan.dataloader import cyclegan_dataset
from mbs.micro_batch_streaming import MicroBatchStreaming

def test_cyclegan_dataset():
    batch_size = 128
    mbs = MicroBatchStreaming()
    dataloader = cyclegan_dataset('../benchmark/cyclegan/dataset/horse2zebra/train', image_size=256, batch_size=batch_size)
    dataloader = mbs.set_dataloader(dataloader, 4)

    for idx, (ze, up, (data0, data1, _, _)) in enumerate(dataloader):
        print(idx+1, ze, up, data0.size(), data1.size())

def test_mnist_dataset():
    batch_size = 64
    mbs = MicroBatchStreaming()
    mnist_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])
    download_root = './'
    train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
    train_loader = mbs.set_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    for idx, (ze, up, (data0, data1)) in enumerate(train_loader):
        print(idx+1, ze, up, data0.size(), data1.size())

def test_cifar10_dataset():
    batch_size = 64
    mbs = MicroBatchStreaming()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    cifar10_datasets = datasets.CIFAR10(
            root='../../common_dataset/cifar10',
            train=True,
            transform=transforms.Compose(
                [
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
        pin_memory=True
    )

    dataloader = mbs.set_dataloader(dataloader, micro_batch_size=16)
    # for idx, (_, up, (data0, data1)) in enumerate(dataloader):
    #     print(idx+1, data0.size(), data1.size())
    #     if up:
    #         break

    for idx, (data0, data1) in enumerate(dataloader):
        print(idx+1, data0.size(), data1.size())
        if idx == 7:
            break

if __name__ == '__main__':
    # test_cyclegan_dataset()
    print('Check...OK!')
    # test_mnist_dataset()
    print('Check...OK!')
    test_cifar10_dataset()
    print('Check...OK!')

