from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
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
    mbs = MicroBatchStreaming(micro_batch_size=8)
    mnist_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])
    download_root = './'
    train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
    train_loader = mbs.set_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    for idx, (ze, up, (data0, data1)) in enumerate(train_loader):
        print(idx+1, ze, up, data0.size(), data1.size())

if __name__ == '__main__':
    test_cyclegan_dataset()
    print('Check...OK!')
    # test_mnist_dataset()

