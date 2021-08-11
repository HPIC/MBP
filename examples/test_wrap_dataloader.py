from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import sys
import math
sys.path.append('../')

from benchmark.data.dataset import cyclegan_dataset
from mbs.micro_batch_streaming import MicroBatchStreaming

def test_cyclegan_dataset():
    batch_size = 128
    mbs = MicroBatchStreaming(micro_batch_size=4)
    dataloader = cyclegan_dataset('../benchmark/data/horse2zebra/train', image_size=256, batch_size=batch_size)
    dataloader = mbs.dataloader(dataloader)

    for idx, (up, data0, data1) in enumerate(dataloader):
        print(idx+1)
        print(up, data0.size(), data1.size())

def test_mnist_dataset():
    pass

if __name__ == '__main__':
    test_cyclegan_dataset()
    print('Check...OK!')
    # test_mnist_dataset()
    # print('Check...OK!')

