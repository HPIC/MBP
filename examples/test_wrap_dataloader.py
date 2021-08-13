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
    mbs = MicroBatchStreaming(micro_batch_size=4)
    dataloader = cyclegan_dataset('../benchmark/cyclegan/dataset/horse2zebra/train', image_size=256, batch_size=batch_size)
    dataloader = mbs.set_dataloader(dataloader)

    for idx, (ze, up, (data0, data1, _, _)) in enumerate(dataloader):
        print(idx+1, ze, up, data0.size(), data1.size())

if __name__ == '__main__':
    test_cyclegan_dataset()
    print('Check...OK!')

