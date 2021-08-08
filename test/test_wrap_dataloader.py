from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import sys
sys.path.append('../')

from benchmark.data.dataset import cyclegan_dataset
from mbs.wrap_dataloader import streaming_dataloader

def test_cyclegan_dataset():
    dataloader = cyclegan_dataset('../benchmark/data/horse2zebra/train', image_size=256, batch_size=128)

    for idx, data in enumerate(dataloader):
        for i, (d0, d1) in enumerate(streaming_dataloader( (data['A'], data['B']), mini_batch_size=128, micro_batch_size=4)):
            pass
        print( f' mini-batch size : {(i+1) * len(d0)}' )
        print( f' micro-batch size : {len(d0)}' )
        print( f' num of streaming : {(i+1)}')
        break

def test_mnist_dataset():
    download_root = '../benchmark/data/MNIST_DATASET'
    mnist_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])
    dataloader = MNIST(download_root, transform=mnist_transform, train=True, download=True)
    dataloader = DataLoader(dataset=dataloader, batch_size=64, shuffle=True)

    for idx, data in enumerate(dataloader):
        for i, (d0, d1) in enumerate(streaming_dataloader( (data[0], data[1]), mini_batch_size=64, micro_batch_size=4)):
            pass
        print( f' mini-batch size : {(i+1) * len(d0)}' )
        print( f' micro-batch size : {len(d0)}' )
        print( f' num of streaming : {(i+1)}')
        break

if __name__ == '__main__':
    test_cyclegan_dataset()
    print('Check...OK!')
    test_mnist_dataset()
    print('Check...OK!')

