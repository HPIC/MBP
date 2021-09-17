import sys
sys.path.append('../')

from mbs.micro_batch_streaming import MicroBatchStreaming

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

'''
    Test code to detect NaN (Why does NaN occur?).
    ---
    - 
'''


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_dataloader(batch_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_datasets = datasets.CIFAR100(
            root='../../common_dataset/cifar100',
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
    train_dataloader = DataLoader(
        train_datasets,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_datasets = datasets.CIFAR100(
            root='../../common_dataset/cifar100',
            train=False,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize
                ]
            )
    )
    test_dataloader = DataLoader(
        test_datasets,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return train_dataloader, train_datasets, test_dataloader, test_datasets


if __name__ ==  '__main__':
    batch_size = 128
    train_dataloader, train_datasets, test_dataloader, test_datasets = get_dataloader(batch_size=batch_size)

    print('dataset : Cifar100')

    print("-----")
    print('total train dataset size :', len(train_datasets))
    print('trian train dataloader size', len(train_dataloader))
    print('total test dataset size :', len(test_datasets))
    print('trian test dataloader size', len(test_dataloader))

    dev = torch.device('cuda:0')

    model = Net().to(dev)
    criterion = nn.CrossEntropyLoss().to(dev)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    mbs = MicroBatchStreaming()
    train_dataloader = mbs.set_dataloader(train_dataloader, micro_batch_size=16)
    optimizer = mbs.set_optimizer(optimizer)
    criterion = mbs.set_loss(criterion)

    print("-----")
    print("MBS train dataloader size :", train_dataloader.micro_len())
    print("-----")

    cur = 0

    for epoch in range(110):
        for idx, (image, label) in enumerate(train_dataloader):
            image = image.to(dev)
            label = label.to(dev)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        #     ''' Check dataloader '''
        #     print(idx+1, loss.item(), image.size(), label.size())
        # print(idx+1, loss.item(), image.size(), label.size())

            ''' Check update error '''
            print(f"{epoch+1}, loss : {loss.detach().item()}, {image.size()}, {label.size()}", end="\r")
        print(f"{epoch+1}, [{idx+1}] loss : {loss.detach().item()}")
