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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ ==  '__main__':
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

    print(len(cifar10_datasets))
    print(len(dataloader))

    dataloader = mbs.set_dataloader(dataloader, micro_batch_size=16)
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = mbs.set_optimizer(optimizer)

    print(len(dataloader))

    for idx, (image, label) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if idx == 0:
            print(idx + 1, loss.detach(), image.size())
        else:
            print(idx + 1, loss.detach(), image.size(), end='\r')
    print(idx + 1, loss.detach(), image.size())
