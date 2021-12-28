import torch
from torch import optim
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import argparse
import random
from time import perf_counter

from utils import cifar_dataset
from mbs import MicroBatchStreaming

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--mbs', action='store_true', default=False )
    parser.add_argument( '-s', '--seed', type=int, default=1000)
    return parser.parse_args()


class simpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.conv3 = nn.Conv2d(6, 6, 5)
        # self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(6, 16, 5)
        self.conv5 = nn.Conv2d(16, 16, 5)
        self.conv6 = nn.Conv2d(16, 16, 5)
        # self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.bn1(x)
        x = self.pool(F.relu(x))

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.bn2(x)
        x = self.pool(F.relu(x))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__=="__main__":
    ''' Get user arguments '''
    args = config_parser()

    ''' Static arguments '''
    epochs = 1

    ''' Setup Random Seed '''
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

    ''' Setup Model, Dataloader, Criterion, and Optimizer '''
    gpu = torch.device("cuda:0")
    net = simpleNet().to(gpu)
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = optim.Adam( net.parameters(), lr=0.01 )

    train_dataset = cifar_dataset(is_train=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=4,
        num_workers=6, pin_memory=True,
        shuffle=True
    )

    ''' Train '''
    if args.mbs:
        trainer, net = MicroBatchStreaming(
            dataloader=train_dataloader,
            model=net,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=4,
            micro_batch_size=2,
        ).get_trainer()
        trainer.train()
    else:
        input: Tensor
        label: Tensor
        output: Tensor
        loss: Tensor

        for didx, (input, label) in enumerate(train_dataloader):
            input = input.to(gpu)
            label = label.to(gpu)

            output = net( input )
            loss = criterion( output, label )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for name, para in net.named_parameters():
        print(name, para.data)

