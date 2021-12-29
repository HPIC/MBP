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
    parser.add_argument( '-b', '--batch', type=int, default=256)
    parser.add_argument( '-i', '--image', type=int, default=32)
    return parser.parse_args()


class simpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.conv2 = nn.Conv2d(6, 6, 5)
        # self.conv3 = nn.Conv2d(6, 6, 5)
        # self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(6, 16, 5)
        # self.conv5 = nn.Conv2d(16, 16, 5)
        # self.conv6 = nn.Conv2d(16, 16, 5)
        # self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.bn1(x)
        x = self.pool(F.relu(x))

        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
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

    train_dataset = cifar_dataset(image_size=args.image, is_train=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch,
        num_workers=6, pin_memory=True,
        shuffle=True
    )

    input: Tensor
    label: Tensor
    output: Tensor
    loss: Tensor

    train_times = []
    data_load_times = []
    data_send_times = []
    iterations = 0
    
    # start = perf_counter()
    data_load_start = perf_counter()
    for didx, (input, label) in enumerate(train_dataloader):
        data_load_end = perf_counter()
        data_load_times.append( data_load_end - data_load_start )
        # data_start = perf_counter()

        input = input.to(gpu)
        label = label.to(gpu)

        # data_end = perf_counter()
        # data_send_times.append( data_end - data_start )

        # train_start = perf_counter()

        output = net( input )
        loss = criterion( output, label )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_end = perf_counter()
        # train_times.append( train_end - train_start )
        
        # iterations = didx + 1

        data_load_start = perf_counter()
    # end = perf_counter()

    # print(f"train time: {end - start:.2f}")
    # print(f"model training time: {sum(train_times)/len(train_times)}")
    # print(f"num of iterations: {iterations}")

    # print(f"data sending time: {sum(data_send_times)/len(data_send_times)}")
    print(f"data loading time: {sum(data_load_times)/len(data_load_times)}")

