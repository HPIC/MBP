import argparse
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from mbs.micro_batch_streaming import MicroBatchStreaming
from mbs.wrap_model import MBSBatchNorm

'''
    Test code to detect NaN (Why does NaN occur?).
    ---
    - 
'''


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.conv3 = nn.Conv2d(6, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(6, 16, 5)
        self.conv5 = nn.Conv2d(16, 16, 5)
        self.conv6 = nn.Conv2d(16, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)


    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.bn1(x)
        x = self.pool(F.relu(x))

        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = self.bn2(x)
        x = self.pool(F.relu(x))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def check(model: Union[nn.Module, nn.Sequential], printing: bool):
    for name, layer in model.named_children():
        assert isinstance(layer, (nn.Module, MBSBatchNorm)), (name, layer)
        if printing:
            if isinstance(layer, MBSBatchNorm):
                print(f"   ({name}): {layer} <- MBS::BatchNorm")
            else:
                print(f"   ({name}): {layer}")

    print("None Error")


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Accumulate test')
    parser.add_argument('-p', '--print', type=bool, default=False)

    args = parser.parse_args()

    mbs = MicroBatchStreaming()
    model = Net()
    model = mbs.set_batch_norm(model)

    check(model, args.print)

