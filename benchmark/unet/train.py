#train.py

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import model
import dataset
from utils import get_test_dataloader, get_training_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='select gpu device')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    _device = 'cuda:' + str(args.gpu_device)
    device = torch.device(_device)
    net = model.UNet(args, device)

    carvana_training_loader = get_training_dataloader(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        batch_size=args.b,
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )
    carvana_test_loader = get_test_dataloader(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        batch_size=args.b,
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )

    carvana_training_loader_dataset_size = len(carvana_training_loader.dataset)
    carvana_test_loader_dataset_size = len(carvana_test_loader.dataset)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)