import argparse
import time
import numpy as np
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim

from mbs.micro_batch_processing import MBP
from utils import get_dataloader
from models import resnet50, resnet101
from models.amoebanet import amoebanetd

@contextmanager
def measure_time(method_name: str, runtimes: list):
    stime = time.perf_counter()
    yield
    etime = time.perf_counter()
    runtime = etime - stime
    runtimes.append(runtime)
    print(f"[{method_name}] Time: {runtime:.1f} (sec)")


def get_arguments():
    paser = argparse.ArgumentParser()
    # Dataset
    paser.add_argument('--path', type=str, default='./datasets')
    paser.add_argument('--type', type=str, default='flower')
    paser.add_argument('--image_size', type=int, default=224)
    paser.add_argument('-b', '--batch_size', type=int, default=16)
    paser.add_argument('--num_class', type=int, default=102)
    paser.add_argument('--num_workers', type=int, default=0)

    paser.add_argument('--version', type=int, default=50)
    paser.add_argument('--device', type=int, default=0)

    paser.add_argument('--epochs', type=int, default=10)
    paser.add_argument('--lr', type=float, default=0.1)
    paser.add_argument('--weight_decay', type=float, default=1e-4)
    paser.add_argument('--momentum', type=float, default=0.9)

    paser.add_argument('--method', type=str, default='default', help='default, dp, mbp')
    paser.add_argument('--micro_batch', type=int, default=1)

    return paser.parse_args()


class DatasetConfig:
    def __init__(
        self, 
        path, 
        type, 
        image_size, 
        batch_size, 
        num_class,
        num_workers,
        is_train=False,
    ):
        # Dataset
        self.type = f"{type}{num_class}"
        if is_train:
            self.path = f"{path}/{self.type}/train"
        else:
            self.path = f"{path}/{self.type}/valid"
        self.image_size = image_size
        self.num_class = num_class

        # DataLoader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = True
        self.shuffle = True


if __name__ == "__main__":
    args = get_arguments()

    dev = torch.device(f"cuda:{args.device}" )
    if args.version == 50:
        model = resnet50(args.num_class)
    elif args.version == 101:
        model = resnet101(args.num_class)
    elif args.version == 0:
        model = amoebanetd(
            num_classes=args.num_class,
            num_layers=6,
            num_filters=190
        )

    trainloader = get_dataloader(
        DatasetConfig(
            path=args.path, type=args.type,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_class=args.num_class,
            num_workers=args.num_workers,
            is_train=True,
        )
    )
    validloader = get_dataloader(
        DatasetConfig(
            path=args.path, type=args.type,
            image_size=args.image_size,
            batch_size=16,
            num_class=args.num_class,
            num_workers=args.num_workers,
            is_train=False,
        )
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    model = model.to(dev)
    if args.method == "dp":
        available_device = [0, 1]
        model = nn.DataParallel(model, device_ids=available_device)
    elif args.method == "mbp":
        model = MBP.hard(
            model=model, 
            loss_fn=criterion, 
            dataloader=trainloader,
            optimizer=optimizer,
            micro_batch_size=args.micro_batch
        )
    elif args.method == "mbp+dp":
        available_device = [0, 1]
        model = MBP.hard(
            model=model, 
            loss_fn=criterion, 
            dataloader=trainloader,
            optimizer=optimizer,
            micro_batch_size=args.micro_batch,
            dp=True,
            device_ids=available_device
        )

    avg_runtime = []
    if args.method == "mbp" or args.method == "mbp+dp":
        for epoch in range(args.epochs):
            with measure_time("MBP", avg_runtime):
                losses, avg_loss = model.train()

            with torch.no_grad():
                total = 0
                correct_top1 = 0
                for _, (input, target) in enumerate(validloader):
                    input, target = input.to(dev), target.to(dev)
                    output = model.model(input)
                    # rank 1
                    _, pred = torch.max(output, 1)
                    total += target.size(0)
                    correct_top1 += (pred == target).sum().item()
                acc = 100 * (correct_top1 / total)
            print(f"[{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, Acc: {acc:.2f}(%)")
        print(f"Runtime: {np.mean(avg_runtime[1:]):.1f} ({np.std(avg_runtime[1:]):.2f}) (sec)")
    else:
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            avg_loss = []
            with measure_time(args.method, avg_runtime):
                for i, (input, target) in enumerate(trainloader):
                    input, target = input.to(dev), target.to(dev)

                    output = model(input)
                    loss = criterion(output, target)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    avg_loss.append(loss.item())
            avg_loss = np.mean(avg_loss)

            with torch.no_grad():
                total = 0
                correct_top1 = 0
                for _, (input, target) in enumerate(validloader):
                    input, target = input.to(dev), target.to(dev)
                    output = model(input)
                    # rank 1
                    _, pred = torch.max(output, 1)
                    total += target.size(0)
                    correct_top1 += (pred == target).sum().item()
                acc = 100 * (correct_top1 / total)
            print(f"[{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, Acc: {acc:.2f}(%)")
        print(f"Runtime: {np.mean(avg_runtime[1:]):.1f} ({np.std(avg_runtime[1:]):.2f}) (sec)")


