import argparse
import time
import numpy as np
from contextlib import contextmanager
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from mbs.micro_batch_processing import MBP
from utils import get_voc_dataloader, carvana_dataset
from models.unet import unet_3156, unet_3356, UNet

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
    paser.add_argument('--path', type=str, default='datasets')
    paser.add_argument('--type', type=str, default='carvana')
    paser.add_argument('--image_size', type=int, default=384)
    paser.add_argument('-b', '--batch_size', type=int, default=16)
    paser.add_argument('--num_class', type=int, default=1)
    paser.add_argument('--num_workers', type=int, default=0)

    paser.add_argument('--device', type=int, default=1)

    paser.add_argument('--epochs', type=int, default=10)
    paser.add_argument('--lr', type=float, default=1e-3)
    paser.add_argument('--weight_decay', type=float, default=5e-4)
    paser.add_argument('--momentum', type=float, default=0.9)

    paser.add_argument('--method', type=str, default='default', help='default, dp, mbp')
    paser.add_argument('--micro_batch', type=int, default=1)

    return paser.parse_args()


class DiceLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', smooth: float = 1. ) -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward( self, preds: Tensor, masks: Tensor ) -> Tuple[Tensor, Tensor]:

        intersection: Tensor = ( preds * masks ).sum( dim=(2,3) )
        union: Tensor = preds.sum( dim=(2,3) ) + masks.sum( dim=(2,3) )
        dice: Tensor = ( 2 * intersection + self.smooth ) / ( union + self.smooth )
        loss: Tensor = 1 - dice
        loss += self.criterion( preds, masks )

        if self.reduction == "mean":
            self.dice = dice.mean()
            return loss.mean()
        elif self.reduction == "sum":
            self.dice = dice.sum()
            return loss.sum()

    def dice_value(self):
        return self.dice


if __name__ == "__main__":
    args = get_arguments()

    dev = torch.device(f"cuda:{args.device}" )
    model = unet_3156()

    trainset, testset = carvana_dataset(
        root=f"{args.path}/{args.type}",
        image_size=args.image_size,
    )
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )
    testloader = DataLoader(
        testset, 
        batch_size=16,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )

    criterion = DiceLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
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

            # with torch.no_grad():
            #     dices = []
            #     for _, (input, target) in enumerate(testloader):
            #         input, target = input.to(dev), target.to(dev)
            #         output = model.model(input)
            #         loss = criterion(output, target)
            #         dices.append( criterion.dice_value().item() )
            #     avg_dice = np.mean(dices)
            # print(f"[{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, IoU: {avg_dice*100:.2f}(%)")
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
                dices = []
                for _, (input, target) in enumerate(testloader):
                    input, target = input.to(dev), target.to(dev)
                    output = model(input)
                    loss = criterion(output, target)
                    dices.append( criterion.dice_value().item() )
                avg_dice = np.mean(dices)
            print(f"[{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, IoU: {avg_dice*100:.2f}(%)")
        print(f"Runtime: {np.mean(avg_runtime[1:]):.1f} ({np.std(avg_runtime[1:]):.2f}) (sec)")


