import os
import argparse
import random
from sys import path
import time
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from collections import OrderedDict

from torch.profiler import profile, ProfilerActivity

import plotext as plx
import numpy as np

from model import ViT
from vgg import select_model
from dataset import get_dataset
from utils import (
    save_parameters,
    load_parameters,
)
''' Micro-Batch Streaming '''
from mbs import MicroBatchStreaming

''' Monitoring Board '''
import wandb
from conf import settings


def train_profile(
    epoch: int, total_epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: _Loss,
    optimizer: Optimizer,
):
    loss: Tensor
    train_losses = []
    dataloader_len = len(dataloader)

    start = time.perf_counter()
    model.train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs: Tensor = inputs.to(device)
            labels: Tensor = labels.to(device)

            optimizer.zero_grad()
            preds: Tensor = model(inputs)
            loss = criterion(preds, labels)

            # train_losses.append( loss.item() )

            loss.backward()
            optimizer.step()

            if idx > 1:
                break
    finish = time.perf_counter()
    prof: profile
    prof.export_chrome_trace("./profiling_origin.json")
    print('\nepoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return (sum(train_losses)/len(train_losses)), (finish - start)

def train(
    epoch: int, total_epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: _Loss,
    optimizer: Optimizer,
):
    loss: Tensor
    train_losses = []
    dataloader_len = len(dataloader)

    start = time.perf_counter()
    model.train()
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs: Tensor = inputs.to(device)
        labels: Tensor = labels.to(device)

        optimizer.zero_grad()
        preds: Tensor = model(inputs)
        # print(preds.data, labels.data)
        loss = criterion(preds, labels)

        train_losses.append( loss.item() )

        loss.backward()
        optimizer.step()

        print(
            f"[{epoch}/{total_epoch}][{idx+1}/{dataloader_len}]  ",
            f"train loss: {(sum(train_losses)/len(train_losses)):.3f}  ",
            f"image size: {inputs.size()}  ",
            end='\r'
        )
    finish = time.perf_counter()
    print('\nepoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return (sum(train_losses)/len(train_losses)), (finish - start)


def eval_training(
    epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: nn.Module,
):
    total = 0
    correct_top1 = 0
    correct_top5 = 0
    model.eval()
    with torch.no_grad():
        for (inputs, labels) in dataloader:
            inputs: Tensor = inputs.to(device)
            labels: Tensor = labels.to(device)
            outputs: Tensor = model(inputs)

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()

            # rank 5
            _, rank5 = outputs.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

            for k in range(6):
                correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_top5 += correct_k.item()

    print(
        f"top-1: {100 * ( correct_top1 / total ):3f}  ",
        f"top-5: {100 * ( correct_top5 / total ):3f}  ",
    )
    return (100 * ( correct_top1 / total )), (100 * ( correct_top5 / total ))


def validation(
    device,
    model: nn.Module,
    dataloader,
    criterion
):
    corrects = 0.
    valid_losses = []
    total = 0
    model.eval()
    with torch.no_grad():
        for (inputs, labels) in dataloader:
            inputs: Tensor = inputs.to(device)
            labels: Tensor = labels.to(device)
            outputs: Tensor = model(inputs)

            loss: Tensor = criterion(outputs, labels)
            valid_losses.append( loss.item() )

            # outputs = torch.exp(outputs)
            # equals = (labels.data == outputs.max(dim=1)[1])
            # # print(labels.data, outputs.data, equals.data)
            # corrects += equals.float().mean().item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (preds == labels).sum().item()

    acc = 100 * corrects / len(dataloader)
    valid_loss = sum(valid_losses) / len(valid_losses)
    print(
            f"valid loss : {valid_loss:.2f}",
            f"acc: {acc:.2f} %"
         )
    return valid_loss, acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='select gpu device')

    ''' Dataloader, Optimizer etc. Arguments '''
    parser.add_argument('-b', type=int, default=8, help='batch size for train dataloader')
    parser.add_argument('-tb', type=int, default=1, help='batch size for test dataloader')
    parser.add_argument('-i', type=int, default=256)
    parser.add_argument('-patch_size', type=int, default=32)
    parser.add_argument('-ncls', type=int, default=10)
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')

    ''' Micro-Batch Streaming Arguments '''
    parser.add_argument('-mbs', action='store_true', default=False, help='training model with MBS')
    parser.add_argument('-mbs_bn', action='store_true', default=False, help='training model with MBS-BN')
    parser.add_argument('-usize', type=int, default=0, help='setting micro-batch size')

    ''' WandB Arguments '''
    parser.add_argument('-wandb', action='store_true', default=False, help='Using Wandb for monitoring')
    parser.add_argument('-para', action='store_true', default=False, help='training model with MBS')

    args = parser.parse_args()
    _device = 'cuda:' + str(args.gpu_device)
    para_path = 'checkpoint/'
    name = None

    if args.mbs:
        name = f'vit(w/ MBS)'
        para_path += f'mbs/mbs_{args.b}_{args.i}_{args.patch_size}.pth'
    elif args.mbs_bn:
        name = f'vit(w/ MBS-BN)'
        para_path += f'bn/bn_{args.b}_{args.i}_{args.patch_size}.pth'
    else:
        name = f'vit(w/ baseline)'
        para_path += f'baseline/baseline_{args.b}_{args.i}_{args.patch_size}.pth'

    ''' Setup WandB '''
    if args.wandb:
        tags = []

        if args.mbs:
            tags.append( f'usize {args.usize}' )
        elif args.mbs_bn:
            tags.append( f'usize {args.usize}' )

        tags.append( f'batch {args.b}' )
        tags.append( f'image {args.i}' )
        tags.append( f'patch {args.patch_size}' )
        if args.ncls == 102:
            tags.append( f'flower102' )
        else:
            tags.append( f'cifar{args.ncls}')

        wandb.init(
            project='mbs_paper_results',
            entity='xypiao97',
            name=f'{name}',
            tags=tags,
        )

    random_seed = 1000000

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(random_seed)

    device = torch.device(_device)
    if args.ncls == 102:
        train_dataset_path = f'./data/flower102/train'
        test_dataset_path = f'./data/flower102/valid'
#        train_dataset = get_dataset( 
#                cls=args.ncls, 
#                path=train_dataset_path,
#                is_train=True,
#                image_size=args.i,
#            )
#        test_dataset = get_dataset( 
#                cls=args.ncls, 
#                path=test_dataset_path,
#                is_train=False,
#                image_size=args.i,
#            )
        normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] ),
                transforms.Resize( (args.i, args.i) ),
                ])
        train_dataset = ImageFolder(train_dataset_path, normalize)
        test_dataset = ImageFolder(test_dataset_path, normalize)
    else:
        dataset_path = f'./data/cifar{args.ncls}'
        train_dataset = get_dataset( 
                cls=args.ncls, 
                path=dataset_path,
                is_train=True,
                image_size=args.i,
            )
        test_dataset = get_dataset( 
                cls=args.ncls, 
                path=dataset_path,
                is_train=False,
                image_size=args.i,
            )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.b,
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.tb,
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )

#    net = ViT(
#            image_size=args.i,
#            patch_size=args.patch_size,
#            num_classes=args.ncls,
#            dim=1024,
#            depth=6,
#            heads=16,
#            mlp_dim=2048,
#            dropout=0.1,
#            emb_dropout=0.1
#        ).to(device)
    net = select_model( False, 16, args.ncls )
    # for params in net.parameters():
    #     params.requries_grad = False

    # new_classifier = nn.Sequential(OrderedDict([
    #         ('fc1', nn.Linear(25088, 4096)),
    #         ('relu', nn.ReLU()),
    #         ('drop', nn.Dropout(p = 0.5)),
    #         ('fc2', nn.Linear(4096, args.ncls)),
    #         ('output', nn.LogSoftmax(dim = 1))
    #     ]))
    # net.classifier = new_classifier
    net = net.to(device)
    loss_function = CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if (args.mbs or args.mbs_bn):
        mbs_trainer, net = MicroBatchStreaming(
            dataloader=train_dataloader,
            model=net,
            criterion=loss_function,
            optimizer=optimizer,
            lr_scheduler=None,
            warmup_factor=None,
            device_index=args.gpu_device,
            batch_size=args.b,
            micro_batch_size=args.usize,
            bn_factor=args.mbs_bn
        ).get_trainer()

    ''' Training model '''
    if (args.mbs or args.mbs_bn):
        for epoch in range(1, settings.EPOCH + 1):
            print(epoch, "/", settings.EPOCH)
            start = time.perf_counter()
            mbs_trainer.train()
            finish = time.perf_counter()
            print('epoch time', finish - start, ' s')
            top1, top5 = eval_training(
                epoch=epoch,
                device=device,
                dataloader=test_dataloader,
                model=net,
            )
            if args.wandb:
                wandb.log( {'train time': finish - start}, step=epoch)
                wandb.log( {'train loss': mbs_trainer.get_loss()}, step=epoch )
                wandb.log( {'top1': top1}, step=epoch )
                wandb.log( {'top5': top5}, step=epoch )
            save_parameters( net, path=para_path )
    else:
        for epoch in range(1, settings.EPOCH + 1):
            train_loss, train_time = train(
                epoch=epoch, total_epoch=settings.EPOCH,
                device=device,
                dataloader=train_dataloader,
                model=net,
                criterion=loss_function,
                optimizer=optimizer
            )
           # top1, top5 = eval_training(
           #     epoch=epoch,
           #     device=device,
           #     dataloader=test_dataloader,
           #     model=net,
           # )
            valid_loss, acc = validation(
                device=device,
                model=net,
                dataloader=test_dataloader,
                criterion=loss_function,
            )
            if args.wandb:
                wandb.log( {'train time': train_time}, step=epoch)
                wandb.log( {'train loss': train_loss}, step=epoch )
                wandb.log( {'acc': acc}, step=epoch )
                wandb.log( {'valid loss': valid_loss}, step=epoch )
            save_parameters( net, path=para_path )

