import os
import argparse
import random
import time
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

import model
import dataset
from utils import (
    DiceLoss,
    dice_loss,
    get_network,
    get_dataset,
    WarmUpLR,
    most_recent_folder, 
    most_recent_weights, 
    last_epoch, 
    best_acc_weights
)
''' Micro-Batch Streaming '''
from mbs import MicroBatchStreaming

''' Monitoring Board '''
import wandb

from conf import settings


def train(
    epoch: int, total_epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: Union[dice_loss, DiceLoss],
    optimizer: Optimizer,
):
    loss: Tensor
    dice: Tensor
    train_losses = []
    train_dices = []
    dataloader_len = len(dataloader)

    start = time.time()
    model.train()
    for idx, (inputs, masks) in enumerate(dataloader):
        inputs: Tensor = inputs.to(device)
        masks: Tensor = masks.to(device)

        optimizer.zero_grad()
        preds: Tensor = model(inputs)
        loss, dice = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        train_losses.append( loss.item() )
        train_dices.append( dice.item() )

        print(
            f"[{epoch}/{total_epoch}][{idx+1}/{dataloader_len}]  ",
            f"train loss: {(sum(train_losses)/len(train_losses)):.3f}  ",
            f"train acc: {(sum(train_dices)/len(train_dices)):.2f} %  ",
            f"image size: {inputs.size()}",
            end='\r'
        )
    finish = time.time()
    print('\nepoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return (sum(train_losses)/len(train_losses)), (sum(train_dices)/len(train_dices))


def eval_training(
    device: torch.device,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: Union[dice_loss, DiceLoss],
):
    loss: Tensor
    dice: Tensor
    test_losses = []
    test_dices = []

    model.eval()
    with torch.no_grad():
        for (inputs, masks) in dataloader:
            inputs: Tensor = inputs.to(device)
            masks: Tensor = masks.to(device)

            preds: Tensor = model(inputs)
            loss, dice = criterion(preds, masks)

            test_losses.append(loss.item())
            test_dices.append(dice.item())

    print(
        f"train loss: {(sum(test_losses)/len(test_losses)):3f}  ",
        f"train acc: {(sum(test_dices)/len(test_dices)):.2f} %  "
        f"image size: {inputs.size()}"
    )

    return (sum(test_losses)/len(test_losses)), (sum(test_dices)/len(test_dices))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='select gpu device')

    ''' Dataloader, Optimizer etc. Arguments '''
    parser.add_argument('-b', type=int, default=8, help='batch size for train dataloader')
    parser.add_argument('-tb', type=int, default=1, help='batch size for test dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-image_factor', type=float, default=1.0, help='the factor for resizing image size')
    parser.add_argument('-val_factor', type=float, default=0.1, help='Setup the factor for validation')

    ''' Micro-Batch Streaming Arguments '''
    parser.add_argument('-mbs', action='store_true', default=False, help='training model with MBS')
    parser.add_argument('-mbs_bn', action='store_true', default=False, help='training model with MBS-BN')
    parser.add_argument('-usize', type=int, default=0, help='setting micro-batch size')

    ''' WandB Arguments '''
    parser.add_argument('-wandb', action='store_true', default=False, help='Using Wandb for monitoring')

    args = parser.parse_args()
    _device = 'cuda:' + str(args.gpu_device)

    ''' Setup WandB '''
    if args.wandb:
        name = None
        tags = []

        if args.mbs:
            name = f'{args.net}(w/ MBS)'
            tags.append( f'usize {args.usize}' )
        elif args.mbs_bn:
            name = f'{args.net}(w/ MBS-BN)'
            tags.append( f'usize {args.usize}' )
        else:
            name = f'{args.net}(w/ baseline)'

        tags.append( f'batch {args.b}' )
        tags.append( f'image {args.image_factor}' )
        tags.append( f'carvana')

        wandb.init(
            project='mbs_paper_results',
            entity='xypiao97',
            name=f'{name}',
            tags=tags,
        )

    random_seed = 42

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(random_seed)

    device = torch.device(_device)

    carvana_dataset = get_dataset(
        settings.CARVANA_MEAN,
        settings.CARVANA_STD,
        scale=args.image_factor
    )
    n_val = int( len(carvana_dataset) * args.val_factor )
    n_train = len(carvana_dataset) - n_val
    train_dataset, val_dataset = random_split(
        carvana_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(random_seed)
    )

    carvana_training_loader = DataLoader(
        train_dataset,
        batch_size=args.b,
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )

    carvana_test_loader = DataLoader(
        val_dataset,
        batch_size=args.tb,
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )

    net = get_network(args).to(device)
    # loss_function = dice_loss
    loss_function = DiceLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if (args.mbs or args.mbs_bn):
        mbs_trainer, net = MicroBatchStreaming(
            dataloader=carvana_training_loader,
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
            mbs_trainer.train()
            val_loss, val_dice = eval_training(
                device=device,
                dataloader=carvana_test_loader,
                model=net,
                criterion=loss_function
            )
            if args.wandb:
                wandb.log( {'train loss': mbs_trainer.get_loss()}, step=epoch )
                # wandb.log( {'accuracy': acc}, step=epoch )
    else:
        for epoch in range(1, settings.EPOCH + 1):
            train_loss, train_dice = train(
                epoch=epoch, total_epoch=settings.EPOCH,
                device=device,
                dataloader=carvana_training_loader,
                model=net,
                criterion=loss_function,
                optimizer=optimizer
            )
            val_loss, val_dice = eval_training(
                device=device,
                dataloader=carvana_test_loader,
                model=net,
                criterion=loss_function
            )
            if args.wandb:
                wandb.log( {'train loss': train_loss}, step=epoch )
                wandb.log( {'train acc': train_dice}, step=epoch )
                wandb.log( {'val loss': val_loss}, step=epoch  )
                wandb.log( {'val acc': val_dice}, step=epoch )
