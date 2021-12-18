import os
import argparse
import random
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

import model
import dataset
from utils import (
    get_network, 
    get_training_dataloader, 
    get_test_dataloader, 
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
from torch.utils.tensorboard import SummaryWriter
import wandb

from conf import settings


def train(
    args,
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: _Loss,
    optimizer: Optimizer,
    scheduler: Optional[WarmUpLR]
):
    losses = 0.0
    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(dataloader):
        images: torch.Tensor
        labels: torch.Tensor

        if args.gpu:
            labels = labels.cuda(device)
            images = images.cuda(device)

        optimizer.zero_grad()
        outputs: torch.Tensor = model(images)
        # print(outputs.size(), labels.size())
        loss: torch.Tensor = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses += loss.detach().item()

        n_iter = (epoch - 1) * len(dataloader) + batch_index + 1

        last_layer = list(model.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print(
            'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tSize:({width},{height})'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(carvana_training_loader.dataset),
                height=images.size(2),
                width=images.size(3)
            ),
            end='\r'
        )

        #update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            scheduler.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()
    print('\nepoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return losses / len(carvana_training_loader)

@torch.no_grad()
def eval_training(
    args,
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: _Loss,
    tb=True
):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in dataloader:
        if args.gpu:
            images = images.cuda(device)
            labels = labels.cuda(device)

        outputs: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(device), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(carvana_test_loader),
        correct.float() / len(carvana_test_loader),
        finish - start
    ))
    print()

    #add informations to tensorboard
    # if tb:
    #     writer.add_scalar('Test/Average loss', test_loss / len(carvana_test_loader.dataset), epoch)
    #     writer.add_scalar('Test/Accuracy', correct.float() / len(carvana_test_loader.dataset), epoch)

    return correct.float() / len(carvana_test_loader)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='select gpu device')

    ''' Dataloader, Optimizer etc. Arguments '''
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-image_factor', type=float, default=1.0, help='the factor for resizing image size')
    parser.add_argument('-val_factor', type=float, default=0.2, help='Setup the factor for validation')

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
    net = get_network(args).cuda(device)
    carvana_dataset = get_dataset(
        settings.CARVANA_TRAIN_MEAN,
        settings.CARVANA_TRAIN_STD,
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
        batch_size=args.b,
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )


    # carvana_training_loader = get_training_dataloader(
    #     settings.CARVANA_TRAIN_MEAN,
    #     settings.CARVANA_TRAIN_STD,
    #     batch_size=args.b,
    #     num_workers=6,
    #     shuffle=True,
    #     pin_memory=True,
    #     scale=args.image_factor
    # )
    # carvana_test_loader = get_test_dataloader(
    #     settings.CARVANA_TRAIN_MEAN,
    #     settings.CARVANA_TRAIN_STD, 
    #     batch_size=args.b,
    #     num_workers=6,
    #     shuffle=True,
    #     pin_memory=True,
    #     scale=args.image_factor
    # )

    carvana_training_loader_dataset_size = len(carvana_training_loader.dataset)
    carvana_test_loader_dataset_size = len(carvana_test_loader.dataset)

    # loss_function = nn.CrossEntropyLoss().cuda(device)
    loss_function = nn.BCEWithLogitsLoss().cuda(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(carvana_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if (args.mbs or args.mbs_bn):
        mbs_trainer, net = MicroBatchStreaming(
            dataloader=carvana_training_loader,
            model=net,
            criterion=loss_function,
            optimizer=optimizer,
            lr_scheduler=warmup_scheduler,
            warmup_factor=args.warm,
            device_index=args.gpu_device,
            batch_size=args.b,
            micro_batch_size=args.usize,
            bn_factor=args.mbs_bn
        ).get_trainer()

    ''' Load checkpoint for training resume '''
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder: raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    ''' Use tensorboard '''
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # #since tensorboard can't overwrite old values
    # #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(
    #         settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(61,1,3,3)
    # if args.gpu:
    #     input_tensor = input_tensor.cuda(device)
    # writer.add_graph(net, input_tensor)

    ''' Create checkpoint folder to save model '''
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


    ''' Load checkpoint for training resume '''
    best_acc = 0.0
    if args.resume:
        print("Load and apply pre-trained parameters")
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    ''' Training model '''
    if (args.mbs or args.mbs_bn):
        for epoch in range(1, settings.EPOCH + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            if args.resume:
                if epoch <= resume_epoch:
                    continue

            mbs_trainer.train()
            acc = eval_training(args, epoch, carvana_test_loader, net, loss_function)
            if args.wandb:
                wandb.log( {'train loss': mbs_trainer.get_loss()}, step=epoch )
                wandb.log( {'accuracy': acc}, step=epoch )

            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
    else:
        for epoch in range(1, settings.EPOCH + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            if args.resume:
                if epoch <= resume_epoch:
                    continue

            epoch_loss = train(args, epoch, carvana_training_loader, net, loss_function, optimizer, warmup_scheduler)
            acc = eval_training(args, epoch, carvana_test_loader, net, loss_function)
            if args.wandb:
                wandb.log( {'train loss': epoch_loss}, step=epoch )
                wandb.log( {'accuracy': acc}, step=epoch )

            #start to save best performance model after learning rate decay to 0.01
            # if epoch > settings.MILESTONES[1] and best_acc < acc:
            #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            #     print('saving weights file to {}'.format(weights_path))
            #     torch.save(net.state_dict(), weights_path)
            #     best_acc = acc
            #     continue

            # if not epoch % settings.SAVE_EPOCH:
            #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            #     print('saving weights file to {}'.format(weights_path))
            #     torch.save(net.state_dict(), weights_path)

    # writer.close()